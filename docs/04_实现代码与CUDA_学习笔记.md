# 04 实现代码与 CUDA 学习笔记

本文说明两件事：这套实现如何映射到 GPU 上执行，以及渲染与安全过滤算法在代码中的逐层实现。所有结论都以 `upstream/` 下的真实源码为依据，代码片段标注文件与行号；可选的验证片段均可直接运行。

仓库是论文（Tong, Dawson, Fan, *Enforcing Safety for Vision-based Controllers via Control Barrier Functions and Neural Radiance Fields*, ICRA 2023）的忠实复现。`upstream/` 为作者公开代码，除一处外未修改：`upstream/single_cbf.py` 第 199 行 `cv2.VideoWriter` 的输出分辨率由 `(240,68)` 改为 `(2400,680)`。该改动只影响视频拼接尺寸，不进入任何 GPU 计算。

---

## 目录

1. [读者定位与前置知识](#1-读者定位与前置知识)
2. [CUDA 执行模型基础](#2-cuda-执行模型基础)
3. [PyTorch 与 CUDA 的对应关系表](#3-pytorch-与-cuda-的对应关系表)
4. [异步执行与同步](#4-异步执行与同步)
5. [显存管理与 OOM 处理](#5-显存管理与-oom-处理)
6. [主机-设备传输与批处理](#6-主机-设备传输与批处理)
7. [渲染算法逐层拆解](#7-渲染算法逐层拆解)
8. [一步决策的完整计算图与 Profiler 验证](#8-一步决策的完整计算图与-profiler-验证)
9. [修改实践：改动清单与约束](#9-修改实践改动清单与约束)
10. [自测题](#10-自测题)
11. [参考资料](#11-参考资料)

---

## 1 读者定位与前置知识

### 1.1 目标读者

本笔记面向满足以下条件的读者：能用 Python 写循环、函数、类；用过 PyTorch 的 `Tensor`、`nn.Module`、`torch.no_grad()`；知道 `x.to('cuda')` 表示把张量放到 GPU 上；未写过 CUDA C/C++，不知道 kernel、block、warp 的确切含义。

读完应能做到：

1. 说出本仓库每一步决策在 GPU 上发生了什么类型的并行工作，以及瓶颈是算力还是显存带宽。
2. 解释为什么本仓库的时间统计天然包含同步点，从而不会误判"GPU 很快"或"CPU 是瓶颈"。
3. 在 8 GB 显存约束下，通过三个层级的参数把一次渲染的显存降下来。
4. 在源码中定位渲染的四个阶段，并说明每个阶段的张量形状与数值语义。
5. 修改 CBF 定义、动力学、采样策略、分辨率、批大小，并知道每项改动引入的约束。

### 1.2 前置知识清单

| 主题 | 需要的程度 | 自检方式 |
| --- | --- | --- |
| Python | 能读类继承与 `while` 循环 | 读懂 `single_cbf.py:104-149`（`find_safe_action`） |
| PyTorch 张量 | 形状、广播、`reshape`/`squeeze`、`.to(device)` | 说出 `(B*H*W, N_samples, 3)` 中三个维度的含义 |
| PyTorch 模块 | `nn.Module`、`nn.ModuleList`、`forward`、`state_dict` | 读懂 `decoder.py:147-150` 的层构造 |
| 相机模型 | 针孔模型、内参 `fx, fy, cx, cy`、相机坐标系沿 −z 观察 | 读懂 `common.py:250-252` 的射线方向构造 |
| 体渲染 | 沿射线采样、α 合成、期望深度 | 读懂 `common.py:211-234` |
| 控制理论 | 离散时间 CBF 条件 `h_{t+1} ≤ α h_t` | 读懂 `single_cbf.py:118` |

不需要的前置知识：CUDA C 编程、cuBLAS、`__shared__` 变量、CUDA 流与图的 API。本仓库没有自定义 CUDA kernel，全部计算由 PyTorch 算子发起。

### 1.3 环境与实测基线数据

实测环境：RTX 5070 Laptop（8 GB 显存）/ torch 2.11.0+cu128 / CUDA 12.8 / Windows 11。

GPU 硬件参数（`torch.cuda.get_device_properties(0)` 读出，下文会用到）：

| 项 | 值 | 项 | 值 |
| --- | --- | --- | --- |
| 流多处理器（SM）数量 | 36 | 每 SM 寄存器数 | 65536 |
| 每 SM 最大驻留线程 | 1536 | 每 block 可用共享内存 | 49152 B |
| warp 大小 | 32 | 每 SM 共享内存 | 102400 B |
| L2 缓存 | 32 MB | 显存容量 | 8150 MiB |

本文引用的实测数据（同一设备、同一环境）：

| 量 | 值 |
| --- | --- |
| 特征网格 `grid_coarse` | `(1, 32, 6, 8, 9)` |
| 特征网格 `grid_middle` | `(1, 32, 18, 28, 31)` |
| 特征网格 `grid_fine` | `(1, 32, 37, 56, 62)` |
| 特征网格 `grid_color` | `(1, 32, 37, 56, 62)` |
| 四个网格元素总数 | 约 8.7 M 个 float32 |
| 四个网格显存占用 | 约 35.5 MB（含解码器参数 0.26 MB） |
| 控制环分辨率 | 120×68 = 8,160 条射线 |
| 每步评估候选数 | 10 个 ⇒ 81,600 条射线 |
| 每条射线采样点 | 32 ⇒ 单步 2,611,200 个空间点 |
| 单步耗时：1 个位姿（261,120 点） | 约 0.012 s |
| 单步耗时：10 个位姿一次性批量（2,611,200 点） | 约 0.103 s |
| 单步耗时：循环渲染 10 次单位姿 | 约 0.121 s |
| 仅 `grid_sample + MLP` 前向（2,611,200 点） | 约 0.101 s |
| 展示渲染 1200×680 两趟 | 约 7 s/帧 |
| `upstream/single_cbf.py` 跑 150 帧 | 约 20 分钟 |
| `upstream/double_cbf.py` 跑 100 步 | 约 12 分钟 |

由表中两行可读出：控制环的安全判定（0.103 s/步）不是仿真步的主要开销，展示渲染（7 s/帧）才是。这一结论在 §8.4 会有耗时构成上的对应证据。

启动方式与 `torch.load` 的前置条件：`upstream/single_cbf.py:52` 用 `torch.load('replica_room1.pth')` 载入整个模块对象，torch ≥ 2.6 的 `weights_only=True` 默认值会拒绝该调用。仓库根目录的 `run_upstream.ps1` 通过环境变量 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` 绕开该限制，不修改上游代码。本文所有验证片段都在该环境变量下运行。

### 1.4 代码地图

| 文件 | 职责 | 是否在控制环热点路径上 |
| --- | --- | --- |
| `upstream/single_cbf.py` | 单积分器实验主程序、CBF 求解、视频输出 | 是（外层循环） |
| `upstream/double_cbf.py` | 双积分器实验主程序 | 是（外层循环） |
| `upstream/src/utils/Renderer.py` | 射线→深度/色彩/不确定度，两级批处理 | 是 |
| `upstream/src/common.py` | 射线生成、体渲染合成、重要性采样、坐标归一化 | 是 |
| `upstream/src/conv_onet/models/decoder.py` | 特征网格采样 + 位置编码 + MLP，四个阶段 | 是 |
| `upstream/src/NICE_SLAM.py` | 装载预训练解码器、生成特征网格、设定场景包围盒 | 否（仅启动） |
| `upstream/src/utils/datasets.py` | 数据集读取（本实验只用第 0 帧做形状校验） | 否（仅启动） |
| `upstream/configs/nice_slam.yaml` | 渲染参数、相机内参、`c_dim`、`grid_len` | 是（参数来源） |

设备选择在本仓库中的实现方式是读取工作目录下的 `device.txt`，出现在四处：`single_cbf.py:23-24`、`double_cbf.py:19-20`、`src/common.py:5-6`、`src/conv_onet/models/decoder.py:6-7`。这四处是模块级代码，因此必须以 `upstream/` 为工作目录导入，否则 `open('device.txt')` 失败。四个模块各读一次，值均为 `cuda:0`。

### 1.5 阅读约定

- 张量形状一律写成 `(d0, d1, ...)`，float32 写 `f32`、float64 写 `f64`。
- "点"指体渲染采样点（`N_points`），"射线"指一次渲染的采样射线（`N_rays`）。
- 行号以当前仓库文件为准，格式为 `文件:起-止`。
- 标注"可运行"的代码片段假定当前工作目录为 `upstream/`，且已设置 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`。
- 片段中出现变量 `r`、`c`、`dec`、`poses`、`ro1`、`rd1` 时，表示已按 §8.6 的脚本完成初始化：`r` 为 `Renderer`、`c` 为四个特征网格的字典、`dec` 为载入的 `NICE` 模块、`poses` 为 `(10, 4, 4)` 的位姿批。该脚本同时打印控制环一步的全部形状，建议先运行一次。

---

## 2 CUDA 执行模型基础

### 2.1 结论先行

本仓库不含任何 `.cu` 文件或自定义 kernel。GPU 上的全部工作由 PyTorch 算子发起：张量算子调用在 GPU 上展开为一次或多次 kernel 启动，kernel 的并行度由输出元素数决定。因此"读 CUDA"在这里等于读懂三件事：一次算子展开成多少 kernel、每个 kernel 的线程如何组织、限制性能的是算力还是显存带宽。

### 2.2 kernel 与线程组织

**kernel（核函数）**：在 GPU 上由大量线程并行执行的函数，是 GPU 调度与资源分配的单位。一次 kernel 启动由主机侧发起，包含参数传递、网格配置、调度入队。

**thread（线程）**：最小执行单元，拥有自己的寄存器与栈。

**block（线程块）**：一组线程，被调度到同一个 SM 上执行，可通过共享内存交换数据，可通过块内同步等待彼此。

**grid（网格）**：一次 kernel 启动内所有 block 的集合。

**warp（线程束）**：block 内 32 个连续的线程，是硬件实际执行的粒度。一个 warp 内的 32 个线程在同一时刻执行同一条指令，操作不同数据（SIMT，Single Instruction Multiple Threads）。出现分支时，若 warp 内线程走不同路径，两条路径被串行执行，两个分支都不空转的部分被屏蔽，这条规则称为分支发散（branch divergence）。

在这份代码里对应的具体对象：

| CUDA 概念 | 本仓库中的对应物 |
| --- | --- |
| kernel | 一次 `F.grid_sample`、一次 `nn.Linear` 的前向、一次 `torch.sigmoid`，各对应一次或多次 kernel 启动 |
| thread | 处理一个采样点的某一个输出元素；例如 `grid_sample` 输出 `(1,32,500000,1,1)`，即约 1.6×10⁷ 个 f32 输出元素 |
| block | 由 PyTorch 的算子实现内部决定（elementwise 类通常每 block 128~512 线程），用户不可见 |
| warp | 32 个连续输出元素被同一 warp 处理；`(N, 32)` 的激活张量在最后一维恰好是 32，一个 warp 覆盖一个点的全部 32 个通道 |
| grid | 一次 kernel 的 block 总数，等于输出元素数除以每线程元素数再除以每 block 线程数 |
| shared memory | 不出现在 Python 代码中；由 cuBLAS 矩阵乘与 `grid_sample` 的插值实现内部使用 |
| 同步 | 块内同步在算子内部；跨 kernel 的依赖由同一条 CUDA 流上的顺序执行保证 |

SIMT 的一个直接后果：融合后的逐元素算子（`relu`、`sigmoid`、加减乘）对 32 个连续元素的处理是"一个 warp 一条指令"，因此这类算子的成本几乎只与访存字节数有关，与指令数无关。`decoder.py:193` 的 `F.relu(h)` 与 `decoder.py:195` 的 `h + self.fc_c[i](c)` 都属于这类算子，其开销在 §3 的实测比例中体现。

### 2.3 显存层次与访问代价

GPU 的存储层次（由快到慢、由小到大）：

| 层次 | 容量级 | 访问延迟级 | 归属 | 本仓库中的内容 |
| --- | --- | --- | --- | --- |
| 寄存器（register） | 每线程数十~255 个 | 1 周期 | 线程私有 | MLP 每层累加中间值；由编译器分配，Python 不可见 |
| 共享内存（shared memory） | 每 block 48 KiB（本机上限） | 20~30 周期 | block 共享 | `grid_sample` 的三线性插值 tile、矩阵乘的分块；由算子实现内部使用 |
| L1/L2 缓存 | L2 = 32 MB（本机） | 约 200 周期（L2） | SM 组共享 | 4 个特征网格（35.5 MB）与权重；`grid_sample` 的邻域访问大量命中 L2 |
| 显存（global memory） | 8 GB（本机） | 400~800 周期 | 全设备 | 全部张量：网格、采样点、激活、深度图、`video.avi` 的像素缓冲区（在 CPU 侧） |

由此得到两条判据，用于 §3 与 §5：

1. 若某算子的输入输出字节数远大于其浮点运算数（算术强度低），它受 HBM 带宽限制，增加算力无用，只能减少数据搬运。
2. 若某算子的工作集超过 L2（32 MB），每次重读都落到 HBM；本仓库 4 个特征网格合计约 35 MB，恰好在 L2 容量附近，因此 `grid_sample` 的命中率对插值顺序敏感。

### 2.4 kernel 启动开销

**kernel 启动开销**指主机发起一次 kernel 到设备开始执行之间的固定成本，含参数打包、驱动提交、调度。它与 kernel 的工作量无关，因此在小张量上占主导，在大张量上可忽略。

在本仓库中，一次 `predict_observation` 调用展开的算子数量在数百量级（5 层 MLP × 6 个点分块 × 每次前向各自的 `.to()` 检查、掩码、切片、拼接、归约）。设单次启动的固定成本为 `c`、算子数为 `K`，则固定开销为 `K·c`。本仓库的 `K` 约为 10²，`c` 取 10 μs 时该项为毫秒量级，`c` 取数十微秒时可达十余毫秒。§2.4 的测量片段给出本机 `c` 的上界，§6.3 的实测差值（0.121 s → 0.103 s）是该量级下的实际收益。

可运行的测量片段（当前工作目录 `upstream/`）：

```python
import torch

x = torch.ones(1, device='cuda:0')
big_a = torch.ones(4 * 1024 * 1024, device='cuda:0')
big_b = torch.ones(4 * 1024 * 1024, device='cuda:0')


def per_call_ms(fn, n):
    for _ in range(200):        # 预热，触发 kernel 模块加载与缓存
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(n):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n


t_view = per_call_ms(lambda: x.view(1), 5000)          # 不产生 kernel
t_tiny = per_call_ms(lambda: x + x, 5000)             # 1 元素 kernel
t_4m = per_call_ms(lambda: big_a + big_b, 200)        # 4M 元素 kernel
print(f"view        {t_view*1000:7.2f} us  (无 kernel)")
print(f"1 元素 add  {t_tiny*1000:7.2f} us")
print(f"4M 元素 add {t_4m*1000:7.2f} us")
print(f"1 元素 add 的固定成本上界 {t_tiny*1000 - t_view*1000:.2f} us")
```

本机在存在并发 GPU 负载的条件下测得：`view` 约 4.3 μs，1 元素 `add` 约 50 μs，4M 元素 `add` 约 333 μs。`view` 不产生 kernel，其时间即框架侧的调度成本；`add` 与 `view` 的差值给出单次 kernel 提交的上界。Windows 的 WDDM 提交路径比 Linux 更长，读者应在自己的机器上复测，而不是引用绝对数值。

### 2.5 一次 PyTorch 算子 = 一次或多次 kernel 启动

判定方法：数输出元素、数中间张量。以 `common.py:260-273` 的 `normalize_3d_coordinate` 为例，一次 Python 调用产生：

```python
# upstream/src/common.py:260-273
def normalize_3d_coordinate(p, bound):
    p = p.reshape(-1, 3)
    p[:, 0] = ((p[:, 0]-bound[0, 0])/(bound[0, 1]-bound[0, 0]))*2-1.0
    p[:, 1] = ((p[:, 1]-bound[1, 0])/(bound[1, 1]-bound[1, 0]))*2-1.0
    p[:, 2] = ((p[:, 2]-bound[2, 0])/(bound[2, 1]-bound[2, 0]))*2-1.0
    return p
```

分解到 kernel 层：

| 步骤 | 张量 | kernel |
| --- | --- | --- |
| `p.reshape(-1, 3)` | 连续张量的视图 | 0（无拷贝） |
| `p[:, 0]-bound[0,0]` 等三则运算 | 每个坐标轴 4 个临时 `(N,)` 或 `(N,1)` 张量 | 每个坐标轴 4 次逐元素 kernel |
| `p[:, 0] = ...` | 写回视图 | 1 次 `copy_`（或 `index_put_`） |
| 三个坐标轴合计 | — | 约 12~15 次 kernel |

同一函数在 `decoder.py:167` 与 `decoder.py:252` 调用时，前面还有 `p.clone()`（`copy_`，1 次 kernel，字节数 = 3×N×4 或 ×8）。因此"一次坐标归一化"对应十余次 kernel 启动。若 N = 500,000，每次 kernel 处理约 0.5 M 个元素，单次启动开销被摊薄，此处不是瓶颈；若 N 降到数百至数千（例如调试时只查询少量射线），固定开销占比上升。

`nn.Linear` 与 `F.grid_sample` 相反：一次调用只产生 1~2 次 kernel，但每次 kernel 内部有完整的分块、共享内存、向量化，属于"高吞吐 kernel"。

对读者的可检验推论：对同一份数据，把 10 次 `(261120, 3)` 的 `normalize_3d_coordinate` 换成 1 次 `(2611200, 3)`，GPU 上的浮点工作量完全相同，但 kernel 启动次数从约 150 降到约 15，wall time 差异即固定开销的规模。§6.6 给出可运行的验证片段。

### 2.6 验证

```python
# 1) 确认 PyTorch 看到的设备与硬件参数
import torch
p = torch.cuda.get_device_properties(0)
print(p.name, p.multi_processor_count, p.max_threads_per_multi_processor,
      p.shared_memory_per_block, p.shared_memory_per_multiprocessor,
      p.regs_per_multiprocessor, p.L2_cache_size / 1024 / 1024, "MB L2")

# 2) 用 profiler 观察一次算子的 kernel 数量（CUDA trace 是否可用见 §8.4）
from torch.profiler import profile, ProfilerActivity
x = torch.rand(1000, 3, device='cuda:0')
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    y = (x - 0.5) / 2 * 3 - 1
for e in prof.key_averages():
    if e.count:
        print(f"{e.count:4d}  {e.key[:60]}")
```

预期：CPU 侧出现 `aten::sub` 2 次、`aten::div` 1 次、`aten::mul` 1 次（本机实测的计数），对应设备侧 4 个逐元素 kernel（eager 模式不做算子融合，每个 Python 算子一次 kernel）。若设备侧条目为空，说明 CUPTI 不可用，见 §8.4。若启用 `torch.compile`，四个算子可能被融合，设备条目数随之下降。

---

## 3 PyTorch 与 CUDA 的对应关系表

### 3.1 结论

本仓库的热点路径只用到五类算子：`F.grid_sample`、`nn.Linear`（GEMM）、逐元素算子、归约算子、`torch.cat` 与形状变换。据此可以判断任一步骤的瓶颈类型，而不需要读 CUDA 源码。

### 3.2 总表

| 算子 | 代码位置 | GPU 侧工作 | 单次调用 kernel 数 | 主要限制 | 本仓库规模 |
| --- | --- | --- | --- | --- | --- |
| `F.grid_sample` | `decoder.py:171-172` | 输出 `(1, 32, N, 1, 1)` 的 f32；每输出元素做 8 邻域读、权重加权求和 | 1（5 维输入按三线性插值） | HBM/L2 随机读；算术强度约 32 FLOP / 128 B | N = 500,000 每分块 |
| `nn.Linear` | `decoder.py:192`、`decoder.py:195`、`decoder.py:198` | `(N, 32) × (32, 32)` 矩阵乘，cuBLAS 实现 | 1~2 | K=32 很小，算术强度约 32 FLOP / 元素；访存受限 | 6 次/分块（5 层 + 输出层） |
| 位置编码 `sin(x @ B)` | `decoder.py:29-30` | `(N, 3) × (3, 93)` GEMM + `sin` | 2 | GEMM 访存受限，`sin` 为逐元素 | 每分块 1 次，输出 `(N, 93)` |
| 逐元素算子（`relu`、`sigmoid`、`+`、`*`、`-`） | `common.py:222`、`common.py:269-272`、`decoder.py:193`、`decoder.py:195` | 每元素 1~2 FLOP，读写 4~8 B | 每个 Python 算子 1 次 | 纯 HBM 带宽 | `(N, 32)`、`(N, 125)`、`(N_rays, 32)` |
| 归约（`.min()`、`torch.sum`、`torch.cumprod`、`torch.norm`） | `single_cbf.py:116`、`single_cbf.py:138`、`common.py:229-234` | 块内共享内存树形归约 + 跨块第二 kernel | 2 次或更多 | 两趟读写；第二趟并行度低（输出仅 `(N_rays,)`） | `(81600, 32)` → `(81600,)` |
| `torch.cat` | `common.py:215-216`、`decoder.py:185`、`decoder.py:197`、`Renderer.py:170`、`Renderer.py:186`、`Renderer.py:250-252`、`Renderer.py:289-290`、`Renderer.py:318-320` | 分配目标张量 + 分段拷贝 | 每段 1 次拷贝 | HBM 带宽（读 2×、写 1×） | 见 §3.4 的 `(N, 125)` 例 |
| `reshape` / `squeeze` / `unsqueeze` / `transpose` | `Renderer.py:174`、`Renderer.py:177`、`decoder.py:178` | 连续张量为视图；非连续时触发拷贝 | 0 或 1 | 无或带宽 | 大量出现，多数为 0 |
| 布尔掩码赋值 | `Renderer.py:44-47`、`Renderer.py:58` | `nonzero` 求索引 + `index_put_` 写回 | 2 | 两趟遍历，索引张量为 int64 | 每分块 3 个掩码 |
| `.to(device)`（同设备） | `decoder.py:171-172`、`decoder.py:189`、`decoder.py:192` | CPU 侧设备比较，不发生任何 GPU 工作 | 0 | 每次调用数微秒的 Python 开销 | 每层每分块调用一次 |
| `.double()` / `.float()` | `Renderer.py:246-247`、`Renderer.py:132`、`decoder.py:187` | 类型转换 kernel | 1 | 带宽（读 4 B 写 8 B） | `(81600,)` 与 `(N, 3)` |

### 3.3 三条判据

1. **算术强度**（每字节访存对应的浮点运算数）低于约 10 的算子受带宽限制：`nn.Linear`（K=32）、全部逐元素算子、`torch.cat`。提高这些算子的吞吐只能减少数据量或减少中间张量数量。
2. `F.grid_sample` 的访问模式是"每点 8 邻域、跨 32 个通道"，通道维连续存储，邻域在空间维跳变。它的效率取决于 L2 命中率，而 4 个网格合计 35.5 MB 与 32 MB 的 L2 同量级，所以它在 2,611,200 点的规模下不是最慢的部分（见 §3.4）。
3. 归约算子的第二次 kernel 输出规模很小（`(81600,)` = 326 KB），SM 利用率低，但它只出现十余次，不属于热点。

### 3.4 实测比例与本仓库的规模

用合成数据在真实形状上分别计时（`N_total = 2,611,200`，`chunk = 500,000`，与 `points_batch_size` 默认值一致），同一进程内重复三轮，相对比例稳定：

| 阶段 | 占 `grid_sample + MLP` 前向时间的比例 |
| --- | --- |
| `normalize_3d_coordinate`（+ `clone`） | 约 2% |
| `F.grid_sample`（middle 网格） | 约 13% |
| 位置编码 + 5 层 MLP + 特征相加 + skip 拼接 + 输出层 | 约 85% |

结论：前向的绝大部分时间在 MLP 的逐层读写上，而不在特征网格插值上。机制解释：每层 MLP 的输入输出都是 `(500000, 32)` 的 f32，即 64 MB，5 层加上 `relu`、特征相加、skip 拼接共约 15 次对同量级张量的读写；`grid_sample` 虽然做 8 邻域插值，但输出只有 `(1, 32, 500000, 1, 1)`，且邻域访问大量命中 L2。该比例解释了 §5 中峰值显存由激活而非模型主导，也解释了 §9 中"减少 MLP 层数或宽度"是提速的主要手段。

`torch.cat` 的具体代价举例：`decoder.py:197` 的 `h = torch.cat([embedded_pts, h], -1)` 在 N = 500,000 时分别读取 `(500000, 93)`（186 MB）与 `(500000, 32)`（64 MB），写出 `(500000, 125)`（250 MB），总搬运 500 MB，只出现在第 3 层之后，因此 `skips=[2]` 的取值直接影响访存量。

### 3.5 验证

```python
# 统计一次控制步里的算子调用次数（需先按 §8.6 完成初始化）
calls = {}
for name in ('middle_decoder', 'fine_decoder', 'color_decoder', 'coarse_decoder'):
    m = getattr(dec, name, None)
    if m is None:
        continue
    calls[name] = 0

    def hook(mod, inp, out, name=name):
        calls[name] += 1

    m.register_forward_hook(hook)

for stage in ('middle', 'fine', 'color'):
    calls.update({k: 0 for k in calls})
    with torch.no_grad():
        r.render_batch_img(c, dec, poses[:1], 'cuda:0', stage=stage, gt_depth=None)
    print(f'{stage:7s} -> {calls}')
```

预期：`middle` 阶段只有 `middle_decoder` 被调用（每点分块 1 次）；`fine` 与 `color` 阶段每次 `eval_points` 调用分别触发 2 个与 3 个解码器（`fine` 除自身网格外还要取 `middle` 网格特征，见 `decoder.py:180-185`），因此调用次数还随点分块数增加。

---

## 4 异步执行与同步

### 4.1 定义

**异步执行**：PyTorch 的张量算子调用只把工作提交到当前 CUDA 流（stream）就返回，主机线程不等待设备完成。返回的张量是"尚未算完"的句柄。

**CUDA 流**：设备上按顺序执行的操作队列。同一流内保序，不同流可并发。PyTorch 默认使用默认流；本仓库未创建任何自定义流，全部工作在同一条流上串行。

**同步**：主机等待设备把队列中的先前工作执行完。等价手段有 `torch.cuda.synchronize()`、`tensor.cpu()`、`tensor.item()`、`float(tensor)`、`print(tensor)`、以及任何依赖张量值的 Python 控制流（`if tensor <= x:`）。

**为什么计时必须同步**：若在提交后立刻读取 `time.time()`，测到的是"提交 200 次 kernel 的耗时"，不是"执行 200 次 kernel 的耗时"。二者相差可达一个数量级。

### 4.2 本仓库的计时为何天然包含同步点

`upstream/single_cbf.py:279-297` 的主循环每步都包含以下必然的同步点：

| 行号 | 代码 | 同步原因 |
| --- | --- | --- |
| `single_cbf.py:118` | `if new_h <= alpha * h:` | 0 维张量参与 Python 布尔判断，必须取值到主机 |
| `single_cbf.py:140` | `if batch_new_h[j] <= alpha * h:` | 同上；`j` 循环 10 次，即 10 次同步 |
| `single_cbf.py:141` | `torch.norm(best_action, p=2) == 0` | 归约结果的 0 维张量参与判断 |
| `single_cbf.py:144` | `if torch.norm(best_action, p=2) > 0:` | 同上 |
| `single_cbf.py:146` | `print('intervention =', float(...))` | `float()` 强制 D2H 并等待 |
| `single_cbf.py:297` | `print('min_depth = {}'.format(depth.min()))` | 格式化 0 维张量触发取值 |
| `single_cbf.py:295-296` | `np.hstack([... .cpu().numpy(), ...])` | 整幅深度图与彩色图 D2H |
| `single_cbf.py:116` | `robot.predict_observation(new_pose).min()` | 归约本身异步，但结果在 `:118` 被取值，等价于同步 |

因此 `single_cbf.py` 打印的 `time cost` 与外部用 `time.time()` 包裹整个 step 得到的数值都包含完整执行时间。这一性质在 `double_cbf.py:211` 与 `double_cbf.py:225` 之间同样成立。

反过来，当需要单独测量 GPU 段时（例如区分"渲染"与"网格采样+MLP"），必须显式同步或用事件计时，否则数字无意义。§4.5 与 §8.5 给出正确写法。

### 4.3 计时必须同步的两种正确写法

```python
import time
import torch

# 写法 A：同步后计时（含主机开销，适合测一个完整 Python 函数）
torch.cuda.synchronize()
t0 = time.time()
out = r.render_batch_img(c, dec, poses, 'cuda:0', stage='middle', gt_depth=None)
torch.cuda.synchronize()
print('wall time per step', time.time() - t0)

# 写法 B：CUDA 事件计时（只测设备侧时间，不含 Python 提交间隔）
e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
e0.record()
out = r.render_batch_img(c, dec, poses, 'cuda:0', stage='middle', gt_depth=None)
e1.record()
torch.cuda.synchronize()
print('device time per step', e0.elapsed_time(e1) / 1000)
```

两者的差值即主机侧的提交开销。在本仓库的实测数据中，10 个位姿一次性批量的 wall time 为 0.103 s，而"仅 `grid_sample + MLP` 前向"为 0.101 s，说明该步几乎全部时间在设备侧，主机只贡献约 2 ms。

### 4.4 每步的同步点数量与影响

以 `single_cbf.py` 的 `while True` 分支为例，一次循环迭代（10 个候选）引入的显式同步点：`batch_new_h[j]` 的判断 10 次、`torch.norm` 判断 2~3 次、`best_action` 打印 1 次。若候选未被接受，迭代继续，同步点随之累加。

同步点本身的开销来自三部分：等待设备排空、D2H 拷贝、主机唤醒。在本机（Windows WDDM）上，单次同步与 §2.4 测得的提交成本同量级（数十微秒），因此每步十余次同步只贡献 0.1~0.5 ms，相对 0.103 s 可忽略。但在把候选数从 10 提升到 100 或把 MLP 换小之后，这一项会成为不可忽略的部分，因为 GPU 段缩短而同步点数量随候选数线性增长。

### 4.5 验证

```python
import time
import torch

# 同一段工作，三种计时方式的差异
def bench(fn, reps=5):
    fn(); torch.cuda.synchronize()

    t0 = time.time()                       # 不同步
    for _ in range(reps):
        fn()
    t_nosync = (time.time() - t0) / reps

    torch.cuda.synchronize()
    t0 = time.time()                       # 同步
    for _ in range(reps):
        fn()
        torch.cuda.synchronize()
    t_sync = (time.time() - t0) / reps

    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        fn()
    e1.record()
    torch.cuda.synchronize()
    t_event = e0.elapsed_time(e1) / reps / 1000

    print(f'异步提交 {t_nosync*1000:7.1f} ms   逐步同步 {t_sync*1000:7.1f} ms   事件计时 {t_event*1000:7.1f} ms')


bench(lambda: r.render_batch_img(c, dec, poses[:1], 'cuda:0', stage='middle', gt_depth=None))
```

预期：异步提交时间最短（只计提交），逐步同步时间最长（含 5 次排空与主机唤醒），事件计时接近真实设备时间。若异步提交显著小于事件计时，说明设备侧工作量大于主机提交成本；若两者接近，说明该段受主机侧限制。

---

## 5 显存管理与 OOM 处理

### 5.1 本仓库的显存构成

显存占用分四部分，量级从大到小：

| 部分 | 内容 | 大小（本机实测/推算） |
| --- | --- | --- |
| 模型常量 | 4 个特征网格 + 4 个解码器参数 | 35.5 MB（网格 34.9 MB + 解码器 0.26 MB） |
| 采样点与中间激活 | 点坐标、`grid_sample` 输出、MLP 各层激活、skip 拼接结果 | 主导项，随 `points_batch_size` 线性变化 |
| 逐射线张量 | `z_vals`、`alpha`、`weights`、`depth_map`、`depth_var`、`rgb` | 与 `N_rays × N_samples` 成正比，f64 时为 8 B/元素 |
| 分配器缓存 | PyTorch caching allocator 保留的空闲块 | 可达活跃量的 1.3~1.5 倍（实测 reserved / allocated 约 1.3~1.4） |

关键实测：控制环 10 位姿一次性渲染（2,611,200 点，`points_batch_size = 500000`）的峰值分配约 0.7 GB，保留量约 1.0 GB；单位姿（261,120 点）峰值约 0.33 GB；展示分辨率 1200×680 的两趟峰值分配约 0.9 GB 与 1.1 GB，保留量最高约 1.5 GB。可见模型常量只占 35.5 MB，其余全部是激活与逐射线张量。

`points_batch_size` 对峰值显存的实测影响（10 位姿，2,611,200 点）：

| `points_batch_size` | 点分块数 | 峰值分配 | 峰值保留 |
| --- | --- | --- | --- |
| 125,000 | 21 | 311 MB | 410 MB |
| 250,000 | 11 | 446 MB | 664 MB |
| 500,000 | 6 | 716 MB | 964 MB |
| 1,000,000 | 3 | 1243 MB | 1686 MB |

四行的耗时都在 0.27~0.30 s 之间，说明在该规模下减少分块数不带来时间收益，代价与收益都体现在显存上。同一组取值在单位姿（261,120 点）下失效：取值 ≥ 261,120 时点分块恒为 1，峰值稳定在约 0.33 GB。

激活量的计算方式：一个分块内 N = 500,000 时，位置编码输出 `(500000, 93)` f32 = 186 MB，第 3 层后的 skip 拼接 `(500000, 125)` f32 = 250 MB，`(500000, 32)` 的激活为 64 MB。因此在 8 GB 卡上，`points_batch_size = 500000` 对应约 0.3~0.7 GB 的峰值，把该值提高到 2,000,000 会使 skip 拼接达到 1 GB 且同时存在多个同量级临时张量。

### 5.2 两级切块的代码

**第一级：射线级切块（`ray_batch_size`）**。`Renderer.py:7-9` 设定默认值，`Renderer.py:233-242` 与 `Renderer.py:302-311` 按该值切分射线。

```python
# upstream/src/utils/Renderer.py:7-9
def __init__(self, cfg, args, slam, points_batch_size=500000, ray_batch_size=100000):
    self.ray_batch_size = ray_batch_size
    self.points_batch_size = points_batch_size
```

```python
# upstream/src/utils/Renderer.py:233-242
ray_batch_size = self.ray_batch_size
if gt_depth is not None:
    gt_depth = gt_depth.reshape(-1)

for i in range(0, rays_d.shape[0], ray_batch_size):
    rays_d_batch = rays_d[i:i+ray_batch_size]
    rays_o_batch = rays_o[i:i+ray_batch_size]
    if gt_depth is None:
        ret = self.render_batch_ray(
            c, decoders, rays_d_batch, rays_o_batch, device, stage, gt_depth=None)
    else:
        gt_depth_batch = gt_depth[i:i+ray_batch_size]
```

**第二级：点级切块（`points_batch_size`）**。`Renderer.py:39-61` 在 `eval_points` 内按该值切分采样点，切块后才调用解码器。

```python
# upstream/src/utils/Renderer.py:39-61
p_split = torch.split(p, self.points_batch_size)
bound = self.bound
rets = []
for pi in p_split:
    mask_x = (pi[:, 0] < bound[0][1]) & (pi[:, 0] > bound[0][0])
    mask_y = (pi[:, 1] < bound[1][1]) & (pi[:, 1] > bound[1][0])
    mask_z = (pi[:, 2] < bound[2][1]) & (pi[:, 2] > bound[2][0])
    mask = mask_x & mask_y & mask_z

    if self.nice:
        ret = decoders(pi, c_grid=c, stage=stage)
    else:
        ret = decoders(pi, c_grid=None)
    ret = ret.squeeze(0)
    if len(ret.shape) == 1 and ret.shape[0] == 4:
        ret = ret.unsqueeze(0)

    ret[~mask, 3] = 100
    rets.append(ret)

ret = torch.cat(rets, dim=0)
```

两级切块的实际切分数（本机配置）：

| 场景 | 射线条数 | 射线分块数（100,000） | 每块点数 | 点分块数（500,000） |
| --- | --- | --- | --- | --- |
| 控制环 1 位姿 | 8,160 | 1 | 261,120 | 1 |
| 控制环 10 位姿 | 81,600 | 1 | 2,611,200 | 6（5×500,000 + 111,200） |
| 展示渲染第 1 趟（32 点/射线） | 816,000 | 9 | 3,200,000 | 7（6×500,000 + 200,000） |
| 展示渲染第 2 趟（48 点/射线） | 816,000 | 9 | 4,800,000 | 10（9×500,000 + 300,000） |

注意第二级切块发生在第一级内部：展示渲染的每个射线块独立完成采样、切点、查询、合成，`ret` 在切块之间通过 `torch.cat` 拼接（`Renderer.py:61`），因此峰值由"一个射线块 × 一个点块"决定，而不由整幅图像的 39,168,000 个点决定。

### 5.3 三档降显存手段

按改动代价从低到高：

| 档位 | 手段 | 代码位置 | 显存效果 | 代价 |
| --- | --- | --- | --- | --- |
| 第一档 | 降低 `points_batch_size` | `Renderer.py:7` 默认值 500000 | 激活量按比例下降，主导项 | 分块数上升，kernel 启动与框架开销上升 |
| 第一档 | 降低 `ray_batch_size` | `Renderer.py:7` 默认值 100000 | 同时降低 `(N_rays, N_samples)` 的逐射线张量与每块的 `points_batch_size` 上限 | 同上 |
| 第二档 | 降低渲染分辨率或采样点数 | `single_cbf.py:54-59`、`configs/nice_slam.yaml:105-108` | 与 `H×W×N_samples` 成正比 | 深度图分辨率下降，CBF 判据的采样密度下降 |
| 第二档 | 减少候选数 | `single_cbf.py:33` 的 `batch_size = 10` | 位姿维 B 下降，射线数线性下降 | 可行动作的搜索密度下降 |
| 第三档 | 更换计算精度或释放缓存 | `Renderer.py:246-247` 的 `.double()`、`torch.cuda.empty_cache()` | 逐射线张量减半（若改回 f32） | 语义变化：`depth_var` 精度下降，需要重新验证数值 |
| 第三档 | 环境变量 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 进程启动环境 | 缓解碎片导致的 reserved 膨胀 | 不减少活跃显存 |

`points_batch_size` 与 `ray_batch_size` 的默认值不能被配置文件覆盖：`NICE_SLAM.py:91` 只传入 `(cfg, args, self)`，两个参数取函数签名默认值。要修改只能改 `Renderer.py:7`，或在 `Robot.__init__`（`single_cbf.py:50-59`）里对 `self.renderer` 的属性赋值。属性赋值是可行路径，因为 `eval_points`（`Renderer.py:39`）与两处切块循环（`Renderer.py:229`、`Renderer.py:297`）都从 `self` 读该值。

另一点与显存相关的实现细节：`single_cbf.py:71-100` 的 `Robot.render` 在进入时把 `renderer.H/W/fx/fy/cx/cy` 改成展示分辨率（1200×680），退出时改回控制分辨率（120×68）。若在这两次赋值之间抛出异常，渲染器会停留在 1200×680 状态，下一步 `predict_observation` 就会按 816,000 射线计算，显存与时间同时上升 100 倍。这是一个仅在异常路径上出现的状态不一致。

### 5.4 OOM 排查顺序

出现 `CUDA out of memory` 时按以下顺序处理：

1. 读报错中的 "allocated / reserved" 数字。reserved 远大于 allocated 表示碎片，用 `torch.cuda.empty_cache()` 或 `expandable_segments`；两者接近表示活跃显存确实超限。
2. 区分场景：控制环渲染（8160~81600 射线）与展示渲染（816,000 射线）。报错发生在 `Robot.render` 内部说明是展示渲染超限，先降 `ray_batch_size`。
3. 注意 `.double()`：`Renderer.py:246-247` 把每个射线块的 `depth`、`uncertainty` 转成 f64 后追加到列表；`depth_list` 的元素个数等于射线块数，展示渲染时为 9 个 `(100000,)` f64，共 7.2 MB，不是主要项，但 `z_vals` 在第二趟因与 f64 的 `z_vals_surface` 拼接而升为 f64（`Renderer.py:169-170`），使每块的点坐标从 f32 变 f64。
4. 最后才考虑减少候选数或分辨率，这两项改变算法语义。

### 5.5 验证

```python
# 可运行：测量不同 points_batch_size 下的峰值显存与耗时（10 位姿，2,611,200 点）
import time
import torch

for pbs in (125000, 250000, 500000, 1000000):
    r.points_batch_size = pbs
    r.render_batch_img(c, dec, poses, 'cuda:0', stage='middle', gt_depth=None)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    r.render_batch_img(c, dec, poses, 'cuda:0', stage='middle', gt_depth=None)
    torch.cuda.synchronize()
    print(f'points_batch_size={pbs:>8d}  {time.time()-t0:.4f} s  '
          f'peak_alloc {torch.cuda.max_memory_allocated()/1024/1024:6.0f} MB  '
          f'peak_reserved {torch.cuda.max_memory_reserved()/1024/1024:6.0f} MB')
```

预期：峰值显存随 `points_batch_size` 单调上升（见 §5.1 的实测表，125,000 → 1,000,000 对应 311 MB → 1243 MB），因为激活量由单个分块的点数决定；耗时在四档之间变化不足 10%，因为分块数从 21 降到 3 所节省的固定开销（每块 6 次 `.to()` 检查、3 次掩码、1 次 `torch.cat`）远小于 500,000 点的实际计算时间。把同一组取值用于单位姿时峰值稳定不变（点分块恒为 1），这是第二级切块只在点数超过 `points_batch_size` 时起作用的直接证据。

---

## 6 主机-设备传输与批处理

### 6.1 定义

**H2D（Host to Device）**：主机内存到显存的拷贝。**D2H**：反向拷贝。PCIe 4.0 ×16 的理论峰值约 32 GB/s，实际单向连续传输通常为 10~25 GB/s，远低于显存带宽（本机 8150 MiB GDDR7，量级为数百 GB/s）。因此**数据搬运的次数**比搬运的字节数更值得关注：小张量的传输由固定延迟（数微秒到数十微秒）主导。

**固定内存（pinned memory）**：页面锁定在物理内存的缓冲区，可与计算重叠传输。本仓库未使用任何固定内存；每次 `.to('cuda:0')` 都从可分页内存拷贝，且小拷贝通常是同步的。

### 6.2 本仓库的传输点清单

| 位置 | 传输内容 | 方向 | 频率 |
| --- | --- | --- | --- |
| `single_cbf.py:37` | `state[i][3:].cpu()` | D2H | 每个位姿 1 次，批量 10 位姿时 10 次 |
| `single_cbf.py:37` | 旋转矩阵 `.to(device)` | H2D | 同上 |
| `single_cbf.py:105` | `pose[:3, :3].cpu()` | D2H | 每步 1 次 |
| `single_cbf.py:30`、`single_cbf.py:106` | `u`、`orient_action` 的 `.to(device)` | H2D | 每步数次 |
| `Renderer.py:100` | `self.bound.unsqueeze(0).to(device)` | H2D | 每个射线块 1 次，10 位姿一次性批量使该调用只发生 1 次 |
| `Renderer.py:246-247`、`Renderer.py:318-320` | `depth.double()`、`torch.cat` | 设备内 | 每块 1 次 |
| `single_cbf.py:295-296` | 深度图与彩色图的 `.cpu().numpy()` | D2H | 每帧 1 次，携带 1200×680×3 f32 与 1200×680 f64 |
| `single_cbf.py:297` | `depth.min()` 的取值 | D2H | 每帧 1 次 |
| `single_cbf.py:119`、`:145`、`:146` | 打印与格式化张量 | D2H | 每次判定 1~3 次 |

按每步的总传输量比较：能进入 `render_batch_img` 的张量只有位姿（10×4×4 f64 = 1.28 KB）与 `bound`（48 B），其余全部在设备内流动。因此**控制环的 H2D 数据量可以忽略**，值得优化的是调用次数：`Renderer.py:100` 在循环内每次重新构造并上传 `bound`，10 次单位姿渲染即上传 10 次。

### 6.3 为什么把 B 个位姿的射线拼成一个大张量

`render_batch_img` 的真实实现（`upstream/src/utils/Renderer.py:277-291`）先对每个位姿单独生成射线再沿第 0 维拼接，然后进入与 `render_img` 相同的切块循环：

```python
# upstream/src/utils/Renderer.py:277-291
with torch.no_grad():
    H = self.H
    W = self.W
    for i in range(c2w.shape[0]):
        rays_o, rays_d = get_rays(H, W, self.fx, self.fy, self.cx, self.cy, c2w[i], device)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)

        if i == 0:
            batch_rays_o = rays_o
            batch_rays_d = rays_d
        else:
            batch_rays_o = torch.cat((batch_rays_o, rays_o), dim=0)
            batch_rays_d = torch.cat((batch_rays_d, rays_d), dim=0)
```

拼接后的形状：`batch_rays_o (B*H*W, 3)`、`batch_rays_d (B*H*W, 3)`，本仓库 B = 10、H×W = 8160，即 `(81600, 3)` f64。随后 `render_batch_ray` 按 `ray_batch_size = 100000` 切分，`(81600, 3)` 落在单个分块内，因此 B 个位姿的全部射线在同一次 `render_batch_ray` 调用中被采样、切点、查询、合成。

收益的来源有三项，全部与"次数"有关：

1. **射线块数**。逐个位姿调用时产生 10 次 `render_batch_ray`，每次内部都执行 `bound` 上传（`Renderer.py:100`）、`z_vals` 构造（`Renderer.py:151-156`）、`torch.cat` 拼接结果（`Renderer.py:61`）、`depth_list.double()`（`Renderer.py:246`）。一次性批量把这些调用数降为 1。
2. **点块数**。逐位姿时每个 `(261120, 3)` 的点张量都小于 `points_batch_size`，切块数为 1，解码器因此被调用 10 次；批量后 2,611,200 个点切为 6 块，解码器调用 6 次。每次解码器调用内部的 `.to(device)` 检查（`decoder.py:171-172`、`decoder.py:189`、`decoder.py:192`、`decoder.py:195`、`decoder.py:198`）随之减少。
3. **归约的规模效应**。`single_cbf.py:138` 的 `.min(dim=-1)[0].min(dim=-1)[0]` 在批量情形下作用于 `(10, 68, 120)`，一次两次归约得到 `(10,)`；逐位姿时需要 10 组归约与 10 次结果拼接。

实测（§1.3 的基线数据）：循环渲染 10 次单位姿 0.121 s，一次性批量 0.103 s，比值约 1.17。收益的绝对值为 18 ms，对应被消除的 9 次 `render_batch_ray` 框架开销与 4 次解码器调用分块差异。该比值与单次工作量成反比：当每次工作量更大（例如把分辨率提高到 1200×680）时，固定开销占比下降，批量收益下降；当候选数提高到 100 时，固定开销占比上升，批量收益上升。

### 6.4 不能无限增大 B 的两个原因

1. 第一级切块的粒度固定为 100,000 射线。B 增大到使 `B×8160 > 100000`（即 B ≥ 13）时，`torch.cat` 之后的射线张量被切成 2 块，第 2 块的解码器调用回到"小块"状态，固定开销重新出现。
2. 第二级切块的块数决定解码器调用次数，块数上升会重新引入每块的 `.to()` 检查与掩码计算。B 增大使总点数上升，块数按 `N_points / points_batch_size` 增加，显存与调用次数同时上升。

因此 B 的合理取值是"总射线数略小于 `ray_batch_size`"，本仓库取 10（81,600 < 100,000）。

### 6.5 与逐位姿实现的语义差异

`render_batch_img` 在切块时按拼好的射线顺序切片（`Renderer.py:302-304`），即第 0 块可能跨越两个位姿。深度结果在 `Renderer.py:325-327` 被 `reshape(B, H, W)` 还原，因此只要 `H×W` 与射线顺序一致，结果的位姿顺序与输入一致。这一点在修改 H/W 时容易被破坏：若在 B 个位姿之间改变 `H` 或 `W`，拼好的射线顺序与 `reshape` 的假设不再一致，深度图会错位。

### 6.6 验证

```python
# 可运行：批量 vs 逐个，用事件计时排除主机抖动
import torch

def timed(fn, reps=5):
    fn(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / reps


t_seq = timed(lambda: [r.render_batch_img(c, dec, poses[i:i+1], 'cuda:0',
                                          stage='middle', gt_depth=None)
                       for i in range(10)])
t_batch = timed(lambda: r.render_batch_img(c, dec, poses, 'cuda:0',
                                           stage='middle', gt_depth=None))
print(f'10 次单位姿 {t_seq:.1f} ms   一次性 10 位姿 {t_batch:.1f} ms   比值 {t_seq/t_batch:.2f}')

# 验证拼接顺序与 reshape 的一致性：把 B 个位姿设成同一个位姿，深度图应完全相同
d, _, _ = r.render_batch_img(c, dec, poses, 'cuda:0', stage='middle', gt_depth=None)
print('各位置姿深度一致:', bool(torch.allclose(d, d[0].expand_as(d), atol=1e-9)))
```

预期：第二个检查输出 `True`（若 `poses` 由同一位姿复制而来），说明 `reshape(B, H, W)` 的映射正确。

---

## 7 渲染算法逐层拆解

### 7.1 总览

一次 `predict_observation`（`single_cbf.py:61-69`）调用 `render_batch_img`，参数为 `stage='middle'`、`gt_depth=None`、`H=68`、`W=120`。数据流如下：

| 阶段 | 代码 | 输入形状 | 输出形状 |
| --- | --- | --- | --- |
| 射线生成 | `common.py:240-257` | `(B, 4, 4)` | `(B, H, W, 3)` ×2 |
| 拼批 | `Renderer.py:280-290` | `(H, W, 3)` ×B | `(B*H*W, 3)` ×2 |
| 包围盒求交 | `Renderer.py:97-104` | `(N_rays, 3)` | `far_bb (N_rays, 1)` |
| 采样 | `Renderer.py:151-156` | `N_samples = 32` | `z_vals (N_rays, 32)` |
| 点展开 | `Renderer.py:172-174` | `(N_rays, 32)` | `pts (N_points, 3)` |
| 网格查询 + MLP | `Renderer.py:176`、`decoder.py:175-201` | `(N_points, 3)` | `raw (N_points, 4)` |
| 体渲染合成 | `common.py:195-236` | `(N_rays, 32, 4)` | `depth (N_rays,)`、`unc (N_rays,)`、`rgb (N_rays, 3)` |
| 还原图像 | `Renderer.py:325-327` | `(B*H*W)` | `(B, H, W)` |

### 7.2 get_rays：像素到射线

**定义**：`get_rays` 把针孔相机的像素坐标 `(i, j)`、内参 `(fx, fy, cx, cy)` 与相机到世界的变换 `c2w` 映射为世界坐标系下的射线原点 `rays_o` 与方向 `rays_d`，满足 `x(t) = rays_o + t · rays_d`，其中 `t` 是射线参数。

```python
# upstream/src/common.py:240-257
def get_rays(H, W, fx, fy, cx, cy, c2w, device):
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w)
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t()
    j = j.t()
    dirs = torch.stack(
        [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
    dirs = dirs.reshape(H, W, 1, 3)
    rays_d = torch.sum(dirs * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d
```

机制要点：

1. 第 3 维取 `-torch.ones_like(i)`，即相机沿 −z 观察；第 2 维取 `-(j-cy)/fy`，说明图像行号向下增长而相机 y 轴向上。
2. `rays_d` 由 `torch.sum(dirs * c2w[:3, :3], -1)` 得到，即用旋转矩阵左乘相机坐标系方向向量。`dirs` 形状 `(H, W, 1, 3)`，与 `(3, 3)` 广播，`sum` 沿最后一维，得到 `(H, W, 3)`。
3. `rays_o` 用 `expand` 得到 `(H, W, 3)`，是广播视图，不占额外显存；下一步 `reshape(-1, 3)`（`Renderer.py:282-283`）会强制物化。
4. **`rays_d` 未归一化**。其长度为 `|((i-cx)/fx, -(j-cy)/fy, -1)|`，在光轴上为 1，在图像角点大于 1。本机在 120×68 控制分辨率下实测区间为 `1.000001 ~ 1.522663`；1200×680 时角点约为 `1.5223`。`t` 与欧氏距离的关系为 `距离 = t · ‖rays_d‖`。

**后果一：`z_vals` 是射线参数而非距离。**`Renderer.py:172-173` 的 `pts = rays_o + rays_d * z_vals` 用未归一化方向，因此 `z_vals` 的物理含义是光轴方向的投影距离，只有在光轴上等于欧氏距离。

**后果二：`depth_map` 与 `dists · ‖rays_d‖` 的区别。**`common.py:213-219` 计算 `dists = z_vals[..., 1:] - z_vals[..., :-1]` 并乘以 `‖rays_d‖`，得到相邻采样点间的欧氏距离：

```python
# upstream/src/common.py:213-219
dists = z_vals[..., 1:] - z_vals[..., :-1]
dists = dists.float()
dists = torch.cat([dists, torch.Tensor([1e10]).float().to(
    device).expand(dists[..., :1].shape)], -1)

# different ray angle corresponds to different unit length
dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
```

`dists` 只用于体积密度分支（`common.py:226` 的 `raw2alpha`）。本仓库 `occupancy: True`（`configs/nice_slam.yaml:5`），走 `common.py:221-223` 的 sigmoid 分支，`dists` 被计算但不参与 α，因此**深度是按射线参数加权的期望值**：

```python
# upstream/src/common.py:232
depth_map = torch.sum(weights * z_vals, -1)  # (N_rays)
```

由此得到的两个可检验事实：(a) `depth_map` 与 GT 深度（沿光轴的传感器深度）同量纲，但图像边缘处二者相差 `‖rays_d‖` 倍；(b) CBF 的 `h = d − depth.min()` 中的 `d = 0.1`（`single_cbf.py:29`）与 `depth` 同为射线参数语义，二者可比，但都不等于欧氏距离。

**验证**

```python
# 可运行：确认射线未归一化，并量化 |rays_d| 的空间分布
import torch
from src.common import get_rays

ro, rd = get_rays(68, 120, 60.0, 60.0, 59.95, 33.95, pose, 'cuda:0')
n = rd.reshape(-1, 3).norm(dim=-1)
print('|rays_d| min / 中心 / max: %.6f / %.6f / %.6f' %
      (n.min(), n[34 * 120 + 60], n.max()))
print('shape', tuple(ro.shape), ro.dtype, '连续:', ro.is_contiguous())

# 深度与欧氏距离的关系：同一条射线上 t 与距离成 |rays_d| 倍
t = torch.linspace(0.5, 3.0, 8, device='cuda:0')
dist = (rd[0, 0][None, :] * t[:, None]).norm(dim=-1)
print('边角像素 距离/t =', (dist / t)[0].item(), ' 该像素 |rays_d| =', rd[0, 0].norm().item())
```

预期：`|rays_d|` 的最小值为 1（光轴）、最大值为 1.52（角点）；`ro` 为 `expand` 得到的非连续视图（`False`）；`距离/t` 与 `‖rays_d‖` 相等，均等于 1.5227 量级。

### 7.3 近远平面与采样点数

**定义**：`near` 与 `far` 定义每条射线上有效采样区间，`N_samples = 32` 是该区间内的均匀采样数（`lindisp=False`，`perturb=0.0`，即不加抖动）；`N_surface = 16` 是额外的表面细化采样数，只在传入 GT 深度时启用；`N_importance = 0` 表示不做重要性采样。

```python
# upstream/src/utils/Renderer.py:87-110
if stage == 'coarse':
    gt_depth = None
if gt_depth is None:
    N_surface = 0
    near = 0.01
else:
    gt_depth = gt_depth.reshape(-1, 1)
    gt_depth_samples = gt_depth.repeat(1, N_samples)
    near = gt_depth_samples*0.01

with torch.no_grad():
    det_rays_o = rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
    det_rays_d = rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
    t = (self.bound.unsqueeze(0).to(device) -
         det_rays_o)/det_rays_d  # (N, 3, 2)
    far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
    far_bb = far_bb.unsqueeze(-1)
    far_bb += 0.01

if gt_depth is not None:
    far = torch.clamp(far_bb, 0,  torch.max(gt_depth*1.2))
else:
    far = far_bb
```

机制要点：

1. **`far` 由包围盒求交得到**。`t = (bound − o) / d` 给出射线与两个包围盒平面相交的参数，`max(dim=2)` 取每个轴上较大的参数，`min(dim=1)` 取三个轴的最小值，即离开轴对齐包围盒的参数。该式假设相机原点在盒内；若相机在盒外，得到的是入射参数。本机在初始位姿（`[-2, 0.2, 0.5, 90°, 0°, 115°]`）下 `far` 的区间为 3.624 ~ 6.539（由 §8.6 的脚本打印）。
2. **`near = 0.01`** 是常数，不随分辨率变化；有 GT 深度时 `near` 随每条射线变为 `0.01 × gt_depth`（形状 `(N_rays, N_samples)`，因 `repeat` 而重复 32 份）。
3. **`N_importance = 0`** 使 `Renderer.py:181-195` 的整段代码不执行，`common.py:20-63` 的 `sample_pdf` 在本流程中从未被调用。这意味着采样密度完全由 `near`、`far` 与 `N_samples` 决定，不存在基于权重的细化。
4. **`N_surface` 的启用条件**是 `gt_depth is not None` 且 `N_surface > 0`（`Renderer.py:111`、`Renderer.py:168`）。控制环的 `predict_observation` 传 `gt_depth=None`，实际采样点为 32；展示渲染的第二趟传 `gt_depth=depth`，采样点为 48。

表面细化的采样位置（`Renderer.py:127-149`）：

```python
# upstream/src/utils/Renderer.py:127-149
gt_none_zero_mask = gt_depth > 0
gt_none_zero = gt_depth[gt_none_zero_mask]
gt_none_zero = gt_none_zero.unsqueeze(-1)
gt_depth_surface = gt_none_zero.repeat(1, N_surface)
t_vals_surface = torch.linspace(
    0., 1., steps=N_surface).double().to(device)
# emperical range 0.05*depth
z_vals_surface_depth_none_zero = 0.95*gt_depth_surface * \
    (1.-t_vals_surface) + 1.05 * \
    gt_depth_surface * (t_vals_surface)
z_vals_surface = torch.zeros(
    gt_depth.shape[0], N_surface).to(device).double()
gt_none_zero_mask = gt_none_zero_mask.squeeze(-1)
z_vals_surface[gt_none_zero_mask,
               :] = z_vals_surface_depth_none_zero
near_surface = 0.001
far_surface = torch.max(gt_depth)
z_vals_surface_depth_zero = near_surface * \
    (1.-t_vals_surface) + far_surface * (t_vals_surface)
z_vals_surface_depth_zero.unsqueeze(
    0).repeat((~gt_none_zero_mask).sum(), 1)
z_vals_surface[~gt_none_zero_mask,
               :] = z_vals_surface_depth_zero
```

16 个表面采样点分布在 `[0.95·d, 1.05·d]` 区间（`d` 为该射线的 GT 深度），即表面附近 ±5%；深度为 0 的像素（无传感器读数）改为在 `[0.001, max(gt_depth)]` 内均匀采样。第 146–147 行的 `repeat` 结果未被赋值给任何变量，对结果无影响。

这 16 点是沿光轴按深度值放置的，而 GT 深度是光轴方向深度，因此与 `z_vals` 的射线参数语义一致（见 §7.2 的后果一）。

**`z_vals` 的构造与类型提升**（`Renderer.py:151-170`）：

```python
# upstream/src/utils/Renderer.py:151-170
t_vals = torch.linspace(0., 1., steps=N_samples, device=device)

if not self.lindisp:
    z_vals = near * (1.-t_vals) + far * (t_vals)
else:
    z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

if self.perturb > 0.:
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], -1)
    lower = torch.cat([z_vals[..., :1], mids], -1)
    t_rand = torch.rand(z_vals.shape).to(device)
    z_vals = lower + (upper - lower) * t_rand

if N_surface > 0:
    z_vals, _ = torch.sort(
        torch.cat([z_vals, z_vals_surface.double()], -1), -1)
```

类型链：`near` 的 dtype 决定 `z_vals`。控制环中 `near = 0.01`（Python float）与 `far`（f64，因为 `rays_o`/`rays_d` 由 f64 位姿得到）运算，`z_vals` 为 f64 `(N_rays, 32)`；展示渲染第二趟因 `z_vals_surface` 显式 `.double()` 而同样是 f64，并在 `sort` 后变为 `(N_rays, 48)`，同时 `sort` 交错了两组采样点。

`pts` 的形状与类型：`(N_rays, 32, 3)` 或 `(N_rays, 48, 3)`，f64；`Renderer.py:174` 的 `reshape(-1, 3)` 得到 `(N_points, 3)`。

**验证**

```python
# 可运行：对比两趟的采样点数与类型
import torch
with torch.no_grad():
    d0, _, _ = r.render_batch_ray(c, dec, rd1, ro1, 'cuda:0', 'color', gt_depth=None)
    d1, _, _ = r.render_batch_ray(c, dec, rd1, ro1, 'cuda:0', 'color', gt_depth=d0.reshape(-1))
print('第一趟 32 点:', tuple(d0.shape), d0.dtype, 'min', float(d0.min()))
print('第二趟 48 点:', tuple(d1.shape), d1.dtype, 'min', float(d1.min()))
```

注意：`render_batch_ray` 期望在外层 `torch.no_grad()` 中调用（`render_img`/`render_batch_img` 都是如此）。直接调用而未关闭梯度时，`Renderer.py:108` 的 `torch.max(gt_depth*1.2)` 会保留梯度，`torch.clamp` 的重载解析会因"标量参数位置收到需要梯度的张量"而抛出 `TypeError`。

### 7.4 网格查询

**定义**：在多分辨率的特征网格上，用采样点的世界坐标做三线性插值，得到该点的 32 维特征向量。网格的存储布局为 `(1, c_dim, D, H, W) = (1, 32, z, y, x)`，最后三维与 `F.grid_sample` 的 `(D, H, W)` 参数位对应。

```python
# upstream/src/conv_onet/models/decoder.py:166-173
def sample_grid_feature(self, p, c):
    p_nor = normalize_3d_coordinate(p.clone(), self.bound)
    p_nor = p_nor.unsqueeze(0)
    vgrid = p_nor[:, :, None, None].float()
    # acutally trilinear interpolation if mode = 'bilinear'
    c = F.grid_sample(c.to(device), vgrid.to(device), padding_mode='border', align_corners=True,
                      mode=self.sample_mode).squeeze(-1).squeeze(-1)
    return c
```

形状链：

| 步骤 | 形状 | 说明 |
| --- | --- | --- |
| `p.clone()` | `(N, 3)` f64 | `clone` 是必需的，因为 `normalize_3d_coordinate` 原地修改输入（`common.py:270-272` 直接对视图赋值） |
| `normalize_3d_coordinate` | `(N, 3)` f64 | 映射到 `[-1, 1]` |
| `.unsqueeze(0)` | `(1, N, 3)` | 增加 batch 维 |
| `[:, :, None, None]` | `(1, N, 1, 1, 3)` | 5 维采样网格，`(D_out, H_out, W_out) = (N, 1, 1)` |
| `.float()` | `(1, N, 1, 1, 3)` f32 | `grid_sample` 要求网格为 f32 |
| `F.grid_sample(...)` | `(1, 32, N, 1, 1)` f32 | 5 维输入按三线性插值 |
| `.squeeze(-1).squeeze(-1)` | `(1, 32, N)` | 去掉 `(1, 1)` 空间维 |
| `.transpose(1, 2).squeeze(0)`（`decoder.py:178`） | `(N, 32)` | 通道维与点维互换 |

四个网格的形状与参数来源（`NICE_SLAM.py:192-247` 的 `grid_init`，`configs/nice_slam.yaml:7-12` 的 `grid_len`，`configs/Replica/room1.yaml:3` 的 `bound`）：

| 网格 | `grid_len` | 由 `bound` 换算的体素数 `(x,y,z)` | 存储形状 `(1,32,z,y,x)` | 元素数 |
| --- | --- | --- | --- | --- |
| coarse | 2.0（另乘 `coarse_bound_enlarge = 2`） | (9, 8, 6) | `(1, 32, 6, 8, 9)` | 13,824 |
| middle | 0.32 | (31, 28, 18) | `(1, 32, 18, 28, 31)` | 499,968 |
| fine | 0.16 | (62, 56, 37) | `(1, 32, 37, 56, 62)` | 4,110,848 |
| color | 0.16 | (62, 56, 37) | `(1, 32, 37, 56, 62)` | 4,110,848 |

`bound` 先按 `grid_len.bound_divisable = 0.32` 向上取整（`NICE_SLAM.py:147-150`），得到 `[[-7.0, 2.92], [-4.6, 4.36], [-3.0, 3.08]]`，即边长 `(9.92, 8.96, 6.08)`。`grid_init` 用 `list(map(int, xyz_len / grid_len))` 取整并交换首末维，因此 `middle` 的 z 方向为 `int(6.08/0.32) = 18`（浮点结果为 18.999…），`fine` 的 z 方向为 `int(6.08/0.16) = 37`。这一取整使体素数比标称值少 1。

**`align_corners=True` 的直接影响**：归一化把 `bound` 的两端映射到 `-1` 与 `+1`，而 `align_corners=True` 表示 `-1` 对应首个体素中心、`+1` 对应末个体素中心，因此相邻体素中心的实际间距为 `边长 / (n − 1)`，而不是配置值 `grid_len = 边长 / n`。middle 网格的有效间距为 `(0.3307, 0.3319, 0.3576)`，配置值为 0.32。

**fine 层拼接 middle 特征**（`decoder.py:180-185`）：fine 解码器的 `c_dim = 64`（`decoder.py:301`），因为它先取 `grid_fine` 的 32 维特征，再以 `torch.no_grad()` 取 `grid_middle` 的 32 维特征并沿通道拼接：

```python
# upstream/src/conv_onet/models/decoder.py:176-185
if self.c_dim != 0:
    c = self.sample_grid_feature(
        p, c_grid['grid_' + self.name]).transpose(1, 2).squeeze(0)

    if self.concat_feature:
        # only happen to fine decoder, get feature from middle level and concat to the current feature
        with torch.no_grad():
            c_middle = self.sample_grid_feature(
                p, c_grid['grid_middle']).transpose(1, 2).squeeze(0)
        c = torch.cat([c, c_middle], dim=1)
```

`torch.no_grad()` 只阻断梯度，不减少 `grid_sample` 的 kernel 数量与访存量；fine 层每次前向因此有 2 次 `grid_sample`。

**验证**

```python
# 可运行：核对归一化与边界填充的实际行为
import torch
from src.common import normalize_3d_coordinate

bound = torch.from_numpy(np.array([[-7.0, 2.92], [-4.6, 4.36], [-3.0, 3.08]]))
p = torch.tensor([[-7.0, -4.6, -3.0], [2.92, 4.36, 3.08], [0.0, 0.0, 0.0]],
                 dtype=torch.float32)
print(normalize_3d_coordinate(p.clone(), bound))   # 期望 [-1,-1,-1] 与 [1,1,1]

# 有效体素间距与名义值不同
for name, k, L in (('middle-x', 31, 9.92), ('middle-y', 28, 8.96), ('middle-z', 18, 6.08)):
    print(f'{name}: 边长/n {L/k:.4f}  边长/(n-1) {L/(k-1):.4f}')
```

预期：两端分别输出 `-1` 与 `1`；`边长/(n−1)` 大于 `边长/n`，且 z 方向偏差最大（因取整损失一个体素：`6.08/18 = 0.3378`，配置值为 0.32）。

### 7.5 MLP 与位置编码

**定义**：解码器把 `(N, 3)` 的点坐标与 `(N, 32)` 的网格特征映射为 `(N, 1)` 的占据 logits 或 `(N, 4)` 的（rgb + 占据）。位置编码把 3 维坐标升到 93 维，使 MLP 能表达高频变化。

```python
# upstream/src/conv_onet/models/decoder.py:9-30
class GaussianFourierFeatureTransform(torch.nn.Module):
    def __init__(self, num_input_channels, mapping_size=93, scale=25, learnable=True):
        super().__init__()

        if learnable:
            self._B = nn.Parameter(torch.randn(
                (num_input_channels, mapping_size)) * scale)
        else:
            self._B = torch.randn((num_input_channels, mapping_size)) * scale

    def forward(self, x):
        x = x.squeeze(0)
        assert x.dim() == 2, 'Expected 2D input (got {}D input)'.format(x.dim())
        x = x @ self._B.to(x.device)
        return torch.sin(x)
```

`mapping_size = 93`、`scale = 25`，`_B` 形状 `(3, 93)`，是**可学习参数**（`nn.Parameter`），因此它出现在 `state_dict` 中并随预训练权重加载（`NICE_SLAM.py:159-190`）。前向为 `(N, 3) × (3, 93) → (N, 93) → sin`。

解码器的层结构（`decoder.py:108-157`），以 middle 解码器为例：

```python
# upstream/src/conv_onet/models/decoder.py:147-157
self.pts_linears = nn.ModuleList(
    [DenseLayer(embedding_size, hidden_size, activation="relu")] +
    [DenseLayer(hidden_size, hidden_size, activation="relu") if i not in self.skips
     else DenseLayer(hidden_size + embedding_size, hidden_size, activation="relu") for i in range(n_blocks-1)])

if self.color:
    self.output_linear = DenseLayer(
        hidden_size, 4, activation="linear")
else:
    self.output_linear = DenseLayer(
        hidden_size, 1, activation="linear")
```

**宽度是 32，不是 256。**`MLP.__init__` 的签名默认 `hidden_size=256`（`decoder.py:109`），但 `NICE.__init__` 的默认参数是 `hidden_size=32`（`decoder.py:291`），并在构造四个解码器时显式传入（`decoder.py:296-306`）；`conv_onet/config.py:24-27` 又没有覆盖该值。预训练权重证实了这一点：`middle_decoder.pts_linears.0.weight` 的形状是 `(32, 93)`、`pts_linears.3.weight` 是 `(32, 125)`。四个解码器共 58,956 个参数。

层配置汇总：

| 项 | 值 | 来源 |
| --- | --- | --- |
| 位置编码输出维 | 93 | `decoder.py:127-129` |
| 层数 `n_blocks` | 5 | `decoder.py:299` |
| 隐藏宽度 | 32 | `decoder.py:291`、`decoder.py:298-306` |
| skip 层索引 | `[2]` | `decoder.py:299` |
| skip 拼接后的输入维 | 125 = 93 + 32 | `decoder.py:197`、权重形状 `(32, 125)` |
| middle 特征维 `c_dim` | 32 | `configs/nice_slam.yaml:113-114` |
| fine 特征维 | 64 = 32 + 32 | `decoder.py:301`、`decoder.py:180-185` |
| 输出维（`color=True`） | 4 | `decoder.py:152-154` |
| 输出维（`color=False`） | 1 | `decoder.py:155-157` |

前向的完整形状流（`decoder.py:175-201`）：

```python
# upstream/src/conv_onet/models/decoder.py:187-201
p = p.float()

embedded_pts = self.embedder(p).to(device)
h = embedded_pts
for i, l in enumerate(self.pts_linears):
    h = self.pts_linears[i].to(device)(h)
    h = F.relu(h)
    if self.c_dim != 0:
        h = h + self.fc_c[i].to(device)(c)
    if i in self.skips:
        h = torch.cat([embedded_pts, h], -1)
out = self.output_linear.to(device)(h)
if not self.color:
    out = out.squeeze(-1)
return out
```

| 变量 | 形状 | 说明 |
| --- | --- | --- |
| `p` | `(N, 3)` f64 → `.float()` → f32 | 从 f64 采样点降为 f32 |
| `c` | `(N, 32)` 或 fine 层 `(N, 64)` | 网格特征 |
| `embedded_pts` | `(N, 93)` | 位置编码 |
| `h`（i = 0） | `(N, 32)` | `93 → 32` |
| `h + fc_c[0](c)` | `(N, 32)` | 特征注入，`fc_c[i]` 为 `(32, 32)` 或 `(64, 32)` |
| `h`（i = 2 之后） | `(N, 125)` | skip 拼接 |
| `out`（`color=False`） | `(N,)` | `output_linear` 后 `squeeze(-1)` |
| `out`（`color=True`） | `(N, 4)` | 不 `squeeze` |

`self.fc_c[i].to(device)` 与 `self.pts_linears[i].to(device)` 在循环体内部：同设备时 `.to()` 不拷贝，只做设备比较，属于主机侧开销（见 §3.2）。

**验证**

```python
# 可运行：确认宽度、skip 与参数总量
m = dec.middle_decoder
print('embedding', m.embedder._B.shape)
for i, l in enumerate(m.pts_linears):
    print(f'layer {i}: {tuple(l.weight.shape)}  in={l.in_features} out={l.out_features}')
print('out', tuple(m.output_linear.weight.shape))
print('decoder params', sum(p.numel() for p in dec.parameters()))
```

预期：5 层，宽度 32，第 3 层的输入维为 125；`decoder params` 为 58,956。

### 7.6 NICE.forward 的四个 stage

**定义**：`stage` 选择查询哪一层解码器。四个阶段对应多分辨率层级，控制环只用 `middle`。

```python
# upstream/src/conv_onet/models/decoder.py:308-338
def forward(self, p, c_grid, stage='middle', **kwargs):
    """
        Output occupancy/color in different stage.
    """
    device = f'cuda:{p.get_device()}'
    if stage == 'coarse':
        occ = self.coarse_decoder(p, c_grid)
        occ = occ.squeeze(0)
        raw = torch.zeros(occ.shape[0], 4).to(device).float()
        raw[..., -1] = occ
        return raw
    elif stage == 'middle':
        middle_occ = self.middle_decoder(p, c_grid)
        middle_occ = middle_occ.squeeze(0)
        raw = torch.zeros(middle_occ.shape[0], 4).to(device).float()
        raw[..., -1] = middle_occ
        return raw
    elif stage == 'fine':
        fine_occ = self.fine_decoder(p, c_grid)
        raw = torch.zeros(fine_occ.shape[0], 4).to(device).float()
        middle_occ = self.middle_decoder(p, c_grid)
        middle_occ = middle_occ.squeeze(0)
        raw[..., -1] = fine_occ+middle_occ
        return raw
    elif stage == 'color':
        fine_occ = self.fine_decoder(p, c_grid)
        raw = self.color_decoder(p, c_grid)
        middle_occ = self.middle_decoder(p, c_grid)
        middle_occ = middle_occ.squeeze(0)
        raw[..., -1] = fine_occ+middle_occ
        return raw
```

| stage | 调用的解码器 | `grid_sample` 次数 | MLP 前向次数 | 用途 |
| --- | --- | --- | --- | --- |
| `coarse` | `coarse_decoder`（`MLP_no_xyz`） | 1 | 1 | 粗层查询，本实验的 CBF 路径不用 |
| `middle` | `middle_decoder` | 1 | 1 | **控制环的预测**（`single_cbf.py:67`、`double_cbf.py:63`） |
| `fine` | `fine_decoder` + `middle_decoder` | 3（fine 自身 + fine 取 middle + middle 自身） | 2 | 精细几何 |
| `color` | `fine_decoder` + `color_decoder` + `middle_decoder` | 4 | 3 | 展示渲染（`single_cbf.py:85`、`single_cbf.py:92`） |

**`raw` 的形状为何是 `(N, 4)`**：体渲染合成的接口（`common.py:195-236`）把前 3 个通道当颜色、最后 1 个通道当占据/密度（`common.py:220-223`）。`middle`、`coarse`、`fine` 三个 stage 不输出颜色，因此先把 `(N, 4)` 置零，只写第 4 通道；`color` stage 中 `color_decoder` 直接输出 `(N, 4)`，随后用 `raw[..., -1] = fine_occ + middle_occ` 覆盖第 4 通道。由此可推断：`middle`/`fine` stage 的 `rgb_map` 恒为 0，`render_img` 返回的 `color` 在这些 stage 下全零，只有 `stage='color'` 才有有效颜色。

`fine` 与 `color` stage 的占据值是 `fine_occ + middle_occ` 之和，两个 logits 直接相加后再经过 `sigmoid(10·x)`，不是凸组合。

`p.get_device()` 说明该模块不能被 CPU 张量调用：CPU 张量的 `get_device()` 返回 `-1`，`f'cuda:{-1}'` 会失败。因此本仓库的渲染路径只有 GPU 版本。

**验证**

```python
# 可运行：四个 stage 的形状与数值范围
p3 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], device='cuda:0')
with torch.no_grad():
    for stage in ('coarse', 'middle', 'fine', 'color'):
        raw = dec(p3, c_grid=c, stage=stage)
        print(f'{stage:7s} raw {tuple(raw.shape)}  occ {raw[..., -1].tolist()}  rgb {raw[..., :3].tolist()}')
```

预期：`raw` 均为 `(2, 4)`；`coarse`/`middle`/`fine` 的 `rgb` 全为 0；`fine` 与 `color` 的占据值**相同**（两处都写 `raw[..., -1] = fine_occ + middle_occ`，见 `decoder.py:330` 与 `decoder.py:337`），区别只在 `color` 阶段的 rgb 非零。

### 7.7 raw2outputs_nerf_color：合成

**定义**：把每条射线上的 `(N_samples, 4)` 预测合成为标量深度、深度方差与 RGB。本仓库 `occupancy=True`，占据概率由 sigmoid 得到，不经过体积密度积分。

```python
# upstream/src/common.py:211-236
def raw2alpha(raw, dists, act_fn=F.relu): return 1. - \
    torch.exp(-act_fn(raw)*dists)
dists = z_vals[..., 1:] - z_vals[..., :-1]
dists = dists.float()
dists = torch.cat([dists, torch.Tensor([1e10]).float().to(
    device).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

# different ray angle corresponds to different unit length
dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
rgb = raw[..., :-1].to(device)
if occupancy:
    raw[..., 3] = torch.sigmoid(10*raw[..., -1])
    alpha = raw[..., -1]
else:
    # original nerf, volume density
    alpha = raw2alpha(raw[..., -1], dists)  # (N_rays, N_samples)
alpha = alpha.to(device)

weights = alpha.float() * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(
    device).float(), (1.-alpha + 1e-10).to(device).float()], -1).float(), -1)[:, :-1]
rgb_map = torch.sum(weights[..., None] * rgb, -2)  # (N_rays, 3)
depth_map = torch.sum(weights * z_vals, -1)  # (N_rays)
tmp = (z_vals-depth_map.unsqueeze(-1))  # (N_rays, N_samples)
depth_var = torch.sum(weights*tmp*tmp, dim=1)  # (N_rays)

return depth_map, depth_var, rgb_map, weights
```

逐项机制与形状（`N_rays = 81600`、`N_samples = 32`）：

| 量 | 公式 | 形状 | 说明 |
| --- | --- | --- | --- |
| `alpha` | `sigmoid(10 · raw[..., 3])` | `(81600, 32)` | 逐点占据概率；`raw[..., 3]` 被原地覆盖（第 222 行），因此该函数**修改输入的 `raw`** |
| `weights` | `w_i = α_i · Π_{j<i}(1 − α_j + 1e−10)` | `(81600, 32)` | 通过 `cat([ones, 1−α])` 后 `cumprod` 再切掉末尾实现"独占前缀积" |
| `rgb_map` | `Σ_i w_i · rgb_i` | `(81600, 3)` | `middle` stage 下 `rgb` 全零，结果为零 |
| `depth_map` | `Σ_i w_i · z_i` | `(81600,)` | 期望深度，单位为射线参数（见 §7.2） |
| `depth_var` | `Σ_i w_i (z_i − depth_map)²` | `(81600,)` | 沿射线的深度方差；`render_img` 以 `uncertainty` 名义返回 |

`1e−10` 的作用是避免 `(1 − α)` 在 α 恰为 1 时把后续所有权重压成精确 0 之外的数值问题；`dists` 的最后一个元素取 `1e10`，使得最后一个采样点的密度分支近似全吸收。

`single_cbf.py` 只用 `depth_map`：`single_cbf.py:116` 的 `d - robot.predict_observation(new_pose).min()`，其中 `.min()` 沿 `(B, H, W)` 的最后两维归约到 `(B,)`，即每个候选位姿的最小深度。`double_cbf.py:132` 在同一表达式上再加 `beta * torch.norm(batch_new_v, dim=1, p=2)`。

一处与数值实现相关的客观事实：`dists` 与 `raw` 都经过 `.float()`，而 `z_vals` 保持 f64，因此 `weights` 是 f32、`z_vals` 是 f64，`depth_map` 的乘法会提升到 f64。这就是 `render_img` 返回的 `depth` 为 f64 的原因（`Renderer.py:246-247` 的 `.double()` 只是显式化）。

**验证**

```python
# 可运行：核对 alpha/weights/depth 的定义
import torch
from src.common import raw2outputs_nerf_color

NR, NS = 4, 8
raw = torch.zeros(NR, NS, 4, device='cuda:0')
raw[..., 3] = torch.linspace(-3, 3, NR * NS, device='cuda:0').reshape(NR, NS)
z_vals = torch.linspace(0.5, 2.0, NS, device='cuda:0').repeat(NR, 1)
rays_d = torch.zeros(NR, 3, device='cuda:0')
rays_d[:, 2] = -1.0

depth, var, rgb, w = raw2outputs_nerf_color(raw.clone(), z_vals, rays_d,
                                             occupancy=True, device='cuda:0')
alpha = torch.sigmoid(10 * raw[..., 3])
print('weights 和 <= 1:', bool((w.sum(-1) <= 1.0 + 1e-6).all()))
print('weights == alpha * 前缀积:', torch.allclose(
    w, alpha * torch.cumprod(torch.cat([torch.ones(NR, 1, device='cuda:0'),
                                        (1 - alpha + 1e-10)], -1), -1)[:, :-1], atol=1e-6))
print('depth == sum(w*z):', torch.allclose(depth, (w * z_vals).sum(-1), atol=1e-6))
print('var == sum(w*(z-depth)^2):', torch.allclose(
    var, (w * (z_vals - depth[:, None]) ** 2).sum(-1), atol=1e-5))
print('rgb 全零（无颜色通道）:', float(rgb.abs().max()) == 0.0)
```

预期：五个比较均为 `True`。

### 7.8 八条可直接核对的实现事实

1. `depth_map` 是射线参数的期望值，不是欧氏距离（§7.2）。
2. `dists` 在 `occupancy=True` 时被计算但不参与 α（§7.3）。
3. `N_importance = 0` 使 `sample_pdf` 在本流程中不会执行（§8.3 的调用统计可验证）。
4. `normalize_3d_coordinate` 原地修改入参，依赖调用方的 `.clone()`（`decoder.py:167`、`decoder.py:252`）。
5. `align_corners=True` 使有效体素间距为 `边长/(n−1)`（§7.4）。
6. 解码器宽度为 32，skip 在索引 2，拼接后为 125 维（§7.5）。
7. `middle`/`fine`/`coarse` stage 的 `rgb_map` 恒为零（§7.6）。
8. `eval_points` 把包围盒外的点第 4 通道置为 100（`Renderer.py:58`），经 `sigmoid(1000)` 后为 1，即包围盒外的采样点被当作完全不透明，其 `z_vals` 直接进入深度期望。

---

## 8 一步决策的完整计算图与 Profiler 验证

### 8.1 控制流

`single_cbf.py:104-149` 的 `find_safe_action` 分三段：

1. 构造意图动作并评估它（`single_cbf.py:105-121`）。单积分器下 `update_dynamics` 为 `state + time_step * action`（`single_cbf.py:45-46`），`time_step = 0.1`。
2. 若不满足 `new_h ≤ alpha * h`，进入 `while True`（`single_cbf.py:122`）：每轮采样 10 个候选动作，一次性渲染 10 个位姿，取满足条件且最接近意图动作的候选（`single_cbf.py:139-147`）。
3. 若本轮没有候选满足条件，重新采样，无迭代上限。

`double_cbf.py` 的差异：状态含速度 `v`，`update_dynamics` 为二阶积分（`double_cbf.py:41-42`），`h` 加 `beta * ‖v‖`（`double_cbf.py:111`、`double_cbf.py:132`），`max_h = 2`（`double_cbf.py:27`），候选循环固定 `range(10)`（`double_cbf.py:121`）。

### 8.2 张量形状流转

以 `single_cbf.py:136-138` 的一次候选批渲染为例（B = 10）：

| 步骤 | 代码位置 | 张量 | 形状 | dtype |
| --- | --- | --- | --- | --- |
| 状态 | `single_cbf.py:105` | `state` | `(6,)` | f32 |
| 候选动作 | `single_cbf.py:123` | `batch_action` | `(10, 6)` | f32 |
| 状态更新 | `single_cbf.py:136` | `batch_new_state` | `(10, 6)` | f32 |
| 位姿构造 | `single_cbf.py:137`、`:35-43` | `batch_new_pose` | `(10, 4, 4)` | f64 |
| 射线生成 | `Renderer.py:281` | `rays_o`、`rays_d` | `(68, 120, 3)` ×10 | f64 |
| 拼批 | `Renderer.py:289-290` | `batch_rays_o/d` | `(81600, 3)` | f64 |
| 包围盒参数 | `Renderer.py:100-101` | `t` | `(81600, 3, 2)` | f64 |
| 出射参数 | `Renderer.py:102-104` | `far_bb` | `(81600, 1)` | f64 |
| 采样 | `Renderer.py:151-156` | `z_vals` | `(81600, 32)` | f64 |
| 点展开 | `Renderer.py:172-174` | `pts` → `pointsf` | `(81600, 32, 3)` → `(2611200, 3)` | f64 |
| 点分块 | `Renderer.py:39` | `pi` | 5×`(500000, 3)` + 1×`(111200, 3)` | f64 |
| 包围盒掩码 | `Renderer.py:44-47` | `mask` | 与 `pi` 同长 | bool |
| 网格特征 | `decoder.py:171-178` | `c` | `(500000, 32)` | f32 |
| 位置编码 | `decoder.py:189` | `embedded_pts` | `(500000, 93)` | f32 |
| 第 1~3 层 | `decoder.py:191-195` | `h` | `(500000, 32)` | f32 |
| skip 拼接 | `decoder.py:197` | `h` | `(500000, 125)` | f32 |
| 输出层 | `decoder.py:198-200` | `out` | `(500000,)` | f32 |
| 拼回 | `decoder.py:322-324`、`Renderer.py:61` | `raw` | `(2611200, 4)` | f32 |
| 还原按射线 | `Renderer.py:177` | `raw` | `(81600, 32, 4)` | f32 |
| α 合成 | `common.py:222-230` | `alpha`、`weights` | `(81600, 32)` | f32 |
| 深度 | `common.py:232-234` | `depth_map`、`depth_var` | `(81600,)` | f64 / f32 |
| 转 f64 并入列 | `Renderer.py:246-247` | `depth_list[*]` | `(81600,)` | f64 |
| 还原图像 | `Renderer.py:325-327` | `depth` | `(10, 68, 120)` | f64 |
| CBF 值 | `single_cbf.py:138` | `batch_new_h` | `(10,)` | f64 |

该表可直接用于核对修改后的形状是否自洽：位姿批维 B 与 `depth` 的第 0 维必须相等；`H`、`W` 必须与射线生成时的参数一致；`N_samples` 必须与 `z_vals` 第 1 维一致。

### 8.3 耗时构成与同步点

一步（10 个候选，批量）的耗时构成：

| 阶段 | 依据 |
| --- | --- |
| `grid_sample + MLP` 前向 | 0.101 s（2,611,200 点） |
| 其余（射线生成、采样、合成、主机侧位姿构造与 `.to()` 检查） | 0.103 s − 0.101 s ≈ 0.002 s |
| 展示渲染（不在控制环内，但每步执行一次） | 约 7 s（1200×680 两趟） |

因此控制环的安全判定本身约 0.1 s，整步约 7.1 s。`single_cbf.py` 跑 150 帧约 20 分钟与此一致（150 × 7.1 s ≈ 18 分钟，加上视频编码与首帧额外开销）。

同步点统计（一次成功判定的路径）：

| 位置 | 次数 | 触发原因 |
| --- | --- | --- |
| `single_cbf.py:118` | 1 | 0 维张量布尔判断 |
| `single_cbf.py:140` | 10 | 候选逐个判断 |
| `single_cbf.py:144` | 1 | 范数判断 |
| `single_cbf.py:146` | 1 | `float()` |
| `single_cbf.py:279` 的参数 `depth.min()` | 1 | 归约结果作为实参被传入并在 `:118` 取值 |
| `single_cbf.py:283` → `:295-296` | 2 | 深度图与彩色图 D2H |
| `single_cbf.py:297` | 1 | 打印 |

合计约 17 次同步，在本机（每次数十微秒量级，见 §2.4）贡献 1 ms 以下，相对 7 s 可忽略；但在把 `batch_size` 提到 100 且把渲染换成 `middle` stage 的小规模试验时，同步点数量与 MLP 前向时间会成为同一量级。

### 8.4 PyTorch Profiler 验证

可运行片段（工作目录 `upstream/`，需先构造 `r`、`c`、`dec`、`poses`，见 §8.6）：

```python
import torch
from torch.profiler import profile, ProfilerActivity

# 预热，避免 kernel 首次编译与 cuBLAS 句柄初始化污染统计
for _ in range(2):
    r.render_batch_img(c, dec, poses, 'cuda:0', stage='middle', gt_depth=None)
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    r.render_batch_img(c, dec, poses, 'cuda:0', stage='middle', gt_depth=None)
    torch.cuda.synchronize()

rows = []
for e in prof.key_averages():
    t = getattr(e, 'self_device_time_total', None)
    if t is None:                      # 旧版本 API 名称
        t = getattr(e, 'self_cuda_time_total', 0.0)
    rows.append((t, e.key, e.count))
rows.sort(reverse=True)
total = sum(acc[0] for acc in rows)
print(f'设备侧自耗时合计 {total/1000:.1f} ms')
for t, k, cnt in rows[:20]:
    if t > 0:
        print(f'{t/1000:8.2f} ms  x{cnt:<5d} {k[:70]}')

prof.export_chrome_trace('trace_step.json')   # 用 chrome://tracing 或 Perfetto 打开
```

读取方式：

1. `self_device_time_total` 高而 `count` 小的条目是"高吞吐 kernel"，优化需改变数据量（例如减少 MLP 层数）。
2. `self_device_time_total` 小但 `count` 很大的条目是"启动开销主导"，优化需减少调用次数（例如增大切块）。
3. `self_cpu_time_total` 与设备时间的差值反映主机侧阻塞，通常来自同步点与 `.to()` 检查。

在本机 Windows 环境下，`ProfilerActivity.CUDA` 可能因 CUPTI 不可用而失效，stderr 会出现：

```text
WARNING: function cbapi->getCuptiStatus() failed with error CUPTI_ERROR_INVALID_DEVICE (2)
WARNING: CUPTI initialization failed - CUDA profiler activities will be missing
```

此时 trace 中只有 CPU 活动，`self_device_time_total` 全为零。可用的替代方案是 §8.5 的手工事件计时；若需要 CUPTI，需以管理员身份运行并确认驱动组件完整（同一条警告提示 `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES` 时指向权限问题）。

### 8.5 手工分段计时

不受 CUPTI 影响的分段测量方式：

```python
import torch

def timed(fn, reps=5):
    fn(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / reps


# 整步
t_step = timed(lambda: r.render_batch_img(c, dec, poses, 'cuda:0', stage='middle', gt_depth=None))

# 仅网格采样 + MLP：跳过射线生成与 α 合成，直接对 2,611,200 个点查询
def query_only():
    with torch.no_grad():
        for k in range(0, 2611200, 500000):
            dec(pointsf[k:k+500000], c_grid=c, stage='middle')

t_fwd = timed(query_only)
print(f'整步 {t_step:.1f} ms   仅 grid_sample+MLP {t_fwd:.1f} ms   '
      f'占比 {t_fwd/t_step*100:.1f}%')
```

其中 `pointsf` 是 `(2611200, 3)` 的采样点（可由 §8.6 的脚本得到）。预期比值接近 0.101/0.103 ≈ 98%，与 §8.3 的构成表一致。

### 8.6 自包含复现脚本

```python
# 文件：upstream/_check_step.py，工作目录 upstream/
# 运行：设置 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 后 python _check_step.py
import sys
import numpy as np
import torch

sys.path.insert(0, '.')
from src.utils.Renderer import Renderer
from src.common import get_rays

DEC = torch.load('replica_room1.pth', map_location='cuda:0', weights_only=False).eval()
C = np.load('replica_room1.npy', allow_pickle=True).item()


class _Slam:
    nice = True
    bound = torch.from_numpy(np.array([[-7.0, 2.92], [-4.6, 4.36], [-3.0, 3.08]]))
    H, W, fx, fy, cx, cy = 68, 120, 60.0, 60.0, 59.95, 33.95


CFG = {'rendering': {'lindisp': False, 'perturb': 0.0,
                     'N_samples': 32, 'N_surface': 16, 'N_importance': 0},
       'scale': 1, 'occupancy': True}
r = Renderer(CFG, None, _Slam)

from scipy.spatial.transform import Rotation
st = torch.tensor([-2.0, 0.2, 0.5, 90.0, 0.0, 115.0], device='cuda:0')
rm = torch.from_numpy(Rotation.from_euler('xyz', st[3:].cpu(), degrees=True).as_matrix()).to('cuda:0')
pose = torch.cat((torch.cat((rm, st[:3].unsqueeze(-1)), 1),
                  torch.tensor([[0., 0, 0, 1]], device='cuda:0')), 0)
poses = pose.unsqueeze(0).repeat(10, 1, 1)

ro, rd = get_rays(68, 120, 60.0, 60.0, 59.95, 33.95, pose, 'cuda:0')
ro1, rd1 = ro.reshape(-1, 3), rd.reshape(-1, 3)
print('c2w', tuple(poses.shape), poses.dtype)
print('rays', tuple(ro1.shape), '|rays_d| 区间',
      float(rd1.norm(dim=-1).min()), float(rd1.norm(dim=-1).max()))

c = C
dec = DEC

# 按 Renderer.render_batch_ray 的方式构造采样点：near = 0.01，far 由包围盒求交得到
with torch.no_grad():
    ro_b = ro1.repeat(10, 1)                                  # (81600, 3)
    rd_b = rd1.repeat(10, 1)                                  # (81600, 3)
    t_bb = (_Slam.bound.unsqueeze(0).to('cuda:0') - ro_b.unsqueeze(-1)) / rd_b.unsqueeze(-1)
    far = torch.min(torch.max(t_bb, dim=2)[0], dim=1)[0].unsqueeze(-1) + 0.01
    t_vals = torch.linspace(0., 1., 32, device='cuda:0')
    z_vals = 0.01 * (1 - t_vals) + far * t_vals                # (81600, 32)
    pointsf = (ro_b[..., None, :] + rd_b[..., None, :] * z_vals[..., :, None]).reshape(-1, 3)
print('far 区间 %.3f ~ %.3f' % (float(far.min()), float(far.max())))
print('pointsf', tuple(pointsf.shape), pointsf.dtype)

torch.cuda.reset_peak_memory_stats()
d, _, _ = r.render_batch_img(c, dec, poses, 'cuda:0', stage='middle', gt_depth=None)
torch.cuda.synchronize()
print('depth', tuple(d.shape), d.dtype,
      'peak_alloc %.0f MB' % (torch.cuda.max_memory_allocated() / 1024 / 1024))
print('h_{t+1} =', [round(float(x), 4) for x in (0.1 - d.min(dim=-1)[0].min(dim=-1)[0])])
```

预期输出：`c2w (10, 4, 4) torch.float64`；`rays (8160, 3)`；`|rays_d|` 区间约为 `1.000001 ~ 1.522663`；`far` 区间约为 `3.624 ~ 6.539`；`pointsf (2611200, 3) torch.float64`；`depth (10, 68, 120) torch.float64`；峰值分配约 0.7~0.8 GB（比单独渲染高，因为上面构造的 `pointsf`、`t_bb` 仍在显存中）。脚本保留 `r`、`c`、`dec`、`poses`、`pointsf`、`ro1`、`rd1` 供 §3~§8 的其他片段直接使用。

---

## 9 修改实践：改动清单与约束

### 9.1 改动清单

| 想改什么 | 文件:行 | 函数 | 必须同时满足的约束 | 验证方法 |
| --- | --- | --- | --- | --- |
| CBF 函数 `h` 的定义 | `single_cbf.py:116`、`single_cbf.py:138` | `find_safe_action` | `h_t` 与 `h_{t+1}` 必须同源：同一 `stage`、同一分辨率、同一 `N_samples`、同一 `gt_depth` 设置。当前实现中二者不同源：`h` 由展示渲染计算（`single_cbf.py:279` 的实参 `d - depth.min()`，`depth` 来自 `single_cbf.py:283` 的 1200×680 `robot.render`），而 `new_h` 由控制环渲染计算（`single_cbf.py:116` 的 120×68 `predict_observation`） | §8.6 脚本打印两组数，比较同一位姿下的取值 |
| 动力学模型 | `single_cbf.py:45-46`；`double_cbf.py:41-42` | `update_dynamics` | 返回值个数必须与调用点匹配（`double_cbf.py:108` 解包两个值）；状态布局固定为 `[x, y, z, roll, pitch, yaw(度)]`（`single_cbf.py:105`、`single_cbf.py:35-43`）；`time_step` 与 CBF 的 `alpha` 需一致（`single_cbf.py:26`、`single_cbf.py:28`） | 单步后打印 `state` 与 `pose[:3, -1]`，核对位移等于 `time_step × action` |
| 候选动作采样 | `single_cbf.py:122-137`；`double_cbf.py:118-130` | `find_safe_action` | `batch_action` 的形状 `(batch_size, 6)` 必须与 `state_to_pose` 的批维一致；`batch_size` 改变后 `batch_new_h` 的第 0 维随之改变；采样分布参数 `mu`、`sigma`（`single_cbf.py:27`、`single_cbf.py:32`）决定搜索密度 | 打印 `batch_new_h` 的分布，检查是否有候选落入安全集 |
| 候选动作的搜索上限 | `single_cbf.py:122` | `find_safe_action` | `while True` 无终止条件：本轮 10 个候选全部不满足时继续重采，无上界也无日志 | 在 `while` 内计数并打印，观察是否存在长时间不退出 |
| 控制环分辨率 | `single_cbf.py:54-59` | `Robot.__init__` | `H`、`W`、`fx`、`fy`、`cx`、`cy` 必须同步修改，否则射线方向与实际视场不一致；改动后 `predict_observation` 的点数变为 `H×W×32` | 用 §6.6 的一致性检查确认各位置姿深度可比较 |
| 展示分辨率 | `single_cbf.py:73-78` | `Robot.render` | 必须在 `single_cbf.py:94-99` 恢复控制分辨率的六个参数，否则下一步的控制环按展示分辨率渲染（射线数增大 100 倍） | 在 `single_cbf.py:100` 前打印 `self.renderer.H/W`，确认为 68/120 |
| 视频输出尺寸 | `single_cbf.py:199` | `VideoWriter` | 输出尺寸须与拼接后的画布一致：单帧为深度图（H×W）与彩色图横向拼接，因此宽为 `2W`、高为 `H`；本仓库已按此改到 `(2400, 680)`，对应 1200×680 | 播放生成的 `video.avi`，确认无拉伸与裁切 |
| 采样点数 | `configs/nice_slam.yaml:105-108` | `Renderer.__init__` 读取 | `N_samples` 改变后每条射线的点数与 `z_vals` 第 1 维随之改变，`h` 的精度随之改变；`N_surface > 0` 只在传入 `gt_depth` 时生效；`N_importance > 0` 会启用 `sample_pdf` 并额外做一次完整查询（`Renderer.py:181-195`） | 打印 `raw` 形状，确认 `(N_rays, N_samples+N_surface, 4)` |
| 提速：减小切块 | `Renderer.py:7` | `Renderer.__init__` | 减小 `points_batch_size`/`ray_batch_size` 降显存但升调用次数；增大则相反；B 使 `B×H×W > ray_batch_size` 时第一级切块增加 | §5.5 的显存/耗时扫描 |
| 提速：减少解码器工作量 | `decoder.py:299-306` | `NICE.__init__` | 改变 `n_blocks`、`hidden_size`、`skips` 会使预训练权重形状不匹配，`load_state_dict` 报错；如需改动必须重新训练或用随机初始化验证形状通路 | 打印 `pts_linears[i].weight.shape`，与 §7.5 的表比对 |
| 提速：换 stage | `single_cbf.py:67` | `Robot.predict_observation` | `middle` 换成 `coarse` 会少一次 MLP 前向但精度下降；换成 `fine`/`color` 会多 1~2 次 MLP 前向与 2~3 次 `grid_sample`，控制环时间上升 2~3 倍 | §7.6 的 hook 统计 |

### 9.2 `h_t` 与 `h_{t+1}` 同源的具体含义

CBF 条件 `h_{t+1} ≤ α h_t`（`single_cbf.py:118`、`single_cbf.py:140`）比较两个标量。若二者由不同的渲染配置产生，比较失去物理意义。当前实现中三处差异同时存在：

| 差异 | `h_t` 的来源 | `h_{t+1}` 的来源 |
| --- | --- | --- |
| 分辨率 | 1200×680（`single_cbf.py:73-78`） | 120×68（`single_cbf.py:54-59`） |
| `stage` | `'color'`（`single_cbf.py:85`、`single_cbf.py:92`） | `'middle'`（`single_cbf.py:67`） |
| `gt_depth` | 第一趟为 `None`，第二趟为第一趟结果（`single_cbf.py:87-93`） | 始终为 `None` |

修改 `h` 的定义时，若只改 `single_cbf.py:116`，`h_t` 与 `h_{t+1}` 的差异被保留；若要让二者同源，需同时改动 `single_cbf.py:279` 传入的 `h`（改为控制分辨率与 `middle` stage 的计算结果）。同一个量还在 `single_cbf.py:204`（初始 `h`）与 `single_cbf.py:302`、`single_cbf.py:320`、`single_cbf.py:338`（另外三个 `direction` 分支）以 `d - depth.min()` 的形式出现，改动必须覆盖全部引用点。

### 9.3 常见错误

| 现象 | 原因 | 定位 |
| --- | --- | --- |
| `FileNotFoundError: device.txt` | 工作目录不是 `upstream/` | `common.py:5`、`decoder.py:6`、`single_cbf.py:23`、`double_cbf.py:19` 使用相对路径 |
| `pickle.UnpicklingError: Weights only load failed` | torch ≥ 2.6 的 `weights_only=True` 默认值 | `single_cbf.py:52`、`double_cbf.py:47`；用 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` |
| `RuntimeError: Attempting to deserialize object on a CUDA device` | 用 CPU-only 版 PyTorch 运行 | 确认 `torch.cuda.is_available()` 为 `True` |
| `TypeError: clamp() received an invalid combination of arguments` | 未在 `torch.no_grad()` 内调用 `render_batch_ray`，`Renderer.py:108` 的 `torch.max` 保留梯度 | 参照 `Renderer.py:216`、`Renderer.py:277` 的外层 `with torch.no_grad()` |
| 深度图数值异常大或全为 `far` | 相机位姿在包围盒外，`Renderer.py:100-102` 求交得到入射参数 | 打印 `far_bb`，与 `bound` 比较 |
| 深度图与彩色图尺寸不符 | 未恢复控制分辨率 | 见 §9.1 的"展示分辨率"行 |
| CBF 值单调下降且动作被持续替换 | 采样分布过宽或 `alpha` 取值不当 | `single_cbf.py:27-32` 的 `mu`、`sigma`、`intend`、`alpha` |

---

## 10 自测题

**题 1** 本仓库中是否存在自定义 CUDA kernel？一次 `F.grid_sample` 调用对应什么粒度的 GPU 工作？

<details><summary>答案</summary>

不存在自定义 kernel：`upstream/` 下没有 `.cu` 文件，全部计算由 PyTorch 算子发起。一次 `F.grid_sample` 对应一次（或含反向的两次）kernel 启动，其并行度由输出元素数决定：本仓库 middle 层单次调用输出 `(1, 32, 500000, 1, 1)`，约 1.6×10⁷ 个 f32 元素，每个元素做 8 邻域插值。代码位置 `decoder.py:171-172`。

</details>

**题 2** 为什么 `single_cbf.py` 打印的 `time cost` 不需要额外加 `torch.cuda.synchronize()` 也有意义？

<details><summary>答案</summary>

因为该 step 内必然存在同步点：`single_cbf.py:118`/`:140`/`:144` 的 0 维张量布尔判断、`:146` 与 `:297` 的 `float()`/格式化、`:295-296` 的 `.cpu().numpy()` 都会等待设备排空并把结果取回主机。因此测量的 wall time 包含完整的设备执行时间。若把某段 GPU 工作单独拿出来计时，则必须显式同步或用 CUDA 事件（§4.3）。

</details>

**题 3** `rays_d` 未归一化。`depth_map` 的数值语义是什么？与欧氏距离的关系如何？

<details><summary>答案</summary>

`depth_map = Σ w_i z_i`（`common.py:232`），`z_i` 是射线参数，满足 `x(t) = rays_o + t·rays_d`（`Renderer.py:172-173`）。`rays_d` 的第 3 个分量为 −1（`common.py:251`），长度为 `|((i−cx)/fx, −(j−cy)/fy, −1)|`，本机在 120×68 下为 1.000001~1.522663。因此欧氏距离 = `t · ‖rays_d‖`，仅在光轴上等于 `depth_map`。`common.py:219` 的 `dists * ‖rays_d‖` 只在体积密度分支使用，`occupancy=True` 时不参与 α。

</details>

**题 4** 控制环一步评估多少个空间点？给出算式。

<details><summary>答案</summary>

120×68 = 8,160 条射线/位姿；10 个候选 ⇒ 81,600 条射线；每条射线 `N_samples = 32`（`gt_depth=None` 使 `N_surface` 被置 0，`Renderer.py:89-91`）⇒ 81,600 × 32 = 2,611,200 个点。单位姿为 8,160 × 32 = 261,120 个点。

</details>

**题 5** `N_importance = 0` 在本流程中意味着什么？如果改成 4 会怎样？

<details><summary>答案</summary>

意味着 `Renderer.py:181-195` 整段不执行，`common.py:20-63` 的 `sample_pdf` 不会被调用，采样位置完全由 `near`、`far`、`N_samples` 与（有 GT 深度时的）`N_surface` 决定。改成 4 后，每条射线会追加 4 个按权重分布的重要性采样点，再对 `(N_rays, N_samples+4+N_surface)` 做第二次完整查询（一次 `eval_points`，即一次 `grid_sample`+MLP 前向），控制环时间近似翻倍。

</details>

**题 6** 为什么 `normalize_3d_coordinate` 的调用点在 `decoder.py:167` 和 `decoder.py:252` 都写成 `p.clone()`？

<details><summary>答案</summary>

因为该函数原地修改入参：`common.py:269` 的 `p = p.reshape(-1, 3)` 在连续张量上返回视图，随后 `p[:, 0] = ...` 等三行直接写入该视图，修改的是原张量的数据。`MLP.forward` 在采样网格特征之后还要用原始坐标做位置编码（`decoder.py:187-189`），若坐标已被归一化，位置编码的输入会变成 `[-1, 1]` 区间，结果错误。`clone` 使原地写只作用于副本。

</details>

**题 7** `align_corners=True` 对有效体素间距的影响是什么？middle 网格的实际间距是多少？

<details><summary>答案</summary>

`normalize_3d_coordinate` 把 `bound` 两端映射到 `-1` 与 `+1`；`align_corners=True` 表示 `-1` 对应首个体素中心、`+1` 对应末个体素中心，因此相邻体素中心的间距为 `边长/(n−1)`，而不是配置值 `grid_len = 边长/n`。middle 网格的 `bound` 边长为 `(9.92, 8.96, 6.08)`，体素数为 `(31, 28, 18)`，有效间距为 `(0.3307, 0.3319, 0.3576)`，配置值为 0.32。z 方向偏差最大，因为 `int(6.08/0.32)` 因浮点结果为 18.999… 而取到 18。

</details>

**题 8** 解码器的隐藏宽度是 256 还是 32？给出两条独立证据。

<details><summary>答案</summary>

是 32。证据一：`NICE.__init__` 的默认参数为 `hidden_size=32`（`decoder.py:291`），并显式传给四个解码器（`decoder.py:296-306`），而 `conv_onet/config.py:24-27` 未覆盖该值；`MLP` 签名里的 `hidden_size=256` 默认值生效不了。证据二：预训练权重形状为 `pts_linears.0.weight (32, 93)`、`pts_linears.3.weight (32, 125)`、`output_linear.weight (1, 32)`，四个解码器合计 58,956 个参数。

</details>

**题 9** 控制环 10 个候选为什么一次性批量渲染比循环 10 次单位姿快？收益的量级与什么有关？

<details><summary>答案</summary>

`render_batch_img`（`Renderer.py:277-291`）把 B 个位姿的射线拼成 `(B*H*W, 3)`，使 B 个位姿落在同一个 `ray_batch_size` 分块内，从而把 `render_batch_ray` 的调用次数（含 `Renderer.py:100` 的 `bound` 上传、`z_vals` 构造、`torch.cat` 拼接结果、`depth.double()`）从 B 次降为 1 次，把解码器调用次数从 B 次降为 `ceil(N_points/points_batch_size)` 次。实测为 0.121 s → 0.103 s（约 1.17 倍）。收益与"固定开销在单次工作中所占比例"成正比：单次工作量越大（分辨率越高）收益越小；`B×H×W` 超过 `ray_batch_size` 后第一级切块恢复，收益进一步减小。

</details>

**题 10** 展示渲染一帧约 7 s，控制环判定一步约 0.1 s。为什么总步时的瓶颈是展示渲染？给出两个可验证的判据。

<details><summary>答案</summary>

展示渲染的射线数 1200×680 = 816,000，是控制环单次判定的 10 倍（81,600），第二趟采样点数为 48 而控制环为 32，因此单帧点数为 39,168,000，是控制环 2,611,200 的 15 倍；且 `stage='color'` 每次 `eval_points` 触发 3 次 MLP 前向与 4 次 `grid_sample`（`decoder.py:332-338`），而 `stage='middle'` 只有 1 次。判据一：用 §8.5 的分段计时分别测 `robot.render` 与 `predict_observation`，前者约为后者 70 倍。判据二：150 帧 × 7.1 s ≈ 18 分钟，与 `single_cbf.py` 跑 150 帧约 20 分钟一致；若控制环是瓶颈，总时长将不足 1 分钟。

</details>

---

## 11 参考资料

1. NVIDIA, *CUDA C++ Programming Guide*（执行模型、内存层次、SIMT 与 warp 的定义）：<https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
2. NVIDIA, *CUDA C++ Best Practices Guide*（算术强度、访存合并、共享内存分块）：<https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/>
3. PyTorch, `torch.nn.functional.grid_sample` 文档（5 维输入的三线性插值、`align_corners`、`padding_mode` 语义）：<https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html>
4. PyTorch, `torch.profiler` 文档（`ProfilerActivity`、`key_averages`、Chrome trace 导出）：<https://pytorch.org/docs/stable/profiler.html>
5. PyTorch, *CUDA semantics* 文档（异步执行、同步、缓存分配器、`PYTORCH_CUDA_ALLOC_CONF`）：<https://pytorch.org/docs/stable/notes/cuda.html>
6. B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, R. Ng, "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis", ECCV 2020, arXiv:2003.08934（体渲染合成、分层采样 `sample_pdf`）。
7. M. Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains", NeurIPS 2020, arXiv:2006.10739（`GaussianFourierFeatureTransform` 的来源，见 `decoder.py:9-15` 的注释）。
8. S. Peng, M. Niemeyer, L. Mescheder, A. Geiger, J. Park, "Convolutional Occupancy Networks", ECCV 2020, arXiv:2003.04618（`decoder.py` 的 `MLP`、`grid_len`、`skips` 设计与预训练权重的来源）。
9. Z. Zhu, S. Peng, V. Larsson, W. Xu, H. Bao, Z. Cui, M. R. Oswald, M. Pollefeys, "NICE-SLAM: Neural Implicit Scalable Encoding for SLAM", CVPR 2022, arXiv:2112.12130（多分辨率特征网格、四个解码器、`stage` 的划分）。
10. E. Sucar, S. Liu, J. Ortiz, A. J. Davison, "iMAP: Implicit Mapping and Positioning in Real-Time", ICCV 2021, arXiv:2103.12352（`imap.yaml` 对应的基线，`conv_onet/config.py:29-31` 的 `decoder_dict['imap']`）。
11. A. D. Ames, X. Xu, J. W. Grizzle, P. Tabuada, "Control Barrier Function Based Quadratic Programs for Safety Critical Systems", IEEE TAC, 2017（离散时间 CBF 条件的形式化）。
12. T. Tong, C. Dawson, C. Fan, "Enforcing Safety for Vision-based Controllers via Control Barrier Functions and Neural Radiance Fields", ICRA 2023（本仓库复现的论文）。
