# 在 PyCharm 里跑通论文结果

> 本仓库的原则是**上游代码一字不动**（唯一例外：`single_cbf.py` 里 `cv2.VideoWriter` 的尺寸从 `(240,68)` 改成 `(2400,680)`，见 README）。
> 所以 PyCharm 里的配置要做的，只是"把上游原本在命令行里做的事搬进 IDE"：设一个环境变量、准备占位数据集、在 `upstream/` 目录下运行。

---

## 0. 三个必须知道的点（否则一定跑不起来）

| 要点 | 值 | 为什么 |
|---|---|---|
| **环境变量** | `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` | torch ≥ 2.6 把 `torch.load` 的默认值改成了 `weights_only=True`，而上游的三个权重文件（`replica_room1.pth`、`pretrained/coarse.pt`、`pretrained/middle_fine.pt`）都是整体 pickle 的对象，不绕过就直接 `UnpicklingError`。这是**唯一**能不改代码的办法。 |
| **工作目录** | `<仓库>/upstream` | 上游用相对路径读 `configs/Replica/room1.yaml`、`device.txt`、`replica_room1.npy/.pth`，还有 `Datasets/Replica/room1`。 |
| **占位数据集** | 先跑一次 `python tools/make_dataset_stub.py` | 上游 `NICE_SLAM.__init__` 会读 `Datasets/Replica/room1`，但实验只用第 0 帧做形状校验。占位目录 3 秒生成，不影响任何数值。 |

---

## 1. 配解释器

```bash
# 建议用 conda(与实测环境一致)
conda env create -f upstream/environment.yaml     # 上游自己那份(Linux 向,Windows 上装不起来的话见下)
```

Windows 上更稳妥的是自建环境（上游 `environment.yaml` 锁了 Linux + Python 3.7 + torch 1.11/cu113，新卡装不上）：

```bash
conda create -n cbf python=3.11 -y
conda activate cbf
pip install torch --index-url https://download.pytorch.org/whl/cu128    # RTX 50 系需要 cu128
pip install numpy==1.26.4 scipy matplotlib opencv-python
```

PyCharm：`File → Settings → Project → Python Interpreter → Add Interpreter`,选到上面那个 `python.exe`。

**验证**（PyCharm 的 Python Console 里）：

```python
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
```

`True` 就说明是 CUDA 版。`False` 说明装成了 CPU 版，虽然也能跑，但每步 7 秒的高分辨率渲染在 CPU 上会变成分钟级。

---

## 2. 准备数据集占位 + 权重

仓库已内置作者发布的权重（`upstream/replica_room1.npy/.pth` + `upstream/pretrained/*.pt`，共约 150 MB），不需要下载。

只需生成数据集占位（在仓库根目录执行一次）：

```bash
python tools/make_dataset_stub.py
```

它会生成：

```
upstream/Datasets/Replica/room1/
├── traj.txt
└── results/
    ├── frame000000.jpg    1200×680 黑图
    └── depth000000.png    1200×680 常量图
```

> 想用真数据就跳过这步，把完整的 Replica `room1` 放到同一路径（实验数值不变，因为脚本只读 `frame_reader[0]` 一个张量）。

---

## 3. 建 Run Configuration

`Run → Edit Configurations → + → Python`。两个实验各一个：

| 字段 | 值（两个配置相同，除了 Script path） |
|---|---|
| **Script path** | `<仓库>\upstream\single_cbf.py`（双积分器用 `double_cbf.py`） |
| **Parameters** | `configs/Replica/room1.yaml --output output/Replica/room1` |
| **Working directory** | `<仓库>\upstream` ← **必须是这个** |
| **Environment variables** | `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1;PYTHONIOENCODING=utf-8` |
| **Python interpreter** | 第 1 步选好的 |
| ☑ Emulate terminal in output console | 建议勾上 |
| ☑ Run with Python Console（可选） | 想在终端里继续敲代码时勾 |

两个配置的耗时（RTX 5070 Laptop）：

| 配置 | 步数（上游写死） | 产出 | 耗时 |
|---|---|---|---|
| `upstream-single` | 150 帧（15 s 仿真） | `upstream/video.avi`（2400×680，论文图 2 的帧序列） | ≈ 20 min |
| `upstream-double` | 100 步（10 s 仿真） | 只有日志（上游把视频写出的两行注释掉了） | ≈ 12 min |

> 上游脚本**没有** `--frames` 之类的参数，步数是写死的（`while frame < 150` / `for i in range(100)`）。想缩短只能改源码——那就不叫"零改动"了。

---

## 4. 跑完出图

上游脚本只打印日志、不画图。出图用新增的脚本（不碰上游文件）：

```bash
python tools/plot_from_log.py logs/single_upstream.log logs/double_upstream.log --csv
```

产出：

- `results/figures/figure3a_single.png` —— 单积分器：最小距离 / CBF 值 / 干预量（对照论文图 3a）
- `results/figures/figure4a_double.png` —— 双积分器：最小距离 / CBF 值 / 速度 / 干预量（对照论文图 4a）
- `results/figures/*_metrics.csv` —— 逐帧数值，便于自己复核

> 在 PyCharm 里也可以直接把 `tools/plot_from_log.py` 配一个 Run Configuration（Parameters 填 `logs/single_upstream.log --csv`），跑完双击 PNG 看。

**一步到位**（不建配置也行，直接在 PyCharm 的 Terminal 里）：

```powershell
powershell -ExecutionPolicy Bypass -File run_upstream.ps1 -Which all
```

---

## 5. 在 PyCharm 里调试

上游代码有 `if __name__ == '__main__':` 保护（`single_cbf.py:152`），所以**可以放心 import**，但断点只能下在 `__main__` 块里。

适合下断点的位置（`single_cbf.py`）：

| 想看什么 | 行号 | 看哪些变量 |
|---|---|---|
| 标称指令是否安全 | 118 `if new_h <= alpha * h:` | `new_h`、`h`、`alpha * h` |
| 10 个候选各自的 `h_{t+1}` | 138 `batch_new_h = ...` | `batch_new_h`（形状 `(10,)`）、`batch_action` |
| 最终选了谁 | 144 `if torch.norm(best_action, p=2) > 0:` | `best_action`、`new_h` |
| 每帧的最近距离 | 297 `print('min_depth = ...')` | `depth.min()` |

**为什么图 3a 的 CBF 曲线可以"算"出来**：主循环把 `d - depth.min()` 当作 `h_t` 传给滤波器，而 `d = 0.1` 是常量，所以 `h_t = 0.1 − min_depth_{t−1}`。`tools/plot_from_log.py` 就是这么重建的。

---

## 6. 常见错误对照表

| 现象 | 原因 | 处理 |
|---|---|---|
| `_pickle.UnpicklingError: Weights only load failed` | 没设 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` | 在 Run Configuration 的 Environment variables 里加上 |
| `FileNotFoundError: 'device.txt'` | 工作目录不是 `upstream/` | 把 Working directory 改成 `upstream` |
| `FileNotFoundError: .../Datasets/Replica/room1/traj.txt` | 没生成占位数据集 | `python tools/make_dataset_stub.py` |
| `AssertionError` 于 `as_euler`/维度相关 | 占位 `traj.txt` 写坏了 | 删掉 `upstream/Datasets` 重新生成 |
| `video.avi` 只有 5.7 KB（空壳） | 用的是未打补丁的上游文件（尺寸 `(240,68)`） | 本仓库已改成 `(2400,680)`；若你自己 clone 的上游原版则会重现这个 bug |
| `torch.cuda.is_available() == False` | 装的是 CPU 版 torch | 重装 CUDA 版 |
| 每帧要 7~20 秒 | 正常：上游每步都做 1200×680 的两趟渲染 | 等；或改用 `double_cbf.py`（也是同样代价） |
| 中文日志乱码 | 控制台编码 | 勾 `Emulate terminal in output console` + `PYTHONIOENCODING=utf-8` |
| 日志文件打开是乱码/带很多空格 | PowerShell `>` 写的是 UTF-16 | `tools/plot_from_log.py` 已自动识别；手工看就用编辑器切换编码 |

---

## 7. 一页速查

```text
环境变量 : TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
工作目录 : <仓库>\upstream
数据集   : python tools/make_dataset_stub.py   (只需一次)
图 3a+2  : upstream\single_cbf.py  configs/Replica/room1.yaml --output output/Replica/room1   (~20 min)
图 4a    : upstream\double_cbf.py  configs/Replica/room1.yaml --output output/Replica/room1   (~12 min)
出图     : python tools/plot_from_log.py logs/*.log --csv
图 5     : python tools/sweep_upstream.py          (~70 min,见脚本内说明)
```
