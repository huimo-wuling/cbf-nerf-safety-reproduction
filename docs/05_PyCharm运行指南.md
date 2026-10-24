# 在 PyCharm 里跑通论文结果

> 本仓库的原则是**上游代码一字不动**（唯一例外：`single_cbf.py` 第 199 行 `cv2.VideoWriter` 的尺寸从 `(240,68)` 改为 `(2400,680)`，见 README）。
> PyCharm 里要做的事只有三件：把目录整份复制进去、选对解释器、在 `upstream/` 目录下运行。

---

## 0. 三个必须知道的点（任一不满足都跑不起来）

| 要点 | 值 | 原因 |
|---|---|---|
| **环境变量** | `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` | torch ≥ 2.6 把 `torch.load` 的默认参数 `weights_only` 改为 `True`。上游加载的三个权重（`replica_room1.pth`、`pretrained/coarse.pt`、`pretrained/middle_fine.pt`）都是整体 pickle 对象，在新默认值下抛 `UnpicklingError`。设该变量是**不改代码**的唯一办法。 |
| **工作目录** | `<仓库>/upstream` | 上游用相对路径读 `configs/Replica/room1.yaml`、`device.txt`、`replica_room1.npy/.pth` 与 `Datasets/Replica/room1`。 |
| **占位数据集** | 运行一次 `python tools/make_dataset_stub.py` | 上游 `NICE_SLAM.__init__` 会读该目录，但实验只用 `frame_reader[0]` 做形状校验，数值不参与计算。 |

---

## 1. 把代码放进 PyCharm 项目

代码必须**整目录复制**。只复制 `single_cbf.py`、`double_cbf.py` 无法运行：它们依赖同级的 `src/`、`configs/`、`device.txt` 以及 147 MB 权重。

### 方式 A：从本机已有副本复制（推荐）

在 PyCharm 的 Terminal 中执行：

```powershell
robocopy "C:\Users\L\Desktop\cbf-nerf-safety-reproduction" `
         "C:\Users\L\PycharmProjects\PythonProject3\cbf-nerf" `
         /E /XD .git output Datasets __pycache__ /XF video.avi
```

`/XD` 排除运行时产物（`output/`、`Datasets/`、缓存），`/XF` 排除视频文件；其余全部复制，约 152 MB。

### 方式 B：从 GitHub 克隆

```powershell
cd C:\Users\L\PycharmProjects\PythonProject3
git clone https://github.com/huimo-wuling/cbf-nerf-safety-reproduction.git cbf-nerf
```

### 复制后的结构

```text
C:\Users\L\PycharmProjects\PythonProject3\
├── main.py                       项目原有文件,与本文无关
├── .idea\
└── cbf-nerf\
    ├── upstream\                 复现代码与权重(151 MB)
    │   ├── single_cbf.py         单积分器实验(图 2 / 3a)
    │   ├── double_cbf.py         双积分器实验(图 4a)
    │   ├── src\ configs\         NICE-SLAM 实现与配置
    │   ├── replica_room1.npy/.pth, pretrained\*.pt
    │   └── device.txt
    ├── tools\                    新增辅助脚本(占位数据集、出图、抽帧、扫参)
    ├── logs\ results\ docs\
    └── run_upstream.ps1 / run_upstream.sh
```

**不要**用编辑器打开文件再"另存"或复制粘贴文件内容——`.npy`、`.pth`、`.pt` 是二进制权重，文本方式复制会损坏；用资源管理器拖拽或上述命令复制。

### 可选：减少索引开销

PyCharm 会索引 `upstream/pretrained/`（117 MB）与 `replica_room1.npy`（33 MB），导致卡顿。右键这两个条目 → `Mark as Excluded`，不影响运行。

---

## 2. 选解释器并验证

### 2.1 不要使用空环境

新建项目时 PyCharm 常默认选一个空的 conda 环境或新建 venv。本项目需要 `torch`、`numpy`、`opencv-python`、`scipy`、`matplotlib`，缺任一项都会在导入阶段失败。

选择：`File → Settings → Project: <项目名> → Python Interpreter → Add Interpreter → Add Local Interpreter → Conda Environment → Existing environment`，指向一个**已装好 CUDA 版 torch** 的 `python.exe`。

### 2.2 一条命令验证

在 PyCharm 的 Python Console 或 Terminal 中执行：

```python
import torch, numpy, cv2, scipy, matplotlib
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(numpy.__version__, cv2.__version__)
```

期望输出（实测环境）：

```text
2.11.0+cu128 12.8 True
1.26.4 4.11.0
```

`is_available()` 为 `False` 表示装的是 CPU 版 torch：仍可运行，但每步 1200×680 渲染会慢一到两个数量级。

### 2.3 若环境未安装依赖

```powershell
conda create -n cbf python=3.11 -y
conda activate cbf
pip install torch --index-url https://download.pytorch.org/whl/cu128   # RTX 50 系需要 cu128
pip install numpy==1.26.4 scipy matplotlib opencv-python
```

上游自带的 `upstream/environment.yaml` 锁定 Linux + Python 3.7 + torch 1.11/cu113，在新显卡与新 Python 上无法安装，不建议使用。

---

## 3. 生成占位数据集（一次）

在 Terminal 中，工作目录为 `cbf-nerf`：

```powershell
cd C:\Users\L\PycharmProjects\PythonProject3\cbf-nerf
python tools/make_dataset_stub.py
```

生成：

```text
upstream/Datasets/Replica/room1/
├── traj.txt
└── results/
    ├── frame000000.jpg    1200×680
    └── depth000000.png    1200×680
```

脚本根据自身位置定位仓库根目录，无需关心当前工作目录。若要与作者环境完全一致，把完整的 Replica `room1` 覆盖到同一路径即可，实验数值不变。

---

## 4. 建 Run Configuration

### 4.1 打开对话框

两种入口，效果相同：

- 菜单 `Run → Edit Configurations...`；
- 或点击窗口右上角运行按钮左侧的下拉框（显示当前配置名）→ `Edit Configurations...`。

已有配置列表在左侧，右侧是该配置的 `Configuration` 面板。

### 4.2 新建第一个配置（单积分器）

按顺序执行：

1. 点击对话框左上角的 `+`（`Add New Configuration`），在弹出列表中选择 **`Python`**。
   - 不要选 `Python tests`、`Python Debug Server`、`Django server` 等；
   - 若列表里没有 `Python`，说明当前项目未被识别为 Python 项目：先回到第 2 节配置解释器。
2. `Name` 输入 `upstream-single`。
3. 确认面板顶部的单选项为 **`Script path`**（另一项是 `Module name`，用于以 `-m 模块名` 方式运行，本项目不适用）。
4. `Script path` 点右侧文件夹图标，选择：

   ```text
   C:\Users\L\PycharmProjects\PythonProject3\cbf-nerf\upstream\single_cbf.py
   ```

5. `Parameters`（脚本参数）文本框填：

   ```text
   configs/Replica/room1.yaml --output output/Replica/room1
   ```

   这两项都是相对路径，相对下面的 `Working directory` 解析：`configs/Replica/room1.yaml` 是场景配置，`--output output/Replica/room1` 指定上游写日志/检查点的目录（会在 `upstream/output/` 下创建）。

6. `Python interpreter` 下拉框选择第 2 节验证通过的那个：

   ```text
   D:\miniconda\envs\nerf-cbf\python.exe
   ```

   若下拉列表中没有它，点下拉框右侧的 `...` 或 `Add Interpreter` 添加（同一对话框可完成，不需退出）。

7. `Working directory`（工作目录）改为：

   ```text
   C:\Users\L\PycharmProjects\PythonProject3\cbf-nerf\upstream
   ```

   该字段默认是项目根目录，必须手工改。上游用相对路径读取 `device.txt`、`configs/`、`replica_room1.npy/.pth` 与 `Datasets/Replica/room1`，工作目录不对会在第一步就报 `FileNotFoundError: 'device.txt'`。

8. `Environment variables`（环境变量）：点击该行右侧的图标（不同版本为 `...` 或文件夹图标），弹出键值表格，逐行添加两对：

   | Name | Value |
   |---|---|
   | `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` | `1` |
   | `PYTHONIOENCODING` | `utf-8` |

   注意是「名称一栏、值一栏」，不要把 `NAME=VALUE` 整串填进 Name。部分版本可直接在单行文本框里用分号书写：`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1;PYTHONIOENCODING=utf-8`。

9. `Emulate terminal in output console`（在输出控制台模拟终端）：
   - 2023.1 及以后版本：先点面板右侧的 `Modify options` 下拉，勾选该项，配置面板中才会出现对应复选框，再勾上；
   - 2022.x 及更早版本：该项直接在 `Configuration` 面板底部，勾上即可。

   勾选后 PyCharm 使用终端渲染输出，中文日志不会乱码。

10. 点击 `Apply` → `OK` 关闭对话框。

### 4.3 复制出第二个配置（双积分器）

在 `Edit Configurations` 对话框中选中 `upstream-single`：

1. 点击对话框左上角的 `Copy Configuration` 按钮（部分版本为 `Ctrl+D`）；
2. 新配置的 `Name` 改为 `upstream-double`；
3. 只修改 `Script path`：

   ```text
   C:\Users\L\PycharmProjects\PythonProject3\cbf-nerf\upstream\double_cbf.py
   ```

4. 其余字段（Parameters、Working directory、Environment variables、Interpreter、勾选项）保持不变；
5. `Apply` → `OK`。

### 4.4 运行前的自检

| 检查项 | 应为 |
|---|---|
| `Script path` 结尾 | `upstream\single_cbf.py`（或 `upstream\double_cbf.py`） |
| `Working directory` 结尾 | `\upstream`（**不是** `cbf-nerf`，也不是项目根） |
| `Parameters` | `configs/Replica/room1.yaml --output output/Replica/room1` |
| 环境变量条数 | 2 条，且包含 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` |
| `Python interpreter` | `nerf-cbf` 环境的 `python.exe` |

点选配置后，对话框底部（或运行窗口顶部）会显示将要执行的完整命令行，形如：

```text
D:\miniconda\envs\nerf-cbf\python.exe C:\Users\L\PycharmProjects\PythonProject3\cbf-nerf\upstream\single_cbf.py
    configs/Replica/room1.yaml --output output/Replica/room1
```

命令行与上表一致即配置正确。运行时 PyCharm 也会把这一行打印在运行控制台的第一行，可用它核对实际使用的解释器与参数。

### 4.5 可选：把出图脚本也做成配置

再新建一个 `Python` 配置：

| 字段 | 值 |
|---|---|
| `Name` | `plot-figures` |
| `Script path` | `...\cbf-nerf\tools\plot_from_log.py` |
| `Parameters` | `logs\single_upstream.log logs\double_upstream.log --csv` |
| `Working directory` | `...\cbf-nerf` |
| `Python interpreter` | 同上 |

该配置不需要环境变量，因为 `plot_from_log.py` 只读日志与写 PNG，不加载 torch 权重。

### 4.6 可选：让实验前自动准备数据集

在实验配置的 `Before launch` 区域点击 `+` → `Run Another Configuration`，选择先把 `make_dataset_stub.py` 配成一个独立配置。这样每次运行实验前都会先生成占位数据集（重复执行是幂等的，会覆盖同名文件）。

### 4.7 与界面相关的常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 配置列表里没有 `Python` 选项 | 项目未配置 Python 解释器 | 先完成第 2 节 |
| 找不到 `Emulate terminal in output console` 复选框 | 新版 PyCharm 把它收进 `Modify options` 下拉 | 点 `Modify options` 勾选后再在面板中勾上 |
| 运行时报 `FileNotFoundError: 'device.txt'` | `Working directory` 仍是项目根 | 改为 `...\upstream` |
| 运行时报环境变量无效 | 把 `NAME=VALUE` 整串写进了 Name 字段 | 拆成 Name / Value 两栏 |
| 运行用了错误的解释器 | 配置里的 `Python interpreter` 与项目默认解释器可以不同 | 在配置内单独指定，或改项目默认解释器 |
| 改了配置却仍按旧参数运行 | 未点 `Apply` | 关闭对话框前点 `Apply` |

---

## 5. 验证顺序与预期输出

| 步 | 操作 | 预期 | 耗时（RTX 5070 Laptop） |
|---|---|---|---|
| 1 | 运行单积分器配置 | 先输出 4 行 `INFO: ...`，随后打印 `1`、`min_depth = 0.95...`、`intervention = 0` | 首帧约 1 分钟内出现 |
| 2 | 等待结束 | 进程正常退出；`upstream\video.avi` 约 **5.3 MB**、150 帧、2400×680 | 约 20 分钟 |
| 3 | 运行双积分器配置 | 逐步打印 `step: i`、`new h = ...`、`min depth = ...`、`new v = ...` | 约 14 分钟 |
| 4 | 出图（Terminal） | `results\figures\figure3a_single.png`、`figure4a_double.png` 及两个 CSV | 数秒 |

若第 2 步的 `video.avi` 只有约 5.7 KB，说明运行的是未修正尺寸的上游原版文件（本仓库已改为 `(2400,680)`）。

单帧耗时约 7 秒，来自上游每步一次 1200×680 的两趟渲染（粗深度 + 表面细化），与算法本身无关。上游脚本没有 `--frames` 参数，步数写死为 `while frame < 150` 与 `for i in range(100)`。

---

## 6. 日志与出图：IDE 运行 vs Terminal 运行

上游脚本只把内容打印到标准输出，**不写文件**。因此：

- 用 Run Configuration 运行：日志仅出现在 PyCharm 控制台，`logs/` 下没有文件，`tools/plot_from_log.py` 将找不到输入；
- 要让日志落盘并出图，在 Terminal 中运行：

```powershell
cd C:\Users\L\PycharmProjects\PythonProject3\cbf-nerf
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = "1"
cd upstream
python -u single_cbf.py configs/Replica/room1.yaml --output output/Replica/room1 2>&1 |
    Tee-Object ..\logs\single_upstream.log
cd ..
python tools/plot_from_log.py logs\single_upstream.log --csv
```

`-u` 使输出不缓冲，日志可实时查看；`Tee-Object` 同时写屏幕与文件（Windows PowerShell 下得到 UTF-16 编码，`plot_from_log.py` 会自动识别）。

也可使用仓库自带的一键脚本（包含环境变量、占位数据集、日志重定向与出图）：

```powershell
powershell -ExecutionPolicy Bypass -File run_upstream.ps1 -Which all
```

出图脚本也可单独配置为 Run Configuration：Script path 选 `tools\plot_from_log.py`，Parameters 填 `logs\single_upstream.log --csv`。

图 2 的 RGBd 序列由视频抽帧生成：

```powershell
python tools\frames_from_video.py upstream\video.avi
```

---

## 7. 调试断点

上游代码有 `if __name__ == '__main__':` 保护（`single_cbf.py:152`），可以安全 import，但断点只能设在 `__main__` 块内。

| 观察对象 | 位置 | 变量 |
|---|---|---|
| 标称指令是否安全 | `single_cbf.py:118` `if new_h <= alpha * h:` | `new_h`、`h`、`alpha * h` |
| 10 个候选各自的 `h_{t+1}` | `single_cbf.py:138` `batch_new_h = ...` | `batch_new_h`（形状 `(10,)`）、`batch_action` |
| 最终选中的动作 | `single_cbf.py:144` `if torch.norm(best_action, p=2) > 0:` | `best_action`、`new_h` |
| 每帧最近距离 | `single_cbf.py:297` `print('min_depth = ...')` | `depth.min()` |
| 双积分器的速度更新 | `double_cbf.py:227` `state, robot.v = update_dynamics(...)` | `robot.v` |

CBF 值的重建依据：主循环把 `d - depth.min()` 传给滤波器，而 `d = 0.1` 为常量，故 `h_t = 0.1 − min_depth_{t−1}`；`tools/plot_from_log.py` 按此式计算。

---

## 8. 常见错误对照表

| 现象 | 原因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch'` / `'numpy'` | PyCharm 选的是空环境（如新建项目默认的 Python 3.13 环境） | 第 2 节切换解释器并验证 |
| `_pickle.UnpicklingError: Weights only load failed` | 未设 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` | 在 Run Configuration 的 Environment variables 中添加 |
| `FileNotFoundError: 'device.txt'` | 工作目录不是 `upstream` | 改 Working directory |
| `FileNotFoundError: .../Datasets/Replica/room1/traj.txt` | 未生成占位数据集 | `python tools/make_dataset_stub.py` |
| `ModuleNotFoundError: No module named 'src'` | Script path 指向了被复制到别处的脚本 | 指向 `cbf-nerf\upstream\single_cbf.py` |
| `video.avi` 仅 5.7 KB | 运行的是未修正尺寸的上游原版 | 使用本仓库的 `upstream/single_cbf.py`（已改为 `2400,680`） |
| `logs/` 为空、出图脚本报找不到日志 | 用 Run Configuration 运行，日志未落盘 | 见第 6 节，用 Terminal + `Tee-Object` |
| `torch.cuda.is_available()` 为 `False` | 装的是 CPU 版 torch | 重装 CUDA 版 |
| 每帧 7–20 秒 | 正常：上游每步一次 1200×680 两趟渲染 | 等待；或改用双积分器实验（同样代价） |
| 中文输出乱码 | 控制台编码 | 勾选 `Emulate terminal in output console`，并设 `PYTHONIOENCODING=utf-8` |
| 日志文件显示为乱码或字符间有空格 | PowerShell 重定向写的是 UTF-16 | `plot_from_log.py` 已自动识别；手工查看时切换编辑器编码 |
| PyCharm 卡顿 | 索引 147 MB 权重 | 右键 `upstream/pretrained`、`replica_room1.npy` → `Mark as Excluded` |

---

## 9. 一页速查

```text
复制     : robocopy <仓库> <项目目录>\cbf-nerf /E /XD .git output Datasets __pycache__
解释器   : 装有 CUDA 版 torch 的环境, 用 import torch 验证
数据集   : python tools/make_dataset_stub.py              (一次)
环境变量 : TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
工作目录 : <项目>\cbf-nerf\upstream
图 3a+2  : upstream\single_cbf.py configs/Replica/room1.yaml --output output/Replica/room1   (~20 min)
图 4a    : upstream\double_cbf.py configs/Replica/room1.yaml --output output/Replica/room1   (~14 min)
出图     : python tools/plot_from_log.py logs\*.log --csv
图 2 抽帧: python tools/frames_from_video.py upstream\video.avi
图 5     : python tools/sweep_upstream.py                 (~70 min)
```
