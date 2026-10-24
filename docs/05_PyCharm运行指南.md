# 在 PyCharm 里复现论文结果

> 本文只给**一条**路径：在 PyCharm 底部的终端里调用 `run_upstream.bat`。
> 不涉及运行配置、编辑器绿三角、中文界面字段等易出错环节。
> 全部上游代码保持原样（唯一改动是 `single_cbf.py` 第 199 行 `cv2.VideoWriter` 的尺寸 `(240,68)` → `(2400,680)`）。

---

## 0. 为什么走这条路径

| 常见故障 | 命令行 + `.bat` 为什么能避开 |
|---|---|
| `error: the following arguments are required: config` | 参数由脚本统一传入。点编辑器绿三角会使用"不带参数"的临时配置，必然报此错。 |
| `字符串缺少终止符` / `缺少右 }`（`.ps1` 解析失败） | Windows PowerShell 5.1 对**无 BOM 的 UTF-8 `.ps1`** 按 GBK 解码，中文注释被打乱后引号失配。`.bat` 全文纯 ASCII，不受代码页影响。 |
| `FileNotFoundError: 'device.txt'` | 脚本内部先 `cd` 到 `upstream`，上游的相对路径才成立。 |
| `_pickle.UnpicklingError: Weights only load failed` | 脚本内部设置 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`。 |
| `ModuleNotFoundError: No module named 'torch'` | 用 `PYTHON` 变量显式指定解释器，不依赖终端默认的 `python`。 |
| 日志与图要手工收集 | 脚本自动把日志写入 `logs/`，跑完自动出图。 |

---

## 1. 前置检查（只做一次）

```powershell
D:\miniconda\envs\nerf-cbf\python.exe -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

期望输出 `2.11.0+cu128 12.8 True`。若该环境不存在，先创建（RTX 50 系需要 cu128 轮子）：

```powershell
conda create -n cbf python=3.11 -y
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy==1.26.4 scipy matplotlib opencv-python
```

上游自带的 `upstream/environment.yaml` 锁定 Linux + Python 3.7 + torch 1.11/cu113，在新环境上无法安装，不使用。

---

## 2. 复制代码（只做一次）

代码必须**整目录复制**：两个实验脚本依赖同级的 `src/`、`configs/`、`device.txt` 以及 147 MB 权重。

```powershell
robocopy "C:\Users\L\Desktop\cbf-nerf-safety-reproduction" `
         "C:\Users\L\PycharmProjects\PythonProject3\cbf-nerf" `
         /E /XD .git output Datasets __pycache__ /XF video.avi
```

复制后：

```text
C:\Users\L\PycharmProjects\PythonProject3\
└── cbf-nerf\
    ├── upstream\           复现代码与权重（151 MB）
    ├── tools\              占位数据集、出图、抽帧、扫参脚本
    ├── run_upstream.bat    本次使用的入口（纯 ASCII）
    ├── run_upstream.ps1    PowerShell 版（UTF-8 带 BOM）
    └── logs\ results\ docs\
```

不要用编辑器打开 `.npy`、`.pth`、`.pt` 再另存——它们是二进制权重，文本方式复制会损坏。

---

## 3. 运行（每次复现都从这里开始）

点击 PyCharm 底部的 `Terminal`（终端），按终端类型选择对应写法。

**终端是 cmd：**

```bat
cd /d C:\Users\L\PycharmProjects\PythonProject3\cbf-nerf
set PYTHON=D:\miniconda\envs\nerf-cbf\python.exe
run_upstream.bat single
```

**终端是 PowerShell：**

```powershell
cd C:\Users\L\PycharmProjects\PythonProject3\cbf-nerf
$env:PYTHON="D:\miniconda\envs\nerf-cbf\python.exe"
.\run_upstream.bat single
```

参数三选一：

| 参数 | 运行内容 | 产出 | 耗时 |
|---|---|---|---|
| `single` | `single_cbf.py`，150 帧 | 图 3a 的图与 CSV、图 2 的视频 | 约 20 分钟 |
| `double` | `double_cbf.py`，100 步 | 图 4a 的图与 CSV | 约 14 分钟 |
| `all` | 两者依次运行 | 以上全部，并自动抽帧生成图 2 的静态拼图 | 约 35 分钟 |

脚本内部依次做三件事：生成 1 帧占位数据集 → 在 `upstream` 目录下以原版命令运行实验并把输出写入 `logs/` → 解析日志生成 PNG 与 CSV。

上游实验的步数写死在源码里（`while frame < 150`、`for i in range(100)`），没有可调参数。

---

## 4. 预期输出与产物

运行时终端应依次出现：

```text
== repo: C:\Users\L\PycharmProjects\PythonProject3\cbf-nerf\
Python 3.11.17

== [1/2] dataset stub
已生成占位数据集:...\upstream\Datasets\Replica\room1

== [2/2] run upstream single_cbf.py  (log: logs\single_upstream.log)
```

第二个 `==` 之后实验开始，终端不再有新输出属正常（输出被重定向到日志文件）。约 20 分钟后打印 `== outputs` 并结束。

| 产物 | 内容 |
|---|---|
| `upstream/video.avi` | 150 帧、2400×680、约 5.3 MB（论文图 2 的帧序列） |
| `logs/single_upstream.log`、`logs/double_upstream.log` | 上游脚本的原始输出 |
| `results/figures/figure3a_single.png` | 最小距离 / CBF 值 / 干预量（论文图 3a） |
| `results/figures/figure4a_double.png` | 最小距离 / CBF 值 / 速度 / 干预量（论文图 4a） |
| `results/figures/*_metrics.csv` | 逐帧数值，用于自行核对 |

单独抽取图 2 的静态拼图（在 `cbf-nerf` 目录下执行）：

```powershell
D:\miniconda\envs\nerf-cbf\python.exe tools\frames_from_video.py upstream\video.avi
```

---

## 5. 失败对照表

| 现象 | 原因 | 处理 |
|---|---|---|
| `error: the following arguments are required: config` | 不是通过 `run_upstream.bat` 启动（例如点了编辑器绿三角） | 按第 3 节在终端执行；绿三角使用不带参数的临时配置，必然报此错 |
| `ModuleNotFoundError: No module named 'torch'` | `PYTHON` 未设置，终端默认 `python` 指向别处 | 设置 `PYTHON` 为第 1 节验证过的 `python.exe` |
| `_pickle.UnpicklingError: Weights only load failed` | 手工运行脚本且未设环境变量 | 用 `run_upstream.bat`；或手工加 `set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` |
| `FileNotFoundError: 'device.txt'` | 工作目录不是 `upstream` | 用 `run_upstream.bat`；或先 `cd upstream` |
| `FileNotFoundError: .../traj.txt` | 占位数据集未生成 | 重新执行 `run_upstream.bat`（会自动生成） |
| `upstream\video.avi` 只有 5.7 KB | 使用的是未修正尺寸的上游原版脚本 | 用本仓库的 `upstream/single_cbf.py`（已改为 `2400,680`） |
| `run_upstream.ps1` 报 `字符串缺少终止符` / `缺少右 }` | `.ps1` 丢失 UTF-8 BOM，PowerShell 5.1 按 GBK 解码 | 改用 `run_upstream.bat`；或确认 `run_upstream.ps1` 前 3 字节为 `EF BB BF` |
| `PermissionError: ... 'output'` | 目标目录不可写 | 换可写目录运行，或确认该目录未被占用 |
| 中文输出乱码 | 控制台代码页 | `run_upstream.bat` 已执行 `chcp 65001` 并设 `PYTHONIOENCODING=utf-8` |

---

## 6. 关于编辑器里的运行按钮

不建议使用。运行配置需同时正确设置脚本路径、工作目录、命令行参数、环境变量四项，任一项缺失都会在启动阶段失败：

- 点编辑器绿三角 → 使用临时配置，**不传参数** → 报 `required: config`；
- 工作目录默认是项目根 → 上游相对路径失效 → 报 `'device.txt'` 找不到；
- 未设环境变量 → 加载权重时抛 `UnpicklingError`。

若确实需要 IDE 内一键运行，四个字段为：

| 字段 | 值 |
|---|---|
| 脚本路径（Script path） | `...\cbf-nerf\upstream\single_cbf.py` |
| 形参（Parameters） | `configs/Replica/room1.yaml --output output/Replica/room1` |
| 工作目录（Working directory） | `...\cbf-nerf\upstream` |
| 环境变量（Environment variables） | `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`；`PYTHONIOENCODING=utf-8` |

---

## 7. 一页速查

```text
解释器   : D:\miniconda\envs\nerf-cbf\python.exe   （先验证 torch.cuda.is_available() 为 True）
复制     : robocopy <仓库> <项目>\cbf-nerf /E /XD .git output Datasets __pycache__
运行     : cd <项目>\cbf-nerf
           set PYTHON=D:\miniconda\envs\nerf-cbf\python.exe      (cmd)
           $env:PYTHON="D:\miniconda\envs\nerf-cbf\python.exe"   (PowerShell)
           run_upstream.bat single | double | all
产出     : upstream\video.avi（图 2）、logs\*.log、results\figures\*.png 与 *.csv
耗时     : single ≈ 20 分钟、double ≈ 14 分钟
不要做   : 不要点绿三角；不要只复制 .py 文件；不要用文本编辑器另存权重文件
```
