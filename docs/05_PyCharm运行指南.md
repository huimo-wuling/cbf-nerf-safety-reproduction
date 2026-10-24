# 在 PyCharm 里复现论文结果

复现论文（ICRA 2023，CBF + NeRF 视觉安全滤波器）的三个仿真结果：图 2、图 3a、图 4a。
做法是在 PyCharm 底部的终端里运行一个脚本，`upstream/` 下的作者原版代码保持不变。

---

## 1. 准备（只做一次）

### 1.1 环境

需要 Python 3.9–3.12 与 **CUDA 版** PyTorch（`opencv-python`、`numpy`、`scipy`、`matplotlib` 一并安装）。本仓库实测环境为 conda 环境 `nerf-cbf`（Python 3.11 + torch 2.11.0+cu128）。

确认方式：

```powershell
D:\miniconda\envs\nerf-cbf\python.exe -c "import torch;print(torch.__version__, torch.cuda.is_available())"
```

输出 `2.11.0+cu128 True` 即为可用环境。运行脚本会自动在常见的 conda 目录中寻找同时满足"装有 CUDA 版 torch"的解释器，因此不必手工指定。

### 1.2 复制代码

```powershell
robocopy "C:\Users\L\Desktop\cbf-nerf-safety-reproduction" `
         "C:\Users\L\PycharmProjects\PythonProject3\cbf-nerf" `
         /E /XD .git output Datasets __pycache__ /XF video.avi
```

复制后的目录：

```text
C:\Users\L\PycharmProjects\PythonProject3\cbf-nerf\
├── upstream\           作者原版代码（single_cbf.py / double_cbf.py / src / configs）
├── tools\              占位数据集、日志出图、视频抽帧、参数扫描
├── run_upstream.bat    复现入口
├── logs\ results\ docs\
└── README.md
```

---

## 2. 运行

点击 PyCharm 底部的 `Terminal`，执行一行：

```bat
cd /d C:\Users\L\PycharmProjects\PythonProject3\cbf-nerf
run_upstream.bat
```

不带参数即运行单积分器实验（图 2 + 图 3a）。需要其它组合时：

| 命令 | 内容 | 耗时 |
|---|---|---|
| `run_upstream.bat` | 单积分器，150 帧（图 2、图 3a） | 约 20 分钟 |
| `run_upstream.bat double` | 双积分器，100 步（图 4a） | 约 14 分钟 |
| `run_upstream.bat all` | 依次运行两者，并生成图 2 的静态拼图 | 约 35 分钟 |

脚本会自动完成三件事：生成一份 1 帧的 Replica 占位数据集、以 `upstream/` 为工作目录按作者原命令运行实验并把输出写入 `logs/`、解析日志生成配图与逐帧指标。

终端输出形如：

```text
== interpreter: D:\miniconda\envs\nerf-cbf\python.exe

== [1/2] placeholder dataset
已生成占位数据集:...\upstream\Datasets\Replica\room1

== [2/2] run upstream single_cbf.py   (log: logs\single_upstream.log)
```

第二条 `==` 之后实验开始，终端不再有新输出属正常，输出被写入日志文件；结束时会打印 `== done. Outputs:`。

---

## 3. 结果

| 文件 | 内容 |
|---|---|
| `upstream\video.avi` | 150 帧、2400×680、约 5.3 MB，深度图 + RGB + CBF 指示条（论文图 2） |
| `results\figures\figure3a_single.png` | 最小距离 / CBF 值 / 干预量随步数变化（论文图 3a） |
| `results\figures\figure4a_double.png` | 最小距离 / CBF 值 / 速度 / 干预量（论文图 4a） |
| `results\figures\single_metrics.csv`、`double_metrics.csv` | 逐帧数值（步号、最近距离、CBF 值、干预量、辅助量） |
| `logs\single_upstream.log`、`logs\double_upstream.log` | 上游脚本的原始输出 |

单积分器实验的参考数值（本机实测，供核对）：初始最近距离 0.9569 m，第 30 帧首次干预（距墙 0.1885 m），最小距离 0.0987 m，`h` 最大值 +0.0013，干预峰值 1.593 m/s。
双积分器：初始 0.9588 m，第 12 步首次干预（距墙 1.2445 m），最小距离 0.1099 m，速度峰值 1.216 → 终值 0.007 m/s。

---

## 4. 一页速查

```text
环境   : Python 3.11 + CUDA 版 torch（自动探测，无需指定）
复制   : robocopy <仓库> <项目>\cbf-nerf /E /XD .git output Datasets __pycache__
运行   : cd /d <项目>\cbf-nerf
         run_upstream.bat            → 图 2 + 图 3a（约 20 分钟）
         run_upstream.bat double     → 图 4a（约 14 分钟）
         run_upstream.bat all        → 全部（约 35 分钟）
结果   : upstream\video.avi、results\figures\*.png、results\figures\*.csv
```
