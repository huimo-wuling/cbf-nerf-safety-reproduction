# 在 PyCharm 里复现论文结果

复现论文（ICRA 2023，CBF + NeRF 视觉安全滤波器）的三个仿真结果：图 2、图 3a、图 4a。
做法是在 PyCharm 底部的终端里运行一个脚本，`upstream/` 下的作者原版代码保持不变。
每一步的终端输出与产物位置见第 2.1 节。

---

## 1. 准备（只做一次）

### 1.1 环境

需要 Python 3.9–3.12 与 **CUDA 版** PyTorch（另装 `opencv-python`、`numpy`、`scipy`、`matplotlib`）。本仓库实测环境为 conda 环境 `nerf-cbf`（Python 3.11 + torch 2.11.0+cu128）。

```powershell
D:\miniconda\envs\nerf-cbf\python.exe -c "import torch;print(torch.__version__, torch.cuda.is_available())"
```

输出 `2.11.0+cu128 True` 即可用。运行脚本会自动在常见 conda 目录中寻找"装有 CUDA 版 torch"的解释器，不需要手工指定。

**终端输出**：一行版本号。**产物**：无文件。

### 1.2 复制代码

```powershell
robocopy "C:\Users\L\Desktop\cbf-nerf-safety-reproduction" `
         "C:\Users\L\PycharmProjects\PythonProject3\cbf-nerf" `
         /E /XD .git output Datasets __pycache__ /XF video.avi
```

**终端输出**（robocopy 自带汇总，实测）：

```text
   ROBOCOPY     ::     Robust File Copy for Windows
  Started : 2026年10月24日 23:55:33

               Total    Copied   Skipped  Mismatch    FAILED    Extras
    Dirs :        31        22         9         0         0         0
   Files :       124       123         1         0         0         0
   Bytes :  159.78 m  154.48 m    5.30 m         0         0         0
   Ended : 2026年10月24日 23:55:34
```

**产物**：目录树落在 `C:\Users\L\PycharmProjects\PythonProject3\cbf-nerf\`：

```text
cbf-nerf\
├── upstream\           作者原版代码（single_cbf.py / double_cbf.py / src / configs / 权重）
├── tools\              占位数据集、日志出图、视频抽帧、参数扫描
├── run_upstream.bat    复现入口
├── logs\               运行日志
├── results\            配图与逐帧指标
└── docs\ README.md
```

---

## 2. 运行

点击 PyCharm 底部的 `Terminal`，执行：

```bat
cd /d C:\Users\L\PycharmProjects\PythonProject3\cbf-nerf
run_upstream.bat
```

不带参数即运行单积分器实验。其它组合：

| 命令 | 运行内容 | 耗时 |
|---|---|---|
| `run_upstream.bat` | 单积分器 150 帧（图 2、图 3a） | 约 20 分钟 |
| `run_upstream.bat double` | 双积分器 100 步（图 4a） | 约 14 分钟 |
| `run_upstream.bat all` | 依次运行两者，并生成图 2 的静态拼图 | 约 35 分钟 |

### 2.1 每一步的终端输出与产物位置

表中路径均相对于 `C:\Users\L\PycharmProjects\PythonProject3\cbf-nerf\`。

| # | 步骤 | 终端输出（实测） | 产物位置 |
|---|---|---|---|
| 1 | 启动脚本，检测解释器 | `== interpreter: D:\miniconda\envs\nerf-cbf\python.exe` | 无文件 |
| 2 | 生成占位数据集 | `== [1/2] placeholder dataset`<br>`已生成占位数据集:...\upstream\Datasets\Replica\room1`<br>`    results\depth000000.png`<br>`    results\frame000000.jpg`<br>`    traj.txt` | `upstream\Datasets\Replica\room1\`（含 `results\` 子目录，共 3 个文件） |
| 3 | 启动上游实验 | `== [2/2] run upstream single_cbf.py   (log: logs\single_upstream.log)`<br>`== launch: single_cbf.py`<br>`== log   : ...\logs\single_upstream.log`<br>`== 每帧一行进度;完整输出写入上面的日志文件` | `logs\single_upstream.log`（开始写入）<br>`upstream\output\Replica\room1\`（上游自建 `ckpts`、`mesh`、`mapping_vis`、`tracking_vis`，本实验均为空） |
| 4 | 逐帧渲染，每 6–8 秒一行 | `[init ]  min_depth=0.9569  iv=0.00`<br>`[  1/150]  min_depth=1.2552  iv=0.00`<br>`[  2/150]  min_depth=0.9747  iv=0.00`<br>`…`<br>`[ 30/150]  min_depth=0.1885  iv=0.86` | `upstream\video.avi`（持续增长，结束时约 5.3 MB）<br>`logs\single_upstream.log`（追加完整原始输出，含 tensor 打印） |
| 5 | 双积分器的同一行 | `[  1/100]  min_depth=1.2445  h=-0.0284  v=1.155  iv=0.86  t=0.15s` | `logs\double_upstream.log` |
| 6 | 实验结束 | `== finished: single_cbf.py  用时 19.8 分钟  退出码 0`<br>`== 最小最近距离: 0.0987 m（d_c = 0.1 m）` | 无新增 |
| 7 | 解析日志出图 | `- single_upstream.log  (single)`<br>`  步数 150 | 最小距离 0.0987 m | …`<br>`  → …\results\figures\figure3a_single.png`<br>`  → …\results\figures\single_metrics.csv` | `results\figures\figure3a_single.png`<br>`results\figures\single_metrics.csv`<br>`all` 时另有 `figure4a_double.png`、`double_metrics.csv`、`figure2_frames.png` |
| 8 | 收尾汇总 | `== done. Outputs:`<br>`   …\upstream\video.avi`<br>`   …\logs\*.log`<br>`   …\results\figures\` | 提示上述全部路径 |

运行中可按 `Ctrl+C` 中止，已写入的日志与视频帧仍保留在磁盘。

---

## 3. 结果

| 文件 | 内容 |
|---|---|
| `upstream\video.avi` | 深度图 + RGB + CBF 指示条（左深度、右渲染），150 帧、2400×680、约 5.3 MB（论文图 2） |
| `results\figures\figure3a_single.png` | 最小距离 / CBF 值 / 干预量随步数变化（论文图 3a） |
| `results\figures\figure4a_double.png` | 最小距离 / CBF 值 / 速度 / 干预量（论文图 4a） |
| `results\figures\single_metrics.csv`、`double_metrics.csv` | 逐帧数值：步号、最近距离、CBF 值、干预量、速度与耗时 |
| `logs\single_upstream.log`、`logs\double_upstream.log` | 上游脚本的完整原始输出 |

单积分器参考数值（本机实测）：初始最近距离 0.9569 m，第 30 帧首次干预（距墙 0.1885 m），最小距离 0.0987 m，`h` 最大值 +0.0013，干预峰值 1.593 m/s。

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
产物   : <项目>\cbf-nerf\upstream\video.avi                 图 2 的帧序列
         <项目>\cbf-nerf\results\figures\*.png 与 *.csv     图 3a / 图 4a 与逐帧数据
         <项目>\cbf-nerf\logs\*.log                         上游原始输出
         <项目>\cbf-nerf\upstream\output\Replica\room1\      上游自建目录（空）
```
