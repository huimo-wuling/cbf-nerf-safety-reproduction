# 运行论文实验(上游原版代码,零改动)
#
#   powershell -ExecutionPolicy Bypass -File run_upstream.ps1            # 单积分器(图 2 / 3a)
#   powershell -ExecutionPolicy Bypass -File run_upstream.ps1 -Which double    # 双积分器(图 4a)
#   powershell -ExecutionPolicy Bypass -File run_upstream.ps1 -Which all       # 两个都跑
#
# 做了什么:
#   1) 设置 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
#      —— torch>=2.6 把 torch.load 的默认 weights_only 改成了 True,而上游的
#         replica_room1.pth / pretrained/*.pt 都是整体 pickle 的对象,不绕过就直接报错。
#         这是**唯一**避免改动上游代码的办法(已在 torch 2.11 上验证)。
#   2) 生成 1 帧占位数据集(上游只用 frame_reader[0] 做形状校验,数值不参与实验)
#   3) 进 upstream/ 目录、按上游 README 的命令跑,日志写到 logs/
#   4) 跑完自动用 tools/plot_from_log.py 出图
param(
    [ValidateSet('single', 'double', 'all')]
    [string]$Which = 'single',
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$upstream = Join-Path $root 'upstream'
$logs = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null

$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = '1'
$env:PYTHONIOENCODING = 'utf-8'

Write-Host "== 仓库: $root" -ForegroundColor Cyan
& $Python --version
& $Python (Join-Path $root 'tools\make_dataset_stub.py')

function Run-One($script, $logname) {
    Write-Host "`n== 运行上游 $script(日志: logs\$logname)" -ForegroundColor Cyan
    Write-Host "   cwd = upstream\ ,命令与上游 README 完全一致" -ForegroundColor DarkGray
    $logPath = Join-Path $logs $logname
    $errPath = Join-Path $logs ($logname -replace '\.log$', '.err.log')
    # 用 Start-Process 直接把子进程的原始字节写盘:
    #   PowerShell 的 `>` / Tee-Object 在 Windows 上默认写 UTF-16(25% 体积 + 解析麻烦),
    #   而子进程自己(设了 PYTHONIOENCODING=utf-8)输出的是 UTF-8,保持原样最省事。
    $proc = Start-Process -FilePath $Python `
        -ArgumentList @('-u', '-W', 'ignore', $script, 'configs/Replica/room1.yaml', '--output', 'output/Replica/room1') `
        -WorkingDirectory $upstream -NoNewWindow -PassThru -Wait `
        -RedirectStandardOutput $logPath -RedirectStandardError $errPath
    Write-Host "   退出码: $($proc.ExitCode)(日志: $logPath)"
}

if ($Which -eq 'single' -or $Which -eq 'all') { Run-One 'single_cbf.py' 'single_upstream.log' }
if ($Which -eq 'double' -or $Which -eq 'all') { Run-One 'double_cbf.py' 'double_upstream.log' }

Write-Host "`n== 出图" -ForegroundColor Cyan
$logsToPlot = @()
if ($Which -eq 'single' -or $Which -eq 'all') { $logsToPlot += (Join-Path $logs 'single_upstream.log') }
if ($Which -eq 'double' -or $Which -eq 'all') { $logsToPlot += (Join-Path $logs 'double_upstream.log') }
& $Python (Join-Path $root 'tools\plot_from_log.py') @logsToPlot --csv

Write-Host "`n== 完成。上游脚本的输出:" -ForegroundColor Green
Write-Host "   upstream\video.avi                    (单积分器,2400x680,论文图 2 的帧序列)"
Write-Host "   upstream\output\Replica\room1\        (上游写的 output 目录)"
Write-Host "   results\figures\figure3a_single.png   论文图 3a 对照"
Write-Host "   results\figures\figure4a_double.png   论文图 4a 对照"
