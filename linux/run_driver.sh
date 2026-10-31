#!/usr/bin/env bash
# 用「逐帧日志驱动版」跑论文实验,产出可直接对照论文图 3a / 4a 的日志。
#
#   bash linux/run_driver.sh single   # 单积分器 150 帧(图 2 / 图 3a)
#   bash linux/run_driver.sh double   # 双积分器 100 步(图 4a)
#   bash linux/run_driver.sh all      # 两个都跑
#   CBF_STEPS=5 bash linux/run_driver.sh single   # 短流程冒烟
#
# 驱动脚本放在 linux/driver/,与 upstream/ 的控制算法逐行等价(见 README 附录),
# 只增加了中文注释、逐帧日志与结果绘图。upstream/ 本身不改动。
set -euo pipefail

WHICH="${1:-single}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
mkdir -p "$ROOT/linux/logs"

export MPLBACKEND=Agg
export PYTHONIOENCODING=utf-8
# torch>=2.6 的 weights_only 兼容(驱动脚本里已显式传参,这里再兜一层保险)
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

if [ ! -f "$ROOT/upstream/Datasets/Replica/room1/traj.txt" ]; then
  echo "== 数据集未就绪,生成 1 帧占位"
  "$PY" "$ROOT/tools/make_dataset_stub.py"
fi

run_one() {
  local script="$1" logname="$2" logfile="$ROOT/linux/logs/$2.log"
  echo
  echo "== 运行驱动 $script(日志: linux/logs/$logname.log)"
  # cwd 必须是 upstream/:脚本按相对路径读取 replica_room1.npy / device.txt / configs/
  ( cd "$ROOT/upstream" && PYTHONPATH="$ROOT/upstream" "$PY" -u "../linux/driver/$script" \
      configs/Replica/room1.yaml --output output/Replica/room1 2>&1 | tee "$logfile" )
}

case "$WHICH" in
  single) run_one single_cbf.py single_linux ;;
  double) run_one double_cbf.py double_linux ;;
  all)    run_one single_cbf.py single_linux
          run_one double_cbf.py double_linux ;;
  *) echo "用法: $0 [single|double|all]"; exit 1 ;;
esac

echo
echo "== 出图"
"$PY" "$ROOT/linux/tools/plot_logs.py"

echo
echo "== 完成"
echo "   linux/logs/*.log                逐帧日志"
echo "   linux/figures/*.png             论文图 3a / 4a 对照"
echo "   upstream/video.avi              单积分器视频(图 2 的帧序列)"
