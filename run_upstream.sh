#!/usr/bin/env bash
# 运行论文实验(上游原版代码,零改动)
#
#   bash run_upstream.sh            # 单积分器(图 2 / 3a)
#   bash run_upstream.sh double     # 双积分器(图 4a)
#   bash run_upstream.sh all        # 两个都跑
#
# 做了什么:
#   1) 设 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
#      —— torch>=2.6 把 torch.load 默认值改成 weights_only=True,而上游的三个权重文件
#         都是整体 pickle 的对象,不绕过就直接报错。这是唯一避免改动上游代码的办法。
#   2) 生成 1 帧占位数据集(上游只用 frame_reader[0] 做形状校验)
#   3) 进 upstream/ 按上游 README 的命令跑,日志写 logs/
#   4) 跑完自动出图
set -euo pipefail

WHICH="${1:-single}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
mkdir -p "$ROOT/logs"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONIOENCODING=utf-8

echo "== 仓库: $ROOT"
"$PY" --version
"$PY" "$ROOT/tools/make_dataset_stub.py"

run_one() {
  local script="$1" logname="$2"
  echo
  echo "== 运行上游 $script(日志: logs/$logname)"
  echo "   cwd = upstream/ ,命令与上游 README 完全一致"
  ( cd "$ROOT/upstream" && "$PY" -u -W ignore "$script" configs/Replica/room1.yaml \
      --output output/Replica/room1 2>&1 | tee "$ROOT/logs/$logname" )
}

LOGS=()
case "$WHICH" in
  single) run_one single_cbf.py single_upstream.log; LOGS+=("$ROOT/logs/single_upstream.log") ;;
  double) run_one double_cbf.py double_upstream.log; LOGS+=("$ROOT/logs/double_upstream.log") ;;
  all)    run_one single_cbf.py single_upstream.log; LOGS+=("$ROOT/logs/single_upstream.log")
          run_one double_cbf.py double_upstream.log; LOGS+=("$ROOT/logs/double_upstream.log") ;;
  *) echo "用法: $0 [single|double|all]"; exit 1 ;;
esac

echo
echo "== 出图"
"$PY" "$ROOT/tools/plot_from_log.py" "${LOGS[@]}" --csv

echo
echo "== 完成。上游脚本的输出:"
echo "   upstream/video.avi                    (单积分器,论文图 2 的帧序列)"
echo "   upstream/output/Replica/room1/        (上游写的 output 目录)"
echo "   results/figures/figure3a_single.png   论文图 3a 对照"
echo "   results/figures/figure4a_double.png   论文图 4a 对照"
