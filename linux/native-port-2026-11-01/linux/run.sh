#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-single}"
case "$mode" in single|double) ;; *) echo 'Use single or double'; exit 2;; esac
export MPLBACKEND=Agg
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
mkdir -p results
.conda/bin/python -u "${mode}_cbf.py" configs/Replica/room1.yaml --output output/Replica/room1 2>&1 | tee "results/${mode}_cbf_log.txt"
.conda/bin/python "parse_${mode}_log.py"
if [[ "$mode" == single ]]; then mv video.avi results/single_cbf.avi; fi
