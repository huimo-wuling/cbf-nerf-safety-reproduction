#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -e Datasets/Replica/room1/traj.txt ]]; then
  echo "Existing dataset retained"
else
  .conda/bin/python ../../tools/make_dataset_stub.py --root Datasets/Replica/room1
fi
.conda/bin/python linux/check_environment.py
