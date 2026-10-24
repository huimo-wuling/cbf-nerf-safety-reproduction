# -*- coding: utf-8 -*-
"""make_dataset_stub.py —— 生成一个"1 帧占位"的 Replica room1 目录

为什么需要
----------
上游 `single_cbf.py` 的启动流程里有一步 `get_dataset()`(在 `NICE_SLAM.__init__` 里),
它会去读 `Datasets/Replica/room1`。而两个实验脚本**只用第 0 帧的一个张量**
(`frame_reader[0]`),读完就扔 —— 场景信息全部来自作者发布的神经隐式地图权重。
所以完整 Replica 数据集(12.4 GB)对实验数值没有任何贡献。

本脚本造一个最小占位目录,让这步启动校验通过:

    room1/
    ├── traj.txt                    1 行,16 个数的单位矩阵
    └── results/
        ├── frame000000.jpg         1200×680 黑图
        └── depth000000.png         1200×680 常量 1000(16 位)

用法
----
    python tools/make_dataset_stub.py                 # 默认写到 upstream/Datasets/Replica/room1
    python tools/make_dataset_stub.py --root 某目录

想用真实数据就跳过本脚本,自行把完整 Replica room1 放到同一路径即可(实验数值不变)。
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
H, W = 680, 1200          # cfg['cam']['H'/'W'],与 configs/nice_slam.yaml 一致


def make(root: Path):
    results = root / 'results'
    results.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(results / 'frame000000.jpg'), np.zeros((H, W, 3), np.uint8))
    cv2.imwrite(str(results / 'depth000000.png'), np.full((H, W), 1000, np.uint16))
    (root / 'traj.txt').write_text('1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n', encoding='ascii')
    return root


def main(argv=None):
    p = argparse.ArgumentParser(description='生成 Replica room1 的最小占位数据集')
    p.add_argument('--root', default=str(ROOT / 'upstream' / 'Datasets' / 'Replica' / 'room1'),
                   help='数据集目录(默认 upstream/Datasets/Replica/room1)')
    args = p.parse_args(argv)
    root = make(Path(args.root))
    files = sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file())
    print('已生成占位数据集:%s' % root)
    for f in files:
        print('   ', f)
    print('\n说明:上游脚本只用 frame_reader[0] 做一个形状校验,数值不参与实验;')
    print('      若要与作者环境完全一致,请把真实的 Replica room1 放到同一路径(覆盖即可)。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
