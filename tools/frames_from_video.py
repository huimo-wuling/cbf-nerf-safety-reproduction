# -*- coding: utf-8 -*-
"""frames_from_video.py —— 从上游写出的 video.avi 中抽取帧,拼成论文图 2 形式的图

背景
----
上游 `single_cbf.py` 每帧把「左:深度图(灰度) + 右:RGB 渲染图 + CBF 指示条」横向拼接后
写入 `video.avi`(10 fps)。论文图 2 是"每 0.5 s 一张"的 RGBd 序列,即每 5 帧取一张。
本脚本按等间隔抽帧并拼成网格图,不改动上游代码。

用法
----
    python tools/frames_from_video.py upstream/video.avi
    python tools/frames_from_video.py upstream/video.avi --every 5 --cols 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None):
    p = argparse.ArgumentParser(description='从 video.avi 抽帧并拼成网格图')
    p.add_argument('video', nargs='?', default=str(ROOT / 'upstream' / 'video.avi'))
    p.add_argument('--every', type=int, default=5, help='每 N 帧取一张(默认 5 = 0.5 s)')
    p.add_argument('--cols', type=int, default=4, help='每行几张')
    p.add_argument('--max', type=int, default=16, help='最多取几张')
    p.add_argument('--out', default=str(ROOT / 'results' / 'figures' / 'figure2_frames.png'))
    args = p.parse_args(argv)

    path = Path(args.video)
    if not path.exists():
        raise SystemExit('找不到 %s(先跑一次 upstream/single_cbf.py)' % path)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit('打不开 %s' % path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('视频:%s | %d 帧 | %.1f fps | 尺寸 %dx%d'
          % (path, total, fps, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    picked, idx = [], 0
    while len(picked) < args.max:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        picked.append((idx, frame))
        idx += args.every
    cap.release()
    if not picked:
        raise SystemExit('没有读到任何帧')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cols = args.cols
    rows = (len(picked) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 1.15 * rows))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis('off')
    for ax, (i, frame) in zip(axes.ravel(), picked):
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title('frame %d (%.1f s)' % (i + 1, (i + 1) / fps), fontsize=7)
    fig.suptitle('Rendered depth / RGB sequence with CBF bar (upstream video.avi, paper Fig.2 analog)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print('已拼图:%s(%d 张,每 %d 帧取 1)' % (out, len(picked), args.every))
    return 0


if __name__ == '__main__':
    sys.exit(main())
