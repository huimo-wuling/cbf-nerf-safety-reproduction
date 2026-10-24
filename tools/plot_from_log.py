# -*- coding: utf-8 -*-
"""plot_from_log.py —— 只解析上游脚本打印的日志,画出论文图 3a / 图 4a

为什么是"解析日志"而不是"改代码插桩"
------------------------------------
本仓库的原则是 **上游代码一字不动**(唯一例外是 `single_cbf.py` 里 VideoWriter 的尺寸),
所以拿不到结构化指标。好在两个脚本本来就把需要的量打印出来了:

`single_cbf.py`(150 帧)每帧打印::

    12                                     ← 帧号
    min_depth = tensor(0.9750, device='cuda:0', dtype=torch.float64)
    intervention = 0.85                    ← 只在"标称指令不安全"时打印

CBF 值 `h` 没有直接打印,但主循环里 `h_t = d - depth.min()`,所以
**h_t = d − min_depth_{t−1}**,可以从日志精确重建。

`double_cbf.py`(100 步)每步打印::

    step: 0
    new state = ...
    new h = -0.857
    min depth = tensor(0.9590, ...)
    new v = 0.100
    time cost = 7.31

四个量(CBF 值 / 最���距离 / 速度 / 耗时)全都有,外加 `intervention = ...`。

用法
----
    python tools/plot_from_log.py logs/single_upstream.log
    python tools/plot_from_log.py logs/double_upstream.log
    python tools/plot_from_log.py logs/*.log --out results/figures

输出:PNG(对照论文图 3a / 4a)+ 同名 `*_metrics.csv`(便于复核)。
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
D_C = 0.1          # 上游 single_cbf.py: d = 0.1
ALPHA = 0.5        # 上游 single_cbf.py: alpha = 0.5

NUM = r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?'
RE_FRAME = re.compile(r'^\s*(\d+)\s*$')
RE_STEP = re.compile(r'^\s*step:\s*(\d+)\s*$')
RE_MIN_SINGLE = re.compile(r'min_depth\s*=\s*(?:tensor\()?(%s)' % NUM)
RE_MIN_DOUBLE = re.compile(r'min depth\s*=\s*(?:tensor\()?(%s)' % NUM)
RE_H = re.compile(r'new h\s*=\s*(%s)' % NUM)
RE_V = re.compile(r'new v\s*=\s*(%s)' % NUM)
RE_T = re.compile(r'time cost\s*=\s*(%s)' % NUM)
RE_IV = re.compile(r'intervention\s*=\s*(%s)' % NUM)


def read_text_any(path: Path):
    """读日志并自动识别编码。

    PowerShell 的 `>` / `Tee-Object` 在 Windows 上默认写 UTF-16LE,
    而 Python 自己重定向出来的通常是 UTF-8,所以这里按 BOM / NUL 字节模式判断。
    """
    raw = path.read_bytes()
    if raw[:2] in (b'\xff\xfe', b'\xfe\xff'):
        return raw.decode('utf-16', errors='replace')
    if raw[:3] == b'\xef\xbb\xbf':
        return raw.decode('utf-8-sig', errors='replace')
    if len(raw) > 1 and raw[1:2] == b'\x00':          # 无 BOM 的 UTF-16LE
        return raw.decode('utf-16-le', errors='replace')
    return raw.decode('utf-8', errors='replace')


def parse(path: Path):
    """返回 (kind, records)。kind ∈ {'single', 'double'}。

    single 的日志结构是:先打印一次"进入循环前"的初始最近深度,然后每帧打印
    帧号(单独一行)与 `min_depth = ...`。因此这里用"帧号行"作为配对锚点,
    初始那次测量记为 step 0(不进入曲线,但用于重建第 1 帧的 h)。
    """
    text = read_text_any(path)
    kind = 'double' if re.search(r'^\s*step:', text, re.M) else 'single'

    recs, pending_iv, pending_h, pending_v, pending_t = [], None, None, None, None
    pending_frame, initial, pending_min = None, None, None
    for line in text.splitlines():
        m = RE_IV.search(line)
        if m:
            pending_iv = float(m.group(1))
            continue
        if kind == 'double':
            # 上游 double_cbf.py 的打印顺序是
            #   new state = ... / new h = ... / min depth = ... / new v = ... / time cost = ...
            # 所以以最后一行的 `time cost` 作为一步结束的标志
            m = RE_H.search(line)
            if m:
                pending_h = float(m.group(1)); continue
            m = RE_MIN_DOUBLE.search(line)
            if m:
                pending_min = float(m.group(1)); continue
            m = RE_V.search(line)
            if m:
                pending_v = float(m.group(1)); continue
            m = RE_T.search(line)
            if m:
                pending_t = float(m.group(1))
                if pending_min is not None:
                    recs.append(dict(step=len(recs) + 1, min_depth=pending_min,
                                     h=pending_h, velocity=pending_v, t_eval=pending_t,
                                     intervention=pending_iv if pending_iv is not None else 0.0))
                pending_iv = pending_h = pending_v = pending_t = None
                pending_min = None
            continue
        # ---- single:帧号行做锚点,min_depth 行结算一步 ----
        m = RE_FRAME.match(line)
        if m:
            pending_frame = int(m.group(1))
            continue
        m = RE_MIN_SINGLE.search(line)
        if m:
            val = float(m.group(1))
            if pending_frame is None:
                initial = val                      # 进入循环前的那次测量
            else:
                recs.append(dict(step=pending_frame, min_depth=val, h=None,
                                 velocity=None, t_eval=None,
                                 intervention=pending_iv if pending_iv is not None else 0.0))
                pending_frame, pending_iv = None, None
    if not recs:
        raise SystemExit('没能从 %s 里解析到任何一步,请检查日志格式' % path)

    # single:主循环把 h_t = d − depth.min()(上一帧的实测值)传给滤波器,故可精确重建
    if kind == 'single':
        prev = initial if initial is not None else recs[0]['min_depth']
        for r in recs:
            r['h'] = D_C - prev
            prev = r['min_depth']
    return kind, recs


def save_csv(recs, out_csv: Path):
    keys = ['step', 'min_depth', 'h', 'velocity', 'intervention', 't_eval']
    with open(out_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in recs:
            w.writerow({k: ('' if r.get(k) is None else r[k]) for k in keys})


def style(ax):
    ax.grid(alpha=0.3, ls=':')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def plot_single(recs, out_png: Path, d_c=D_C):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    st = [r['step'] for r in recs]
    dep = [r['min_depth'] for r in recs]
    h = [r['h'] for r in recs]
    iv = [r['intervention'] for r in recs]

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 8.4), sharex=True)
    axes[0].plot(st, dep, 'b.-', ms=3)
    axes[0].axhline(d_c, color='crimson', ls='--', lw=1.2, label=r'$d_c$')
    axes[0].set_ylabel('min depth (m)'); axes[0].legend(fontsize=8)
    axes[1].plot(st, h, 'g.-', ms=3)
    axes[1].axhline(0, color='k', lw=1, ls=':')
    axes[1].set_ylabel('CBF value $h$')
    axes[2].plot(st, iv, 'r.-', ms=3)
    axes[2].set_ylabel('intervention (m/s)'); axes[2].set_xlabel('time step (0.1 s each)')
    for ax in axes:
        style(ax)
    fig.suptitle('Single-integrator CBF (upstream single_cbf.py, paper Fig.3a analog)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=130); plt.close(fig)


def plot_double(recs, out_png: Path, d_c=D_C):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    st = [r['step'] for r in recs]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True)
    axes[0, 0].plot(st, [r['min_depth'] for r in recs], 'b.-', ms=3)
    axes[0, 0].axhline(d_c, color='crimson', ls='--', lw=1.2, label=r'$d_c$')
    axes[0, 0].set_ylabel('min depth (m)'); axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(st, [r['h'] for r in recs], 'g.-', ms=3)
    axes[0, 1].axhline(0, color='k', lw=1, ls=':')
    axes[0, 1].set_ylabel('CBF value $h$')
    axes[1, 0].plot(st, [r['velocity'] for r in recs], '.-', ms=3, color='#8c564b')
    axes[1, 0].set_ylabel('velocity (m/s)'); axes[1, 0].set_xlabel('time step (0.1 s each)')
    axes[1, 1].plot(st, [r['intervention'] for r in recs], 'r.-', ms=3)
    axes[1, 1].set_ylabel('intervention (m/s²)'); axes[1, 1].set_xlabel('time step (0.1 s each)')
    for ax in axes.ravel():
        style(ax)
    fig.suptitle('Double-integrator CBF (upstream double_cbf.py, paper Fig.4a analog)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=130); plt.close(fig)


def summarize(kind, recs, path):
    dep = np.array([r['min_depth'] for r in recs])
    h = np.array([r['h'] for r in recs if r['h'] is not None])
    iv = np.array([r['intervention'] for r in recs])
    print('-' * 76)
    print('%s  (%s)' % (path.name, kind))
    print('  步数 %d | 最小距离 %.4f m | 终值 %.4f | h 最大 %+.4f | h 为正的步数 %d'
          % (len(recs), dep.min(), dep[-1], h.max(), int((h > 0).sum())))
    print('  干预:>0 %d 步 | 均值 %.3f | 峰值 %.3f'
          % (int((iv > 0).sum()), iv.mean(), iv.max()))
    if kind == 'double':
        v = np.array([r['velocity'] for r in recs])
        t = np.array([r['t_eval'] for r in recs if r['t_eval'] is not None])
        print('  速度:峰值 %.3f → 终值 %.3f m/s' % (v.max(), v[-1]))
        if t.size:
            print('  耗时:均值 %.3f s/步 | 峰值 %.3f s' % (t.mean(), t.max()))
    if (dep < D_C).any():
        print('  ⚠ 最小距离跌破 d_c = %.2f m' % D_C)


def main(argv=None):
    p = argparse.ArgumentParser(description='从上游脚本日志画论文图 3a / 4a')
    p.add_argument('logs', nargs='+', help='日志文件(可多个)')
    p.add_argument('--out', default=str(ROOT / 'results' / 'figures'), help='输出目录')
    p.add_argument('--csv', action='store_true', help='同时写出 *_metrics.csv')
    args = p.parse_args(argv)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    for lp in args.logs:
        path = Path(lp)
        kind, recs = parse(path)
        summarize(kind, recs, path)
        tag = 'single' if kind == 'single' else 'double'
        png = out_dir / ('figure3a_%s.png' % tag if kind == 'single' else 'figure4a_%s.png' % tag)
        (plot_single if kind == 'single' else plot_double)(recs, png)
        print('  → %s' % png)
        if args.csv:
            save_csv(recs, out_dir / ('%s_metrics.csv' % tag))
            print('  → %s' % (out_dir / ('%s_metrics.csv' % tag)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
