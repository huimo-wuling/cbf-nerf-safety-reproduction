# -*- coding: utf-8 -*-
"""把逐帧日志解析成论文图 3a / 4a 的对照图。

日志由 linux/driver/ 下的脚本产生,每帧/每步一行:

    [帧  30/150] 最近深度 = 0.189 m | h = -0.089 | 干预 = 0.62 m/s | 拦截替换
    [步  12/100] 最近深度 = 1.244 m | h = +0.152 | 速度 = 1.106 m/s | 干预 = 0.35 m/s² | 放行

用法(在仓库根目录):
    python linux/tools/plot_logs.py                 # 用默认路径
    python linux/tools/plot_logs.py --single a.log --double b.log --out linux/figures
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

SINGLE_PAT = re.compile(
    r"\[帧\s*(\d+)/(\d+)\] 最近深度 = ([\d.eE+-]+) m \| h = ([+-]?[\d.eE+-]+) "
    r"\| 干预 = ([\d.eE+-]+)"
)
DOUBLE_PAT = re.compile(
    r"\[步\s*(\d+)/(\d+)\] 最近深度 = ([\d.eE+-]+) m \| h = ([+-]?[\d.eE+-]+) "
    r"\| 速度 = ([\d.eE+-]+) m/s \| 干预 = ([\d.eE+-]+)"
)


def read(path):
    return Path(path).read_text(encoding="utf-8")


def parse_single(text):
    out = []
    for m in SINGLE_PAT.finditer(text):
        out.append(
            dict(step=int(m[1]), total=int(m[2]), depth=float(m[3]), h=float(m[4]), iv=float(m[5]))
        )
    return out


def parse_double(text):
    out = []
    for m in DOUBLE_PAT.finditer(text):
        out.append(
            dict(
                step=int(m[1]), total=int(m[2]), depth=float(m[3]),
                h=float(m[4]), v=float(m[5]), iv=float(m[6]),
            )
        )
    return out


def plot_single(rows, out_png, dc=0.1):
    s = [r["step"] for r in rows]
    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    ax[0].plot(s, [r["depth"] for r in rows], "b.-", ms=3)
    ax[0].axhline(dc, color="r", ls="--", label=f"$d_c={dc}$")
    ax[0].set_ylabel("min depth (m)")
    ax[0].set_title("Single-integrator CBF (paper Fig. 3a analog)")
    ax[0].legend()
    ax[1].plot(s, [r["h"] for r in rows], "g.-", ms=3)
    ax[1].axhline(0.0, color="k", ls=":")
    ax[1].set_ylabel("CBF $h$")
    ax[2].plot(s, [r["iv"] for r in rows], "r.-", ms=3)
    ax[2].set_ylabel("intervention (m/s)")
    ax[2].set_xlabel("time step (0.1 s each)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)

    depths = [r["depth"] for r in rows]
    hs = [r["h"] for r in rows]
    ivs = [r["iv"] for r in rows]
    print(f"[单积分器] {len(rows)} 帧")
    print(f"  最小距离 {min(depths):.4f} m | 末帧 {depths[-1]:.4f} m | 第 1 帧 {depths[0]:.4f} m")
    print(f"  max h = {max(hs):+.4f}(要求 ≤ 0:{'✓' if max(hs) <= 0 else '✗'})")
    first = next((r for r in rows if r["iv"] > 0), None)
    if first:
        print(f"  首次干预 第 {first['step']} 帧,距离 {first['depth']:.4f} m")
    print(f"  干预 {sum(1 for x in ivs if x > 0)}/{len(rows)} 帧 | 均值 {sum(ivs)/len(ivs):.3f} | 峰值 {max(ivs):.3f} m/s")


def plot_double(rows, out_png, dc=0.1):
    s = [r["step"] for r in rows]
    fig, ax = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    ax[0].plot(s, [r["depth"] for r in rows], "b.-", ms=3)
    ax[0].axhline(dc, color="r", ls="--", label=f"$d_c={dc}$")
    ax[0].set_ylabel("min depth (m)")
    ax[0].set_title("Double-integrator CBF (paper Fig. 4a analog)")
    ax[0].legend()
    ax[1].plot(s, [r["h"] for r in rows], "g.-", ms=3)
    ax[1].axhline(0.0, color="k", ls=":")
    ax[1].set_ylabel("CBF $h$")
    ax[2].plot(s, [r["v"] for r in rows], "m.-", ms=3)
    ax[2].set_ylabel("|v| (m/s)")
    ax[3].plot(s, [r["iv"] for r in rows], "r.-", ms=3)
    ax[3].set_ylabel(r"intervention (m/s$^2$)")
    ax[3].set_xlabel("time step (0.1 s each)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)

    depths = [r["depth"] for r in rows]
    hs = [r["h"] for r in rows]
    vs = [r["v"] for r in rows]
    ivs = [r["iv"] for r in rows]
    print(f"[双积分器] {len(rows)} 步")
    print(f"  最小距离 {min(depths):.4f} m | 末帧 {depths[-1]:.4f} m | 第 1 帧 {depths[0]:.4f} m")
    print(f"  速度 峰值 {max(vs):.3f} → 终值 {vs[-1]:.3f} m/s")
    print(f"  max h = {max(hs):+.4f}(要求 ≤ 0:{'✓' if max(hs) <= 0 else '✗'})")
    first = next((r for r in rows if r["iv"] > 0), None)
    if first:
        print(f"  首次干预 第 {first['step']} 步,距离 {first['depth']:.4f} m")
    print(f"  干预 {sum(1 for x in ivs if x > 0)}/{len(rows)} 步 | 均值 {sum(ivs)/len(ivs):.3f} | 峰值 {max(ivs):.3f} m/s²")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", default=str(REPO / "linux/logs/single_linux.log"))
    ap.add_argument("--double", default=str(REPO / "linux/logs/double_linux.log"))
    ap.add_argument("--out", default=str(REPO / "linux/figures"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if Path(args.single).is_file():
        rows = parse_single(read(args.single))
        if rows:
            plot_single(rows, out / "single_trajectory.png")
    else:
        print("跳过单积分器:日志不存在", file=sys.stderr)

    if Path(args.double).is_file():
        rows = parse_double(read(args.double))
        if rows:
            plot_double(rows, out / "double_trajectory.png")
    else:
        print("跳过双积分器:日志不存在", file=sys.stderr)

    print("\n图已写入:", out)


if __name__ == "__main__":
    main()
