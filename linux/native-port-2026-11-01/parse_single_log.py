# -*- coding: utf-8 -*-
"""解析 single_cbf_log.txt(新版简洁日志),绘制轨迹图对照论文图 3a。

日志逐帧格式:
[帧  12/150] 最近深度 = 0.821 m | h = -0.721 | 干预 = 0.00 m/s | 放行(标称指令安全)
[帧  13/150] 最近深度 = 0.774 m | h = -0.674 | 干预 = 0.62 m/s | 拦截替换
"""
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log_path = r"results/single_cbf_log.txt"
text = open(log_path, encoding="utf-8").read()

pat = re.compile(
    r"\[帧\s*(\d+)/(\d+)\] 最近深度 = ([\d.eE+-]+) m \| h = (-?[\d.eE+-]+) \| 干预 = ([\d.eE+-]+)")
records = [
    {"step": int(m.group(1)), "total": int(m.group(2)), "min_depth": float(m.group(3)),
     "h": float(m.group(4)), "intervention": float(m.group(5))}
    for m in pat.finditer(text)
]
n = len(records)
print(f"解析到 {n} 帧")
if n == 0:
    raise SystemExit("未解析到任何帧,请检查日志格式")

s = [r["step"] for r in records]
d = [r["min_depth"] for r in records]
h = [r["h"] for r in records]
iv = [r["intervention"] for r in records]

fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
axes[0].plot(s, d, "b.-", ms=3)
axes[0].axhline(0.1, color="r", ls="--", label=r"$d_c=0.1$")
axes[0].set_ylabel("min depth (m)")
axes[0].set_title("Single-integrator CBF (reproduction, paper Fig. 3a analog)")
axes[0].legend()
axes[1].plot(s, h, "g.-", ms=3)
axes[1].axhline(0.0, color="k", ls=":")
axes[1].set_ylabel("CBF h")
axes[2].plot(s, iv, "r.-", ms=3)
axes[2].set_ylabel("intervention (m/s)")
axes[2].set_xlabel("time step (0.1 s each)")
fig.tight_layout()
fig.savefig(r"results/single_cbf_trajectory.png", dpi=120)

print(f"min_depth: start={d[0]:.3f}  min={min(d):.3f}  end={d[-1]:.3f}  (d_c=0.1)")
print(f"h: start={h[0]:.3f}  max={max(h):.4f}")
first_iv = next((r["step"] for r, x in zip(records, iv) if x > 0), None)
if first_iv is not None:
    fd = next(r for r in records if r["step"] == first_iv)
    print(f"首次干预: 第 {first_iv} 帧(当时最近深度 {fd['min_depth']:.3f} m)")
print(f"干预帧数: {sum(1 for x in iv if x > 0)}/{n} | 平均干预 {sum(iv)/len(iv):.3f} | 峰值 {max(iv):.3f} m/s")
