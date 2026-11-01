# -*- coding: utf-8 -*-
"""解析 double_cbf_log.txt(新版简洁日志),绘制轨迹图对照论文图 4a。

日志逐步格式:
[步  12/100] 最近深度 = 0.456 m | h = -0.102 | 速度 = 0.683 m/s | 干预 = 0.35 m/s² | 放行
"""
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log_path = r"results/double_cbf_log.txt"
text = open(log_path, encoding="utf-8").read()

pat = re.compile(
    r"\[步\s*(\d+)/(\d+)\] 最近深度 = ([\d.eE+-]+) m \| h = (-?[\d.eE+-]+) \| 速度 = ([\d.eE+-]+) m/s \| 干预 = ([\d.eE+-]+)")
records = [
    {"step": int(m.group(1)), "total": int(m.group(2)), "min_depth": float(m.group(3)),
     "h": float(m.group(4)), "v": float(m.group(5)), "intervention": float(m.group(6))}
    for m in pat.finditer(text)
]
n = len(records)
print(f"解析到 {n} 步")
if n == 0:
    raise SystemExit("未解析到任何步,请检查日志格式")

s = [r["step"] for r in records]
d = [r["min_depth"] for r in records]
h = [r["h"] for r in records]
v = [r["v"] for r in records]
iv = [r["intervention"] for r in records]

fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
axes[0].plot(s, d, "b.-", ms=3)
axes[0].axhline(0.1, color="r", ls="--", label=r"$d_c=0.1$")
axes[0].set_ylabel("min depth (m)")
axes[0].set_title("Double-integrator CBF (reproduction, paper Fig. 4a analog)")
axes[0].legend()
axes[1].plot(s, h, "g.-", ms=3)
axes[1].axhline(0.0, color="k", ls=":")
axes[1].set_ylabel("CBF h")
axes[2].plot(s, v, "m.-", ms=3)
axes[2].set_ylabel("|v| (m/s)")
axes[3].plot(s, iv, "r.-", ms=3)
axes[3].set_ylabel(r"intervention (m/s$^2$)")
axes[3].set_xlabel("time step (0.1 s each)")
fig.tight_layout()
fig.savefig(r"results/double_cbf_trajectory.png", dpi=120)

print(f"min_depth: start={d[0]:.3f}  min={min(d):.3f}  end={d[-1]:.3f}  (d_c=0.1)")
print(f"velocity:  start={v[0]:.3f}  max={max(v):.3f}  end={v[-1]:.3f}")
print(f"h:         start={h[0]:.3f}  max={max(h):.4f}")
print(f"干预步数: {sum(1 for x in iv if x > 0)}/{n} | 平均干预 {sum(iv)/len(iv):.3f} | 峰值 {max(iv):.3f} m/s²")
first_iv = next((r["step"] for r, x in zip(records, iv) if x > 0), None)
print(f"首次干预: 第 {first_iv} 步")
