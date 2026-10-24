# -*- coding: utf-8 -*-
"""sweep_upstream.py —— 论文图 5 的参数扫描(不改上游任何文件)

上游 `single_cbf.py` 把 `d_c` / `alpha` 写成模块级全局::

    # single_cbf.py 第 26-33 行
    time_step = 0.1
    mu = 0
    alpha = 0.5
    d = 0.1
    ...

而 `find_safe_action()` 里直接读这两个全局。于是**不需要改文件**:把上游模块
import 进来,改它的模块属性即可(第 152 行有 `if __name__ == '__main__':` 保护,
import 不会触发实验)。本脚本就是这样一个"外挂驱动":

    import single_cbf as S
    S.d = 0.5        # 等价于把源码里的 d = 0.5
    S.alpha = 0.25

扫描设置与论文 V-D 节一致:`alpha=0.5` 时 `d_c ∈ {0.1, 0.5, 0.9}`;
`d_c=0.1` 时 `alpha ∈ {0.25, 0.5, 0.75}`(论文正文写 0.5,图注写 0.1,这里按图注)。

**耗时提醒**:每个配置 = 100 步 × (控制环 0.1 s + 每步一次 1200×680 展示渲染 ≈ 7 s)
≈ 12 分钟;6 个配置 ≈ 70 分钟。想快就把 --steps 调小(但机器人可能还没走到墙)。

用法::

    python tools/sweep_upstream.py                      # 6 个配置 × 1 次,约 70 分钟
    python tools/sweep_upstream.py --steps 60           # 只跑 60 步(约 40 分钟)
    python tools/sweep_upstream.py --only dc            # 只扫 d_c
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = ROOT / 'upstream'
INIT_STATE = [-2, 0.2, 0.5, 90, 0, 115]     # 与上游 single_cbf.py 一致
SWEEP_DC = [0.1, 0.5, 0.9]
SWEEP_ALPHA = [0.25, 0.5, 0.75]
ALPHA_FIXED = 0.5
DC_FIXED = 0.1      # 扫 alpha 时固定的 d_c(论文图注)


def load_upstream():
    """import 上游 single_cbf 模块,并补齐它期望的全局 args。"""
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    os.chdir(UPSTREAM)                        # 上游用相对路径读 configs/ 与 device.txt
    sys.path.insert(0, str(UPSTREAM))
    import single_cbf as S

    class _Args:                              # 只被 NICE_SLAM.__init__ / get_dataset 读取
        nice = True
        output = 'output/Replica/room1'
        input_folder = None
        save_rendering = False
        vis_input_frame = False
        no_gt_traj = False
    if not hasattr(S, 'args'):
        S.args = _Args()
    return S


def rollout(S, robot, steps, dc, alpha, seed=0):
    """复刻上游 single_cbf.py 的主循环(只取数值,不写视频)。"""
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    S.d = dc
    S.alpha = alpha

    pose = S.state_to_pose(torch.tensor(INIT_STATE).unsqueeze(0).to(S.device)).squeeze()
    with contextlib.redirect_stdout(io.StringIO()):        # 关掉上游的逐帧 print
        depth, _ = robot.render(pose.to(S.device))
    recs = []
    for frame in range(1, steps + 1):
        with contextlib.redirect_stdout(io.StringIO()):
            action, h, is_safe = S.find_safe_action(
                robot, pose, S.d - depth.min(), -S.intend, 'up')
            iv = 0.0 if is_safe else float(torch.norm(
                torch.tensor(-S.intend * torch.mm(pose[:3, :3].float(), S.u).squeeze().to(S.device)) - action, p=2))
            state = torch.cat((pose[:3, -1].to(S.device), torch.from_numpy(
                S.R.from_matrix(pose[:3, :3].cpu()).as_euler('xyz', degrees=True)).to(S.device)), dim=0).to(S.device)
            state = S.update_dynamics(state, action)
            pose = S.state_to_pose(state.unsqueeze(0)).squeeze()
            depth, _ = robot.render(pose.to(S.device))
        md = float(depth.min())
        recs.append(dict(step=frame, min_depth=md, h=float(S.d - md),
                         intervention=iv, velocity=None, t_eval=None))
    return recs


def stats(recs, dc):
    dep = np.array([r['min_depth'] for r in recs])
    iv = np.array([r['intervention'] for r in recs])
    first = next((r['step'] for r in recs if r['intervention'] > 0), None)
    return dict(min_depth=float(dep.min()), iv_mean=float(iv.mean()),
                iv_max=float(iv.max()), first_iv=first, dc=dc)


def main(argv=None):
    p = argparse.ArgumentParser(description='论文图 5 参数扫描(外挂驱动,不改上游代码)')
    p.add_argument('--steps', type=int, default=100, help='每个配置的步数(默认 100 = 10 s)')
    p.add_argument('--only', choices=('dc', 'alpha', 'both'), default='both')
    p.add_argument('--out', default=str(ROOT / 'results' / 'figures'))
    args = p.parse_args(argv)

    S = load_upstream()
    cfg = S.config.load_config('configs/Replica/room1.yaml', 'configs/nice_slam.yaml')
    print('构造 Robot(会加载作者发布的 Room1 权重)…')
    robot = S.Robot(cfg)

    result = {'dc': {}, 'alpha': {}}
    plan = []
    if args.only in ('dc', 'both'):
        plan += [('dc', dc, ALPHA_FIXED) for dc in SWEEP_DC]
    if args.only in ('alpha', 'both'):
        plan += [('alpha', a, DC_FIXED) for a in SWEEP_ALPHA]

    for i, (grp, val, other) in enumerate(plan, 1):
        dc = val if grp == 'dc' else other
        alpha = other if grp == 'dc' else val
        print('\n[%d/%d] %s 扫描:%s = %.2f(另一个固定为 %.2f),%d 步…'
              % (i, len(plan), grp, 'd_c' if grp == 'dc' else 'α', val, other, args.steps))
        recs = rollout(S, robot, args.steps, dc, alpha)
        st = stats(recs, dc)
        result[grp][val] = st
        print('      最小距离 %.4f m | 平均干预 %.2f | 最大干预 %.2f | 首次干预 %s'
              % (st['min_depth'], st['iv_mean'], st['iv_max'], st['first_iv']))

    # 复用 plot_from_log 的风格画图
    sys.path.insert(0, str(ROOT / 'tools'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, key, title in ((axes[0], 'dc', r'sweep $d_c$ ($\alpha=0.5$)'),
                           (axes[1], 'alpha', r'sweep $\alpha$ ($d_c=%.1f$)' % DC_FIXED)):
        cur = result[key]
        xs = sorted(cur)
        ax.plot(xs, [cur[x]['iv_mean'] for x in xs], 'o-', label='mean intervention')
        ax.plot(xs, [cur[x]['iv_max'] for x in xs], 's--', label='max intervention')
        ax.plot(xs, [cur[x]['min_depth'] for x in xs], '^-', label='min distance (m)')
        ax.set_xlabel(r'$d_c$' if key == 'dc' else r'$\alpha$')
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, ls=':')
        ax.legend(fontsize=8)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    fig.suptitle('Parameter sweep (upstream single_cbf.py, paper Fig.5 analog)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    png = out / 'figure5_sweep.png'
    fig.savefig(png, dpi=130)
    plt.close(fig)

    print('\n' + '-' * 76)
    print(' %-8s %-8s %12s %12s %12s' % ('组', '取值', '最小距离', '平均干预', '最大干预'))
    for key in ('dc', 'alpha'):
        for v in sorted(result[key]):
            s = result[key][v]
            print(' %-8s %-8.2f %12.4f %12.2f %12.2f'
                  % (key, v, s['min_depth'], s['iv_mean'], s['iv_max']))
    print('-' * 76)
    print('图表:', png)
    return 0


if __name__ == '__main__':
    sys.exit(main())
