# -*- coding: utf-8 -*-
"""
双积分器 CBF 视觉安全滤波实验(论文复现脚本)
================================================================
对应论文:Tong, Dawson, Fan.《Enforcing Safety for Vision-based Controllers
via Control Barrier Functions and Neural Radiance Fields》, ICRA 2023
第 V-B-b 节实验,结果对应论文图 4a(CBF 值/干预量/速度/最小距离曲线)。

与单积分器版本(single_cbf.py)的区别
------------------------------------
动力学换为带惯性的双积分器(控制量是加速度):

    x_{t+1} = x_t + v_t·dt + (1/2)·a·dt²
    v_{t+1} = v_t + a·dt

CBF 在距离项基础上增加速度项(论文 IV-B 节):

    h(y, v) = d_c - min_depth(y) + β·‖v‖      (d_c = 0.1 m, β = 1)

速度越快 h 越大 → 滤波器在碰撞尚未来临时就"外推"并提前干预减速。
标称策略为"1 m/s² 恒定加速冲墙"。

运行方式
--------
    python -W ignore double_cbf.py configs/Replica/room1.yaml --output output/Replica/room1

输出:逐行简洁日志(最近深度 / h / 速度 / 干预量 / 控制耗时)与结束统计。
(原码的视频写出代码被作者注释,本实验以数值轨迹为结果;
 需要视频可仿照 single_cbf.py 恢复,注意 VideoWriter 尺寸应为 (2400, 680)。)

说明:本脚本在官方源码基础上做了兼容性修补(torch.load 的 weights_only、
num_workers=0),实验逻辑与原码完全一致;并移除了产生混乱 tensor 输出的
调试打印。
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader

from src import config
from src.utils.datasets import get_dataset
from src.NICE_SLAM import NICE_SLAM
from scipy.spatial.transform import Rotation as R

# 日志文件将以 UTF-8 写出(中文不乱码);行缓冲保证重定向到文件时也逐行即时落盘
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

# ------------------------------------------------------------------
# 全局配置
# ------------------------------------------------------------------
with open('device.txt', encoding='utf-8') as file:
    device = file.read().strip()          # 运算设备,仓库自带 "cuda:0"

time_step = 0.1      # 控制周期 dt = 0.1 s → 控制频率 10 Hz
mu = 0               # 候选动作高斯采样均值
alpha = 0.5          # CBF 衰减率 α
d = 0.1              # 最小安全间隙 d_c = 0.1 m
u = torch.tensor([[0, 0, 1]]).t().float().to(device)  # 机体前向轴(相机坐标系 z 轴)
max_h = 2            # (指示条量程,视频未启用,保留原码设定)
intend = sigma = 1   # 标称加速度 1 m/s²;采样标准差 σ = 1
beta = 1             # CBF 速度项权重 β = 1(论文 V-A)


def state_to_pose(state):
    """状态向量 → 4×4 相机位姿矩阵(位置 + 欧拉角,支持批量 (B,6) → (B,4,4))。"""
    for i in range(state.shape[0]):
        rm = torch.from_numpy(R.from_euler('xyz', state[i][3:].cpu(), degrees=True).as_matrix()).to(device)
        pose = torch.cat((torch.cat((rm, state[i][:3].unsqueeze(-1)), 1), torch.tensor([[0, 0, 0, 1]]).to(device)), 0).unsqueeze(0)
        if i == 0:
            batch_pose = pose
        else:
            batch_pose = torch.cat((batch_pose, pose), dim=0)
    return batch_pose


def update_dynamics(state, v, action):
    """双积分器动力学一步更新(半欧拉积分):
    x_{t+1} = x_t + v_t·dt + a·dt²/2;v_{t+1} = v_t + a·dt。返回 (新状态, 新速度)。"""
    return state + time_step * v + 0.5 * action * (time_step ** 2), v + time_step * action


def pose_to_state(pose):
    """4×4 位姿矩阵 → 状态向量(位置 + 欧拉角)。"""
    return torch.cat((pose[:3, -1].to(device),
                      torch.from_numpy(R.from_matrix(pose[:3, :3].cpu()).as_euler('xyz', degrees=True)).to(device)), dim=0)


class Robot:
    """装备 NICE-SLAM 神经隐式地图的机器人(同 single_cbf.py,另维护速度状态 v)。"""

    def __init__(self, cfg):
        self.c = np.load('replica_room1.npy', allow_pickle=True).item()
        self.decoders = torch.load('replica_room1.pth', weights_only=False)  # weights_only: 兼容 torch≥2.6
        self.renderer = NICE_SLAM(cfg, args).renderer
        self._set_cam(low_res=True)
        self.v = torch.zeros(6).to(device)    # 当前速度(6 维,本实验只有前向平移分量被激发)

    def _set_cam(self, low_res):
        """切换相机内参:低分辨率 120×68(控制评估)/ 高分辨率 1200×680(视频)。"""
        if low_res:
            self.renderer.H, self.renderer.W = 68, 120
            self.renderer.fx = self.renderer.fy = 60
            self.renderer.cx, self.renderer.cy = 59.95, 33.95
        else:
            self.renderer.H, self.renderer.W = 680, 1200
            self.renderer.fx = self.renderer.fy = 600.0
            self.renderer.cx, self.renderer.cy = 599.5, 339.5

    def predict_observation(self, pose):
        """NSPredict:在给定(批量)位姿处渲染深度图。"""
        depth, _, _ = self.renderer.render_batch_img(
            self.c, self.decoders, pose, device, stage='middle', gt_depth=None)
        return depth

    def render(self, pose):
        """高分辨率渲染当前位姿的(深度, RGB)图,仅用于演示,不参与控制。"""
        self._set_cam(low_res=False)
        depth, _, _ = self.renderer.render_img(
            self.c, self.decoders, pose, device, stage='color', gt_depth=None)
        depth, uncertainty, color = self.renderer.render_img(
            self.c, self.decoders, pose, device, stage='color', gt_depth=depth)
        self._set_cam(low_res=True)
        return depth, color


def find_safe_action(robot, pose, h, intended_action, direction):
    """算法 1(双积分器版):审查标称加速度指令,不安全则搜索安全替代。

    与单积分器版的差别:
      · h 中含速度项 h(y,v) = d_c - min_depth(y) + β·‖v‖;
      · 候选采样无拒绝步骤(允许任意方向的加速度候选);
      · 评估候选时用更新后的速度 v_{t+1} 计算 h_{t+1}。

    返回 (action, new_h, is_safe, intervention),new_h 已含新速度的贡献。
    """
    state = pose_to_state(pose)
    # 标量指令 → 6 维动作向量('up' = 沿机体前向加速)
    orient_action = torch.zeros(6).to(device)
    if direction in ['up', 'down']:
        unit = intended_action * torch.mm(pose[:3, :3].float(), u).squeeze().to(device)
        orient_action[0] = unit[0]
        orient_action[1] = unit[1]
        orient_action[2] = unit[2]
    elif direction in ['left', 'right']:
        orient_action[5] = intended_action * 10

    # ---- 第一步:试标称指令(干预量 0)----
    new_state, new_v = update_dynamics(state, robot.v, orient_action)
    new_pose = state_to_pose(new_state.unsqueeze(0))
    new_h = d - robot.predict_observation(new_pose).min().unsqueeze(0) + beta * torch.norm(new_v, p=2)
    best_action = torch.zeros(6).to(device)
    if new_h <= alpha * h:
        return orient_action, new_h, True, 0.0    # 标称指令安全 → 放行

    # ---- 第二步:批量采样 10 个候选,并行渲染验证 ----
    while True:
        batch_action = torch.zeros((10, 6)).to(device)
        if direction in {'up', 'down'}:
            for j in range(10):
                value = np.random.normal(mu, sigma)
                unit = value * torch.mm(pose[:3, :3].float(), u).squeeze().to(device)
                batch_action[j][0] = unit[0]
                batch_action[j][1] = unit[1]
                batch_action[j][2] = unit[2]
        else:
            for j in range(10):
                batch_action[j][5] = np.random.normal(mu, sigma) * 10

        batch_new_state, batch_new_v = update_dynamics(state, robot.v, batch_action)
        batch_new_pose = state_to_pose(batch_new_state)
        # 每个候选的 h_{t+1} = d_c - min_depth(预测深度) + β·‖v_{t+1}‖
        batch_new_h = d - robot.predict_observation(batch_new_pose).min(dim=-1)[0].min(dim=-1)[0] + beta * torch.norm(batch_new_v, dim=1, p=2)

        for j in range(10):
            if batch_new_h[j] <= alpha * h:
                if torch.norm(best_action, p=2) == 0 or torch.norm(batch_action[j] - orient_action, p=2) < torch.norm(best_action - orient_action, p=2):
                    best_action = batch_action[j]
                    new_h = batch_new_h[j]
        if torch.norm(best_action, p=2) > 0:
            intervention = float(torch.norm(orient_action - best_action, p=2))
            return best_action, new_h, False, intervention


if __name__ == '__main__':
    # ---------------- 命令行参数(与官方仓库保持一致) ----------------
    parser = argparse.ArgumentParser(description='双积分器 CBF 视觉安全滤波实验。')
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str, help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str, help='output folder, this have higher priority, can overwrite the one in config file')
    nice_parser = parser.add_mutually_exclusive_group(required=False)
    nice_parser.add_argument('--nice', dest='nice', action='store_true')
    nice_parser.add_argument('--imap', dest='nice', action='store_false')
    parser.set_defaults(nice=True)
    parser.add_argument('--save_rendering', action='store_true', help='save rendering video to `vis.mp4` in output folder ')
    parser.add_argument('--vis_input_frame', action='store_true', help='visualize input frames')
    parser.add_argument('--no_gt_traj', action='store_true', help='not visualize gt trajectory')
    args = parser.parse_args()

    cfg = config.load_config(args.config, 'configs/nice_slam.yaml' if args.nice else 'configs/imap.yaml')
    scale = cfg['scale']
    output = cfg['data']['output'] if args.output is None else args.output

    # 数据集仅用于启动校验(读第 0 帧),其取值不参与实验
    frame_reader = get_dataset(cfg, args, scale, device=device)
    frame_loader = DataLoader(frame_reader, batch_size=1, shuffle=False, num_workers=0)
    ckptsdir = f'{output}/ckpts'          # (沿用原码:若有 SLAM 断点则恢复,本实验不会产生)
    if os.path.exists(ckptsdir):
        ckpts = [os.path.join(ckptsdir, f)
                 for f in sorted(os.listdir(ckptsdir)) if 'tar' in f]
        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            print('Get ckpt :', ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=torch.device(device), weights_only=False)
            estimate_c2w_list = ckpt['estimate_c2w_list']
            gt_c2w_list = ckpt['gt_c2w_list']
            N = ckpt['idx']
            estimate_c2w_list[:, :3, 3] /= scale
            gt_c2w_list[:, :3, 3] /= scale
    # The original first dataset frame is discarded; use the pretrained scene directly.

    robot = Robot(cfg)

    # 初始位姿(与单积分器实验相同):位置 (-2, 0.2, 0.5),欧拉角 (90°, 0°, 115°)
    pose = state_to_pose(torch.tensor([-2, 0.2, 0.5, 90, 0, 115]).unsqueeze(0).to(device)).squeeze()
    depth, color = robot.render(pose.to(device))
    # 初始 h_t:速度为 0,速度项不起作用
    h = d - depth.min().unsqueeze(0)

    # ---------------- 实验主循环(100 步 = 10 s 仿真时间) ----------------
    N_STEPS = int(os.environ.get("CBF_STEPS", "100"))
    stats = []    # 每步记录: (step, min_depth, h, v_norm, intervention, is_safe, t_eval)
    h0 = float(h)
    print('=' * 64)
    print(' 双积分器 CBF 视觉安全滤波实验(论文 V-B-b,对应图 4a)')
    print('-' * 64)
    print(' 场景    : Replica Room1(NICE-SLAM 隐式地图,作者预训练权重)')
    print(' 动力学  : 双积分器(控制量 = 加速度),dt = 0.1 s(10 Hz)')
    print(' CBF     : h(y,v) = d_c - min_depth(y) + β·‖v‖,d_c = 0.1 m,β = 1')
    print(' 标称策略: 1 m/s² 恒定加速冲墙(最坏情况压力测试)')
    print(' 安全判据: h_(t+1) ≤ α·h_t,不满足则采样 10 个候选并行验证')
    print(' 注      : h 为滤波器内部值(基于 120×68 快速渲染预测);')
    print('           最近深度为 1200×680 展示渲染,两者略有差异属正常')
    print('-' * 64)
    print(' 初始最近深度 = %.3f m(h = %.3f,速度 = 0)' % (float(depth.min()), h0))
    print('-' * 64)

    for step in range(1, N_STEPS + 1):
        # --- 控制决策:审查"1 m/s² 加速冲墙";当前 h_t 含速度项 β‖robot.v‖ ---
        t0 = time.time()
        action, h, is_safe, intervention = find_safe_action(
            robot, pose, d - depth.min() + beta * torch.norm(robot.v, p=2), -intend, 'up')
        t_eval = time.time() - t0

        # --- 执行动作:同时更新位置与速度 ---
        state = pose_to_state(pose)
        state, robot.v = update_dynamics(state, robot.v, action)
        pose = state_to_pose(state.unsqueeze(0)).squeeze()

        # --- 渲染新观测(仅用于下一步的 h 计算,本实验不存视频) ---
        depth, color = robot.render(pose.to(device))

        min_depth = float(depth.min())
        v_norm = float(torch.norm(robot.v, p=2))
        stats.append((step, min_depth, float(h), v_norm, intervention, is_safe, t_eval))
        verdict = '放行' if is_safe else '拦截'
        print('[步 %3d/%d] 最近深度 = %.3f m | h = %+.3f | 速度 = %.3f m/s | 干预 = %.2f m/s² | %s'
              % (step, N_STEPS, min_depth, float(h), v_norm, intervention, verdict))

    # ---------------- 结束统计 ----------------
    depths = [s[1] for s in stats]
    hs = [s[2] for s in stats]
    vs = [s[3] for s in stats]
    ivs = [s[4] for s in stats]
    tvals = [s[6] for s in stats]
    n_unsafe = sum(1 for s in stats if not s[5])
    first_iv = next((s[0] for s in stats if s[4] > 0), None)
    collided = min(depths) <= d
    print('=' * 64)
    print(' 实验结束:%d 步(%.1f s 仿真时间)' % (N_STEPS, N_STEPS * time_step))
    print(' 碰撞判定 : %s(最近深度最小值 %.3f m %s d_c = %.1f m)'
          % ('发生碰撞' if collided else '未碰撞 ✓', min(depths), '<=' if collided else '>', d))
    print(' 最近深度 : 最小 %.3f m | 终值 %.3f m | 初始 %.3f m' % (min(depths), depths[-1], d - h0))
    print(' CBF 值 h : 最大 %+.4f(理论要求始终 ≤ 0)' % max(hs))
    print(' 速度     : 峰值 %.3f m/s | 终值 %.3f m/s(滤波器将冲墙速度刹停)'
          % (max(vs), vs[-1]))
    if first_iv is not None:
        fd = next(s for s in stats if s[0] == first_iv)
        print(' 干预     : 首次第 %d 步(当时最近深度 %.3f m,速度 %.3f m/s)| 拦截 %d/%d 步 | 平均 %.2f | 峰值 %.2f m/s²'
              % (first_iv, fd[1], fd[3], n_unsafe, N_STEPS, sum(ivs) / len(ivs), max(ivs)))
    else:
        print(' 干预     : 全程未干预')
    print(' 控制耗时 : 平均 %.3f s/步 | 峰值 %.3f s(10 Hz 要求 ≤ 0.1 s 完成决策)'
          % (sum(tvals) / len(tvals), max(tvals)))
    print(' 输出     : 数值轨迹(可用 parse_double_log.py 绘图对照论文图 4a)')
    print('=' * 64)
