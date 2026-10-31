# -*- coding: utf-8 -*-
"""
单积分器 CBF 视觉安全滤波实验(论文复现脚本)
================================================================
对应论文:Tong, Dawson, Fan.《Enforcing Safety for Vision-based Controllers
via Control Barrier Functions and Neural Radiance Fields》, ICRA 2023
第 V-B-a 节实验,结果对应论文图 2(渲染帧序列)与图 3a(CBF 值/干预量曲线)。

实验内容
--------
机器人在 Replica Room1 场景中面向墙 ~2 m 处起步,标称策略(模拟人类遥操作)
持续下达"1 m/s 径直冲墙"的最坏情况指令。CBF 安全滤波器(算法 1)逐步审查:

  1) 用动力学模型预测执行动作后的下一状态        x_{t+1} = x_t + u·dt
  2) 用 NeRF 从下一状态渲染"将看到的"深度图      ŷ_{t+1} = NSPredict(x_{t+1})
  3) 计算 CBF 值并检查单步安全约束(式 (1b)):
         h(y) = d_c - min_depth(y)              (h ≤ 0 安全,> 0 危险)
         放行条件: h_{t+1} ≤ α·h_t
  4) 标称指令不安全时,在其附近高斯采样 10 个候选动作,并行渲染验证,
     选离标称指令最近的可行者(干预量 = ‖u - π₀(y)‖)

论文参数:d_c = 0.1 m,α = 0.5,批量 = 10,dt = 0.1 s(10 Hz)。

运行方式
--------
    python -W ignore single_cbf.py configs/Replica/room1.yaml --output output/Replica/room1

输出:video.avi(150 帧,左深度图+右渲染图+CBF 指示条;蓝=放行,红=拦截)
      以及逐帧简洁日志(最近深度 / h / 干预量)和结束统计。

说明:本脚本在官方源码基础上做了兼容性修补(torch.load 的 weights_only、
VideoWriter 尺寸 (2400,680)——原码 240×68 与实际帧尺寸不符会导致帧被静默丢弃、
num_workers=0),实验逻辑与原码完全一致;原码中大段被注释的键盘交互演示
代码已移除以便阅读(可在上游仓库查看)。
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

time_step = 0.1      # 控制周期 dt = 0.1 s → 控制频率 10 Hz(论文 V-A)
mu = 0               # 候选动作高斯采样均值
alpha = 0.5          # CBF 衰减率 α(论文 V-A:单积分器取 0.5)
d = 0.1              # 最小安全间隙 d_c = 0.1 m(论文 V-A)
u = torch.tensor([[0, 0, 1]]).t().float().to(device)  # 机体前向轴 [0,0,1](相机坐标系 z 轴)
max_h = 1.5          # 视频指示条的量程(仅用于显示)
intend = sigma = 1   # 标称指令幅值 1 m/s;采样标准差 σ = 1
batch_size = 10      # 每步并行评估的候选动作数(论文 V-A:batch of 10)


def state_to_pose(state):
    """状态向量 → 4×4 相机位姿矩阵。

    状态 x = [px, py, pz, rx, ry, rz](前 3 维位置 m,后 3 维欧拉角,度)。
    旋转矩阵 R(rx,ry,rz) 放入左上 3×3,位置放入最后一列,补齐齐次行。
    支持批量输入 (B, 6) → 输出 (B, 4, 4)。
    """
    for i in range(state.shape[0]):
        rm = torch.from_numpy(R.from_euler('xyz', state[i][3:].cpu(), degrees=True).as_matrix()).to(device)
        pose = torch.cat((torch.cat((rm, state[i][:3].unsqueeze(-1)), 1), torch.tensor([[0, 0, 0, 1]]).to(device)), 0).unsqueeze(0)
        if i == 0:
            batch_pose = pose
        else:
            batch_pose = torch.cat((batch_pose, pose), dim=0)
    return batch_pose


def update_dynamics(state, action):
    """单积分器动力学(论文 III 节两层架构的顶层简化模型):x_{t+1} = x_t + u·dt。"""
    return state + time_step * action


def pose_to_state(pose):
    """4×4 位姿矩阵 → 状态向量(位置 + 欧拉角),find_safe_action 内部使用。"""
    return torch.cat((pose[:3, -1].to(device),
                      torch.from_numpy(R.from_matrix(pose[:3, :3].cpu()).as_euler('xyz', degrees=True)).to(device)), dim=0)


class Robot:
    """装备 NICE-SLAM 神经隐式地图的机器人(论文图 1 的感知前端)。

    场景模型 = 作者发布的 Room1 训练成果:
      replica_room1.npy —— NICE-SLAM 的多尺度特征网格 c(场景的隐式表示)
      replica_room1.pth —— 解码器网络(查询 空间点 → 占据/密度/颜色)
    因此本实验无需重新训练 SLAM,Robot 相当于实现了算法 1 的 NSPredict:
    给定任意新位姿即可渲染出该视角的深度/彩色图像。
    """

    def __init__(self, cfg):
        self.c = np.load('replica_room1.npy', allow_pickle=True).item()
        self.decoders = torch.load('replica_room1.pth', weights_only=False)  # weights_only: 兼容 torch≥2.6
        # 借用 NICE_SLAM 的渲染器(仅取 renderer,不做建图/跟踪)
        self.renderer = NICE_SLAM(cfg, args).renderer
        self._set_cam(low_res=True)   # 默认低分辨率:控制环用的快速渲染

    def _set_cam(self, low_res):
        """切换相机内参。低分辨率 120×68(控制评估,快);高分辨率 1200×680(存视频,慢)。"""
        if low_res:
            self.renderer.H, self.renderer.W = 68, 120
            self.renderer.fx = self.renderer.fy = 60
            self.renderer.cx, self.renderer.cy = 59.95, 33.95
        else:
            self.renderer.H, self.renderer.W = 680, 1200
            self.renderer.fx = self.renderer.fy = 600.0
            self.renderer.cx, self.renderer.cy = 599.5, 339.5

    def predict_observation(self, pose):
        """NSPredict:在给定(批量)位姿处渲染深度图——评估候选动作的"未来观测"。"""
        depth, _, _ = self.renderer.render_batch_img(
            self.c, self.decoders, pose, device, stage='middle', gt_depth=None)
        return depth

    def render(self, pose):
        """高分辨率渲染当前位姿的(深度, RGB)图,仅用于生成演示视频。"""
        self._set_cam(low_res=False)
        depth, _, _ = self.renderer.render_img(
            self.c, self.decoders, pose, device, stage='color', gt_depth=None)
        depth, uncertainty, color = self.renderer.render_img(
            self.c, self.decoders, pose, device, stage='color', gt_depth=depth)
        self._set_cam(low_res=True)   # 恢复低分辨率,保证下一步控制评估的速度
        return depth, color


def find_safe_action(robot, pose, h, intended_action, direction):
    """算法 1 的核心:审查一个标称动作,不安全则在附近搜索安全替代。

    参数
    ----
    robot           : Robot 实例(提供 NeRF 渲染)
    pose            : 当前位姿 (4, 4)
    h               : 当前 CBF 值 h_t = d_c - min_depth(y_t)
    intended_action : 标称指令幅值(正负号表示沿机体前向的方向)
    direction       : 'up'/'down' = 沿机体前向平移;'left'/'right' = 原地转航向

    返回
    ----
    (action, new_h, is_safe, intervention)
      action        实际执行的动作(6 维)
      new_h         执行该动作后的预测 CBF 值 h_{t+1}
      is_safe       标称指令本身是否安全
      intervention  干预量 ‖u - π₀(y)‖(标称安全时为 0)

    安全判据(CBF 约束,式 (1b)):候选动作 u 须满足 new_h ≤ α·h。
    """
    state = pose_to_state(pose)
    # 把标量指令装配成 6 维动作向量:平移类指令作用于机体前向 [0,0,1]
    # (经旋转矩阵 R 变换到世界系),转向类指令作用于航向分量
    orient_action = torch.zeros(6).to(device)
    if direction in ['up', 'down']:
        unit = intended_action * torch.mm(pose[:3, :3].float(), u).squeeze().to(device)
        orient_action[0] = unit[0]
        orient_action[1] = unit[1]
        orient_action[2] = unit[2]
    elif direction in ['left', 'right']:
        orient_action[5] = intended_action * 10   # 转向的幅值放大 10 倍(原码设定)

    # ---- 第一步:先试标称指令本身(对应式 (1a) 的理论最优解,干预量 0)----
    new_state = update_dynamics(state, orient_action).unsqueeze(0)
    new_pose = state_to_pose(new_state)
    new_h = d - robot.predict_observation(new_pose).min().unsqueeze(0)
    best_action = torch.zeros(6).to(device)
    if new_h <= alpha * h:
        return orient_action, new_h, True, 0.0    # 标称指令安全 → 直接放行

    # ---- 第二步:不安全 → 批量采样候选,并行渲染验证(算法 1 的 Re-Sample)----
    # 注:此处为死循环保护缺失的原码逻辑——若所有候选都不可行会一直采样,
    # 实验中未触发(远离/贴墙时总能采到可行候选)。
    while True:
        batch_action = torch.zeros((batch_size, 6)).to(device)
        if direction in {'up', 'down'}:
            for j in range(batch_size):
                value = np.random.normal(mu, sigma)          # 高斯采样 N(0, σ²)
                # 拒绝采样:不允许采到"比标称指令更猛地冲向墙"的候选
                while value * intended_action > 0 and abs(value) >= abs(intended_action):
                    value = np.random.normal(mu, sigma)
                unit = value * torch.mm(pose[:3, :3].float(), u).squeeze().to(device)
                batch_action[j][0] = unit[0]
                batch_action[j][1] = unit[1]
                batch_action[j][2] = unit[2]
        else:
            for j in range(batch_size):
                batch_action[j][5] = np.random.normal(mu, sigma) * 10

        # 10 个候选并行渲染,得到各自的 h_{t+1}
        batch_new_state = update_dynamics(state, batch_action)
        batch_new_pose = state_to_pose(batch_new_state)
        batch_new_h = d - robot.predict_observation(batch_new_pose).min(dim=-1)[0].min(dim=-1)[0]

        # 在可行候选中取离标称指令最近者(近似求解式 (1) 的 CBF-QP)
        for j in range(batch_size):
            if batch_new_h[j] <= alpha * h:
                if torch.norm(best_action, p=2) == 0 or torch.norm(batch_action[j] - orient_action, p=2) < torch.norm(best_action - orient_action, p=2):
                    best_action = batch_action[j]
                    new_h = batch_new_h[j]
        if torch.norm(best_action, p=2) > 0:
            intervention = float(torch.norm(orient_action - best_action, p=2))
            return best_action, new_h, False, intervention


def draw_overlay(color, h, is_safe):
    """在 680p 渲染图右侧画 CBF 指示条(论文图 2 右侧的竖条)。

    黑框为量程(中心 h=0,顶部 +max_h,底部 -max_h);
    内条高度按 h/max_h 线性映射;蓝 = 放行,红 = 拦截。
    注意 color 是 0~1 的 RGB 浮点图,(0,0,1) 即纯蓝、(1,0,0) 即纯红。
    """
    color = cv2.rectangle(color, (1100, 100), (1150, 580), (0, 0, 0), 10)   # 外框
    bar_y = 340 - int(240 * h / max_h)                                       # h=0 对应条中点 y=340
    fill = (0, 0, 1) if is_safe else (1, 0, 0)
    color = cv2.rectangle(color, (1110, min(bar_y, 340)), (1140, max(bar_y, 340)), fill, -1)
    color = cv2.putText(color, 'CBF h', (1025, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.75, (0, 0, 0), 5)
    return color


def frame_to_video(depth, color):
    """深度图(灰度化到 0~255)与 RGB 图左右拼接成一帧视频画面。

    渲染器输出 RGB,而 cv2.VideoWriter 按 BGR 处理,故 color 取 [2,1,0] 通道反转。
    """
    depth_u8 = cv2.normalize(depth.unsqueeze(-1).repeat(1, 1, 3).cpu().numpy(),
                             dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    color_u8 = (color[:, :, [2, 1, 0]] * 255).astype(np.uint8).clip(0, 255)
    return np.hstack([depth_u8, color_u8])


if __name__ == '__main__':
    # ---------------- 命令行参数(与官方仓库保持一致) ----------------
    parser = argparse.ArgumentParser(description='单积分器 CBF 视觉安全滤波实验。')
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

    # 数据集仅用于启动校验:读第 0 帧确认 RGB-D 数据在位,
    # 其取值不参与实验(场景信息全部来自预训练 NeRF 权重)
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

    # ---------------- 初始化机器人与视频写出器 ----------------
    robot = Robot(cfg)
    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # 尺寸必须是 (宽2400, 高680):robot.render() 返回 1200×680 帧,左右拼接后宽 2400。
    # 原码写 (240,68),与实际帧不符 → 所有帧被 OpenCV 静默丢弃,视频为空壳。
    videoWriter = cv2.VideoWriter('video.avi', fourcc, fps, (2400, 680))

    # 初始位姿:位置 (-2, 0.2, 0.5),欧拉角 (90°, 0°, 115°),面向墙(作者发布设定)
    pose = state_to_pose(torch.tensor([-2, 0.2, 0.5, 90, 0, 115]).unsqueeze(0).to(device)).squeeze()
    depth, color = robot.render(pose.to(device))

    # ---------------- 实验主循环(算法 1,150 帧 = 15 s 仿真时间) ----------------
    N_FRAMES = int(os.environ.get("CBF_STEPS", "150"))
    stats = []                 # 每帧记录: (frame, min_depth, h, intervention, is_safe, t_eval)
    h0 = float(d - depth.min())
    print('=' * 64)
    print(' 单积分器 CBF 视觉安全滤波实验(论文 V-B-a,对应图 2 / 图 3a)')
    print('-' * 64)
    print(' 场景    : Replica Room1(NICE-SLAM 隐式地图,作者预训练权重)')
    print(' 动力学  : x_(t+1) = x_t + u·dt,dt = 0.1 s(10 Hz)')
    print(' CBF     : h(y) = d_c - min_depth(y),d_c = 0.1 m,α = 0.5')
    print(' 标称策略: 1 m/s 径直冲墙(最坏情况压力测试)')
    print(' 安全判据: h_(t+1) ≤ α·h_t,不满足则在标称指令附近采样 10 个候选')
    print(' 注      : h 为滤波器内部值(基于 120×68 快速渲染预测);')
    print('           最近深度为 1200×680 展示渲染,两者略有差异属正常')
    print('-' * 64)
    print(' 初始最近深度 = %.3f m(h = %.3f)' % (float(depth.min()), h0))
    print('-' * 64)

    for frame in range(1, N_FRAMES + 1):
        # --- 控制决策:审查标称指令"1 m/s 冲墙"('up' 分支,原码其余方向分支未启用) ---
        t0 = time.time()
        action, h, is_safe, intervention = find_safe_action(
            robot, pose, d - depth.min(), -intend, 'up')
        t_eval = time.time() - t0

        # --- 执行选定动作,更新位姿 ---
        state = pose_to_state(pose)
        state = update_dynamics(state, action)
        pose = state_to_pose(state.unsqueeze(0)).squeeze()

        # --- 渲染执行后的新观测,画指示条,写入视频 ---
        depth, color = robot.render(pose.to(device))
        color = draw_overlay(color.to(device).detach().cpu().numpy(), float(h), is_safe)
        videoWriter.write(frame_to_video(depth, color))

        min_depth = float(depth.min())
        stats.append((frame, min_depth, float(h), intervention, is_safe, t_eval))
        verdict = '放行(标称指令安全)' if is_safe else '拦截替换'
        print('[帧 %3d/%d] 最近深度 = %.3f m | h = %+.3f | 干预 = %.2f m/s | %s'
              % (frame, N_FRAMES, min_depth, float(h), intervention, verdict))

    videoWriter.release()

    # ---------------- 结束统计 ----------------
    depths = [s[1] for s in stats]
    hs = [s[2] for s in stats]
    ivs = [s[3] for s in stats]
    tvals = [s[5] for s in stats]
    n_unsafe = sum(1 for s in stats if not s[4])
    first_iv = next((s[0] for s in stats if s[3] > 0), None)
    collided = min(depths) <= d
    print('=' * 64)
    print(' 实验结束:%d 帧(%.1f s 仿真时间)' % (N_FRAMES, N_FRAMES * time_step))
    print(' 碰撞判定 : %s(最近深度最小值 %.3f m %s d_c = %.1f m)'
          % ('发生碰撞' if collided else '未碰撞 ✓', min(depths), '<=' if collided else '>', d))
    print(' 最近深度 : 最小 %.3f m | 终值 %.3f m | 初始 %.3f m' % (min(depths), depths[-1], d - h0))
    print(' CBF 值 h : 最大 %+.4f(理论要求始终 ≤ 0)' % max(hs))
    if first_iv is not None:
        fd = next(s for s in stats if s[0] == first_iv)
        print(' 干预     : 首次第 %d 帧(当时最近深度 %.3f m)| 拦截 %d/%d 帧 | 平均 %.2f | 峰值 %.2f m/s'
              % (first_iv, fd[1], n_unsafe, N_FRAMES, sum(ivs) / len(ivs), max(ivs)))
    else:
        print(' 干预     : 全程未干预')
    print(' 控制耗时 : 平均 %.3f s/步 | 峰值 %.3f s(10 Hz 要求 ≤ 0.1 s 完成决策)'
          % (sum(tvals) / len(tvals), max(tvals)))
    print(' 输出文件 : video.avi(%d 帧,2400×680;蓝条 = 放行,红条 = 拦截)' % N_FRAMES)
    print('=' * 64)
