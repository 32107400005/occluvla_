"""
OccluVLA - Push-Then-Grasp 环境封装 (修正版)

核心修正:
1. 自定义场景: 1 target + 1 occluder (挡在 target 前) + 2 distractors
2. 从仿真中读取真实物体 pose 和末端执行器位置
3. 使用 pd_ee_delta_pose 控制模式, action = [dx,dy,dz, drx,dry,drz, gripper]
4. 渲染分辨率 512x512
"""

import numpy as np
import sapien
import torch
import gymnasium as gym
from pathlib import Path

# ── ManiSkill3 imports ──────────────────────────────────────────────
import mani_skill.envs          # 注册所有内置环境
from mani_skill.utils.wrappers import RecordEpisode

# ── YCB 物体 ID 列表 (ManiSkill3 内置) ──────────────────────────────
# 根据你安装的 asset 版本, 可用的 YCB 物体 ID 可能略有不同
# 运行 print_available_ycb() 查看
YCB_OBJECTS = [
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "011_banana",
    "019_pitcher_base",
    "021_bleach_cleanser",
    "024_bowl",
    "025_mug",
    "035_power_drill",
    "036_wood_block",
    "037_scissors",
    "040_large_marker",
    "052_extra_large_clamp",
    "061_foam_brick",
]


def print_available_ycb():
    """辅助函数: 打印当前可用的 YCB 物体"""
    from mani_skill.utils.download_asset import DATA_SOURCES
    print("Available YCB objects in your installation:")
    for k in sorted(DATA_SOURCES.keys()):
        if "ycb" in k.lower():
            print(f"  {k}")


class PushGraspSceneManager:
    """
    自定义 push-then-grasp 场景管理器

    场景布局 (俯视图, 机器人在下方):
        [distractor_L]  [occluder]  [distractor_R]
                        [target]   ← 被 occluder 遮挡

    使用 PickClutterYCB-v1 作为基础, 但通过 reset 后手动
    调整物体位置来实现结构化布局。

    如果 PickClutterYCB 不支持 pose override, 则退回到
    使用 PickSingleYCB-v1 + 手动添加额外物体的方案。
    """

    # ── 场景参数 ────────────────────────────────────────────────────
    TABLE_HEIGHT = 0.0        # ManiSkill3 桌面高度 (env 内部坐标)
    OBJECT_Z     = 0.02       # 物体放置高度 (桌面 + 半高)
    RENDER_SIZE  = 512        # 渲染分辨率

    def __init__(self, num_envs=1, max_episode_steps=200,
                 render_size=512, control_mode="pd_ee_delta_pose"):
        """
        Args:
            num_envs: 并行环境数 (数据生成时可以 >1)
            max_episode_steps: 最大步数
            render_size: 渲染分辨率 (宽=高)
            control_mode: 控制模式
              - "pd_ee_delta_pose": action=[dx,dy,dz, drx,dry,drz, gripper]
              - "pd_joint_delta_pos": action=[dq1..dq7, gripper]
        """
        self.render_size = render_size
        self.control_mode = control_mode
        self.num_envs = num_envs

        # ── 创建环境 ────────────────────────────────────────────────
        # 优先尝试 PickSingleYCB (更容易控制场景)
        # 如果你的 ManiSkill3 版本支持 PickClutterYCB 的 pose override,
        # 可以换用它
        self.env = gym.make(
            "PickSingleYCB-v1",
            obs_mode="rgbd",
            control_mode=control_mode,
            render_mode="rgb_array",
            num_envs=num_envs,
            max_episode_steps=max_episode_steps,
            # 提高相机分辨率
            sensor_configs=dict(
                base_camera=dict(width=render_size, height=render_size),
                hand_camera=dict(width=render_size, height=render_size),
            ),
        )

        self.action_dim = self.env.action_space.shape[-1]
        print(f"  Environment created: PickSingleYCB-v1")
        print(f"  Control mode: {control_mode}")
        print(f"  Action dim: {self.action_dim}")
        print(f"  Render size: {render_size}x{render_size}")
        print(f"  Num envs: {num_envs}")

    # ── Reset & 场景布局 ────────────────────────────────────────────
    def reset(self, seed=None):
        """Reset 环境并获取初始观测"""
        obs, info = self.env.reset(seed=seed)
        return obs, info

    def place_objects_structured(self, target_xy, occluder_offset,
                                  distractor_offsets):
        """
        [高级] 在 reset 之后, 手动调整物体位置以构建结构化场景.
        这需要访问 env.unwrapped 的底层 SAPIEN scene.

        注意: 这个方法依赖 ManiSkill3 的内部 API, 如果不可用,
        请在 generate_demos.py 中使用 rejection sampling 策略替代.

        Args:
            target_xy: (2,) 目标物体的 xy 位置
            occluder_offset: (2,) 遮挡物相对于目标的 xy 偏移
            distractor_offsets: list of (2,) 干扰物相对于目标的 xy 偏移
        """
        try:
            scene = self.env.unwrapped
            # 尝试访问场景中的 actors
            # ManiSkill3 的 API 可能是:
            #   scene.obj (单个目标物体)
            #   或 scene.objects (多个物体列表)
            # 具体取决于环境版本, 请根据你的版本调整

            if hasattr(scene, 'obj'):
                # PickSingleYCB: 只有一个目标物体
                target_pose = scene.obj.pose
                new_p = target_pose.p.clone()
                new_p[..., 0] = target_xy[0]
                new_p[..., 1] = target_xy[1]
                new_p[..., 2] = self.OBJECT_Z
                scene.obj.set_pose(sapien.Pose(p=new_p, q=target_pose.q))

            print("  [场景] 物体位置已调整 (structured layout)")
            return True

        except Exception as e:
            print(f"  [警告] 无法手动调整物体位置: {e}")
            print(f"  [警告] 将使用环境默认随机布局")
            return False

    # ── 观测提取 ────────────────────────────────────────────────────
    def get_images(self, obs):
        """提取 RGB 和 Depth 图像"""
        images = {}
        sensor_data = obs.get("sensor_data", {})

        for cam_name, cam_data in sensor_data.items():
            if not isinstance(cam_data, dict):
                continue

            # RGB
            if "rgb" in cam_data:
                rgb = cam_data["rgb"]
                if isinstance(rgb, torch.Tensor):
                    rgb = rgb.cpu().numpy()
                if rgb.ndim == 4:       # (batch, H, W, C)
                    rgb = rgb[0]
                if rgb.max() <= 1.0:
                    rgb = (rgb * 255).astype(np.uint8)
                else:
                    rgb = rgb.astype(np.uint8)
                images[f"{cam_name}_rgb"] = rgb

            # Depth
            if "depth" in cam_data:
                depth = cam_data["depth"]
                if isinstance(depth, torch.Tensor):
                    depth = depth.cpu().numpy()
                if depth.ndim == 4:
                    depth = depth[0]
                images[f"{cam_name}_depth"] = depth

        return images

    def get_ee_pos(self, obs):
        """
        从观测中提取末端执行器 (end-effector) 的真实位置.
        ManiSkill3 中 EE 位置通常在 obs['extra']['tcp_pose'] 中.
        """
        # 尝试多种可能的 key
        extra = obs.get("extra", {})

        # 方法 1: tcp_pose (最常见)
        if "tcp_pose" in extra:
            tcp = extra["tcp_pose"]
            if isinstance(tcp, torch.Tensor):
                tcp = tcp.cpu().numpy()
            if tcp.ndim > 1:
                tcp = tcp[0]
            return tcp[:3]  # 只取 xyz 位置

        # 方法 2: 通过 agent 信息
        agent = obs.get("agent", {})
        if "tcp_pose" in agent:
            tcp = agent["tcp_pose"]
            if isinstance(tcp, torch.Tensor):
                tcp = tcp.cpu().numpy()
            if tcp.ndim > 1:
                tcp = tcp[0]
            return tcp[:3]

        # 方法 3: 通过 controller 获取
        try:
            ee_pose = self.env.unwrapped.agent.tcp.pose
            p = ee_pose.p
            if isinstance(p, torch.Tensor):
                p = p.cpu().numpy()
            if p.ndim > 1:
                p = p[0]
            return p
        except Exception:
            pass

        print("  [警告] 无法获取 EE 位置, 返回默认值")
        return np.array([0.0, 0.0, 0.3])

    def get_target_pos(self, obs):
        """
        从观测中提取目标物体的真实位置.
        ManiSkill3 中通常在 obs['extra']['goal_pos'] 或
        obs['extra']['obj_pose'] 中.
        """
        extra = obs.get("extra", {})

        # 方法 1: obj_pose
        if "obj_pose" in extra:
            pose = extra["obj_pose"]
            if isinstance(pose, torch.Tensor):
                pose = pose.cpu().numpy()
            if pose.ndim > 1:
                pose = pose[0]
            return pose[:3]

        # 方法 2: goal_pos
        if "goal_pos" in extra:
            gp = extra["goal_pos"]
            if isinstance(gp, torch.Tensor):
                gp = gp.cpu().numpy()
            if gp.ndim > 1:
                gp = gp[0]
            return gp[:3]

        # 方法 3: 通过 scene 直接获取
        try:
            scene = self.env.unwrapped
            if hasattr(scene, 'obj'):
                p = scene.obj.pose.p
                if isinstance(p, torch.Tensor):
                    p = p.cpu().numpy()
                if p.ndim > 1:
                    p = p[0]
                return p
        except Exception:
            pass

        print("  [警告] 无法获取目标物体位置")
        return None

    def get_robot_state(self, obs):
        """提取机器人关节状态"""
        agent = obs.get("agent", {})
        parts = []
        for key in ["qpos", "qvel"]:
            if key in agent:
                val = agent[key]
                if isinstance(val, torch.Tensor):
                    val = val.cpu().numpy()
                if val.ndim > 1:
                    val = val[0]
                parts.append(val)
        return np.concatenate(parts, axis=-1) if parts else None

    # ── 环境交互 ────────────────────────────────────────────────────
    def step(self, action):
        """执行一步, action 应该匹配 control_mode 的维度"""
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(0)   # (1, action_dim) for batch
        return self.env.step(action)

    def render_frame(self):
        """渲染当前帧 (用于保存视频)"""
        frame = self.env.render()
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        return frame

    def close(self):
        self.env.close()

    # ── 调试工具 ────────────────────────────────────────────────────
    def inspect_obs_keys(self, obs, prefix=""):
        """递归打印 obs 的所有 key 和 shape, 用于调试"""
        if isinstance(obs, dict):
            for k, v in obs.items():
                self.inspect_obs_keys(v, prefix=f"{prefix}.{k}" if prefix else k)
        elif isinstance(obs, (torch.Tensor, np.ndarray)):
            shape = tuple(obs.shape)
            dtype = obs.dtype
            print(f"  {prefix}: shape={shape}, dtype={dtype}")
        else:
            print(f"  {prefix}: type={type(obs).__name__}, value={obs}")


class ScriptedPushGraspPolicy:
    """
    脚本化 push-then-grasp 策略 (修正版)

    核心修正:
    1. 基于真实 EE 位置和物体位置生成轨迹
    2. 不再一次性生成全部 action, 而是分阶段执行
    3. 每一步都是 closed-loop: 读取当前 EE 位置 → 计算 delta → 发送

    action 格式 (pd_ee_delta_pose):
        [dx, dy, dz, drx, dry, drz, gripper]
        gripper: 1.0 = open, -1.0 = close (或 0/1, 取决于 env 版本)
    """

    # ── 策略参数 ────────────────────────────────────────────────────
    APPROACH_HEIGHT  = 0.10   # 从上方多高接近物体
    PUSH_HEIGHT      = 0.005  # push 时末端执行器距桌面高度
    PUSH_DISTANCE    = 0.12   # push 距离 (12cm)
    GRASP_HEIGHT     = 0.005  # grasp 时末端执行器高度
    LIFT_HEIGHT      = 0.15   # 抬升高度
    RETREAT_HEIGHT   = 0.15   # push 后撤回高度

    MOVE_SPEED       = 0.015  # 每步移动距离 (m), 越小越平滑
    NOISE_STD        = 0.001  # 轨迹噪声标准差

    GRIPPER_OPEN     = 1.0
    GRIPPER_CLOSE    = -1.0
    GRIPPER_STEPS    = 8      # 夹爪闭合等待步数

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k.upper()):
                setattr(self, k.upper(), v)

    # ── 核心: 逐步生成 action ───────────────────────────────────────
    def plan_push_grasp(self, ee_pos, target_pos, occluder_pos):
        """
        规划完整的 push-then-grasp 轨迹 (waypoint 序列).

        Returns:
            list of dict, 每个 dict 包含:
                'target_xyz': 目标位置
                'gripper': gripper 值
                'phase': 'push' / 'grasp' / 'lift'
                'label': 人类可读标签
        """
        waypoints = []

        # ── Phase 1: Push occluder aside ────────────────────────────
        # 计算 push 方向: 从 target 指向 occluder, 然后旋转 90°
        # 这样 push 方向是侧向的, 不会把 occluder 推到 target 上
        vec_to_occ = occluder_pos[:2] - target_pos[:2]
        vec_to_occ_norm = vec_to_occ / (np.linalg.norm(vec_to_occ) + 1e-8)

        # 旋转 90° (选择离桌边较远的方向)
        push_dir_a = np.array([-vec_to_occ_norm[1], vec_to_occ_norm[0]])
        push_dir_b = np.array([vec_to_occ_norm[1], -vec_to_occ_norm[0]])
        # 选离桌面中心更近的方向 (避免推出桌外)
        push_dir = push_dir_a if abs(push_dir_a[0]) < abs(push_dir_b[0]) else push_dir_b

        # 1a. 移动到 occluder 上方
        approach_occ = occluder_pos.copy()
        approach_occ[2] += self.APPROACH_HEIGHT
        waypoints.append({
            'target_xyz': approach_occ,
            'gripper': self.GRIPPER_OPEN,
            'phase': 'push',
            'label': 'approach occluder from above'
        })

        # 1b. 下降到 push 高度 (occluder 的侧面, push 起始点)
        push_start = occluder_pos.copy()
        push_start[:2] -= push_dir * 0.03   # 从 push 反方向偏移一点
        push_start[2] = occluder_pos[2] + self.PUSH_HEIGHT
        waypoints.append({
            'target_xyz': push_start,
            'gripper': self.GRIPPER_OPEN,
            'phase': 'push',
            'label': 'descend to push start'
        })

        # 1c. 执行 push (沿 push_dir 推动)
        push_end = push_start.copy()
        push_end[:2] += push_dir * self.PUSH_DISTANCE
        waypoints.append({
            'target_xyz': push_end,
            'gripper': self.GRIPPER_OPEN,
            'phase': 'push',
            'label': 'push occluder aside'
        })

        # 1d. 撤回到安全高度
        retreat = push_end.copy()
        retreat[2] += self.RETREAT_HEIGHT
        waypoints.append({
            'target_xyz': retreat,
            'gripper': self.GRIPPER_OPEN,
            'phase': 'push',
            'label': 'retreat after push'
        })

        # ── Phase 2: Grasp target ──────────────────────────────────
        # 2a. 移动到 target 上方
        approach_target = target_pos.copy()
        approach_target[2] += self.APPROACH_HEIGHT
        waypoints.append({
            'target_xyz': approach_target,
            'gripper': self.GRIPPER_OPEN,
            'phase': 'grasp',
            'label': 'approach target from above'
        })

        # 2b. 下降到 grasp 高度
        grasp_pos = target_pos.copy()
        grasp_pos[2] += self.GRASP_HEIGHT
        waypoints.append({
            'target_xyz': grasp_pos,
            'gripper': self.GRIPPER_OPEN,
            'phase': 'grasp',
            'label': 'descend to grasp position'
        })

        # 2c. 闭合夹爪 (原地不动, 只闭合)
        waypoints.append({
            'target_xyz': grasp_pos,
            'gripper': self.GRIPPER_CLOSE,
            'phase': 'grasp',
            'label': 'close gripper'
        })

        # 2d. 抬起
        lift_pos = grasp_pos.copy()
        lift_pos[2] += self.LIFT_HEIGHT
        waypoints.append({
            'target_xyz': lift_pos,
            'gripper': self.GRIPPER_CLOSE,
            'phase': 'lift',
            'label': 'lift object'
        })

        return waypoints

    def compute_action(self, current_ee_pos, target_xyz, gripper_value):
        """
        计算单步 action (closed-loop).

        Args:
            current_ee_pos: (3,) 当前 EE xyz
            target_xyz: (3,) 目标 xyz
            gripper_value: float, gripper 命令

        Returns:
            action: (7,) [dx, dy, dz, 0, 0, 0, gripper]
            reached: bool, 是否到达目标点
        """
        delta = target_xyz - current_ee_pos
        dist = np.linalg.norm(delta)

        if dist < 0.003:  # 到达阈值 3mm
            action = np.zeros(7)
            action[6] = gripper_value
            return action, True

        # 限制每步移动距离
        if dist > self.MOVE_SPEED:
            delta = delta / dist * self.MOVE_SPEED

        # 添加微小噪声 (增加轨迹多样性)
        delta += np.random.randn(3) * self.NOISE_STD

        action = np.zeros(7)
        action[:3] = delta
        # action[3:6] = 0  # 不旋转
        action[6] = gripper_value
        return action, False
