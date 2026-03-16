"""
OccluVLA - Push-Then-Grasp 环境封装
obs key: obs['sensor_data']['base_camera']['rgb'] → [batch, 128, 128, 3]
"""
import numpy as np
import gymnasium as gym
import mani_skill.envs
import torch


class PushGraspSceneManager:
    def __init__(self, obs_mode="rgbd", num_envs=1, max_episode_steps=200):
        self.env = gym.make(
            "PickClutterYCB-v1",
            obs_mode=obs_mode,
            render_mode="rgb_array",
            num_envs=num_envs,
            max_episode_steps=max_episode_steps,
        )
        self.action_space = self.env.action_space

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def step(self, action):
        return self.env.step(action)

    def get_images(self, obs):
        """提取 RGB 和 Depth — 用正确的 key: sensor_data"""
        images = {}
        sensor_data = obs.get("sensor_data", {})
        for cam_name, cam_data in sensor_data.items():
            # RGB
            if "rgb" in cam_data:
                rgb = cam_data["rgb"]
                if isinstance(rgb, torch.Tensor):
                    rgb = rgb.cpu().numpy()
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
                images[f"{cam_name}_depth"] = depth

        return images

    def get_state(self, obs):
        """提取机器人关节状态"""
        if "agent" in obs:
            agent = obs["agent"]
            parts = []
            for key in ["qpos", "qvel"]:
                if key in agent:
                    val = agent[key]
                    if isinstance(val, torch.Tensor):
                        val = val.cpu().numpy()
                    parts.append(val)
            return np.concatenate(parts, axis=-1) if parts else None
        return None

    def close(self):
        self.env.close()


class ScriptedPushGraspPolicy:
    """脚本化 push-then-grasp 策略"""

    def __init__(self, push_distance=0.12, approach_height=0.08,
                 grasp_height=0.01, lift_height=0.15,
                 move_speed=0.02, noise_std=0.002):
        self.push_distance = push_distance
        self.approach_height = approach_height
        self.grasp_height = grasp_height
        self.lift_height = lift_height
        self.move_speed = move_speed
        self.noise_std = noise_std

    def generate_trajectory(self, ee_pos, target_pos, occluder_pos):
        trajectory = []

        # Phase 1: Push occluder
        push_vec = occluder_pos[:2] - target_pos[:2]
        push_vec = push_vec / (np.linalg.norm(push_vec) + 1e-8)
        push_dir = np.array([-push_vec[1], push_vec[0]])

        approach = occluder_pos.copy(); approach[2] += self.approach_height
        trajectory.extend(self._move_to(ee_pos, approach, True))

        push_start = occluder_pos.copy(); push_start[2] += 0.01
        trajectory.extend(self._move_to(approach, push_start, True))

        push_end = push_start.copy(); push_end[:2] += push_dir * self.push_distance
        trajectory.extend(self._move_to(push_start, push_end, True))

        retreat = push_end.copy(); retreat[2] += self.approach_height + 0.05
        trajectory.extend(self._move_to(push_end, retreat, True))

        # Phase 2: Grasp target
        grasp_approach = target_pos.copy(); grasp_approach[2] += self.approach_height
        trajectory.extend(self._move_to(retreat, grasp_approach, True))

        grasp_pos = target_pos.copy(); grasp_pos[2] += self.grasp_height
        trajectory.extend(self._move_to(grasp_approach, grasp_pos, True))

        for _ in range(5):
            a = np.zeros(8); a[6] = -1.0; a[7] = -1.0
            trajectory.append(a)

        lift = grasp_pos.copy(); lift[2] += self.lift_height
        trajectory.extend(self._move_to(grasp_pos, lift, False))

        return [a + self._noise() for a in trajectory]

    def _move_to(self, start, end, gripper_open):
        dist = np.linalg.norm(end - start)
        n = max(1, int(dist / self.move_speed))
        actions = []
        for i in range(n):
            delta = (end - start) / n
            a = np.zeros(8)
            a[:3] = delta
            a[6] = 1.0 if gripper_open else -1.0; a[7] = a[6]
            actions.append(a)
        return actions

    def _noise(self):
        n = np.zeros(8)
        n[:3] = np.random.randn(3) * self.noise_std  # only add noise to position
        return n
