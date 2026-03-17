"""
OccluVLA - Push-Then-Grasp 环境封装 (基于 OccluBench 自定义环境)

底层环境: OccluBench-Level1-v0 / OccluBench-Level2-v0
  shoulder_camera: 固定在机器人后上方 (通过 ManiSkill3 sensor_configs)
  hand_camera:     mount 到 panda_hand TCP link (通过 _after_loading)

API:
  EE 位置:        env.unwrapped.agent.tcp.pose.p
  Target 物体:    env.unwrapped.target_object
  Occluder 列表:  env.unwrapped.occluder_objects
  All 物体:       env.unwrapped.all_objects
  Action:         7-DoF pd_ee_delta_pose [dx,dy,dz, drx,dry,drz, gripper]
  机器人 base:    x=-0.615, y=0
"""

import numpy as np
import torch
import gymnasium as gym

# 注册 OccluBench 环境 (import 触发 @register_env)
import occlubench_env  # noqa: F401


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.asarray(x)


# 相机名常量 — 与 occlubench_env 中定义的一致
CAM_SHOULDER = "shoulder_camera"
CAM_HAND     = "hand_camera"


class OccluVLAEnv:
    """
    封装 OccluBench 环境，提供:
      - 双相机图像提取 (shoulder + hand, 按相机分开)
      - target / occluder 位置读取
      - 统一的 step / reset 接口
    """

    def __init__(
        self,
        level: int = 1,
        num_envs: int = 1,
        max_episode_steps: int = 200,
        render_size: int = 256,
    ):
        self.render_size = render_size
        self.level = level

        env_id = f"OccluBench-Level{level}-v0"
        if level == 2:
            max_episode_steps = max(max_episode_steps, 300)

        self.env = gym.make(
            env_id,
            obs_mode="rgbd",
            control_mode="pd_ee_delta_pose",
            render_mode="rgb_array",
            num_envs=num_envs,
            max_episode_steps=max_episode_steps,
        )
        self.base_env = self.env.unwrapped
        self.action_dim = 7

        # ── 探测可用相机 ─────────────────────────────────────────
        self._available_cams = []
        test_obs, _ = self.env.reset(seed=0)
        sd = test_obs.get('sensor_data', {})
        for cam_name in [CAM_SHOULDER, CAM_HAND]:
            if cam_name in sd and isinstance(sd[cam_name], dict):
                self._available_cams.append(cam_name)
        # 如果自定义相机名不在 sensor_data 中，回退检测所有
        if not self._available_cams:
            for k in sd:
                if isinstance(sd[k], dict) and 'rgb' in sd[k]:
                    self._available_cams.append(k)

        print(f"  Env: {env_id} | action_dim={self.action_dim} "
              f"| render={render_size}")
        print(f"  Available cameras: {self._available_cams}")

    @property
    def camera_names(self) -> list:
        return list(self._available_cams)

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        return self.env.step(action)

    # ── 位置读取 ─────────────────────────────────────────────────
    def get_ee_pos(self):
        p = to_np(self.base_env.agent.tcp.pose.p)
        return p[0] if p.ndim > 1 else p

    def get_target_object(self):
        """返回 {'name': str, 'pos': np.array(3), 'ycb_id': str}"""
        tgt = self.base_env.target_object
        p = to_np(tgt.pose.p)
        if p.ndim > 1: p = p[0]
        return {
            'name': tgt.name,
            'pos': p,
            'ycb_id': self.base_env.target_ycb_id,
        }

    def get_occluder_objects(self):
        """返回 [{'name': str, 'pos': np.array(3), 'ycb_id': str}, ...]"""
        occluders = self.base_env.occluder_objects
        ycb_ids = self.base_env.occluder_ycb_ids
        result = []
        for occ, yid in zip(occluders, ycb_ids):
            p = to_np(occ.pose.p)
            if p.ndim > 1: p = p[0]
            result.append({'name': occ.name, 'pos': p, 'ycb_id': yid})
        return result

    def get_all_object_poses(self):
        """所有动态物体的名称和位置"""
        objects = []
        for obj in self.base_env.all_objects:
            p = to_np(obj.pose.p)
            if p.ndim > 1: p = p[0]
            objects.append({'name': obj.name, 'pos': p})
        return objects

    def find_nearest_occluder(self, target_pos):
        """
        找最近的 occluder (挡在 target 前面的)。
        OccluBench 的 occluder 保证在 target 前方，直接取最近的。
        """
        occluders = self.get_occluder_objects()
        if not occluders:
            return None
        # 按到 target 的距离排序 (最靠近 target 的 = 最需要先推开的)
        occluders.sort(key=lambda o: np.linalg.norm(o['pos'][:2] - target_pos[:2]))
        return occluders[0]

    # ── 图像提取 (按相机分开) ────────────────────────────────────
    def get_images(self, obs) -> dict:
        """
        返回: {
            'shoulder_camera': {'rgb': (H,W,3) uint8, 'depth': (H,W,1)},
            'hand_camera':     {'rgb': (H,W,3) uint8, 'depth': (H,W,1)},
        }
        """
        result = {}
        sensor_data = obs.get('sensor_data', {})

        # 1) 从 ManiSkill sensor_data 读取 (shoulder_camera 等)
        for cam_name in self._available_cams:
            if cam_name in sensor_data:
                data = sensor_data[cam_name]
                if not isinstance(data, dict):
                    continue
                cam_result = {}
                if 'rgb' in data:
                    rgb = to_np(data['rgb'])
                    if rgb.ndim == 4: rgb = rgb[0]
                    if rgb.max() <= 1.0 and rgb.dtype in [np.float32, np.float64]:
                        rgb = (rgb * 255).astype(np.uint8)
                    else:
                        rgb = rgb.astype(np.uint8)
                    if rgb.shape[-1] == 4:
                        rgb = rgb[..., :3]
                    cam_result['rgb'] = rgb
                if 'depth' in data:
                    depth = to_np(data['depth'])
                    if depth.ndim == 4: depth = depth[0]
                    cam_result['depth'] = depth
                if cam_result:
                    result[cam_name] = cam_result

        return result

    def get_render_frame(self):
        """第三人称 render view"""
        f = self.env.render()
        if isinstance(f, torch.Tensor): f = f.cpu().numpy()
        if f is not None and f.ndim == 4: f = f[0]
        return f

    def get_qpos(self, obs):
        q = obs.get('agent', {}).get('qpos', None)
        if q is not None:
            q = to_np(q)
            if q.ndim > 1: q = q[0]
        return q

    def get_success(self, info):
        s = info.get('success', False)
        return bool(s.item() if isinstance(s, torch.Tensor) else s)

    def close(self):
        self.env.close()

    def print_scene(self):
        ee = self.get_ee_pos()
        tgt = self.get_target_object()
        occs = self.get_occluder_objects()
        objs = self.get_all_object_poses()

        print(f"  EE:      [{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}]")
        print(f"  Cameras: {self._available_cams}")
        print(f"  Target:  {tgt['name']} (ycb={tgt['ycb_id']}) "
              f"[{tgt['pos'][0]:.3f}, {tgt['pos'][1]:.3f}, {tgt['pos'][2]:.3f}]")
        for oc in occs:
            print(f"  Occluder:{oc['name']} (ycb={oc['ycb_id']}) "
                  f"[{oc['pos'][0]:.3f}, {oc['pos'][1]:.3f}, {oc['pos'][2]:.3f}]")
        for o in objs:
            tag = " ← TARGET" if o['name'] == tgt['name'] else ""
            print(f"    {o['name']:30s} "
                  f"[{o['pos'][0]:.3f}, {o['pos'][1]:.3f}, {o['pos'][2]:.3f}]{tag}")


class ScriptedPushGraspPolicy:
    """Closed-loop push-then-grasp scripted policy"""

    APPROACH_H   = 0.10
    PUSH_H       = 0.01
    PUSH_DIST    = 0.10
    GRASP_H      = 0.01
    LIFT_H       = 0.15
    RETREAT_H    = 0.12
    SPEED        = 0.02
    NOISE        = 0.001
    REACH_THR    = 0.005
    GRIP_OPEN    = 1.0
    GRIP_CLOSE   = -1.0
    GRIP_WAIT    = 8

    def push_direction(self, target_pos, occluder_pos):
        v = occluder_pos[:2] - target_pos[:2]
        v = v / (np.linalg.norm(v) + 1e-8)
        da, db = np.array([-v[1], v[0]]), np.array([v[1], -v[0]])
        pa = occluder_pos[:2] + da * self.PUSH_DIST
        pb = occluder_pos[:2] + db * self.PUSH_DIST
        return da if np.linalg.norm(pa) < np.linalg.norm(pb) else db

    def make_action(self, ee, goal, gripper):
        delta = goal - ee
        dist = np.linalg.norm(delta)
        if dist < self.REACH_THR:
            a = np.zeros(7); a[6] = gripper
            return a, True
        if dist > self.SPEED:
            delta = delta / dist * self.SPEED
        delta += np.random.randn(3) * self.NOISE
        a = np.zeros(7); a[:3] = delta; a[6] = gripper
        return a, False

    def plan(self, ee, target_pos, occluder_pos=None):
        wps = []
        if occluder_pos is not None:
            pd = self.push_direction(target_pos, occluder_pos)
            w = occluder_pos.copy(); w[2] += self.APPROACH_H
            wps.append(('push', w, self.GRIP_OPEN, 'approach occ'))
            w = occluder_pos.copy(); w[2] += self.PUSH_H
            wps.append(('push', w, self.GRIP_OPEN, 'descend'))
            w2 = w.copy(); w2[:2] += pd * self.PUSH_DIST
            wps.append(('push', w2, self.GRIP_OPEN, 'push'))
            w3 = w2.copy(); w3[2] += self.RETREAT_H
            wps.append(('push', w3, self.GRIP_OPEN, 'retreat'))

        w = target_pos.copy(); w[2] += self.APPROACH_H
        wps.append(('grasp', w, self.GRIP_OPEN, 'approach tgt'))
        w = target_pos.copy(); w[2] += self.GRASP_H
        wps.append(('grasp', w, self.GRIP_OPEN, 'descend'))
        wps.append(('grasp', w.copy(), self.GRIP_CLOSE, 'close'))
        w = target_pos.copy(); w[2] += self.LIFT_H
        wps.append(('lift', w, self.GRIP_CLOSE, 'lift'))
        return wps