"""
OccluVLA - Push-Then-Grasp 环境封装 (基于诊断结果确认的 API)

确认的 API (PickClutterYCB-v1):
  EE 位置:       env.unwrapped.agent.tcp.pose.p          → (1,3)
  所有物体位置:   env.unwrapped.all_objects.pose.p        → (N,3)
  Target 物体:   env.unwrapped.target_object              → Actor
  Goal 位置:     env.unwrapped.goal_pos                   → tensor
  Segmentation:  env.unwrapped.segmentation_id_map        → {id: Actor/Link}
  Action:        7-DoF pd_ee_delta_pose [dx,dy,dz, drx,dry,drz, gripper]
  机器人 base:   x=-0.615, y=0
  默认 base_cam: pos=[0.3, 0, 0.6] → 对面(作弊!), 仅用于 debug
"""

import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.structs import Actor


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.asarray(x)


class OccluVLAEnv:

    def __init__(self, num_envs=1, max_episode_steps=200, render_size=512):
        self.render_size = render_size
        self.env = gym.make(
            "PickClutterYCB-v1",
            obs_mode="rgbd",
            control_mode="pd_ee_delta_pose",
            render_mode="rgb_array",
            num_envs=num_envs,
            max_episode_steps=max_episode_steps,
            sensor_configs=dict(
                base_camera=dict(width=render_size, height=render_size),
            ),
        )
        self.base_env = self.env.unwrapped
        self.action_dim = 7
        print(f"  Env: PickClutterYCB-v1 | action_dim={self.action_dim} | render={render_size}")

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        return self.env.step(action)

    # ── 真实位置 (从诊断确认的 API) ──────────────────────────────
    def get_ee_pos(self):
        p = to_np(self.base_env.agent.tcp.pose.p)
        return p[0] if p.ndim > 1 else p

    def get_all_object_poses(self):
        objects = []
        for seg_id, obj in self.base_env.segmentation_id_map.items():
            if not isinstance(obj, Actor):
                continue
            if any(s in obj.name for s in ['table', 'ground', 'goal']):
                continue
            p = to_np(obj.pose.p)
            if p.ndim > 1: p = p[0]
            objects.append({'seg_id': seg_id, 'name': obj.name, 'pos': p})
        return objects

    def get_target_object(self):
        try:
            tgt = self.base_env.target_object
            p = to_np(tgt.pose.p)
            if p.ndim > 1: p = p[0]
            return {'name': tgt.name, 'pos': p}
        except:
            return self._find_target_by_goal()

    def _find_target_by_goal(self):
        try:
            gp = to_np(self.base_env.goal_site.pose.p)
            if gp.ndim > 1: gp = gp[0]
        except:
            return None
        objects = self.get_all_object_poses()
        if not objects: return None
        best = min(objects, key=lambda o: np.linalg.norm(o['pos'][:2] - gp[:2]))
        return best

    def find_occluder(self, target_pos):
        """找挡在 target 前面(y 更小=更靠近机器人)的物体"""
        tgt = self.get_target_object()
        target_name = tgt['name'] if tgt else None
        candidates = []
        for obj in self.get_all_object_poses():
            if target_name and obj['name'] == target_name:
                continue
            dy = target_pos[1] - obj['pos'][1]   # >0 means obj is closer to robot
            dx = abs(target_pos[0] - obj['pos'][0])
            if dy > 0.02 and dx < 0.12:
                candidates.append({**obj, 'score': dy / (dx + 0.01)})
        if not candidates: return None
        return max(candidates, key=lambda c: c['score'])

    # ── 图像提取 ─────────────────────────────────────────────────
    def get_images(self, obs):
        images = {}
        for cam, data in obs.get('sensor_data', {}).items():
            if not isinstance(data, dict): continue
            if 'rgb' in data:
                rgb = to_np(data['rgb'])
                if rgb.ndim == 4: rgb = rgb[0]
                if rgb.max() <= 1.0 and rgb.dtype in [np.float32, np.float64]:
                    rgb = (rgb * 255).astype(np.uint8)
                else:
                    rgb = rgb.astype(np.uint8)
                images[f'{cam}_rgb'] = rgb
            if 'depth' in data:
                d = to_np(data['depth'])
                if d.ndim == 4: d = d[0]
                images[f'{cam}_depth'] = d
        return images

    def get_render_frame(self):
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
        objs = self.get_all_object_poses()
        print(f"  EE:      [{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}]")
        if tgt:
            print(f"  Target:  {tgt['name']} [{tgt['pos'][0]:.3f}, {tgt['pos'][1]:.3f}, {tgt['pos'][2]:.3f}]")
            occ = self.find_occluder(tgt['pos'])
            if occ:
                print(f"  Occluder:{occ['name']} [{occ['pos'][0]:.3f}, {occ['pos'][1]:.3f}, {occ['pos'][2]:.3f}]")
            else:
                print(f"  Occluder: None")
        for o in objs:
            tag = " ← TARGET" if (tgt and o['name'] == tgt['name']) else ""
            print(f"    {o['name']:30s} [{o['pos'][0]:.3f}, {o['pos'][1]:.3f}, {o['pos'][2]:.3f}]{tag}")


class ScriptedPushGraspPolicy:
    """Closed-loop push-then-grasp policy"""

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
