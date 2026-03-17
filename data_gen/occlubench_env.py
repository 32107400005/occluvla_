"""
OccluBench: Occlusion-Aware Push-Then-Grasp Benchmark for ManiSkill3.

ManiSkill3 version: 3.0.0b22 (API 已通过诊断确认)

确认的 API:
  SimConfig:       from mani_skill.utils.structs.types import SimConfig
  YCB builder:     from mani_skill.utils.building.actors.ycb import get_ycb_builder
                   builder = get_ycb_builder(scene=self.scene, id="003_cracker_box")
                   actor = builder.build(name="my_object")
  Robot:           robot_uids="panda" 通过 __init__ → super().__init__(robot_uids=...)
  CameraConfig:    支持 mount: Union[Actor, Link] 参数
  SUPPORTED_ROBOTS: ["panda"]

Levels:
  Level 1: 1 target + 1 occluder + 2 distractors
  Level 2: 1 target + 2-3 occluders + 2 distractors

Cameras:
  shoulder_camera — fixed, robot rear-top overview
  hand_camera     — mounted on panda_hand link (via CameraConfig mount)
"""

from __future__ import annotations
from typing import Any

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.actors.ycb import get_ycb_builder
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.types import SimConfig


# ---------------------------------------------------------------------------
#  YCB object pools
# ---------------------------------------------------------------------------
TARGET_YCB_IDS: list[str] = [
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "025_mug",
]

OCCLUDER_YCB_IDS: list[str] = [
    "003_cracker_box",
    "004_sugar_box",
    "006_mustard_bottle",
    "021_bleach_cleanser",
    "035_power_drill",
    "036_wood_block",
]

DISTRACTOR_YCB_IDS: list[str] = [
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "011_banana",
    "012_strawberry",
    "013_apple",
    "014_lemon",
    "017_orange",
    "025_mug",
]


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def _sample_unique(pool: list[str], n: int, rng: np.random.Generator) -> list[str]:
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return [pool[i] for i in idx]


def _xy_no_collision(
    existing: list[np.ndarray],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    min_dist: float,
    rng: np.random.Generator,
    max_attempts: int = 200,
) -> np.ndarray:
    xy = None
    for _ in range(max_attempts):
        xy = np.array([rng.uniform(*x_range), rng.uniform(*y_range)])
        if all(np.linalg.norm(xy - e) >= min_dist for e in existing):
            return xy
    return xy


def _build_ycb_actor(scene, ycb_id: str, name: str):
    """用 get_ycb_builder 构建 YCB 物体并返回 Actor。"""
    builder = get_ycb_builder(scene=scene, id=ycb_id)
    return builder.build(name=name)


def _p(pose_p):
    """
    Extract position as flat numpy (3,) from ManiSkill3 batch pose.
    pose.p can be torch.Tensor shape (1,3) or (3,) — always return np (3,).
    """
    p = pose_p
    if isinstance(p, torch.Tensor):
        p = p.cpu().numpy()
    p = np.asarray(p, dtype=np.float64)
    if p.ndim > 1:
        p = p[0]
    return p


# ---------------------------------------------------------------------------
#  Level 1 Environment
# ---------------------------------------------------------------------------
@register_env("OccluBench-Level1-v0", max_episode_steps=200)
class OccluBenchLevel1Env(BaseEnv):
    """
    Level 1 — single occluder.

    Scene layout (world frame, robot base at x=-0.615):
      - target:     farther from robot (larger x)
      - occluder:   between robot and target
      - distractors: sides, not blocking
    """

    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    TABLE_HEIGHT: float = 0.0
    X_RANGE  = (0.0, 0.15)
    Y_RANGE  = (-0.20, 0.20)
    MIN_OBJ_DIST: float = 0.08
    GRASP_THRESH: float = 0.02
    LIFT_THRESH:  float = 0.05

    _target_ycb_id:   str = ""
    _occluder_ycb_id: str = ""

    # ------------------------------------------------------------------
    #  __init__: 传入 robot_uids="panda"
    #  (参照 PickClutterYCBEnv 的模式)
    # ------------------------------------------------------------------
    def __init__(self, *args, robot_uids="panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ------------------------------------------------------------------
    #  SimConfig — 必须返回 SimConfig 对象
    # ------------------------------------------------------------------
    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=500, control_freq=20)

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.6, 0.6, 0.8], target=[0.0, 0.0, 0.1])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512,
            fov=1.0, near=0.01, far=100,
        )

    # ------------------------------------------------------------------
    #  Sensor cameras: shoulder (fixed) + hand (mount to TCP link)
    #
    #  CameraConfig 的 mount 参数接受 Link 对象。
    #  _default_sensor_configs 在 agent 加载之后被调用，
    #  所以此时 self.agent 已经可用。
    # ------------------------------------------------------------------
    @property
    def _default_sensor_configs(self):
        # --- shoulder camera: 机器人后上方固定 ---
        shoulder_pose = sapien_utils.look_at(
            eye=[-0.4, 0.0, 0.7],
            target=[0.1, 0.0, 0.0],
        )
        shoulder_cam = CameraConfig(
            "shoulder_camera",
            pose=shoulder_pose,
            width=320, height=240,
            fov=np.deg2rad(68),
            near=0.01, far=10,
        )

        configs = [shoulder_cam]

        # --- hand camera: mount 到 panda_hand link ---
        # agent 可能还没加载 (首次构造时)，需要安全检查
        hand_link = self._get_hand_link()
        if hand_link is not None:
            hand_pose = sapien.Pose(
                p=[0.0, 0.0, 0.04],
                q=euler2quat(0, np.deg2rad(180), 0),
            )
            hand_cam = CameraConfig(
                "hand_camera",
                pose=hand_pose,
                width=320, height=240,
                fov=np.deg2rad(75),
                near=0.01, far=10,
                mount=hand_link,
            )
            configs.append(hand_cam)

        return configs

    def _get_hand_link(self):
        """安全获取 panda_hand link，agent 未加载时返回 None。"""
        try:
            for link in self.agent.robot.get_links():
                if link.name == "panda_hand":
                    return link
            # fallback: 最后一个 link
            return self.agent.robot.get_links()[-1]
        except Exception:
            return None

    # ------------------------------------------------------------------
    #  Scene: table + YCB objects
    #
    #  用 get_ycb_builder(scene, id) → builder.build(name) 加载 YCB
    # ------------------------------------------------------------------
    def _load_scene(self, options: dict):
        self._rng = np.random.default_rng(self._episode_seed)

        # TableSceneBuilder creates table + ground + lighting
        # (same as PickClutterYCB)
        self.scene_builder = TableSceneBuilder(self, robot_init_qpos_noise=0.02)
        self.scene_builder.build()

        # Sample YCB IDs
        self._target_ycb_id = _sample_unique(TARGET_YCB_IDS, 1, self._rng)[0]
        self._occluder_ycb_id = _sample_unique(
            [o for o in OCCLUDER_YCB_IDS if o != self._target_ycb_id],
            1, self._rng,
        )[0]
        distractor_ids = _sample_unique(
            [d for d in DISTRACTOR_YCB_IDS
             if d not in {self._target_ycb_id, self._occluder_ycb_id}],
            2, self._rng,
        )

        # Build actors via get_ycb_builder
        self._target = _build_ycb_actor(self.scene, self._target_ycb_id, "target")
        self._occluder = _build_ycb_actor(self.scene, self._occluder_ycb_id, "occluder")
        self._distractors = []
        for i, did in enumerate(distractor_ids):
            a = _build_ycb_actor(self.scene, did, f"distractor_{i}")
            self._distractors.append(a)

        self._all_objects = [self._target, self._occluder] + self._distractors

    # ------------------------------------------------------------------
    #  Public accessors (for data gen scripts)
    # ------------------------------------------------------------------
    @property
    def target_object(self):
        return self._target

    @property
    def occluder_objects(self) -> list:
        return [self._occluder]

    @property
    def all_objects(self):
        return self._all_objects

    @property
    def target_ycb_id(self) -> str:
        return self._target_ycb_id

    @property
    def occluder_ycb_ids(self) -> list[str]:
        return [self._occluder_ycb_id]

    # ------------------------------------------------------------------
    #  Episode init: structured randomisation
    # ------------------------------------------------------------------
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        rng = np.random.default_rng(self._episode_seed)
        placed: list[np.ndarray] = []

        # Target — far from robot
        target_xy = _xy_no_collision(
            placed, x_range=(0.05, 0.15), y_range=(-0.12, 0.12),
            min_dist=self.MIN_OBJ_DIST, rng=rng,
        )
        placed.append(target_xy)
        target_z = 0.02
        self._target.set_pose(sapien.Pose(
            p=[target_xy[0], target_xy[1], target_z],
            q=euler2quat(0, 0, rng.uniform(0, 2 * np.pi)),
        ))

        # Occluder — between robot and target
        occ_x = target_xy[0] - rng.uniform(0.06, 0.10)
        occ_y = target_xy[1] + rng.uniform(-0.03, 0.03)
        occ_xy = np.array([occ_x, occ_y])
        placed.append(occ_xy)
        self._occluder.set_pose(sapien.Pose(
            p=[occ_xy[0], occ_xy[1], 0.02],
            q=euler2quat(0, 0, rng.uniform(0, 2 * np.pi)),
        ))

        # Distractors — sides
        for da in self._distractors:
            dxy = _xy_no_collision(
                placed, x_range=self.X_RANGE, y_range=self.Y_RANGE,
                min_dist=self.MIN_OBJ_DIST, rng=rng,
            )
            placed.append(dxy)
            da.set_pose(sapien.Pose(
                p=[dxy[0], dxy[1], 0.02],
                q=euler2quat(0, 0, rng.uniform(0, 2 * np.pi)),
            ))

        self._target_init_z = target_z

    # ------------------------------------------------------------------
    #  Evaluation
    # ------------------------------------------------------------------
    def evaluate(self) -> dict:
        tcp_pos = _p(self.agent.tcp.pose.p)
        target_pos = _p(self._target.pose.p)
        dist = float(np.linalg.norm(tcp_pos[:2] - target_pos[:2]))
        lifted = (target_pos[2] - self._target_init_z) > self.LIFT_THRESH
        success = dist < self.GRASP_THRESH and lifted
        return {
            "success": torch.tensor([bool(success)]),
            "tcp_to_target_dist": torch.tensor([dist]),
            "target_lifted": torch.tensor([bool(lifted)]),
            "target_z": torch.tensor([float(target_pos[2])]),
        }

    def _get_obs_extra(self, info: dict) -> dict:
        tcp_pos = _p(self.agent.tcp.pose.p)
        tcp_q = self.agent.tcp.pose.q
        if isinstance(tcp_q, torch.Tensor):
            tcp_q = tcp_q.cpu().numpy()
        tcp_q = np.asarray(tcp_q).flatten()
        target_pos = _p(self._target.pose.p)
        occluder_pos = _p(self._occluder.pose.p)
        return {
            "tcp_pose":        np.concatenate([tcp_pos, tcp_q]),
            "target_pos":      target_pos,
            "occluder_pos":    occluder_pos,
            "tcp_to_target":   target_pos - tcp_pos,
            "tcp_to_occluder": occluder_pos - tcp_pos,
        }

    # ------------------------------------------------------------------
    #  Reward
    # ------------------------------------------------------------------
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict) -> float:
        tcp_pos      = _p(self.agent.tcp.pose.p)
        target_pos   = _p(self._target.pose.p)
        occluder_pos = _p(self._occluder.pose.p)

        occluder_blocks = (
            abs(occluder_pos[1] - target_pos[1]) < 0.06
            and occluder_pos[0] < target_pos[0]
        )
        reward = 0.0
        if occluder_blocks:
            dist_to_occ = float(np.linalg.norm(tcp_pos - occluder_pos))
            reward += max(0, 1.0 - dist_to_occ * 5.0)
            reward += abs(occluder_pos[1] - target_pos[1]) * 2.0
        else:
            reward += 1.0
            dist_to_target = float(np.linalg.norm(tcp_pos - target_pos))
            reward += max(0, 2.0 - dist_to_target * 10.0)
            lift = max(0.0, float(target_pos[2] - self._target_init_z))
            reward += lift * 20.0
        if info.get("success", False):
            reward += 10.0
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict,
    ) -> float:
        return self.compute_dense_reward(obs, action, info) / 14.0


# ---------------------------------------------------------------------------
#  Level 2 Environment
# ---------------------------------------------------------------------------
@register_env("OccluBench-Level2-v0", max_episode_steps=300)
class OccluBenchLevel2Env(OccluBenchLevel1Env):
    """Level 2 — 2-3 occluders, sequential push."""

    NUM_OCCLUDERS_RANGE = (2, 3)

    def _load_scene(self, options: dict):
        self._rng = np.random.default_rng(self._episode_seed)

        self.scene_builder = TableSceneBuilder(self, robot_init_qpos_noise=0.02)
        self.scene_builder.build()

        n_occ = self._rng.integers(*self.NUM_OCCLUDERS_RANGE, endpoint=True)

        self._target_ycb_id = _sample_unique(TARGET_YCB_IDS, 1, self._rng)[0]
        used = {self._target_ycb_id}

        occ_pool = [o for o in OCCLUDER_YCB_IDS if o not in used]
        occ_ids = _sample_unique(occ_pool, n_occ, self._rng)
        used.update(occ_ids)
        self._occluder_ycb_ids_list = list(occ_ids)

        dist_pool = [d for d in DISTRACTOR_YCB_IDS if d not in used]
        dist_ids = _sample_unique(dist_pool, 2, self._rng)

        self._target = _build_ycb_actor(self.scene, self._target_ycb_id, "target")

        self._occluders_list: list = []
        for i, oid in enumerate(occ_ids):
            a = _build_ycb_actor(self.scene, oid, f"occluder_{i}")
            self._occluders_list.append(a)
        self._occluder = self._occluders_list[0]

        self._distractors = []
        for i, did in enumerate(dist_ids):
            a = _build_ycb_actor(self.scene, did, f"distractor_{i}")
            self._distractors.append(a)

        self._all_objects = [self._target] + self._occluders_list + self._distractors

    @property
    def occluder_objects(self) -> list:
        return self._occluders_list

    @property
    def occluder_ycb_ids(self) -> list[str]:
        return self._occluder_ycb_ids_list

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        rng = np.random.default_rng(self._episode_seed)
        placed: list[np.ndarray] = []

        target_xy = _xy_no_collision(
            placed, x_range=(0.08, 0.18), y_range=(-0.10, 0.10),
            min_dist=self.MIN_OBJ_DIST, rng=rng,
        )
        placed.append(target_xy)
        target_z = 0.02
        self._target.set_pose(sapien.Pose(
            p=[target_xy[0], target_xy[1], target_z],
            q=euler2quat(0, 0, rng.uniform(0, 2 * np.pi)),
        ))

        n_occ = len(self._occluders_list)
        x_offsets = np.linspace(0.06, 0.06 * n_occ, n_occ)
        for i, occ_actor in enumerate(self._occluders_list):
            occ_x = target_xy[0] - x_offsets[i] + rng.uniform(-0.02, 0.02)
            occ_y = target_xy[1] + rng.uniform(-0.04, 0.04)
            occ_xy = np.array([occ_x, occ_y])
            for _ in range(50):
                if all(np.linalg.norm(occ_xy - p) >= self.MIN_OBJ_DIST for p in placed):
                    break
                occ_y = target_xy[1] + rng.uniform(-0.06, 0.06)
                occ_xy = np.array([occ_x, occ_y])
            placed.append(occ_xy)
            occ_actor.set_pose(sapien.Pose(
                p=[occ_xy[0], occ_xy[1], 0.02],
                q=euler2quat(0, 0, rng.uniform(0, 2 * np.pi)),
            ))

        for da in self._distractors:
            dxy = _xy_no_collision(
                placed, x_range=self.X_RANGE, y_range=self.Y_RANGE,
                min_dist=self.MIN_OBJ_DIST, rng=rng,
            )
            placed.append(dxy)
            da.set_pose(sapien.Pose(
                p=[dxy[0], dxy[1], 0.02],
                q=euler2quat(0, 0, rng.uniform(0, 2 * np.pi)),
            ))

        self._target_init_z = target_z

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict) -> float:
        tcp_pos    = _p(self.agent.tcp.pose.p)
        target_pos = _p(self._target.pose.p)

        blocking = []
        for occ in self._occluders_list:
            op = _p(occ.pose.p)
            if abs(op[1] - target_pos[1]) < 0.06 and op[0] < target_pos[0]:
                blocking.append(occ)

        n_cleared = len(self._occluders_list) - len(blocking)
        reward = float(n_cleared) * 0.5

        if blocking:
            dists = [float(np.linalg.norm(tcp_pos - _p(o.pose.p))) for o in blocking]
            reward += max(0, 1.0 - min(dists) * 5.0)
        else:
            reward += len(self._occluders_list) * 0.5
            dist_to_target = float(np.linalg.norm(tcp_pos - target_pos))
            reward += max(0, 2.0 - dist_to_target * 10.0)
            lift = max(0.0, float(target_pos[2] - self._target_init_z))
            reward += lift * 20.0

        if info.get("success", False):
            reward += 10.0
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict,
    ) -> float:
        return self.compute_dense_reward(obs, action, info) / 15.0


# ---------------------------------------------------------------------------
#  Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import gymnasium as gym

    for env_id in ["OccluBench-Level1-v0", "OccluBench-Level2-v0"]:
        print(f"\n{'='*60}")
        print(f"  Testing: {env_id}")
        print(f"{'='*60}")

        env = gym.make(env_id, obs_mode="rgbd", render_mode="rgb_array")
        obs, info = env.reset(seed=42)

        if isinstance(obs, dict):
            print(f"  Top-level keys: {list(obs.keys())}")
            sd = obs.get('sensor_data', {})
            print(f"  sensor_data cameras: {list(sd.keys())}")
            for cam, data in sd.items():
                if isinstance(data, dict):
                    for k, v in data.items():
                        shape = v.shape if hasattr(v, 'shape') else type(v)
                        print(f"    {cam}/{k}: {shape}")

        uw = env.unwrapped
        print(f"  target_object:   {uw.target_object.name}")
        print(f"  occluder_objects: {[o.name for o in uw.occluder_objects]}")
        print(f"  target_ycb_id:   {uw.target_ycb_id}")
        print(f"  Action space:    {env.action_space}")

        for step in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  Step {step}: reward={reward:.3f}, info={info}")
            if terminated or truncated:
                break

        env.close()
        print(f"  ✅ {env_id} OK")