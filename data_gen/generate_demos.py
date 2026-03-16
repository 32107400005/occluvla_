"""
OccluVLA - 生成 ~100 条 push-then-grasp 演示轨迹 (修正版)

核心修正:
1. Closed-loop 控制: 每步读取真实 EE 位置, 计算 delta
2. 从仿真中读取目标/遮挡物的真实 pose
3. 512x512 高分辨率渲染
4. 结构化场景 (而非完全随机)
5. 自动语言标注

用法:
    python generate_demos.py --num_episodes 100 --save_video
    python generate_demos.py --num_episodes 5 --save_video --debug
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from push_grasp_env import PushGraspSceneManager, ScriptedPushGraspPolicy

import torch


# ── 场景配置 ─────────────────────────────────────────────────────────
# Level 1: 1 target + 1 occluder in front + 2 distractors
# 物体在桌面上的典型 xy 范围 (ManiSkill3 PickSingleYCB 坐标系)
# 你需要根据实际运行结果微调这些范围

TARGET_XY_RANGE   = [(-0.05, 0.05), (-0.05, 0.05)]   # target 的 xy 范围
OCCLUDER_OFFSET_Y = (-0.08, -0.05)                     # occluder 在 target 前方 (y 更小)
OCCLUDER_OFFSET_X = (-0.02, 0.02)                      # occluder 的 x 偏移
TABLE_Z           = 0.02                                # 物体的大致 z 高度

# 语言模板
LANGUAGE_TEMPLATES = [
    "push the blocking object aside, then pick up the target object",
    "move the occluding object out of the way and grasp the target",
    "clear the obstacle by pushing it, then grab the target object",
    "push away the object in front, then pick up the target",
    "first push the blocking object to the side, then grasp the target",
]


def make_occluder_pos(target_pos, rng):
    """在 target 前方生成一个 occluder 位置"""
    offset_x = rng.uniform(*OCCLUDER_OFFSET_X)
    offset_y = rng.uniform(*OCCLUDER_OFFSET_Y)
    occluder_pos = target_pos.copy()
    occluder_pos[0] += offset_x
    occluder_pos[1] += offset_y
    return occluder_pos


def collect_episode(env_mgr, policy, seed, save_video=False, debug=False):
    """
    收集一条 push-then-grasp 演示轨迹.

    核心流程:
    1. Reset 环境, 获取真实物体位置
    2. 规划 waypoint 序列
    3. Closed-loop 执行: 每步读取 EE 位置 → 计算 delta → step

    Returns:
        episode_data: dict
        success: bool
        frames: list of RGB frames (for video)
    """
    rng = np.random.RandomState(seed)
    obs, info = env_mgr.reset(seed=seed)

    # ── 1. 获取真实物体位置 ──────────────────────────────────────────
    # 从仿真中读取 target 位置
    target_pos = env_mgr.get_target_pos(obs)

    if target_pos is None:
        # Fallback: 如果无法读取, 尝试从 env.unwrapped 获取
        print(f"  [seed={seed}] 无法从 obs 中获取 target 位置, 尝试 fallback...")
        try:
            scene = env_mgr.env.unwrapped
            if hasattr(scene, 'obj'):
                p = scene.obj.pose.p
                if isinstance(p, torch.Tensor):
                    p = p.cpu().numpy()
                target_pos = p[0] if p.ndim > 1 else p
        except Exception as e:
            print(f"  [seed={seed}] Fallback 也失败: {e}")
            return None, False, []

    if debug:
        print(f"  [seed={seed}] Target pos: {target_pos}")
        env_mgr.inspect_obs_keys(obs)

    # ── 2. 确定 occluder 位置 ────────────────────────────────────────
    # 在 PickSingleYCB 中只有一个目标物体, 没有额外的 occluder.
    # 两种策略:
    #   A) 如果你已经创建了自定义环境 (有多个物体), 从 scene 中读取
    #   B) 如果用 PickSingleYCB, 我们假设一个 "虚拟" occluder 位置
    #      来测试 scripted policy 的 push 轨迹规划是否正确
    #      (push 阶段不会真的推到东西, 但 grasp 阶段是真实的)
    #
    # ⚠️ 重要: 要获得真正的 push-then-grasp 行为, 你最终需要:
    #   - 自定义一个 ManiSkill3 环境, 场景中有多个物体
    #   - 或者使用 PickClutterYCB 并从 scene 中识别哪个物体挡住了 target
    #
    # 这里我们先实现策略 B, 确保代码框架正确, 后续替换为策略 A

    occluder_pos = make_occluder_pos(target_pos, rng)

    if debug:
        print(f"  [seed={seed}] Occluder pos: {occluder_pos}")

    # ── 3. 获取初始 EE 位置 ──────────────────────────────────────────
    ee_pos = env_mgr.get_ee_pos(obs)
    if debug:
        print(f"  [seed={seed}] Initial EE pos: {ee_pos}")

    # ── 4. 规划 waypoint 序列 ────────────────────────────────────────
    waypoints = policy.plan_push_grasp(ee_pos, target_pos, occluder_pos)

    if debug:
        print(f"  [seed={seed}] Planned {len(waypoints)} waypoints:")
        for i, wp in enumerate(waypoints):
            print(f"    [{i}] {wp['label']}: {wp['target_xyz']}")

    # ── 5. Closed-loop 执行 ──────────────────────────────────────────
    ep_rgb       = []
    ep_depth     = []
    ep_states    = []
    ep_actions   = []
    ep_rewards   = []
    ep_phases    = []
    frames       = []   # for video

    max_steps_per_waypoint = 80
    total_steps = 0
    terminated = False
    truncated = False

    for wp_idx, wp in enumerate(waypoints):
        target_xyz = wp['target_xyz']
        gripper_val = wp['gripper']
        phase = wp['phase']

        # 如果是 "close gripper" waypoint, 原地等待几步
        if wp['label'] == 'close gripper':
            for _ in range(policy.GRIPPER_STEPS):
                # 记录数据
                imgs = env_mgr.get_images(obs)
                for k, v in imgs.items():
                    if "rgb" in k:
                        ep_rgb.append(v)
                        if save_video and "base" in k:
                            frames.append(v)
                    elif "depth" in k:
                        ep_depth.append(v)

                state = env_mgr.get_robot_state(obs)
                if state is not None:
                    ep_states.append(state)

                action = np.zeros(env_mgr.action_dim)
                action[-1] = gripper_val   # 只闭合夹爪
                ep_actions.append(action)
                ep_phases.append(phase)

                obs, reward, terminated, truncated, info = env_mgr.step(action)
                r = reward.item() if isinstance(reward, torch.Tensor) else float(reward)
                ep_rewards.append(r)
                total_steps += 1

                if terminated or truncated:
                    break
            continue

        # 正常的移动 waypoint: closed-loop 逼近
        for step_i in range(max_steps_per_waypoint):
            # 记录当前帧数据
            imgs = env_mgr.get_images(obs)
            for k, v in imgs.items():
                if "rgb" in k:
                    ep_rgb.append(v)
                    if save_video and "base" in k:
                        frames.append(v)
                elif "depth" in k:
                    ep_depth.append(v)

            state = env_mgr.get_robot_state(obs)
            if state is not None:
                ep_states.append(state)

            # 读取当前 EE 位置 (closed-loop!)
            current_ee = env_mgr.get_ee_pos(obs)

            # 计算 action
            action, reached = policy.compute_action(current_ee, target_xyz, gripper_val)

            # 确保 action 维度正确
            if len(action) < env_mgr.action_dim:
                full_action = np.zeros(env_mgr.action_dim)
                full_action[:len(action)] = action
                action = full_action
            elif len(action) > env_mgr.action_dim:
                action = action[:env_mgr.action_dim]

            ep_actions.append(action.copy())
            ep_phases.append(phase)

            # 执行
            obs, reward, terminated, truncated, info = env_mgr.step(action)
            r = reward.item() if isinstance(reward, torch.Tensor) else float(reward)
            ep_rewards.append(r)
            total_steps += 1

            if reached or terminated or truncated:
                break

        if terminated or truncated:
            break

    # ── 6. 检查成功 ──────────────────────────────────────────────────
    success = False
    if isinstance(info, dict) and "success" in info:
        s = info["success"]
        success = bool(s.item() if isinstance(s, torch.Tensor) else s)

    # ── 7. 语言标注 ──────────────────────────────────────────────────
    language = rng.choice(LANGUAGE_TEMPLATES)

    # ── 8. 打包数据 ──────────────────────────────────────────────────
    episode_data = {
        "rgb":      np.array(ep_rgb) if ep_rgb else np.array([]),
        "depth":    np.array(ep_depth) if ep_depth else np.array([]),
        "states":   np.array(ep_states) if ep_states else np.array([]),
        "actions":  np.array(ep_actions) if ep_actions else np.array([]),
        "rewards":  np.array(ep_rewards),
        "phases":   np.array(ep_phases),
        "language":  language,
        "seed":      seed,
        "num_steps": total_steps,
        "success":   success,
        "target_pos":   target_pos,
        "occluder_pos": occluder_pos,
    }

    return episode_data, success, frames


def main():
    parser = argparse.ArgumentParser(description="OccluVLA push-then-grasp demo generation")
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="Number of episodes to generate")
    parser.add_argument("--save_video", action="store_true",
                        help="Save MP4 videos for visualization")
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--render_size", type=int, default=512,
                        help="Render resolution (width=height)")
    parser.add_argument("--debug", action="store_true",
                        help="Print detailed debug info for first few episodes")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: ../data/push_grasp_demos)")
    args = parser.parse_args()

    # ── 路径设置 ─────────────────────────────────────────────────────
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent.parent / "data" / "push_grasp_demos"
    vis_dir = Path(__file__).parent.parent / "data" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  OccluVLA: Push-Then-Grasp Demo Generation")
    print("=" * 60)
    print(f"  Episodes: {args.num_episodes}")
    print(f"  Render:   {args.render_size}x{args.render_size}")
    print(f"  Video:    {'Yes' if args.save_video else 'No'}")
    print(f"  Output:   {output_dir}")
    print()

    # ── 创建环境和策略 ───────────────────────────────────────────────
    env_mgr = PushGraspSceneManager(
        num_envs=1,
        render_size=args.render_size,
        control_mode="pd_ee_delta_pose",
    )
    policy = ScriptedPushGraspPolicy()

    # ── 首次运行: 检查 obs 结构 ──────────────────────────────────────
    print("\n[检查] 探查 obs 结构...")
    test_obs, test_info = env_mgr.reset(seed=42)
    env_mgr.inspect_obs_keys(test_obs)

    test_ee = env_mgr.get_ee_pos(test_obs)
    test_tgt = env_mgr.get_target_pos(test_obs)
    print(f"\n  EE position:     {test_ee}")
    print(f"  Target position: {test_tgt}")

    test_imgs = env_mgr.get_images(test_obs)
    for k, v in test_imgs.items():
        print(f"  Image '{k}': shape={v.shape}, dtype={v.dtype}")

    print()

    # ── 主循环: 生成轨迹 ─────────────────────────────────────────────
    total_success = 0
    total_steps   = 0
    failed_seeds  = []
    start_time    = time.time()

    for i in range(args.num_episodes):
        seed = args.start_seed + i
        do_debug = args.debug and i < 3   # 前3条打印详细信息

        ep_data, success, frames = collect_episode(
            env_mgr, policy, seed,
            save_video=args.save_video,
            debug=do_debug,
        )

        if ep_data is None:
            print(f"  [!] Episode {i} (seed={seed}) FAILED - skipping")
            failed_seeds.append(seed)
            continue

        # 保存数据
        np.savez_compressed(
            output_dir / f"episode_{i:04d}.npz",
            **{k: v for k, v in ep_data.items()
               if isinstance(v, (np.ndarray, str, int, float, bool))}
        )

        # 保存视频 (512x512 MP4)
        if args.save_video and frames and len(frames) > 0:
            try:
                import imageio.v3 as iio
                video_path = vis_dir / f"episode_{i:04d}.mp4"
                # 确保帧尺寸一致
                frames_arr = np.array(frames)
                iio.imwrite(
                    str(video_path),
                    frames_arr,
                    fps=15,
                    codec="libx264",
                    plugin="pyav",
                )
            except ImportError:
                try:
                    import imageio
                    imageio.mimsave(
                        str(vis_dir / f"episode_{i:04d}.mp4"),
                        frames, fps=15
                    )
                except Exception as e:
                    print(f"  [!] 保存视频失败: {e}")

        total_success += int(success)
        total_steps   += ep_data["num_steps"]

        # 进度报告
        elapsed = time.time() - start_time
        speed = (i + 1) / elapsed
        eta = (args.num_episodes - i - 1) / speed if speed > 0 else 0

        if (i + 1) % 10 == 0 or i == 0 or do_debug:
            print(
                f"  [{i+1:3d}/{args.num_episodes}]  "
                f"steps={ep_data['num_steps']:3d}  "
                f"success={success}  "
                f"speed={speed:.1f} ep/s  "
                f"ETA={eta:.0f}s"
            )

    env_mgr.close()

    # ── 保存元数据 ───────────────────────────────────────────────────
    completed = args.num_episodes - len(failed_seeds)
    metadata = {
        "num_episodes":    args.num_episodes,
        "completed":       completed,
        "total_success":   total_success,
        "success_rate":    total_success / completed if completed > 0 else 0,
        "avg_steps":       total_steps / completed if completed > 0 else 0,
        "total_time_sec":  time.time() - start_time,
        "render_size":     args.render_size,
        "control_mode":    "pd_ee_delta_pose",
        "failed_seeds":    failed_seeds,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # ── 汇总 ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  ✅ 完成!")
    print(f"  总轨迹:   {completed} / {args.num_episodes}")
    print(f"  成功率:   {metadata['success_rate']:.1%}")
    print(f"  平均步数: {metadata['avg_steps']:.1f}")
    print(f"  总耗时:   {metadata['total_time_sec']:.1f}s")
    print(f"  数据:     {output_dir}")
    if args.save_video:
        print(f"  视频:     {vis_dir}")
    if failed_seeds:
        print(f"  失败 seed: {failed_seeds}")
    print("=" * 60)


if __name__ == "__main__":
    main()
