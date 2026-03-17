"""
OccluVLA - 生成 push-then-grasp 演示轨迹

特性:
  - Closed-loop: 每步从仿真读取真实 EE 位置
  - 从 env.unwrapped.target_object 读取 target
  - 自动检测 occluder 并规划 push 方向
  - 同时录制 base_camera + render_view
  - 语言标注包含物体名称
  - 混合数据: 有 occluder → push+grasp, 无 occluder → direct grasp

用法:
    python generate_demos.py --num 5 --debug          # 调试
    python generate_demos.py --num 100 --save_video   # 正式
"""

import os, sys, json, time, argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from push_grasp_env import OccluVLAEnv, ScriptedPushGraspPolicy, to_np


LANG_PUSH_GRASP = [
    "push the {occ} aside, then pick up the {tgt}",
    "move the {occ} out of the way and grasp the {tgt}",
    "clear the {occ} by pushing, then grab the {tgt}",
]
LANG_DIRECT_GRASP = [
    "pick up the {tgt}",
    "grasp the {tgt} from the table",
    "grab the {tgt}",
]


def clean_name(raw_name):
    """set_0_065-j_cups → cups (j)"""
    # 去掉 set_N_ 前缀
    name = raw_name
    if name.startswith('set_'):
        parts = name.split('_', 2)
        if len(parts) >= 3:
            name = parts[2]
    # 简化 YCB 名称
    name = name.replace('_', ' ')
    return name


def collect_episode(env, policy, seed, save_video=False, debug=False):
    rng = np.random.RandomState(seed)
    obs, info = env.reset(seed=seed)

    if debug:
        print(f"\n  [seed={seed}] Scene:")
        env.print_scene()

    # ── 读取 target 和 occluder ──────────────────────────────────
    tgt = env.get_target_object()
    if tgt is None:
        print(f"  [seed={seed}] Cannot find target, skipping")
        return None, False, []

    target_pos = tgt['pos'].copy()
    target_name = clean_name(tgt['name'])

    occluder = env.find_occluder(target_pos)
    has_occluder = occluder is not None

    if has_occluder:
        occ_pos = occluder['pos'].copy()
        occ_name = clean_name(occluder['name'])
        lang = rng.choice(LANG_PUSH_GRASP).format(occ=occ_name, tgt=target_name)
    else:
        occ_pos = None
        occ_name = None
        lang = rng.choice(LANG_DIRECT_GRASP).format(tgt=target_name)

    if debug:
        print(f"  Mode: {'push+grasp' if has_occluder else 'direct grasp'}")
        print(f"  Language: {lang}")

    # ── 规划 waypoints ───────────────────────────────────────────
    ee_pos = env.get_ee_pos()
    waypoints = policy.plan(ee_pos, target_pos, occ_pos)

    if debug:
        for i, (phase, wp, grip, label) in enumerate(waypoints):
            print(f"    WP[{i}] {phase:5s} {label:20s} → "
                  f"[{wp[0]:.3f},{wp[1]:.3f},{wp[2]:.3f}] grip={grip}")

    # ── Closed-loop 执行 ─────────────────────────────────────────
    ep_data = {
        'rgb_frames': [], 'render_frames': [], 'actions': [], 'states': [],
        'phases': [], 'rewards': [],
    }
    max_steps_per_wp = 80
    total_steps = 0

    for wp_idx, (phase, target_xyz, gripper, label) in enumerate(waypoints):

        if label == 'close':
            # 闭合夹爪: 原地等待
            for _ in range(policy.GRIP_WAIT):
                imgs = env.get_images(obs)
                for k, v in imgs.items():
                    if 'rgb' in k:
                        ep_data['rgb_frames'].append(v)

                if save_video:
                    rf = env.get_render_frame()
                    if rf is not None:
                        ep_data['render_frames'].append(rf)

                qpos = env.get_qpos(obs)
                if qpos is not None:
                    ep_data['states'].append(qpos)

                action = np.zeros(env.action_dim)
                action[6] = gripper
                ep_data['actions'].append(action)
                ep_data['phases'].append(phase)

                obs, reward, terminated, truncated, info = env.step(action)
                ep_data['rewards'].append(
                    reward.item() if isinstance(reward, torch.Tensor) else float(reward)
                )
                total_steps += 1
                if terminated or truncated: break
            continue

        # 正常 waypoint: closed-loop 移动
        for step_i in range(max_steps_per_wp):
            # 记录
            imgs = env.get_images(obs)
            for k, v in imgs.items():
                if 'rgb' in k:
                    ep_data['rgb_frames'].append(v)

            if save_video:
                rf = env.get_render_frame()
                if rf is not None:
                    ep_data['render_frames'].append(rf)

            qpos = env.get_qpos(obs)
            if qpos is not None:
                ep_data['states'].append(qpos)

            # Closed-loop: 读取当前真实 EE 位置
            current_ee = env.get_ee_pos()
            action, reached = policy.make_action(current_ee, target_xyz, gripper)

            # 确保维度
            if len(action) < env.action_dim:
                full = np.zeros(env.action_dim)
                full[:len(action)] = action
                action = full

            ep_data['actions'].append(action.copy())
            ep_data['phases'].append(phase)

            obs, reward, terminated, truncated, info = env.step(action)
            import torch
            ep_data['rewards'].append(
                reward.item() if isinstance(reward, torch.Tensor) else float(reward)
            )
            total_steps += 1

            if reached or terminated or truncated:
                break

        if terminated or truncated:
            break

    # ── 结果 ─────────────────────────────────────────────────────
    success = env.get_success(info)

    episode = {
        'rgb':      np.array(ep_data['rgb_frames']) if ep_data['rgb_frames'] else np.array([]),
        'actions':  np.array(ep_data['actions']) if ep_data['actions'] else np.array([]),
        'states':   np.array(ep_data['states']) if ep_data['states'] else np.array([]),
        'rewards':  np.array(ep_data['rewards']),
        'phases':   np.array(ep_data['phases']),
        'language':  lang,
        'seed':      seed,
        'num_steps': total_steps,
        'success':   success,
        'has_occluder': has_occluder,
        'target_name': tgt['name'],
        'target_pos':  target_pos,
        'occluder_name': occluder['name'] if occluder else '',
        'occluder_pos':  occ_pos if occ_pos is not None else np.zeros(3),
    }

    render_frames = ep_data['render_frames'] if save_video else []
    return episode, success, render_frames


def save_video(frames, path, fps=15):
    if not frames: return
    try:
        import imageio
        arr = np.array(frames)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        imageio.mimsave(str(path), arr, fps=fps)
    except Exception as e:
        print(f"  [!] Video save failed: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--render_size", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(__file__).parent.parent / "data" / "push_grasp_demos"
    vid_dir = Path(__file__).parent.parent / "data" / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  OccluVLA: Push-Then-Grasp Demo Generation")
    print(f"  Episodes: {args.num} | Video: {args.save_video}")
    print(f"  Render: {args.render_size} | Debug: {args.debug}")
    print("=" * 60)

    env = OccluVLAEnv(num_envs=1, render_size=args.render_size)
    policy = ScriptedPushGraspPolicy()

    total_success = 0
    total_push_grasp = 0
    total_direct = 0
    failed = []
    t0 = time.time()

    import torch  # for reward handling

    for i in range(args.num):
        seed = args.start_seed + i
        do_debug = args.debug and i < 3

        ep, success, frames = collect_episode(
            env, policy, seed,
            save_video=args.save_video,
            debug=do_debug,
        )

        if ep is None:
            failed.append(seed)
            continue

        # 保存 npz
        save_dict = {}
        for k, v in ep.items():
            if isinstance(v, np.ndarray):
                save_dict[k] = v
            elif isinstance(v, (str, int, float, bool)):
                save_dict[k] = v
        np.savez_compressed(out_dir / f"episode_{i:04d}.npz", **save_dict)

        # 保存视频 (render view, 512x512)
        if args.save_video and frames:
            save_video(frames, vid_dir / f"episode_{i:04d}.mp4")

        total_success += int(success)
        if ep['has_occluder']:
            total_push_grasp += 1
        else:
            total_direct += 1

        elapsed = time.time() - t0
        speed = (i + 1) / elapsed
        if (i + 1) % 10 == 0 or i == 0 or do_debug:
            print(f"  [{i+1:3d}/{args.num}] steps={ep['num_steps']:3d} "
                  f"{'push+grasp' if ep['has_occluder'] else 'direct    '} "
                  f"success={success} {speed:.1f}ep/s")

    env.close()

    # 元数据
    done = args.num - len(failed)
    meta = {
        "num_episodes": args.num,
        "completed": done,
        "success_rate": total_success / done if done > 0 else 0,
        "push_grasp_count": total_push_grasp,
        "direct_grasp_count": total_direct,
        "failed_seeds": failed,
        "render_size": args.render_size,
        "time_sec": time.time() - t0,
    }
    json.dump(meta, open(out_dir / "metadata.json", "w"), indent=2)

    print(f"\n{'='*60}")
    print(f"  Done! {done}/{args.num} episodes")
    print(f"  Success: {total_success}/{done} = {meta['success_rate']:.1%}")
    print(f"  Push+grasp: {total_push_grasp} | Direct: {total_direct}")
    print(f"  Data: {out_dir}")
    if args.save_video:
        print(f"  Video: {vid_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
