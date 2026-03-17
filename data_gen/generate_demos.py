"""
OccluVLA - 生成 push-then-grasp 演示轨迹

基于 OccluBench 自定义环境 (shoulder_camera + hand_camera)

特性:
  - Closed-loop: 每步从仿真读取真实 EE 位置
  - 从 OccluBench 的 target_object / occluder_objects 读取场景信息
  - 双相机数据分开存储: {cam}_rgb, {cam}_depth
  - 每个相机独立视频 + shoulder|hand 并排合成
  - 支持 Level 1 / Level 2

用法:
    python generate_demos.py --num 5 --level 1 --debug
    python generate_demos.py --num 100 --level 1 --save_video
    python generate_demos.py --num 100 --level 2 --save_video
"""

import os, sys, json, time, argparse
import numpy as np
import torch
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


def clean_ycb_name(ycb_id: str) -> str:
    """002_master_chef_can → master chef can"""
    parts = ycb_id.split('_', 1)
    if len(parts) == 2 and parts[0].isdigit():
        return parts[1].replace('_', ' ')
    return ycb_id.replace('_', ' ')


def collect_episode(env, policy, seed, save_video=False, debug=False):
    """收集一条 push-then-grasp 轨迹，返回 episode dict + video frames。"""
    rng = np.random.RandomState(seed)
    obs, info = env.reset(seed=seed)

    if debug:
        print(f"\n  [seed={seed}] Scene:")
        env.print_scene()

    # ── 场景信息 ─────────────────────────────────────────────────
    tgt = env.get_target_object()
    target_pos = tgt['pos'].copy()
    target_name = clean_ycb_name(tgt['ycb_id'])

    occluders = env.get_occluder_objects()
    # 取最近的 occluder 来 push
    nearest_occ = env.find_nearest_occluder(target_pos)
    has_occluder = nearest_occ is not None

    if has_occluder:
        occ_pos = nearest_occ['pos'].copy()
        occ_name = clean_ycb_name(nearest_occ['ycb_id'])
        lang = rng.choice(LANG_PUSH_GRASP).format(occ=occ_name, tgt=target_name)
    else:
        occ_pos = None
        occ_name = None
        lang = rng.choice(LANG_DIRECT_GRASP).format(tgt=target_name)

    if debug:
        print(f"  Target: {target_name} | Occluders: {len(occluders)}")
        print(f"  Mode: {'push+grasp' if has_occluder else 'direct grasp'}")
        print(f"  Language: {lang}")

    # ── Waypoints ────────────────────────────────────────────────
    ee_pos = env.get_ee_pos()
    waypoints = policy.plan(ee_pos, target_pos, occ_pos)

    if debug:
        for i, (phase, wp, grip, label) in enumerate(waypoints):
            print(f"    WP[{i}] {phase:5s} {label:20s} → "
                  f"[{wp[0]:.3f},{wp[1]:.3f},{wp[2]:.3f}] grip={grip}")

    # ── 初始化存储 ───────────────────────────────────────────────
    cam_names = env.camera_names
    ep_data = {'actions': [], 'states': [], 'phases': [], 'rewards': []}
    for cam in cam_names:
        ep_data[f'{cam}_rgb'] = []
        ep_data[f'{cam}_depth'] = []

    video_frames = {}
    if save_video:
        for cam in cam_names:
            video_frames[cam] = []
        video_frames['render'] = []

    max_steps_per_wp = 80
    total_steps = 0

    # ── 单步录制 ─────────────────────────────────────────────────
    def record_step(obs_t, action_t, phase_t):
        imgs = env.get_images(obs_t)
        for cam in cam_names:
            if cam in imgs:
                cd = imgs[cam]
                ep_data[f'{cam}_rgb'].append(cd.get('rgb'))
                ep_data[f'{cam}_depth'].append(cd.get('depth'))
                if save_video and cd.get('rgb') is not None:
                    video_frames[cam].append(cd['rgb'].copy())
            else:
                ep_data[f'{cam}_rgb'].append(None)
                ep_data[f'{cam}_depth'].append(None)

        if save_video:
            rf = env.get_render_frame()
            if rf is not None:
                video_frames['render'].append(rf)

        qpos = env.get_qpos(obs_t)
        if qpos is not None:
            ep_data['states'].append(qpos)

        ep_data['actions'].append(action_t.copy())
        ep_data['phases'].append(phase_t)

    # ── 执行 ─────────────────────────────────────────────────────
    for wp_idx, (phase, target_xyz, gripper, label) in enumerate(waypoints):

        if label == 'close':
            for _ in range(policy.GRIP_WAIT):
                action = np.zeros(env.action_dim)
                action[6] = gripper
                record_step(obs, action, phase)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_data['rewards'].append(
                    reward.item() if isinstance(reward, torch.Tensor) else float(reward))
                total_steps += 1
                if terminated or truncated: break
            continue

        for step_i in range(max_steps_per_wp):
            current_ee = env.get_ee_pos()
            action, reached = policy.make_action(current_ee, target_xyz, gripper)
            if len(action) < env.action_dim:
                full = np.zeros(env.action_dim)
                full[:len(action)] = action
                action = full

            record_step(obs, action, phase)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_data['rewards'].append(
                reward.item() if isinstance(reward, torch.Tensor) else float(reward))
            total_steps += 1
            if reached or terminated or truncated: break

        if terminated or truncated: break

    # ── 组装 episode ─────────────────────────────────────────────
    success = env.get_success(info)

    episode = {
        'actions':       np.array(ep_data['actions']) if ep_data['actions'] else np.array([]),
        'states':        np.array(ep_data['states'])  if ep_data['states']  else np.array([]),
        'rewards':       np.array(ep_data['rewards']),
        'phases':        np.array(ep_data['phases']),
        'language':      lang,
        'seed':          seed,
        'num_steps':     total_steps,
        'success':       success,
        'has_occluder':  has_occluder,
        'num_occluders': len(occluders),
        'target_name':   tgt['name'],
        'target_ycb_id': tgt['ycb_id'],
        'target_pos':    target_pos,
        'occluder_name': nearest_occ['name'] if nearest_occ else '',
        'occluder_pos':  occ_pos if occ_pos is not None else np.zeros(3),
        'camera_names':  ','.join(cam_names),
    }

    for cam in cam_names:
        rgb_list = [f for f in ep_data[f'{cam}_rgb'] if f is not None]
        depth_list = [f for f in ep_data[f'{cam}_depth'] if f is not None]
        if rgb_list:
            episode[f'{cam}_rgb'] = np.stack(rgb_list, axis=0)
        if depth_list:
            episode[f'{cam}_depth'] = np.stack(depth_list, axis=0)

    return episode, success, video_frames


# ── 视频保存 ─────────────────────────────────────────────────────
def save_video_file(frames, path, fps=15):
    if not frames: return
    try:
        import imageio
        arr = np.array(frames)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        imageio.mimsave(str(path), arr, fps=fps)
    except Exception as e:
        print(f"  [!] Video save failed ({path}): {e}")


def save_side_by_side(frame_dict, cam_names, path, fps=15):
    """多相机并排视频"""
    all_frames = [frame_dict[c] for c in cam_names if c in frame_dict and frame_dict[c]]
    if not all_frames: return

    min_len = min(len(f) for f in all_frames)
    if min_len == 0: return

    combined = []
    for i in range(min_len):
        row = []
        for frames in all_frames:
            img = frames[i]
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            if img.shape[-1] == 4:
                img = img[..., :3]
            row.append(img)
        # 统一高度
        min_h = min(r.shape[0] for r in row)
        resized = []
        for r in row:
            if r.shape[0] != min_h:
                start = (r.shape[0] - min_h) // 2
                r = r[start:start + min_h]
            resized.append(r)
        combined.append(np.concatenate(resized, axis=1))

    save_video_file(combined, path, fps=fps)


# ── Main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="OccluVLA: Demo Generation (OccluBench + dual camera)")
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--level", type=int, default=1, choices=[1, 2],
                        help="OccluBench difficulty level")
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--render_size", type=int, default=256)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(__file__).parent.parent / "data" / f"push_grasp_level{args.level}"
    vid_dir = Path(__file__).parent.parent / "data" / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  OccluVLA: Push-Then-Grasp Demo Gen (Level {args.level})")
    print(f"  Env: OccluBench-Level{args.level}-v0")
    print(f"  Episodes: {args.num} | Video: {args.save_video}")
    print(f"  Render: {args.render_size}x{args.render_size}")
    print(f"  Output: {out_dir}")
    print("=" * 60)

    env = OccluVLAEnv(level=args.level, num_envs=1, render_size=args.render_size)
    policy = ScriptedPushGraspPolicy()
    cam_names = env.camera_names

    total_success = 0
    total_push_grasp = 0
    total_direct = 0
    failed = []
    t0 = time.time()

    for i in range(args.num):
        seed = args.start_seed + i
        do_debug = args.debug and i < 3

        ep, success, vid_frames = collect_episode(
            env, policy, seed,
            save_video=args.save_video, debug=do_debug,
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

        # 保存视频
        if args.save_video and vid_frames:
            for cam in cam_names:
                if cam in vid_frames and vid_frames[cam]:
                    save_video_file(
                        vid_frames[cam],
                        vid_dir / f"ep{i:04d}_L{args.level}_{cam}.mp4",
                    )
            if vid_frames.get('render'):
                save_video_file(
                    vid_frames['render'],
                    vid_dir / f"ep{i:04d}_L{args.level}_render.mp4",
                )
            if len(cam_names) >= 2:
                save_side_by_side(
                    vid_frames, cam_names,
                    vid_dir / f"ep{i:04d}_L{args.level}_sidebyside.mp4",
                )

        total_success += int(success)
        if ep['has_occluder']:
            total_push_grasp += 1
        else:
            total_direct += 1

        elapsed = time.time() - t0
        speed = (i + 1) / elapsed
        if (i + 1) % 10 == 0 or i == 0 or do_debug:
            cam_info = " ".join(
                f"{c}={ep.get(f'{c}_rgb', np.array([])).shape[0] if isinstance(ep.get(f'{c}_rgb'), np.ndarray) else 0}f"
                for c in cam_names
            )
            print(f"  [{i+1:3d}/{args.num}] steps={ep['num_steps']:3d} "
                  f"occ={ep['num_occluders']} "
                  f"{'push+grasp' if ep['has_occluder'] else 'direct    '} "
                  f"success={success} {cam_info} {speed:.1f}ep/s")

    env.close()

    done = args.num - len(failed)
    meta = {
        "env_id": f"OccluBench-Level{args.level}-v0",
        "level": args.level,
        "num_episodes": args.num,
        "completed": done,
        "success_rate": total_success / done if done > 0 else 0,
        "push_grasp_count": total_push_grasp,
        "direct_grasp_count": total_direct,
        "cameras": cam_names,
        "render_size": args.render_size,
        "failed_seeds": failed,
        "time_sec": time.time() - t0,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Done! {done}/{args.num} episodes (Level {args.level})")
    print(f"  Success: {total_success}/{done} = {meta['success_rate']:.1%}")
    print(f"  Push+grasp: {total_push_grasp} | Direct: {total_direct}")
    print(f"  Cameras: {cam_names}")
    print(f"  Data: {out_dir}")
    if args.save_video:
        print(f"  Videos: {vid_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
