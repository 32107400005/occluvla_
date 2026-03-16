"""生成 ~100 条 push-then-grasp 演示轨迹"""
import os, sys, json, time, argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from push_grasp_env import PushGraspSceneManager, ScriptedPushGraspPolicy
import torch


def collect_episode(env_mgr, policy, seed, save_video=False):
    obs, info = env_mgr.reset(seed=seed)
    np.random.seed(seed)

    target_pos = np.array([np.random.uniform(-0.08, 0.08),
                           np.random.uniform(0.02, 0.10), 0.02])
    occluder_pos = target_pos + np.array([np.random.uniform(-0.03, 0.03),
                                          np.random.uniform(-0.07, -0.03), 0.0])
    ee_pos = np.array([0.0, 0.0, 0.25])

    trajectory = policy.generate_trajectory(ee_pos, target_pos, occluder_pos)

    ep_rgb, ep_depth, ep_states, ep_actions, ep_rewards, frames = [], [], [], [], [], []

    for action in trajectory:
        imgs = env_mgr.get_images(obs)
        for k, v in imgs.items():
            if "rgb" in k:
                frame = v[0] if v.ndim == 4 else v
                ep_rgb.append(frame)
                if save_video and "base" in k:
                    frames.append(frame)
            elif "depth" in k:
                ep_depth.append(v[0] if v.ndim == 4 else v)

        st = env_mgr.get_state(obs)
        if st is not None:
            ep_states.append(st[0] if st.ndim > 1 else st)

        ep_actions.append(action)
        action_t = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        obs, reward, terminated, truncated, info = env_mgr.step(action_t)
        r = reward.item() if isinstance(reward, torch.Tensor) else float(reward)
        ep_rewards.append(r)

        if terminated or truncated:
            break

    success = False
    if isinstance(info, dict) and "success" in info:
        s = info["success"]
        success = bool(s.item() if isinstance(s, torch.Tensor) else s)

    return {
        "rgb": np.array(ep_rgb) if ep_rgb else np.array([]),
        "depth": np.array(ep_depth) if ep_depth else np.array([]),
        "states": np.array(ep_states) if ep_states else np.array([]),
        "actions": np.array(ep_actions),
        "rewards": np.array(ep_rewards),
        "language": "push the occluding object aside, then pick up the target object",
        "seed": seed, "num_steps": len(ep_actions), "success": success,
    }, success, frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--start_seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(__file__).parent.parent / "data" / "push_grasp_demos"
    vis_dir = Path(__file__).parent.parent / "data" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    env_mgr = PushGraspSceneManager(obs_mode="rgbd", num_envs=1)
    policy = ScriptedPushGraspPolicy()

    total_success, total_steps = 0, 0
    start = time.time()

    for i in range(args.num_episodes):
        seed = args.start_seed + i
        ep, success, frames = collect_episode(env_mgr, policy, seed, args.save_video)

        np.savez_compressed(output_dir / f"episode_{i:04d}.npz", **ep)

        if args.save_video and frames:
            try:
                import imageio
                imageio.mimsave(str(vis_dir / f"episode_{i:04d}.mp4"), frames, fps=10)
            except: pass

        total_success += int(success)
        total_steps += ep["num_steps"]

        if (i+1) % 10 == 0 or i == 0:
            elapsed = time.time() - start
            print(f"  [{i+1}/{args.num_episodes}] steps={ep['num_steps']}, "
                  f"speed={((i+1)/elapsed):.1f} ep/s, ETA={((args.num_episodes-i-1)/((i+1)/elapsed)):.0f}s")

    env_mgr.close()

    meta = {"num_episodes": args.num_episodes, "total_success": total_success,
            "success_rate": total_success/args.num_episodes,
            "avg_steps": total_steps/args.num_episodes,
            "total_time_sec": time.time()-start}
    json.dump(meta, open(output_dir/"metadata.json","w"), indent=2)

    print(f"\n  ✅ 完成! {args.num_episodes} 条轨迹, 平均 {meta['avg_steps']:.1f} 步, 保存于 {output_dir}")

if __name__ == "__main__":
    main()
