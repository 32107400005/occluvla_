#!/usr/bin/env python3
"""
OccluVLA - Step 0: Environment Diagnostic Script
=================================================
在写任何 scripted policy 之前，先运行这个脚本搞清楚：
  1. obs 里有哪些 key，shape 是什么
  2. 怎么拿到 target 物体的 pose
  3. 怎么拿到所有物体的 pose（occluder, distractors）
  4. 怎么拿到 EE (end-effector) 的位置
  5. action space 的维度和含义
  6. 相机有哪些，分辨率是多少
  7. 渲染出来的图片长什么样

用法:
    conda activate occluvla-sim
    python 0_diagnose_env.py
    python 0_diagnose_env.py --env PickSingleYCB-v1
    python 0_diagnose_env.py --env PickClutterYCB-v1 --save_images
"""

import argparse
import gymnasium as gym
import numpy as np
import torch
import json
from pathlib import Path

import mani_skill.envs  # 注册所有 ManiSkill 环境


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.asarray(x)


def inspect_dict(d, prefix="", depth=0, max_depth=5):
    """递归打印 dict/tensor 结构"""
    if depth > max_depth:
        print(f"{'  ' * depth}{prefix}: ... (max depth reached)")
        return
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            inspect_dict(v, key, depth + 1, max_depth)
    elif isinstance(d, (torch.Tensor, np.ndarray)):
        arr = to_numpy(d)
        print(f"{'  ' * depth}{prefix}: shape={arr.shape}, dtype={arr.dtype}, "
              f"range=[{arr.min():.4f}, {arr.max():.4f}]")
    elif isinstance(d, (list, tuple)):
        print(f"{'  ' * depth}{prefix}: {type(d).__name__}, len={len(d)}")
    else:
        val_str = str(d)
        if len(val_str) > 80:
            val_str = val_str[:80] + "..."
        print(f"{'  ' * depth}{prefix}: {type(d).__name__} = {val_str}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="PickClutterYCB-v1",
                        help="Environment ID")
    parser.add_argument("--obs_mode", default="rgbd",
                        help="Observation mode: state_dict, rgbd, state_dict+rgbd")
    parser.add_argument("--control_mode", default="pd_ee_delta_pose",
                        help="Control mode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_images", action="store_true",
                        help="Save camera images as PNG files")
    parser.add_argument("--output_dir", default="./diagnostic_output")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  OccluVLA Environment Diagnostic")
    print(f"  Env:          {args.env}")
    print(f"  Obs mode:     {args.obs_mode}")
    print(f"  Control mode: {args.control_mode}")
    print(f"  Seed:         {args.seed}")
    print("=" * 70)

    # ══════════════════════════════════════════════════════════════════
    # 1. 创建环境
    # ══════════════════════════════════════════════════════════════════
    print("\n[1] Creating environment...")
    env = gym.make(
        args.env,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render_mode="rgb_array",
        num_envs=1,
        max_episode_steps=200,
    )

    print(f"  Action space: {env.action_space}")
    print(f"  Action shape: {env.action_space.shape}")
    print(f"  Action range: [{env.action_space.low.min():.2f}, {env.action_space.high.max():.2f}]")

    # ══════════════════════════════════════════════════════════════════
    # 2. Reset 并检查 obs 结构
    # ══════════════════════════════════════════════════════════════════
    print("\n[2] Resetting environment...")
    obs, info = env.reset(seed=args.seed)

    print("\n  ── obs structure ──")
    inspect_dict(obs)

    print("\n  ── info structure ──")
    inspect_dict(info)

    # ══════════════════════════════════════════════════════════════════
    # 3. 访问 env.unwrapped 获取内部状态
    # ══════════════════════════════════════════════════════════════════
    print("\n[3] Inspecting env.unwrapped...")
    base_env = env.unwrapped

    # 3a. 目标物体
    print("\n  ── Target object ──")
    if hasattr(base_env, 'obj'):
        obj = base_env.obj
        pose = obj.pose
        p = to_numpy(pose.p)
        q = to_numpy(pose.q)
        print(f"  env.unwrapped.obj exists!")
        print(f"  Name: {obj.name}")
        print(f"  Pose.p (position): {p}")
        print(f"  Pose.q (quaternion): {q}")
    else:
        print("  env.unwrapped.obj does NOT exist")

    # 3b. 所有物体 (PickClutterYCB 可能有 .objects 列表)
    print("\n  ── All objects ──")
    for attr_name in ['objects', 'objs', 'all_objects', 'actors',
                       'object_list', 'ycb_objects']:
        if hasattr(base_env, attr_name):
            objs = getattr(base_env, attr_name)
            print(f"  env.unwrapped.{attr_name} exists! type={type(objs).__name__}")
            if isinstance(objs, (list, tuple)):
                for i, o in enumerate(objs):
                    try:
                        p = to_numpy(o.pose.p)
                        print(f"    [{i}] name={o.name}, pos={p}")
                    except Exception as e:
                        print(f"    [{i}] {type(o).__name__}: {e}")
            elif hasattr(objs, '__len__'):
                print(f"    Length: {len(objs)}")
            break
    else:
        print("  No standard object list attribute found")
        print("  Searching all attributes...")
        for attr in sorted(dir(base_env)):
            if attr.startswith('_'):
                continue
            try:
                val = getattr(base_env, attr)
                if hasattr(val, 'pose') and hasattr(val, 'name'):
                    p = to_numpy(val.pose.p)
                    print(f"  env.unwrapped.{attr}: name={val.name}, pos={p}")
            except:
                pass

    # 3c. Segmentation map (哪个 ID 对应哪个物体)
    print("\n  ── Segmentation ID map ──")
    if hasattr(base_env, 'segmentation_id_map'):
        from mani_skill.utils.structs import Actor, Link
        for obj_id, obj in sorted(base_env.segmentation_id_map.items()):
            if isinstance(obj, Actor):
                try:
                    p = to_numpy(obj.pose.p)
                    print(f"  ID {obj_id}: Actor '{obj.name}', pos={p}")
                except:
                    print(f"  ID {obj_id}: Actor '{obj.name}'")
            elif isinstance(obj, Link):
                print(f"  ID {obj_id}: Link '{obj.name}'")
    else:
        print("  segmentation_id_map not found")

    # 3d. TCP / End-effector
    print("\n  ── End-effector (TCP) ──")
    try:
        tcp = base_env.agent.tcp
        tcp_pose = tcp.pose
        p = to_numpy(tcp_pose.p)
        print(f"  TCP position: {p}")
    except Exception as e:
        print(f"  Cannot access TCP: {e}")

    # 3e. 尝试从 obs['extra'] 或 obs['agent'] 获取
    print("\n  ── Useful obs keys ──")
    if isinstance(obs, dict):
        for top_key in ['extra', 'agent']:
            if top_key in obs and isinstance(obs[top_key], dict):
                for k, v in obs[top_key].items():
                    arr = to_numpy(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v
                    if isinstance(arr, np.ndarray):
                        print(f"  obs['{top_key}']['{k}']: shape={arr.shape}, "
                              f"sample={arr.flatten()[:6]}")
                    else:
                        print(f"  obs['{top_key}']['{k}']: {arr}")

    # ══════════════════════════════════════════════════════════════════
    # 4. 相机信息
    # ══════════════════════════════════════════════════════════════════
    print("\n[4] Camera information...")
    if isinstance(obs, dict) and 'sensor_data' in obs:
        for cam_name, cam_data in obs['sensor_data'].items():
            if isinstance(cam_data, dict):
                print(f"\n  Camera: '{cam_name}'")
                for k, v in cam_data.items():
                    arr = to_numpy(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v
                    if isinstance(arr, np.ndarray):
                        print(f"    {k}: shape={arr.shape}, dtype={arr.dtype}")
                    else:
                        print(f"    {k}: {type(arr).__name__}")

    if 'sensor_param' in obs:
        for cam_name, params in obs['sensor_param'].items():
            if isinstance(params, dict):
                print(f"\n  Camera params: '{cam_name}'")
                for k, v in params.items():
                    arr = to_numpy(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v
                    if isinstance(arr, np.ndarray):
                        print(f"    {k}: shape={arr.shape}")

    # ══════════════════════════════════════════════════════════════════
    # 5. 保存图像
    # ══════════════════════════════════════════════════════════════════
    if args.save_images:
        print("\n[5] Saving images...")
        try:
            from PIL import Image
        except ImportError:
            print("  pip install Pillow first!")
            return

        # 5a. 从 sensor_data 保存各相机的 RGB
        if isinstance(obs, dict) and 'sensor_data' in obs:
            for cam_name, cam_data in obs['sensor_data'].items():
                if isinstance(cam_data, dict) and 'rgb' in cam_data:
                    rgb = to_numpy(cam_data['rgb'])
                    if rgb.ndim == 4:
                        rgb = rgb[0]  # remove batch dim
                    if rgb.max() <= 1.0:
                        rgb = (rgb * 255).astype(np.uint8)
                    else:
                        rgb = rgb.astype(np.uint8)
                    img = Image.fromarray(rgb[:, :, :3])
                    path = output_dir / f"{cam_name}_rgb.png"
                    img.save(str(path))
                    print(f"  Saved {path} ({rgb.shape})")

                if isinstance(cam_data, dict) and 'depth' in cam_data:
                    depth = to_numpy(cam_data['depth'])
                    if depth.ndim == 4:
                        depth = depth[0]
                    if depth.ndim == 3:
                        depth = depth[:, :, 0]
                    # Normalize for visualization
                    valid = depth[np.isfinite(depth)]
                    if len(valid) > 0:
                        d_min, d_max = valid.min(), valid.max()
                        depth_norm = np.clip((depth - d_min) / (d_max - d_min + 1e-8), 0, 1)
                        depth_vis = (depth_norm * 255).astype(np.uint8)
                        img = Image.fromarray(depth_vis, mode='L')
                        path = output_dir / f"{cam_name}_depth.png"
                        img.save(str(path))
                        print(f"  Saved {path}")

        # 5b. 从 env.render() 保存渲染图
        frame = env.render()
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        if frame is not None and frame.size > 0:
            if frame.ndim == 4:
                frame = frame[0]
            img = Image.fromarray(frame.astype(np.uint8))
            path = output_dir / "render_view.png"
            img.save(str(path))
            print(f"  Saved {path} ({frame.shape})")

    # ══════════════════════════════════════════════════════════════════
    # 6. 测试 action: 做一步 random action 看看变化
    # ══════════════════════════════════════════════════════════════════
    print("\n[6] Testing one random action step...")
    action = env.action_space.sample()
    if isinstance(action, np.ndarray):
        action = torch.tensor(action, dtype=torch.float32)
    print(f"  Action: shape={action.shape}, sample={to_numpy(action).flatten()[:8]}")

    obs2, reward, terminated, truncated, info2 = env.step(action)
    print(f"  Reward: {to_numpy(reward)}")
    print(f"  Terminated: {to_numpy(terminated)}")

    # Check EE moved
    try:
        tcp2 = to_numpy(base_env.agent.tcp.pose.p)
        print(f"  TCP after step: {tcp2}")
    except:
        pass

    # ══════════════════════════════════════════════════════════════════
    # 7. 总结: 写出关键 API 路径
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  SUMMARY: Copy these API paths into your scripts")
    print("=" * 70)

    summary = {}

    # EE position
    try:
        p = to_numpy(base_env.agent.tcp.pose.p)
        summary["ee_pos"] = "env.unwrapped.agent.tcp.pose.p"
        print(f"\n  EE position:  env.unwrapped.agent.tcp.pose.p → {p}")
    except:
        print(f"\n  EE position:  ❌ cannot access")

    # Target object
    if hasattr(base_env, 'obj'):
        p = to_numpy(base_env.obj.pose.p)
        summary["target_pos"] = "env.unwrapped.obj.pose.p"
        summary["target_name"] = base_env.obj.name
        print(f"  Target pos:   env.unwrapped.obj.pose.p → {p}")
        print(f"  Target name:  env.unwrapped.obj.name → {base_env.obj.name}")

    # All objects
    for attr in ['objects', 'objs', 'all_objects']:
        if hasattr(base_env, attr):
            objs = getattr(base_env, attr)
            summary["all_objects"] = f"env.unwrapped.{attr}"
            print(f"  All objects:  env.unwrapped.{attr} → {len(objs)} objects")
            break

    # Action dim
    summary["action_dim"] = int(env.action_space.shape[-1])
    print(f"  Action dim:   {summary['action_dim']}")
    print(f"  Control mode: {args.control_mode}")

    # Save summary
    with open(output_dir / "env_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved to {output_dir / 'env_summary.json'}")

    env.close()
    print("\n  ✅ Diagnostic complete!")
    print(f"  Output: {output_dir}/")
    print(f"\n  Next step: read the summary above, then run the data generation script")


if __name__ == "__main__":
    main()
