"""
OccluVLA - 数据质量检查 (修正版)

检查内容:
1. RGB 图像: shape, dtype, value range, 分辨率
2. Depth 图像: shape, NaN/Inf 检查
3. Actions: 维度, NaN, 范围合理性
4. 轨迹: 步数分布, phase 覆盖
5. 可视化: 保存采样帧以便人工检查

用法:
    python inspect_data.py
    python inspect_data.py --data_dir /path/to/demos --save_samples
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path


def inspect_episode(filepath):
    """检查单条轨迹的数据质量"""
    d = np.load(filepath, allow_pickle=True)
    report = {
        "filename": filepath.name,
        "issues": [],
        "warnings": [],
    }

    # ── RGB 检查 ─────────────────────────────────────────────────────
    if "rgb" in d and d["rgb"].size > 0:
        rgb = d["rgb"]
        report["rgb_shape"] = list(rgb.shape)
        report["rgb_dtype"] = str(rgb.dtype)

        # 检查分辨率
        if rgb.ndim >= 3:
            h, w = rgb.shape[-3], rgb.shape[-2]
            if h < 256 or w < 256:
                report["warnings"].append(
                    f"Low resolution: {h}x{w} (recommend >=512x512)"
                )
            report["resolution"] = f"{h}x{w}"

        # 检查值范围
        if rgb.max() > 255:
            report["issues"].append(f"RGB max={rgb.max()} > 255")
        if rgb.min() < 0:
            report["issues"].append(f"RGB min={rgb.min()} < 0")
        if rgb.max() <= 1.0 and rgb.dtype in [np.float32, np.float64]:
            report["warnings"].append("RGB in [0,1] float range (expected uint8)")

        # 检查是否全黑/全白
        if rgb.max() == 0:
            report["issues"].append("RGB is all black (rendering failed?)")
        if rgb.min() == 255:
            report["issues"].append("RGB is all white")
    else:
        report["issues"].append("RGB data missing or empty")
        report["rgb_shape"] = "N/A"

    # ── Depth 检查 ───────────────────────────────────────────────────
    if "depth" in d and d["depth"].size > 0:
        depth = d["depth"]
        report["depth_shape"] = list(depth.shape)
        if np.any(np.isnan(depth)):
            report["warnings"].append(
                f"Depth has NaN ({np.isnan(depth).sum()} values)"
            )
        if np.any(np.isinf(depth)):
            report["warnings"].append(
                f"Depth has Inf ({np.isinf(depth).sum()} values)"
            )
        report["depth_range"] = f"[{np.nanmin(depth):.3f}, {np.nanmax(depth):.3f}]"
    else:
        report["warnings"].append("Depth data missing or empty")
        report["depth_shape"] = "N/A"

    # ── Actions 检查 ─────────────────────────────────────────────────
    if "actions" in d and d["actions"].size > 0:
        actions = d["actions"]
        report["actions_shape"] = list(actions.shape)
        report["num_steps"] = len(actions)

        # 维度检查
        if actions.ndim >= 2 and actions.shape[-1] != 7:
            report["issues"].append(
                f"Action dim={actions.shape[-1]} (expected 7 for ee_delta_pose)"
            )

        # NaN 检查
        if np.any(np.isnan(actions)):
            report["issues"].append(
                f"Actions contain NaN ({np.isnan(actions).sum()} values)"
            )

        # 范围检查 (delta pose 通常很小)
        max_abs = np.abs(actions[:, :3]).max() if actions.ndim >= 2 else 0
        if max_abs > 0.5:
            report["warnings"].append(
                f"Large position delta: max |action[:3]|={max_abs:.3f}"
            )

        # 检查 gripper 值
        if actions.ndim >= 2 and actions.shape[-1] >= 7:
            gripper_vals = np.unique(actions[:, 6])
            report["gripper_values"] = [float(v) for v in gripper_vals]

        # 检查 action 是否全零 (可能 policy 没有正确计算)
        zero_ratio = (np.abs(actions[:, :3]).sum(axis=-1) < 1e-6).mean()
        if zero_ratio > 0.5:
            report["warnings"].append(
                f"High zero-action ratio: {zero_ratio:.1%} (policy not moving?)"
            )
    else:
        report["issues"].append("Actions data missing or empty")
        report["actions_shape"] = "N/A"
        report["num_steps"] = 0

    # ── Phase 检查 ───────────────────────────────────────────────────
    if "phases" in d and d["phases"].size > 0:
        phases = d["phases"]
        unique_phases = np.unique(phases)
        report["phases"] = list(unique_phases)

        has_push  = any("push" in str(p) for p in unique_phases)
        has_grasp = any("grasp" in str(p) for p in unique_phases)
        has_lift  = any("lift" in str(p) for p in unique_phases)

        if not has_push:
            report["issues"].append("No 'push' phase found")
        if not has_grasp:
            report["issues"].append("No 'grasp' phase found")
        if not has_lift:
            report["warnings"].append("No 'lift' phase found (may have terminated early)")

    # ── States 检查 ──────────────────────────────────────────────────
    if "states" in d and d["states"].size > 0:
        states = d["states"]
        report["states_shape"] = list(states.shape)
        if np.any(np.isnan(states)):
            report["issues"].append("States contain NaN")
    else:
        report["warnings"].append("States data missing or empty")

    # ── 其他元数据 ───────────────────────────────────────────────────
    if "language" in d:
        lang = str(d["language"])
        report["language"] = lang
        if len(lang) < 10:
            report["warnings"].append(f"Language too short: '{lang}'")

    if "success" in d:
        report["success"] = bool(d["success"])

    if "seed" in d:
        report["seed"] = int(d["seed"])

    report["severity"] = "ERROR" if report["issues"] else (
        "WARN" if report["warnings"] else "OK"
    )

    return report


def save_sample_frames(filepath, output_dir, num_frames=5):
    """从一条轨迹中保存采样帧, 用于人工目视检查"""
    try:
        from PIL import Image
    except ImportError:
        print("  [!] 需要 Pillow 来保存采样帧: pip install Pillow")
        return

    d = np.load(filepath, allow_pickle=True)
    if "rgb" not in d or d["rgb"].size == 0:
        return

    rgb = d["rgb"]
    total = len(rgb)
    if total == 0:
        return

    # 均匀采样帧
    indices = np.linspace(0, total - 1, num_frames, dtype=int)

    ep_name = filepath.stem
    frame_dir = output_dir / "sample_frames" / ep_name
    frame_dir.mkdir(parents=True, exist_ok=True)

    for idx in indices:
        frame = rgb[idx]
        if frame.ndim == 2:  # grayscale
            img = Image.fromarray(frame, mode="L")
        elif frame.shape[-1] == 3:
            img = Image.fromarray(frame, mode="RGB")
        elif frame.shape[-1] == 4:
            img = Image.fromarray(frame, mode="RGBA")
        else:
            continue
        img.save(frame_dir / f"frame_{idx:04d}.png")

    print(f"    Saved {len(indices)} sample frames to {frame_dir}")


def main():
    parser = argparse.ArgumentParser(description="OccluVLA data quality inspection")
    parser.add_argument(
        "--data_dir",
        default=os.path.expanduser(
            "~/workspace/project/occluvla/data/push_grasp_demos"
        ),
    )
    parser.add_argument(
        "--save_samples", action="store_true",
        help="Save sample RGB frames for visual inspection",
    )
    parser.add_argument(
        "--max_display", type=int, default=10,
        help="Max episodes to display in detail",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    episodes = sorted(data_dir.glob("episode_*.npz"))

    if not episodes:
        print(f"  ✗ No episodes found in {data_dir}")
        print(f"    请先运行: python generate_demos.py --num_episodes 100 --save_video")
        return

    print("=" * 60)
    print(f"  OccluVLA: Data Quality Inspection")
    print(f"  Directory: {data_dir}")
    print(f"  Episodes:  {len(episodes)}")
    print("=" * 60)
    print()

    # ── 逐条检查 ─────────────────────────────────────────────────────
    all_reports = []
    error_count = 0
    warn_count  = 0
    ok_count    = 0

    for f in episodes:
        report = inspect_episode(f)
        all_reports.append(report)

        if report["severity"] == "ERROR":
            error_count += 1
        elif report["severity"] == "WARN":
            warn_count += 1
        else:
            ok_count += 1

    # ── 详细输出 (前 N 条) ───────────────────────────────────────────
    print(f"[详细检查] 前 {min(args.max_display, len(all_reports))} 条:\n")

    for report in all_reports[:args.max_display]:
        severity_icon = {"OK": "✓", "WARN": "⚠", "ERROR": "✗"}[report["severity"]]
        print(f"  {severity_icon} {report['filename']}:")
        print(f"      RGB:     {report.get('rgb_shape', 'N/A')}  "
              f"({report.get('resolution', '?')})")
        print(f"      Actions: {report.get('actions_shape', 'N/A')}  "
              f"({report.get('num_steps', 0)} steps)")
        print(f"      Depth:   {report.get('depth_shape', 'N/A')}  "
              f"{report.get('depth_range', '')}")

        if "phases" in report:
            print(f"      Phases:  {report['phases']}")
        if "gripper_values" in report:
            print(f"      Gripper: {report['gripper_values']}")
        if "success" in report:
            print(f"      Success: {report['success']}")

        if report["issues"]:
            for issue in report["issues"]:
                print(f"      ❌ {issue}")
        if report["warnings"]:
            for warn in report["warnings"]:
                print(f"      ⚠️  {warn}")

        if not report["issues"] and not report["warnings"]:
            print(f"      ✅ All checks passed")
        print()

    # ── 统计汇总 ─────────────────────────────────────────────────────
    steps = [r["num_steps"] for r in all_reports if r["num_steps"] > 0]

    print("=" * 60)
    print(f"  汇总:")
    print(f"    OK:      {ok_count} / {len(all_reports)}")
    print(f"    Warning: {warn_count} / {len(all_reports)}")
    print(f"    Error:   {error_count} / {len(all_reports)}")
    print()

    if steps:
        print(f"    Steps:   {np.mean(steps):.1f} ± {np.std(steps):.1f}  "
              f"(min={np.min(steps)}, max={np.max(steps)})")

    # 成功率
    successes = [r.get("success", False) for r in all_reports]
    success_rate = sum(successes) / len(successes) if successes else 0
    print(f"    Success: {sum(successes)} / {len(successes)} = {success_rate:.1%}")

    # 分辨率统计
    resolutions = [r.get("resolution", "?") for r in all_reports if "resolution" in r]
    if resolutions:
        from collections import Counter
        res_counts = Counter(resolutions)
        print(f"    Resolution: {dict(res_counts)}")

    # 常见问题汇总
    all_issues = []
    for r in all_reports:
        all_issues.extend(r["issues"])
    if all_issues:
        from collections import Counter
        issue_counts = Counter(all_issues)
        print(f"\n    常见问题:")
        for issue, count in issue_counts.most_common(5):
            print(f"      [{count}x] {issue}")

    print("=" * 60)

    # ── 保存采样帧 ───────────────────────────────────────────────────
    if args.save_samples:
        print("\n[保存] 采样帧 (用于目视检查)...")
        sample_indices = np.linspace(0, len(episodes) - 1, min(5, len(episodes)),
                                      dtype=int)
        for idx in sample_indices:
            save_sample_frames(episodes[idx], data_dir)

    # ── 加载 metadata.json (如果存在) ────────────────────────────────
    meta_path = data_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"\n  metadata.json:")
        for k, v in meta.items():
            if k != "failed_seeds":
                print(f"    {k}: {v}")
        if meta.get("failed_seeds"):
            print(f"    failed_seeds: {meta['failed_seeds'][:10]}...")

    # ── 最终判断 ─────────────────────────────────────────────────────
    print()
    if error_count == 0 and warn_count == 0:
        print("  ✅ 所有数据检查通过!")
    elif error_count == 0:
        print(f"  ⚠️  数据基本可用, 但有 {warn_count} 条警告, 建议检查")
    else:
        print(f"  ❌ 有 {error_count} 条严重错误, 请修复后重新生成")


if __name__ == "__main__":
    main()
