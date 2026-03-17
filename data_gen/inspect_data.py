"""
OccluVLA - 数据质量检查 (OccluBench 双相机版本)

检查每个相机的 RGB/depth 完整性，验证 action 维度，帧数一致性。

用法:
    python inspect_data.py --data_dir /path/to/demos
    python inspect_data.py --data_dir /path/to/demos --save_samples
"""

import os, json, argparse
import numpy as np
from pathlib import Path
from collections import Counter


EXPECTED_CAMS = ["shoulder_camera", "hand_camera"]


def inspect_episode(f):
    d = np.load(f, allow_pickle=True)
    r = {'filename': f.name, 'issues': [], 'warnings': [], 'cameras': {}}

    # ── 相机列表 ─────────────────────────────────────────────────
    if 'camera_names' in d:
        cams = str(d['camera_names']).split(',')
    else:
        cams = list(set(
            k.rsplit('_rgb', 1)[0] for k in d.files if k.endswith('_rgb')
        ))
        if not cams:
            cams = ['base_camera'] if 'rgb' in d else []

    r['camera_list'] = cams

    # ── 每个相机 ─────────────────────────────────────────────────
    for cam in cams:
        rgb_key = f'{cam}_rgb'
        depth_key = f'{cam}_depth'
        cr = {}

        if rgb_key in d and d[rgb_key].size > 0:
            rgb = d[rgb_key]
            cr['rgb_shape'] = list(rgb.shape)
            h, w = rgb.shape[-3], rgb.shape[-2]
            cr['resolution'] = f"{h}x{w}"
            cr['num_frames'] = rgb.shape[0]
            if h < 128: r['warnings'].append(f"{cam}: low res {h}x{w}")
            if rgb.max() > 255: r['issues'].append(f"{cam}: RGB>255")
            if rgb.max() == 0: r['issues'].append(f"{cam}: all black")
        else:
            r['issues'].append(f"{cam}: no RGB data")
            cr['rgb_shape'] = 'N/A'
            cr['num_frames'] = 0

        if depth_key in d and d[depth_key].size > 0:
            cr['depth_shape'] = list(d[depth_key].shape)
        else:
            r['warnings'].append(f"{cam}: no depth")
            cr['depth_shape'] = 'N/A'

        r['cameras'][cam] = cr

    for expected in EXPECTED_CAMS:
        if expected not in cams:
            r['warnings'].append(f"Missing expected camera: {expected}")

    # ── Actions ──────────────────────────────────────────────────
    if 'actions' in d and d['actions'].size > 0:
        act = d['actions']
        r['actions_shape'] = list(act.shape)
        r['num_steps'] = len(act)
        if act.ndim >= 2 and act.shape[-1] != 7:
            r['issues'].append(f"action dim={act.shape[-1]}, expected 7")
        if np.any(np.isnan(act)): r['issues'].append("NaN in actions")
        zero = (np.abs(act[:, :3]).sum(-1) < 1e-6).mean()
        if zero > 0.5: r['warnings'].append(f"Zero-action {zero:.0%}")
    else:
        r['issues'].append("No actions")
        r['actions_shape'] = 'N/A'
        r['num_steps'] = 0

    # 帧/action 一致性
    for cam in cams:
        nf = r['cameras'].get(cam, {}).get('num_frames', 0)
        if nf > 0 and r['num_steps'] > 0 and abs(nf - r['num_steps']) > 2:
            r['warnings'].append(
                f"{cam}: frame/action mismatch ({nf} vs {r['num_steps']})")

    # ── 元信息 ───────────────────────────────────────────────────
    if 'phases' in d and d['phases'].size > 0:
        r['phases'] = list(np.unique(d['phases']))
    else:
        r['warnings'].append("No phases")
        r['phases'] = []

    r['language'] = str(d['language']) if 'language' in d else ''
    r['success'] = bool(d['success']) if 'success' in d else None
    r['has_occluder'] = bool(d['has_occluder']) if 'has_occluder' in d else None
    r['num_occluders'] = int(d['num_occluders']) if 'num_occluders' in d else None

    r['severity'] = 'ERROR' if r['issues'] else ('WARN' if r['warnings'] else 'OK')
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",
                   default=os.path.expanduser(
                       "~/workspace/project/occluvla/data/push_grasp_level1"))
    p.add_argument("--save_samples", action="store_true")
    p.add_argument("--max_show", type=int, default=10)
    args = p.parse_args()

    eps = sorted(Path(args.data_dir).glob("episode_*.npz"))
    if not eps:
        print(f"  No episodes in {args.data_dir}")
        return

    print(f"{'='*60}")
    print(f"  OccluBench Data Inspection: {len(eps)} episodes")
    print(f"  Dir: {args.data_dir}")
    print(f"{'='*60}\n")

    reports = [inspect_episode(f) for f in eps]
    ok   = sum(1 for r in reports if r['severity'] == 'OK')
    warn = sum(1 for r in reports if r['severity'] == 'WARN')
    err  = sum(1 for r in reports if r['severity'] == 'ERROR')

    for r in reports[:args.max_show]:
        icon = {'OK': '+', 'WARN': '!', 'ERROR': 'X'}[r['severity']]
        cam_summary = " | ".join(
            f"{c}: {r['cameras'][c].get('num_frames',0)}f "
            f"{r['cameras'][c].get('resolution','?')}"
            for c in r['camera_list'] if c in r['cameras']
        )
        print(f"  [{icon}] {r['filename']}: steps={r.get('num_steps',0)} "
              f"occ={r.get('num_occluders','?')} "
              f"success={r.get('success','?')}")
        print(f"       Cameras: {cam_summary}")
        for iss in r['issues']:  print(f"       X {iss}")
        for w in r['warnings']:  print(f"       ! {w}")

    # ── 汇总 ─────────────────────────────────────────────────────
    print(f"\n  Summary: OK={ok} WARN={warn} ERROR={err}")

    steps = [r['num_steps'] for r in reports if r['num_steps'] > 0]
    successes = [r.get('success', False) for r in reports]
    occ_count = sum(1 for r in reports if r.get('has_occluder', False))

    if steps:
        print(f"  Steps: mean={np.mean(steps):.0f} ± {np.std(steps):.0f} "
              f"(range {min(steps)}-{max(steps)})")
    print(f"  Success: {sum(successes)}/{len(successes)}")
    print(f"  Push+grasp: {occ_count} | Direct: {len(reports)-occ_count}")

    for cam in EXPECTED_CAMS:
        resolutions = Counter(
            r['cameras'].get(cam, {}).get('resolution', 'missing')
            for r in reports
        )
        print(f"  {cam} resolutions: {dict(resolutions)}")

    # ── 样本帧 ───────────────────────────────────────────────────
    if args.save_samples:
        try:
            from PIL import Image
            sample_dir = Path(args.data_dir) / "sample_frames"
            sample_dir.mkdir(exist_ok=True)
            indices = np.linspace(0, len(eps) - 1, min(5, len(eps)), dtype=int)
            for idx in indices:
                d = np.load(eps[idx], allow_pickle=True)
                for cam in EXPECTED_CAMS:
                    key = f'{cam}_rgb'
                    if key not in d or d[key].size == 0: continue
                    rgb = d[key]
                    for fi in np.linspace(0, len(rgb) - 1, 5, dtype=int):
                        frame = rgb[fi]
                        if frame.shape[-1] >= 3:
                            Image.fromarray(frame[:, :, :3]).save(
                                str(sample_dir / f"ep{idx:04d}_{cam}_f{fi:04d}.png"))
            print(f"  Samples saved to {sample_dir}")
        except ImportError:
            print("  pip install Pillow for sample saving")

    meta = Path(args.data_dir) / "metadata.json"
    if meta.exists():
        m = json.load(open(meta))
        print(f"\n  metadata: {json.dumps(m, indent=2)}")


if __name__ == "__main__":
    main()
