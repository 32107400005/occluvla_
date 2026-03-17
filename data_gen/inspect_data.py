"""
OccluVLA - 数据质量检查

用法:
    python inspect_data.py
    python inspect_data.py --data_dir /path/to/demos --save_samples
"""

import os, json, argparse
import numpy as np
from pathlib import Path
from collections import Counter


def inspect_episode(f):
    d = np.load(f, allow_pickle=True)
    r = {'filename': f.name, 'issues': [], 'warnings': []}

    # RGB
    if 'rgb' in d and d['rgb'].size > 0:
        rgb = d['rgb']
        r['rgb_shape'] = list(rgb.shape)
        h, w = rgb.shape[-3], rgb.shape[-2]
        r['resolution'] = f"{h}x{w}"
        if h < 256: r['warnings'].append(f"Low res: {h}x{w}")
        if rgb.max() > 255: r['issues'].append("RGB>255")
        if rgb.max() == 0: r['issues'].append("All black")
    else:
        r['issues'].append("No RGB"); r['rgb_shape'] = 'N/A'; r['resolution'] = '?'

    # Actions
    if 'actions' in d and d['actions'].size > 0:
        act = d['actions']
        r['actions_shape'] = list(act.shape)
        r['num_steps'] = len(act)
        if act.ndim >= 2 and act.shape[-1] != 7:
            r['issues'].append(f"action dim={act.shape[-1]}")
        if np.any(np.isnan(act)): r['issues'].append("NaN actions")
        zero = (np.abs(act[:, :3]).sum(-1) < 1e-6).mean()
        if zero > 0.5: r['warnings'].append(f"Zero-action {zero:.0%}")
    else:
        r['issues'].append("No actions"); r['actions_shape'] = 'N/A'; r['num_steps'] = 0

    # Phases
    if 'phases' in d and d['phases'].size > 0:
        phases = list(np.unique(d['phases']))
        r['phases'] = phases
        if not any('push' in str(p) for p in phases) and not any('grasp' in str(p) for p in phases):
            r['issues'].append("No push or grasp phase")
    else:
        r['warnings'].append("No phases")

    # Language
    if 'language' in d:
        r['language'] = str(d['language'])
    # Success
    if 'success' in d:
        r['success'] = bool(d['success'])
    # Has occluder
    if 'has_occluder' in d:
        r['has_occluder'] = bool(d['has_occluder'])

    r['severity'] = 'ERROR' if r['issues'] else ('WARN' if r['warnings'] else 'OK')
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",
                   default=os.path.expanduser("~/workspace/project/occluvla/data/push_grasp_demos"))
    p.add_argument("--save_samples", action="store_true")
    p.add_argument("--max_show", type=int, default=10)
    args = p.parse_args()

    eps = sorted(Path(args.data_dir).glob("episode_*.npz"))
    if not eps:
        print(f"  No episodes in {args.data_dir}")
        return

    print(f"{'='*60}")
    print(f"  Data inspection: {len(eps)} episodes")
    print(f"  Dir: {args.data_dir}")
    print(f"{'='*60}\n")

    reports = [inspect_episode(f) for f in eps]
    ok = sum(1 for r in reports if r['severity'] == 'OK')
    warn = sum(1 for r in reports if r['severity'] == 'WARN')
    err = sum(1 for r in reports if r['severity'] == 'ERROR')

    for r in reports[:args.max_show]:
        icon = {'OK': '+', 'WARN': '!', 'ERROR': 'X'}[r['severity']]
        print(f"  [{icon}] {r['filename']}: RGB={r.get('rgb_shape','?')} "
              f"steps={r.get('num_steps',0)} "
              f"phases={r.get('phases','?')} "
              f"occ={r.get('has_occluder','?')} "
              f"success={r.get('success','?')}")
        for iss in r['issues']:  print(f"      X {iss}")
        for w in r['warnings']:  print(f"      ! {w}")

    # Stats
    steps = [r['num_steps'] for r in reports if r['num_steps'] > 0]
    successes = [r.get('success', False) for r in reports]
    occ_count = sum(1 for r in reports if r.get('has_occluder', False))

    print(f"\n  OK={ok} WARN={warn} ERROR={err}")
    if steps:
        print(f"  Steps: {np.mean(steps):.0f} +/- {np.std(steps):.0f}")
    print(f"  Success: {sum(successes)}/{len(successes)}")
    print(f"  Push+grasp: {occ_count} | Direct: {len(reports)-occ_count}")

    resolutions = Counter(r.get('resolution', '?') for r in reports)
    print(f"  Resolutions: {dict(resolutions)}")

    # Save samples
    if args.save_samples:
        try:
            from PIL import Image
            sample_dir = Path(args.data_dir) / "sample_frames"
            sample_dir.mkdir(exist_ok=True)
            indices = np.linspace(0, len(eps)-1, min(5, len(eps)), dtype=int)
            for idx in indices:
                d = np.load(eps[idx], allow_pickle=True)
                if 'rgb' not in d or d['rgb'].size == 0: continue
                rgb = d['rgb']
                for fi in np.linspace(0, len(rgb)-1, 5, dtype=int):
                    frame = rgb[fi]
                    if frame.shape[-1] >= 3:
                        Image.fromarray(frame[:,:,:3]).save(
                            str(sample_dir / f"ep{idx:04d}_f{fi:04d}.png"))
            print(f"  Samples saved to {sample_dir}")
        except ImportError:
            print("  pip install Pillow for sample saving")

    meta = Path(args.data_dir) / "metadata.json"
    if meta.exists():
        m = json.load(open(meta))
        print(f"\n  metadata: {json.dumps(m, indent=2)}")


if __name__ == "__main__":
    main()
