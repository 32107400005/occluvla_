"""检查生成的 demo 数据质量"""
import os, json, argparse
import numpy as np
from pathlib import Path

def inspect(f):
    d = np.load(f, allow_pickle=True)
    issues = []
    if "rgb" in d and d["rgb"].size > 0:
        if d["rgb"].max() > 255: issues.append("RGB>255")
    else: issues.append("RGB empty")
    if "actions" in d and d["actions"].size > 0:
        if d["actions"].shape[-1] != 8: issues.append(f"action dim={d['actions'].shape[-1]}")
        if np.any(np.isnan(d["actions"])): issues.append("NaN actions")
    return {"rgb": d["rgb"].shape if "rgb" in d and d["rgb"].size>0 else "N/A",
            "actions": d["actions"].shape if "actions" in d else "N/A",
            "steps": len(d["actions"]) if "actions" in d else 0,
            "issues": issues}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=os.path.expanduser("~/workspace/project/occluvla/data/push_grasp_demos"))
    args = p.parse_args()
    eps = sorted(Path(args.data_dir).glob("episode_*.npz"))
    if not eps: print("  ✗ No episodes found!"); return

    print(f"  Found {len(eps)} episodes\n")
    issues_total = 0
    for f in eps[:5]:
        info = inspect(f)
        print(f"  {f.name}: RGB={info['rgb']}, Actions={info['actions']}, Steps={info['steps']}")
        if info["issues"]: print(f"    ⚠ {info['issues']}"); issues_total += len(info["issues"])
        else: print(f"    ✓ OK")

    steps = [inspect(f)["steps"] for f in eps]
    print(f"\n  Avg steps: {np.mean(steps):.1f} ± {np.std(steps):.1f}")
    print(f"  Total issues: {issues_total}")
    if issues_total == 0: print("  ✅ All data OK!")

if __name__ == "__main__":
    main()
