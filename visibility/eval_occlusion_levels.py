"""
Test GroundingDINO + SAM2 on ManiSkill3 rendered images
at different occlusion levels (0%, 25%, 50%, 75%, 100%)
"""
import os, sys, json, time
import numpy as np
from pathlib import Path
from PIL import Image

import gymnasium as gym
import mani_skill.envs
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visibility_assessor import VisibilityAssessor


def render_scenes(n=30):
    """Render scenes from ManiSkill3 using correct obs keys"""
    print("  Rendering ManiSkill3 scenes...")
    env = gym.make("PickClutterYCB-v1", obs_mode="rgbd",
                    render_mode="rgb_array", num_envs=1)
    scenes = []
    for i in range(n):
        obs, _ = env.reset(seed=i*7)
        # Correct key: sensor_data -> base_camera -> rgb
        rgb = obs["sensor_data"]["base_camera"]["rgb"][0].cpu().numpy()
        rgb = (rgb*255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
        depth = obs["sensor_data"]["base_camera"]["depth"][0].cpu().numpy()
        scenes.append({"rgb": rgb, "depth": depth, "seed": i*7})
    env.close()
    print(f"  ✓ Rendered {len(scenes)} scenes")
    return scenes


def test_basic_detection(assessor, scenes):
    """Test basic detection ability on clean scenes"""
    targets = ["cup", "bowl", "bottle", "box", "can", "object"]
    detected, total, confs = 0, 0, []

    for i, s in enumerate(scenes[:15]):
        for t in targets[:3]:
            total += 1
            try:
                r = assessor.assess(s["rgb"], t)
                confs.append(r["target_confidence"])
                if r["target_visible"]: detected += 1
            except Exception as e:
                print(f"  ⚠ scene {i}, target '{t}': {e}")
        if (i+1) % 5 == 0:
            print(f"    [{i+1}/15] detection rate: {detected/total:.1%}")

    return {"total": total, "detected": detected,
            "rate": detected/max(total,1),
            "avg_conf": np.mean(confs) if confs else 0}


def test_occlusion_levels(assessor, scenes):
    """Simulate occlusion by covering parts of image"""
    levels = [0.0, 0.25, 0.50, 0.75, 1.0]
    results = {}

    for level in levels:
        detected, total, confs = 0, 0, []
        for s in scenes[:20]:
            rgb = s["rgb"].copy()
            H, W = rgb.shape[:2]

            if level > 0:
                ch, cw = H//2, W//2
                half = int(min(H,W)*0.3)
                rows = int(half*2*level)
                color = np.random.randint(80, 180, 3, dtype=np.uint8)
                r0 = max(0, ch-half)
                r1 = min(H, r0+rows)
                c0 = max(0, cw-half)
                c1 = min(W, cw+half)
                rgb[r0:r1, c0:c1, :] = color

            total += 1
            try:
                r = assessor.assess(rgb, "object")
                if r["target_visible"]: detected += 1
                confs.append(r["target_confidence"])
            except: pass

        rate = detected/max(total,1)
        avg_c = np.mean(confs) if confs else 0
        results[f"{int(level*100)}%"] = {
            "occlusion": level, "detected": detected,
            "total": total, "rate": rate, "avg_conf": avg_c
        }
        bar = "█" * int(rate*20)
        print(f"  Occlusion {int(level*100):>3}%: {bar:<20} {rate:.1%} (conf={avg_c:.3f})")

    return results


def main():
    out = Path(os.path.expanduser("~/workspace/project/occluvla/data/visibility_eval"))
    out.mkdir(parents=True, exist_ok=True)

    scenes = render_scenes(30)

    # Save sample images
    for i, s in enumerate(scenes[:5]):
        Image.fromarray(s["rgb"]).save(out / f"sample_{i:03d}.png")
    print(f"  Sample images saved to {out}")

    print("\n  Initializing VisibilityAssessor...")
    assessor = VisibilityAssessor()

    print("\n  === Test 1: Basic Detection ===")
    basic = test_basic_detection(assessor, scenes)
    print(f"  Detection rate: {basic['rate']:.1%}, Avg conf: {basic['avg_conf']:.3f}")

    print("\n  === Test 2: Occlusion Levels ===")
    occ = test_occlusion_levels(assessor, scenes)

    # Example report
    print("\n  === Example Visibility Report ===")
    r = assessor.assess(scenes[0]["rgb"], "cup")
    print(assessor.format_report(r, "cup"))

    # Save results
    results = {"basic": basic, "occlusion_levels": occ,
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(results, open(out/"visibility_results.json","w"), indent=2, default=str)

    print(f"\n  Results saved: {out}/visibility_results.json")
    print("\n" + "="*50)
    print("  ✅ Visibility Module Prototyping 完成!")
    print("="*50)

if __name__ == "__main__":
    main()
