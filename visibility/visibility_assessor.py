"""
OccluVLA - Visibility Assessor
GroundingDINO + SAM2 based target visibility assessment
"""
import os, sys
import numpy as np
import torch
from PIL import Image
from typing import Dict, Optional

from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class VisibilityAssessor:
    def __init__(self, gdino_config=None, gdino_weights=None,
                 sam2_config=None, sam2_weights=None,
                 device="cuda", box_threshold=0.25, text_threshold=0.25):
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        home = os.path.expanduser("~")

        # GroundingDINO paths - try groundingdino-py package first
        if gdino_config is None:
            # groundingdino-py installs config to package directory
            import groundingdino
            pkg_dir = os.path.dirname(groundingdino.__file__)
            gdino_config = os.path.join(pkg_dir, "config", "GroundingDINO_SwinT_OGC.py")
            if not os.path.exists(gdino_config):
                # fallback to source install
                gdino_config = f"{home}/workspace/project/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        if gdino_weights is None:
            gdino_weights = f"{home}/workspace/models/groundingdino/groundingdino_swint_ogc.pth"
        if sam2_config is None:
            sam2_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
        if sam2_weights is None:
            sam2_weights = f"{home}/workspace/models/sam2/sam2.1_hiera_large.pt"

        print("  Loading GroundingDINO...")
        self.gdino_model = load_model(gdino_config, gdino_weights)
        print("  ✓ GroundingDINO loaded")

        print("  Loading SAM2...")
        sam2 = build_sam2(sam2_config, sam2_weights)
        self.sam2_predictor = SAM2ImagePredictor(sam2)
        print("  ✓ SAM2 loaded")

        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def assess(self, rgb, target_name, depth=None):
        H, W = rgb.shape[:2]
        image_pil = Image.fromarray(rgb)
        image_tensor, _ = self.transform(image_pil, None)

        # Detect target
        boxes, logits, phrases = predict(
            model=self.gdino_model, image=image_tensor,
            caption=target_name,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )

        # Detect all objects
        all_boxes, all_logits, all_phrases = predict(
            model=self.gdino_model, image=image_tensor,
            caption="object . box . cup . bottle . bowl . can",
            box_threshold=0.2, text_threshold=0.2,
        )

        all_dets = [{"name": all_phrases[i], "confidence": float(all_logits[i]),
                      "bbox": (all_boxes[i].cpu().numpy()*[W,H,W,H]).tolist()}
                     for i in range(len(all_boxes))]

        if len(boxes) == 0:
            return {"target_visible": False, "target_confidence": 0.0,
                    "target_bbox": None, "target_mask": None,
                    "visible_area_ratio": 0.0,
                    "occluders": [d["name"] for d in all_dets[:3]],
                    "all_detections": all_dets}

        best_idx = logits.argmax().item()
        best_conf = float(logits[best_idx])
        best_box = boxes[best_idx].cpu().numpy() * np.array([W, H, W, H])

        if best_conf < 0.3:
            return {"target_visible": False, "target_confidence": best_conf,
                    "target_bbox": best_box.tolist(), "target_mask": None,
                    "visible_area_ratio": 0.0,
                    "occluders": [d["name"] for d in all_dets[:3]],
                    "all_detections": all_dets}

        # SAM2 segmentation
        self.sam2_predictor.set_image(rgb)
        masks, scores, _ = self.sam2_predictor.predict(box=best_box, multimask_output=False)
        mask = masks[0]

        visible_pixels = mask.sum()
        bbox_area = (best_box[2]-best_box[0]) * (best_box[3]-best_box[1])
        visible_ratio = min(1.0, visible_pixels / max(bbox_area * 0.65, 1))

        return {"target_visible": True, "target_confidence": best_conf,
                "target_bbox": best_box.tolist(), "target_mask": mask,
                "visible_area_ratio": float(visible_ratio),
                "occluders": [], "all_detections": all_dets}

    def format_report(self, assessment, target_name):
        lines = ["Visibility Report:"]
        lines.append(f"- Target: {target_name} | Visible: {'YES' if assessment['target_visible'] else 'NO'} | Conf: {assessment['target_confidence']:.2f}")
        if assessment["target_visible"]:
            lines.append(f"- Visible area: {assessment['visible_area_ratio']:.1%}")
        if assessment["occluders"]:
            lines.append(f"- Potential occluders: {', '.join(assessment['occluders'][:3])}")
        lines.append(f"- Total objects detected: {len(assessment['all_detections'])}")
        return "\n".join(lines)
