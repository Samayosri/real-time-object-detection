"""Build the Faster R-CNN models once and pickle them to disk.

    python save_model.py            # saves both: fast and accurate
    python save_model.py fast       # only the fast one
    python save_model.py accurate   # only the accurate one

app.py loads these files and lets the user pick the mode in the browser.
"""
import sys

import torch
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)


def build(mode):
    if mode == "fast":
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        # min_size / max_size are baked into the saved model: smaller = faster
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights, min_size=480, max_size=800)
    else:
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=weights, min_size=600, max_size=1000)
    return model.eval(), weights.meta["categories"]


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    modes = ["fast", "accurate"] if arg == "all" else [arg]
    for mode in modes:
        if mode not in ("fast", "accurate"):
            sys.exit("Usage: python save_model.py [fast|accurate|all]")
        print(f"Building {mode} model (first time downloads weights)...", flush=True)
        model, categories = build(mode)
        path = f"faster_rcnn_{mode}.pkl"
        torch.save({"model": model, "categories": categories, "mode": mode}, path)
        print(f"Saved {path}", flush=True)