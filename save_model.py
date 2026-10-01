"""Download the Objects365-pretrained YOLO26 weights once, so app.py starts fast.

    pip install ultralytics
    python save_model.py            # downloads both: fast and accurate
    python save_model.py fast       # only the fast one
    python save_model.py accurate   # only the accurate one

Run it from the same folder as app.py (weights are saved in the current folder).
Edit MODEL_FILES in app.py AND here if you want a different size (n/s/m/l/x).
"""
import sys

from ultralytics import YOLO

MODEL_FILES = {
    "fast": "yolo26s-objv1-150.pt",
    "accurate": "yolo26x-objv1-150.pt",
}

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    modes = list(MODEL_FILES) if arg == "all" else [arg]
    for mode in modes:
        if mode not in MODEL_FILES:
            sys.exit("Usage: python save_model.py [fast|accurate|all]")
        name = MODEL_FILES[mode]
        print(f"Fetching {mode} model: {name} ...", flush=True)
        model = YOLO(name)  # downloads if missing
        print(f"OK: {name} ({len(model.names)} classes)", flush=True)