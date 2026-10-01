import asyncio
import json
import os
import threading
import time

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Pickles created by save_model.py. The browser picks which one to use.
MODEL_FILES = {
    "fast": "faster_rcnn_fast.pkl",
    "accurate": "faster_rcnn_accurate.pkl",
}
DEFAULT_MODE = "fast"
DEFAULT_CONF = 0.5
USE_FP16 = True  # mixed precision on GPU

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"GPU enabled: {torch.cuda.get_device_name(0)}", flush=True)
else:
    device = torch.device("cpu")
    USE_FP16 = False
    print("WARNING: CUDA not available, running on CPU (slow).", flush=True)

# ---- Black-box model cache: each mode is loaded once, on first use ----
_models = {}
_lock = threading.Lock()


def get_model(mode):
    with _lock:
        if mode in _models:
            return _models[mode]
        path = MODEL_FILES[mode]
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Run: python save_model.py {mode}")
        print(f"Loading {mode} model...", flush=True)
        # weights_only=False is required for full-model pickles.
        # Only load pickle files you created yourself or fully trust.
        bundle = torch.load(path, map_location=device, weights_only=False)
        model = bundle["model"].to(device).eval()
        with torch.inference_mode():  # warm-up
            model([torch.zeros(3, 480, 640, device=device)])
            if device.type == "cuda":
                torch.cuda.synchronize()
        _models[mode] = (model, bundle["categories"])
        print(f"{mode} model ready.", flush=True)
        return _models[mode]


get_model(DEFAULT_MODE)  # preload the default so the first connection is fast


def infer(frame_bgr, mode, conf_threshold):
    """Black-box call: BGR frame in, list of detections out."""
    model, categories = get_model(mode)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(device)
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=USE_FP16):
        out = model([tensor])[0]

    detections = []
    for box, score, label in zip(out["boxes"], out["scores"], out["labels"]):
        conf = float(score.item())
        if conf < conf_threshold:
            continue
        x1, y1, x2, y2 = box.tolist()
        cls_id = int(label.item())
        detections.append({
            "box": [round(x1), round(y1), round(x2), round(y2)],
            "confidence": round(conf, 2),
            "label": categories[cls_id],
            "class": cls_id,
        })
    return detections


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.get("/")
async def index():
    # Serving the page from the backend keeps page and WebSocket on the same origin
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


@app.websocket("/ws/detect")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    settings = {"mode": DEFAULT_MODE, "conf": DEFAULT_CONF}  # per connection
    try:
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.disconnect":
                break

            # Text message = settings from the browser
            if msg.get("text") is not None:
                try:
                    req = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue

                if "conf" in req:
                    try:
                        settings["conf"] = min(max(float(req["conf"]), 0.0), 1.0)
                    except (TypeError, ValueError):
                        pass

                new_mode = req.get("mode")
                if new_mode in MODEL_FILES and new_mode != settings["mode"]:
                    if new_mode not in _models:
                        await websocket.send_json({"status": f"Loading {new_mode} model..."})
                    try:
                        await asyncio.to_thread(get_model, new_mode)
                        settings["mode"] = new_mode
                        await websocket.send_json(
                            {"status": f"Mode: {new_mode}", "mode": new_mode}
                        )
                    except Exception as e:
                        await websocket.send_json(
                            {"error": str(e), "mode": settings["mode"]}
                        )
                continue

            # Binary message = a JPEG frame
            data = msg.get("bytes")
            if data is None:
                continue
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            t0 = time.perf_counter()
            detections = await asyncio.to_thread(
                infer, frame, settings["mode"], settings["conf"]
            )
            await websocket.send_json({
                "detections": detections,
                "mode": settings["mode"],
                "ms": round((time.perf_counter() - t0) * 1000),
            })
    except WebSocketDisconnect:
        pass
    print("Client disconnected")


if __name__ == "__main__":
    cert, key = os.path.join(BASE_DIR, "cert.pem"), os.path.join(BASE_DIR, "key.pem")
    if os.path.exists(cert) and os.path.exists(key):
        print("HTTPS on. Open https://<this-pc-ip>:8000 (or https://localhost:8000)", flush=True)
        uvicorn.run(app, host="0.0.0.0", port=8000, ssl_certfile=cert, ssl_keyfile=key)
    else:
        print("No cert.pem/key.pem: HTTP only, the camera works on localhost only.", flush=True)
        print("For phones, run: python gen_cert.py", flush=True)
        uvicorn.run(app, host="0.0.0.0", port=8000)