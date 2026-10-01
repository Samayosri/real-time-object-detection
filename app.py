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
from ultralytics import YOLO

MODEL_FILES = {
    "fast": "yolo26s-objv1-150.pt",
    "accurate": "yolo26x-objv1-150.pt",
}
DEFAULT_MODE = "fast"
DEFAULT_CONF = 0.40
IMGSZ = 480 

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if torch.cuda.is_available():
    DEVICE = 0
    PRECISION = {"quantize": 16}  
    print(f"GPU enabled: {torch.cuda.get_device_name(0)}", flush=True)
else:
    DEVICE = "cpu"
    PRECISION = {}  # FP32 on CPU
    print("WARNING: CUDA not available, running on CPU (slow).", flush=True)

_models = {}
_lock = threading.Lock()        # guards loading
_infer_lock = threading.Lock()  # Ultralytics predictors are not thread-safe


def get_model(mode):
    with _lock:
        if mode in _models:
            return _models[mode]
        name = MODEL_FILES[mode]
        print(f"Loading {mode} model ({name})...", flush=True)
        model = YOLO(name)
        model.predict(np.zeros((480, 640, 3), np.uint8), imgsz=IMGSZ,
                      device=DEVICE, verbose=False, **PRECISION)
        _models[mode] = model
        print(f"{mode} model ready.", flush=True)
        return model


get_model(DEFAULT_MODE)  

def infer(frame_bgr, mode, conf_threshold):
    """BGR frame in, list of detections out."""
    model = get_model(mode)
    with _infer_lock:
        res = model.predict(
            frame_bgr, conf=conf_threshold, imgsz=IMGSZ,
            device=DEVICE, verbose=False, **PRECISION,
        )[0]

    detections = []
    if res.boxes is not None and len(res.boxes):
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
            detections.append({
                "box": [round(float(x1)), round(float(y1)), round(float(x2)), round(float(y2))],
                "confidence": round(float(conf), 2),
                "label": res.names[int(cls_id)],
                "class": int(cls_id),
            })
    return detections


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.get("/")
async def index():
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