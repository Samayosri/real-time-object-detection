# Real-Time Object Detection Web App

A high-performance, real-time object detection web application powered by **FastAPI**, **WebSockets**, and **YOLO** (Objects365 pretrained models). It streams live camera feeds directly from any browser (desktop or mobile) over low-latency WebSockets, performs GPU/CPU inference on the backend, and renders dynamic bounding boxes in real time.

---

## 🌟 Features

- **Ultra-Low Latency Streaming:** Sends JPEG frames over binary WebSockets with pipelined in-flight frame throttling for smooth real-time performance.
- **Pretrained on Objects365 (365 Classes):** Detects a wide variety of everyday objects far beyond standard 80-class COCO models.
- **On-the-Fly Model Switching:** Switch seamlessly between:
  - **Fast Mode:** `yolo26s` for high FPS and low latency.
  - **Accurate Mode:** `yolo26x` for maximum precision.
- **Adjustable Confidence:** Live slider to tune the confidence threshold (10% to 95%) in real time without refreshing.
- **Hardware Acceleration:** Automatically leverages NVIDIA CUDA with FP16 precision if available; gracefully falls back to CPU (FP32).
- **Mobile & Multi-Camera Ready:** Switch between front and back (environment) cameras on phones and tablets.
- **Built-in HTTPS Generator:** Includes a helper script to generate self-signed SSL certificates for mobile browser camera access over local Wi-Fi.

---

## 🏗️ Architecture

```
┌──────────────────────────┐                   ┌──────────────────────────────┐
│     Client (Browser)     │                   │       FastAPI Backend        │
│                          │                   │                              │
│  - getUserMedia (Camera) │  Binary (JPEG)    │  - WebSocket Server          │
│  - HTML5 Canvas overlay  │ ────────────────> │  - OpenCV Frame Decode       │
│  - Controls & Settings   │ <──────────────── │  - YOLO Inference (CUDA/CPU) │
│                          │   JSON (Boxes)    │                              │
└──────────────────────────┘                   └──────────────────────────────┘
```

---

## 📁 Project Structure

```
real-time-object-detection/
├── app.py              # Main FastAPI application and WebSocket inference server
├── index.html          # Frontend UI with camera handling, Canvas, and WebSocket client
├── gen_cert.py         # Utility to generate self-signed SSL certificates for LAN/mobile
├── save_model.py       # Helper script to download YOLO model weights in advance
├── yolo26s-objv1-150.pt# Fast model weights (Objects365v1)
├── yolo26x-objv1-150.pt# Accurate model weights (Objects365v1)
├── cert.pem / key.pem  # (Optional) Self-signed SSL certificate and private key
└── README.md           # Documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites

- **Python 3.10+**
- (Optional, recommended) **NVIDIA GPU** with CUDA support and matching PyTorch installation.

### 2. Environment Setup

Clone the repository and set up a virtual environment:

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies

Install the required packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

> **Note for CUDA users:** If you have an NVIDIA GPU, make sure to install the CUDA-enabled build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## 📦 Model Weights Setup

The application will automatically download missing weights on the first run. However, you can pre-download them using `save_model.py`:

```bash
# Download both Fast and Accurate models
python save_model.py all

# Or download individually:
python save_model.py fast
python save_model.py accurate
```

---

## 🏃 Running the Application

### Option A: Local PC Only (HTTP)

If running and viewing exclusively on the same computer:

1. Ensure no `cert.pem` or `key.pem` files are in the root directory (or rename them to `.bak`).
2. Start the server:
   ```bash
   python app.py
   ```
3. Open your browser and navigate to:
   ```text
   http://localhost:8000
   ```

---

### Option B: Mobile / Local Network (LAN with HTTPS)

Modern mobile browsers (Chrome, Safari) **block camera access** on non-localhost addresses unless the connection is encrypted via **HTTPS**.

1. **Generate the SSL certificate** (detects your PC's local Wi-Fi IP automatically):
   ```bash
   python gen_cert.py
   ```
   *(This creates `cert.pem` and `key.pem` in your project folder).*

2. **Start the server:**
   ```bash
   python app.py
   ```
   *You will see:* `HTTPS on. Open https://<your-pc-ip>:8000`

3. **Open on your phone:**
   - Connect your phone to the same Wi-Fi network as your PC.
   - In your mobile browser, go to `https://<your-pc-ip>:8000` (e.g., `https://192.168.1.3:8000`).
   - Tap **Advanced** > **Proceed** to bypass the self-signed certificate warning.
   - Allow camera permissions.

---

### Option C: Public Access via Tunneling

If you wish to expose the app over the internet:

- **Using ngrok (recommended for HTTPS upstreams):**
  ```bash
  ngrok http https://localhost:8000
  ```
- **Using localtunnel:**
  *Requires running `app.py` in HTTP mode (rename `cert.pem` and `key.pem` first)*:
  ```bash
  npx localtunnel --port 8000 --local-host 127.0.0.1
  ```

---

## ⚙️ Configuration & Tuning

In [app.py](file:///d:/real-time%20detection/real-time-object-detection/app.py):
- `DEFAULT_MODE`: Change default model (`"fast"` or `"accurate"`).
- `DEFAULT_CONF`: Default confidence threshold (default: `0.40`).
- `IMGSZ`: YOLO inference resolution (default: `480`).

In [index.html](file:///d:/real-time%20detection/real-time-object-detection/index.html):
- `MAX_IN_FLIGHT`: Number of concurrent frames waiting for backend response (default: `2`).
- `SEND_SCALE`: Scaling factor before uploading frames (default: `0.75`).
- `JPEG_QUALITY`: Compression quality of uploaded frames (default: `0.6`).

---

## 🔧 Troubleshooting

| Issue | Cause | Solution |
|---|---|---|
| **Camera access denied / not working on phone** | Mobile browsers block camera on insecure HTTP origins. | Run `python gen_cert.py` to enable HTTPS, then use `https://<pc-ip>:8000`. |
| **Localtunnel shows "502 Bad Gateway"** | Localtunnel is sending HTTP requests to an HTTPS-enabled server, or failing on IPv6. | Temporarily rename `cert.pem` and `key.pem` so `app.py` runs on HTTP, and use `--local-host 127.0.0.1`. |
| **Low FPS / High Latency** | Running on CPU or sending full-resolution frames. | Enable CUDA GPU, switch mode to **Fast**, or lower `SEND_SCALE` and `IMGSZ`. |
| **"spawn code-tunnel.exe ENOENT" in IDE** | IDE attempted automatic port forwarding without the tunnel binary. | Disable `remote.autoForwardPorts` in IDE settings or open `http://localhost:8000` directly in your browser. |

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
