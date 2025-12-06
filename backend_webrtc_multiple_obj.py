from sam2.build_sam import build_sam2_video_predictor
import argparse
import asyncio
import json
import logging
import os
import sys
import time
import cv2
import uvicorn
import threading
import queue
import torch
import torch.nn.functional as F
import numpy as np
from fractions import Fraction
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from av import VideoFrame
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

# Ensure we can import sam2 from current directory
sys.path.append(os.getcwd())

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("edgetam_webrtc")

# --- Model Configuration & Loading ---
CHECKPOINT = "checkpoints/edgetam.pt"
CONFIG = "edgetam_for_gradio_app.yaml"

if torch.cuda.is_available():
    DEVICE = "cuda"
    os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

logger.info(f"Loading EdgeTAM model on {DEVICE}...")
predictor = build_sam2_video_predictor(CONFIG, CHECKPOINT, device=DEVICE)
IMAGE_SIZE = predictor.image_size
logger.info("Model loaded.")

# GPU Constants
GPU_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(3, 1, 1)
GPU_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(3, 1, 1)
COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0)
]

# --- Helper Functions ---


def preprocess_frame_gpu(image_bgr):
    # Convert BGR (OpenCV) -> RGB
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).to(
        DEVICE, non_blocking=True).permute(2, 0, 1).float()
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = F.interpolate(img_tensor, size=(
        IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)
    img_tensor = img_tensor.squeeze(0)
    img_tensor = img_tensor / 255.0
    img_tensor = (img_tensor - GPU_MEAN) / GPU_STD
    return img_tensor


def get_mask_overlay(image, masks_tensor, obj_ids, alpha=0.5):
    # image is BGR
    if isinstance(masks_tensor, torch.Tensor):
        masks_np = masks_tensor.cpu().numpy()
    else:
        masks_np = masks_tensor

    if masks_np is None or len(masks_np) == 0:
        return image

    overlay = image.copy()
    for i, obj_id in enumerate(obj_ids):
        if i >= len(masks_np):
            continue
        mask = masks_np[i]
        if mask.ndim == 3:
            mask = mask.squeeze()
        mask_bool = mask > 0
        if not mask_bool.any():
            continue

        # Colors are RGB, need BGR for OpenCV
        rgb = COLORS[obj_id % len(COLORS)]
        bgr = (rgb[2], rgb[1], rgb[0])

        contours, _ = cv2.findContours(mask_bool.astype(
            np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, bgr, 2)

        # Optimization: Vectorized blending
        # Only blend where mask is true
        roi = overlay[mask_bool]
        # roi is (N, 3), bgr is (3,)
        # We want: roi * (1-alpha) + bgr * alpha
        # Note: cv2.addWeighted is usually faster but requires same size images.
        # Manual numpy blending on ROI:
        blended = roi.astype(float) * (1 - alpha) + np.array(bgr) * alpha
        overlay[mask_bool] = blended.astype(np.uint8)

    return overlay


def determine_target_obj_id(click_x, click_y, last_masks):
    if not last_masks:
        return None
    for obj_id in sorted(last_masks.keys(), reverse=True):
        mask = last_masks[obj_id]
        h, w = mask.shape
        if 0 <= click_x < w and 0 <= click_y < h:
            if mask[int(click_y), int(click_x)] > 0:
                return obj_id
    return None


# Global reference to the active track for interaction
active_track = None


class EdgeTAMTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self):
        super().__init__()
        self.frame_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.start_time = time.time()
        self.frame_count = 0

        # Tracking State
        self.inference_state = None
        self.is_tracking = False
        self.frame_idx = 0
        self.active_objects = {}
        self.last_masks_cache = {}
        self.click_queue = []
        self.lock = threading.Lock()  # Protect shared state

        self.capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _capture_loop(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            return

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                continue
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)
        cap.release()

    def add_click(self, x, y, label=1):
        with self.lock:
            self.click_queue.append((x, y, label))

    def reset_state(self):
        with self.lock:
            self.inference_state = None
            self.is_tracking = False
            self.frame_idx = 0
            self.active_objects = {}
            self.last_masks_cache = {}
            self.click_queue = []
        logger.info("State Reset")

    async def recv(self):
        pts = int(time.time() * 90000)
        time_base = Fraction(1, 90000)

        loop = asyncio.get_event_loop()
        try:
            frame_bgr = await loop.run_in_executor(None, self.frame_queue.get)
        except Exception:
            frame_bgr = np.zeros((720, 1280, 3), dtype='uint8')

        frame_h, frame_w = frame_bgr.shape[:2]

        # --- Inference Logic (Thread-safe access) ---
        current_display_masks_tensor = None
        current_display_obj_ids = []

        # Only acquire lock for state updates, try to keep it short
        # But since predictor state is not thread safe, we effectively single-thread the logic here
        with self.lock:
            # 1. Handle Clicks
            new_clicks_triggered_reinit = False
            if self.click_queue:
                while self.click_queue:
                    cx, cy, clabel = self.click_queue.pop(0)

                    target_obj_id = determine_target_obj_id(
                        cx, cy, self.last_masks_cache)
                    if target_obj_id is None:
                        if clabel == 1:  # Only support include for now as requested
                            if self.active_objects:
                                target_obj_id = max(
                                    self.active_objects.keys()) + 1
                            else:
                                target_obj_id = 0
                            self.active_objects[target_obj_id] = {
                                'points': [], 'labels': []}
                        else:
                            continue

                    self.active_objects[target_obj_id]['points'].append([
                                                                        cx, cy])
                    self.active_objects[target_obj_id]['labels'].append(clabel)
                    new_clicks_triggered_reinit = True

            # 2. Run Tracking
            # Preprocess only if needed
            frame_tensor = None
            if self.is_tracking or new_clicks_triggered_reinit:
                frame_tensor = preprocess_frame_gpu(frame_bgr)

            if new_clicks_triggered_reinit or (not self.is_tracking and self.active_objects):
                if not self.active_objects:
                    self.is_tracking = False
                    self.inference_state = None
                    self.last_masks_cache = {}
                else:
                    # Re-init
                    self.inference_state = predictor.init_state(
                        images=[frame_tensor],
                        video_height=frame_h,
                        video_width=frame_w
                    )
                    self.is_tracking = True
                    self.frame_idx = 0

                    temp_masks_list = []
                    temp_ids_list = []
                    for obj_id in sorted(self.active_objects.keys()):
                        data = self.active_objects[obj_id]
                        point_coords = np.array(
                            data['points'], dtype=np.float32)
                        lbls = np.array(data['labels'], dtype=np.int32)
                        _, _, out_masks = predictor.add_new_points_or_box(
                            inference_state=self.inference_state,
                            frame_idx=0,
                            obj_id=obj_id,
                            points=point_coords,
                            labels=lbls
                        )
                        temp_masks_list.append(out_masks)
                        temp_ids_list.append(obj_id)

                    if temp_masks_list:
                        current_display_masks_tensor = torch.cat(
                            temp_masks_list, dim=0)
                        current_display_obj_ids = temp_ids_list
                        self.last_masks_cache = {}
                        masks_np = current_display_masks_tensor.cpu().numpy()
                        for i, oid in enumerate(current_display_obj_ids):
                            if i < len(masks_np):
                                self.last_masks_cache[oid] = masks_np[i].squeeze(
                                )

            elif self.is_tracking and self.active_objects:
                predictor.append_frame(self.inference_state, frame_tensor)
                self.frame_idx += 1
                out_masks = predictor.track_new_frame(
                    self.inference_state, self.frame_idx)
                current_display_masks_tensor = out_masks
                current_display_obj_ids = self.inference_state["obj_ids"]

                # Update cache for next click
                # NOTE: Optimization: maybe don't download every frame if we only click occasionally?
                # But we need masks_np for determine_target_obj_id.
                # For 30fps, downloading mask (small) is okay.
                self.last_masks_cache = {}
                if current_display_masks_tensor is not None:
                    masks_np = current_display_masks_tensor.cpu().numpy()
                    for i, oid in enumerate(current_display_obj_ids):
                        if i < len(masks_np):
                            self.last_masks_cache[oid] = masks_np[i].squeeze()

        # 3. Visualize (Draw masks on BGR frame)
        display_frame = frame_bgr.copy()
        if current_display_masks_tensor is not None and len(current_display_obj_ids) > 0:
            display_frame = get_mask_overlay(
                display_frame, current_display_masks_tensor, current_display_obj_ids)

        # FPS
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # WebRTC expects RGB
        display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

        new_frame = VideoFrame.from_ndarray(display_frame_rgb, format="rgb24")
        new_frame.pts = pts
        new_frame.time_base = time_base
        return new_frame

    def stop(self):
        self.stop_event.set()
        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        super().stop()


# --- FastAPI ---
app = FastAPI()
pcs = set()


@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
    <head>
        <title>EdgeTAM WebRTC</title>
        <style>
            body { font-family: sans-serif; text-align: center; background: #f0f0f0; }
            #video-container { position: relative; display: inline-block; }
            video { border: 2px solid #333; background: #000; max-width: 100%; }
            .btn { padding: 10px 20px; font-size: 16px; margin: 10px; cursor: pointer; }
            .btn-reset { background: #f44336; color: white; border: none; }
            .btn-start { background: #4CAF50; color: white; border: none; }
        </style>
    </head>
    <body>
        <h1>EdgeTAM Tracker (30 FPS)</h1>
        
        <div id="video-container">
            <video id="video" autoplay playsinline muted style="width: 1280px; height: 720px;"></video>
        </div>
        <br>
        <button class="btn btn-start" onclick="start()">Start Camera</button>
        <button class="btn btn-reset" onclick="reset()">Reset (R)</button>
        <div id="log"></div>
        
        <script>
            var pc = null;
            
            function log(msg) { console.log(msg); }

            async function start() {
                var config = {
                    sdpSemantics: 'unified-plan',
                    iceServers: [{ urls: ['stun:stun.l.google.com:19302'] }]
                };

                if (pc) pc.close();
                pc = new RTCPeerConnection(config);

                // Explicitly add a transceiver to ensure m-line is generated
                pc.addTransceiver('video', {direction: 'recvonly'});

                pc.addEventListener('track', function(evt) {
                    if (evt.track.kind == 'video') {
                        document.getElementById('video').srcObject = evt.streams[0];
                    }
                });

                var offer = await pc.createOffer();
                await pc.setLocalDescription(offer);

                // Quick timeout for ICE
                await new Promise(r => setTimeout(r, 500));
                
                var baseUrl = window.location.origin; // Get current host and port
                
                var response = await fetch(baseUrl + '/offer', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        sdp: pc.localDescription.sdp,
                        type: pc.localDescription.type
                    })
                });
                
                if (!response.ok) {
                    throw new Error("Server responded with " + response.status);
                }

                var answer = await response.json();
                await pc.setRemoteDescription(answer);
            }

            async function reset() {
                var baseUrl = window.location.origin;
                await fetch(baseUrl + '/reset', { method: 'POST' });
            }

            // Click Handler
            document.getElementById('video').addEventListener('mousedown', async function(e) {
                var rect = e.target.getBoundingClientRect();
                var videoW = 1280;
                var videoH = 720;
                
                if (e.target.naturalWidth > 0) {
                    videoW = e.target.naturalWidth;
                    videoH = e.target.naturalHeight;
                }

                var scaleX = videoW / rect.width;
                var scaleY = videoH / rect.height;
                
                var x = (e.clientX - rect.left) * scaleX;
                var y = (e.clientY - rect.top) * scaleY;
                
                var baseUrl = window.location.origin;
                await fetch(baseUrl + '/click', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ x: x, y: y, label: 1 })
                });
            });
        </script>
    </body>
    </html>
    """


@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    # Debug: Print received SDP to analyze direction issues
    print("--- Received SDP Offer ---")
    print(params["sdp"])
    print("--------------------------")

    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    global active_track
    video = EdgeTAMTrack()
    active_track = video  # Save reference for interaction

    # Explicitly add transceiver to avoid direction negotiation issues
    pc.addTransceiver(video, direction="sendonly")
    # pc.addTrack(video) # addTransceiver adds the track implicitly

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return JSONResponse({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})


@app.post("/click")
async def click(request: Request):
    params = await request.json()
    x = params.get("x")
    y = params.get("y")
    if active_track and x is not None:
        active_track.add_click(x, y, label=1)
        return JSONResponse({"status": "ok"})
    return JSONResponse({"status": "error"})


@app.post("/reset")
async def reset_handler(request: Request):
    if active_track:
        active_track.reset_state()
    return JSONResponse({"status": "reset"})


@app.on_event("shutdown")
async def on_shutdown():
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
