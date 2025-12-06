"""
EdgeTAM Backend Server
负责摄像头采集、模型推理、视频流推送
"""
from sam2.build_sam import build_sam2_video_predictor
import asyncio
import logging
import os
import sys
import time
import cv2
import threading
import queue
import torch
import torch.nn.functional as F
import numpy as np
from fractions import Fraction
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from av import VideoFrame
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Ensure we can import sam2 from current directory
sys.path.append(os.getcwd())

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("edgetam_backend")

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
    """将 BGR 图像预处理为模型输入张量"""
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
    """在图像上叠加分割 mask"""
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

        # Vectorized blending
        roi = overlay[mask_bool]
        blended = roi.astype(float) * (1 - alpha) + np.array(bgr) * alpha
        overlay[mask_bool] = blended.astype(np.uint8)

    return overlay


def determine_target_obj_id(click_x, click_y, last_masks):
    """根据点击坐标确定目标对象 ID"""
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
    """WebRTC 视频轨道，负责摄像头采集和模型推理"""
    kind = "video"

    def __init__(self):
        super().__init__()
        logger.info("Initializing EdgeTAMTrack...")
        self.frame_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        
        # FPS 计算（实时）
        self.last_frame_time = time.time()
        self.fps = 0.0
        self.fps_alpha = 0.1  # 平滑系数，用于指数移动平均

        # Tracking State
        self.inference_state = None
        self.is_tracking = False
        self.frame_idx = 0
        self.active_objects = {}
        self.last_masks_cache = {}
        self.click_queue = []
        self.lock = threading.Lock()

        logger.info("Starting camera capture thread...")
        self.capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        logger.info("EdgeTAMTrack initialized")

    def _capture_loop(self):
        """摄像头采集线程"""
        logger.info("Attempting to open camera with CAP_DSHOW...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            logger.warning("CAP_DSHOW failed, trying default backend...")
            cap = cv2.VideoCapture(0)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            logger.error("Failed to open camera")
            return

        actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Camera opened successfully: {actual_w}x{actual_h} @ {actual_fps} FPS")
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
        logger.info("Camera released")

    def add_click(self, x, y, label=1):
        """添加点击事件到队列"""
        with self.lock:
            self.click_queue.append((x, y, label))
            logger.info(f"Click added: ({x}, {y})")

    def reset_state(self):
        """重置所有跟踪状态"""
        with self.lock:
            self.inference_state = None
            self.is_tracking = False
            self.frame_idx = 0
            self.active_objects = {}
            self.last_masks_cache = {}
            self.click_queue = []
        logger.info("Tracking state reset")

    async def recv(self):
        """接收并处理视频帧"""
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

        with self.lock:
            # 1. Handle Clicks
            new_clicks_triggered_reinit = False
            if self.click_queue:
                while self.click_queue:
                    cx, cy, clabel = self.click_queue.pop(0)

                    target_obj_id = determine_target_obj_id(
                        cx, cy, self.last_masks_cache)
                    if target_obj_id is None:
                        # 创建新对象
                        if self.active_objects:
                            target_obj_id = max(
                                self.active_objects.keys()) + 1
                        else:
                            target_obj_id = 0
                        self.active_objects[target_obj_id] = {
                            'points': [], 'labels': []}
                        logger.info(f"New object created: {target_obj_id}")

                    self.active_objects[target_obj_id]['points'].append([cx, cy])
                    self.active_objects[target_obj_id]['labels'].append(clabel)
                    new_clicks_triggered_reinit = True

            # 2. Run Tracking
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
                    logger.info("Inference state initialized")

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
                                self.last_masks_cache[oid] = masks_np[i].squeeze()

            elif self.is_tracking and self.active_objects:
                predictor.append_frame(self.inference_state, frame_tensor)
                self.frame_idx += 1
                out_masks = predictor.track_new_frame(
                    self.inference_state, self.frame_idx)
                current_display_masks_tensor = out_masks
                current_display_obj_ids = self.inference_state["obj_ids"]

                # Update cache
                self.last_masks_cache = {}
                if current_display_masks_tensor is not None:
                    masks_np = current_display_masks_tensor.cpu().numpy()
                    for i, oid in enumerate(current_display_obj_ids):
                        if i < len(masks_np):
                            self.last_masks_cache[oid] = masks_np[i].squeeze()

        # 3. Visualize
        display_frame = frame_bgr.copy()
        if current_display_masks_tensor is not None and len(current_display_obj_ids) > 0:
            display_frame = get_mask_overlay(
                display_frame, current_display_masks_tensor, current_display_obj_ids)

        # 计算实时 FPS（基于帧间隔）
        current_time = time.time()
        frame_interval = current_time - self.last_frame_time
        if frame_interval > 0:
            instant_fps = 1.0 / frame_interval
            # 使用指数移动平均平滑 FPS
            self.fps = self.fps_alpha * instant_fps + (1 - self.fps_alpha) * self.fps
        self.last_frame_time = current_time
        
        cv2.putText(display_frame, f"FPS: {self.fps:.1f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # WebRTC expects RGB
        display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

        new_frame = VideoFrame.from_ndarray(display_frame_rgb, format="rgb24")
        new_frame.pts = pts
        new_frame.time_base = time_base
        return new_frame

    def stop(self):
        """停止视频轨道"""
        self.stop_event.set()
        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        super().stop()


# --- FastAPI Backend ---
app = FastAPI()

# 添加 CORS 支持，允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pcs = set()


@app.post("/offer")
async def offer(request: Request):
    """处理 WebRTC Offer，建立连接"""
    params = await request.json()
    logger.info("=" * 60)
    logger.info("Received WebRTC offer from frontend")
    logger.info(f"SDP type: {params['type']}")

    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)
    logger.info(f"Created RTCPeerConnection, total connections: {len(pcs)}")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"WebRTC Connection state changed: {pc.connectionState}")
        if pc.connectionState == "failed":
            logger.error("WebRTC connection failed!")
            await pc.close()
            pcs.discard(pc)

    global active_track
    logger.info("Creating EdgeTAMTrack instance...")
    video = EdgeTAMTrack()
    active_track = video

    logger.info("Adding video track to peer connection...")
    pc.addTransceiver(video, direction="sendonly")

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    logger.info("WebRTC answer created and set")
    logger.info("=" * 60)
    return JSONResponse({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})


@app.post("/click")
async def click(request: Request):
    """接收前端点击坐标"""
    params = await request.json()
    x = params.get("x")
    y = params.get("y")
    if active_track and x is not None and y is not None:
        active_track.add_click(int(x), int(y), label=1)
        return JSONResponse({"status": "ok"})
    return JSONResponse({"status": "error", "message": "Invalid click data"})


@app.post("/reset")
async def reset_handler(request: Request):
    """重置跟踪状态"""
    if active_track:
        active_track.reset_state()
        logger.info("Reset request received")
        return JSONResponse({"status": "reset"})
    return JSONResponse({"status": "error", "message": "No active track"})


@app.on_event("shutdown")
async def on_shutdown():
    """关闭所有连接"""
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    logger.info("Server shutdown")


if __name__ == "__main__":
    logger.info("Starting EdgeTAM Backend Server on http://0.0.0.0:7860")
    uvicorn.run(app, host="0.0.0.0", port=7860)
