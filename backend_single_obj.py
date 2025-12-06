"""
EdgeTAM Backend Server (Simplified)
简化版本：
1. 最多只追踪一个物品
2. 已经在追踪时，忽略新的点击
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
logger = logging.getLogger("edgetam_backend_simple")

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

# 单一颜色用于单对象追踪
TRACK_COLOR_BGR = (0, 255, 0)  # 绿色

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


def get_mask_overlay(image, mask_tensor, alpha=0.5):
    """在图像上叠加单个分割 mask"""
    if mask_tensor is None:
        return image
    
    # mask_tensor shape: (1, 1, 480, 640), dtype: float32
    mask_np = mask_tensor.cpu().numpy()
    mask = mask_np[0, 0]  # 提取 (480, 640)
    
    mask_bool = mask > 0
    if not mask_bool.any():
        return image

    overlay = image.copy()
    
    # 绘制轮廓
    mask_uint8 = mask_bool.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, TRACK_COLOR_BGR, 2)

    # 填充半透明颜色
    roi = overlay[mask_bool]
    blended = roi.astype(float) * (1 - alpha) + np.array(TRACK_COLOR_BGR) * alpha
    overlay[mask_bool] = blended.astype(np.uint8)

    return overlay


# Global reference to the active track for interaction
active_track = None


class EdgeTAMTrackSimple(MediaStreamTrack):
    """WebRTC 视频轨道（简化版）- 只追踪单个物体"""
    kind = "video"

    def __init__(self):
        super().__init__()
        logger.info("Initializing EdgeTAMTrackSimple (single object tracking)...")
        self.frame_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        
        # FPS 计算（实时）
        self.last_frame_time = time.time()
        self.fps = 0.0
        self.fps_alpha = 0.1  # 平滑系数

        # Tracking State (简化：只追踪一个物体)
        self.inference_state = None
        self.is_tracking = False
        self.frame_idx = 0
        self.target_point = None  # 只存储一个点 (x, y)
        self.lock = threading.Lock()

        logger.info("Starting camera capture thread...")
        self.capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        logger.info("EdgeTAMTrackSimple initialized")

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

    def add_click(self, x, y):
        """添加点击事件（简化：只在未追踪时接受）"""
        with self.lock:
            if self.is_tracking:
                logger.warning(f"Already tracking! Ignoring click at ({x}, {y})")
                return False
            
            self.target_point = (x, y)
            logger.info(f"Click accepted: ({x}, {y})")
            return True

    def reset_state(self):
        """重置所有跟踪状态"""
        with self.lock:
            self.inference_state = None
            self.is_tracking = False
            self.frame_idx = 0
            self.target_point = None
        logger.info("Tracking state reset")

    async def recv(self):
        """接收并处理视频帧"""
        pts = int(time.time() * 90000)
        time_base = Fraction(1, 90000)

        loop = asyncio.get_event_loop()
        try:
            frame_bgr = await loop.run_in_executor(None, self.frame_queue.get)
        except Exception as e:
            logger.error(f"Error getting frame from queue: {e}")
            frame_bgr = np.zeros((480, 640, 3), dtype='uint8')

        frame_h, frame_w = frame_bgr.shape[:2]

        # --- Simplified Inference Logic ---
        current_mask = None

        try:
            with self.lock:
                # 预处理帧（只在需要时）
                frame_tensor = None
                if self.is_tracking or self.target_point is not None:
                    frame_tensor = preprocess_frame_gpu(frame_bgr)

                # 初始化追踪（只在第一次点击时）
                if self.target_point is not None and not self.is_tracking:
                    x, y = self.target_point
                    
                    # 初始化状态
                    self.inference_state = predictor.init_state(
                        images=[frame_tensor],
                        video_height=frame_h,
                        video_width=frame_w
                    )
                    self.is_tracking = True
                    self.frame_idx = 0
                    logger.info(f"Tracking initialized for object at ({x}, {y})")

                    # 添加点到模型
                    point_coords = np.array([[x, y]], dtype=np.float32)
                    labels = np.array([1], dtype=np.int32)  # 1 = include
                    
                    _, _, out_masks = predictor.add_new_points_or_box(
                        inference_state=self.inference_state,
                        frame_idx=0,
                        obj_id=0,  # 固定使用 obj_id=0
                        points=point_coords,
                        labels=labels
                    )
                    current_mask = out_masks
                    
                    # 清除点击状态
                    self.target_point = None

                # 追踪现有物体
                elif self.is_tracking:
                    predictor.append_frame(self.inference_state, frame_tensor)
                    self.frame_idx += 1
                    
                    out_masks = predictor.track_new_frame(
                        self.inference_state, self.frame_idx)
                    current_mask = out_masks
        
        except Exception as e:
            logger.error(f"Error during inference: {e}", exc_info=True)
            current_mask = None

        # 可视化
        display_frame = frame_bgr.copy()
        if current_mask is not None:
            display_frame = get_mask_overlay(display_frame, current_mask)

        # 计算实时 FPS
        current_time = time.time()
        frame_interval = current_time - self.last_frame_time
        if frame_interval > 0:
            instant_fps = 1.0 / frame_interval
            self.fps = self.fps_alpha * instant_fps + (1 - self.fps_alpha) * self.fps
        self.last_frame_time = current_time
        
        # 显示 FPS 和状态
        cv2.putText(display_frame, f"FPS: {self.fps:.1f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        status_text = "Tracking" if self.is_tracking else "Click to track"
        cv2.putText(display_frame, status_text, (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

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

# 添加 CORS 支持
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

    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)
    logger.info(f"Created RTCPeerConnection, total connections: {len(pcs)}")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"WebRTC Connection state: {pc.connectionState}")
        if pc.connectionState == "failed":
            logger.error("WebRTC connection failed!")
            await pc.close()
            pcs.discard(pc)

    global active_track
    logger.info("Creating EdgeTAMTrackSimple instance...")
    video = EdgeTAMTrackSimple()
    active_track = video

    logger.info("Adding video track to peer connection...")
    pc.addTransceiver(video, direction="sendonly")

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    logger.info("WebRTC connection established")
    logger.info("=" * 60)
    return JSONResponse({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})


@app.post("/click")
async def click(request: Request):
    """接收前端点击坐标（简化：只在未追踪时接受）"""
    params = await request.json()
    x = params.get("x")
    y = params.get("y")
    
    if active_track and x is not None and y is not None:
        accepted = active_track.add_click(int(x), int(y))
        if accepted:
            return JSONResponse({"status": "ok", "message": "Tracking started"})
        else:
            return JSONResponse({"status": "ignored", "message": "Already tracking, click ignored"})
    
    return JSONResponse({"status": "error", "message": "Invalid click data"})


@app.post("/reset")
async def reset_handler(request: Request):
    """重置跟踪状态"""
    if active_track:
        active_track.reset_state()
        logger.info("Reset request received")
        return JSONResponse({"status": "reset", "message": "Tracking reset"})
    return JSONResponse({"status": "error", "message": "No active track"})


@app.on_event("shutdown")
async def on_shutdown():
    """关闭所有连接"""
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    logger.info("Server shutdown")


if __name__ == "__main__":
    logger.info("Starting EdgeTAM Backend Server (Simplified) on http://0.0.0.0:7860")
    logger.info("Features:")
    logger.info("  - Single object tracking only")
    logger.info("  - Ignores clicks while tracking")
    logger.info("  - Press Reset to track a new object")
    uvicorn.run(app, host="0.0.0.0", port=7860)
