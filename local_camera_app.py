import os
import sys
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import time
import random

# Ensure we can import sam2 from current directory
sys.path.append(os.getcwd())

from sam2.build_sam import build_sam2_video_predictor

# --- Configuration ---
CHECKPOINT = "checkpoints/edgetam.pt"
CONFIG = "edgetam_for_gradio_app.yaml"

# Setup device
if torch.cuda.is_available():
    DEVICE = "cuda"
    # Optimizations for CUDA (only for Ampere GPUs or newer)
    os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Loading EdgeTAM model on {DEVICE}...")
predictor = build_sam2_video_predictor(CONFIG, CHECKPOINT, device=DEVICE)
IMAGE_SIZE = predictor.image_size # Model's internal image size (e.g., 1024)
print("Model loaded successfully.")

# --- Constants ---
WINDOW_NAME = 'EdgeTAM Multi-Object Tracker'

# Colors for different objects (RGB format for drawing)
COLORS = [
    (0, 255, 0),    # Green (for obj_id 0, 8, ...)
    (255, 0, 0),    # Blue (for obj_id 1, 9, ...)
    (0, 0, 255),    # Red (for obj_id 2, 10, ...)
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 255),  # Purple
    (255, 128, 0),  # Orange
    (0, 128, 255),  # Light Blue
    (128, 255, 0),  # Lime Green
    (255, 100, 100),# Light Red
    (100, 100, 255),# Light Blue
]

# --- Global State Variables ---
inference_state = None      # Stores the SAM2 predictor's internal state
is_tracking = False         # Flag to indicate if tracking is active
frame_idx = 0               # Current frame index in the tracking sequence
click_queue = []            # Mouse click events pending processing: [(orig_x, orig_y, label), ...]
active_objects = {}         # Registry of all objects being tracked: { obj_id: {'points': [[x,y],...], 'labels': [1,0,...]} }
last_masks_cache = {}       # Last known masks for hit-testing clicks: { obj_id: mask_numpy_2d }
display_params = {"scale": 1.0, "pad_x": 0, "pad_y": 0} # For coordinate mapping (window to original frame)

# --- Helper Functions ---
def preprocess_frame_gpu(image_np, device, mean, std):
    """
    Preprocess a raw numpy image frame on GPU for the EdgeTAM model.
    1. Uploads raw uint8 image to GPU.
    2. Resizes to IMAGE_SIZE using interpolation on GPU.
    3. Normalizes pixel values.
    """
    # 1. Upload to GPU: (H, W, C) -> (C, H, W)
    # We keep it as uint8 during transfer to save bandwidth, then convert to float on GPU
    img_tensor = torch.from_numpy(image_np).to(device, non_blocking=True).permute(2, 0, 1).float()
    
    # 2. Resize to model input size (1024x1024)
    # interpolate expects (N, C, H, W), so we unsqueeze and then squeeze back
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = F.interpolate(img_tensor, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)
    img_tensor = img_tensor.squeeze(0)
    
    # 3. Normalize (0-255 -> 0-1) and apply ImageNet mean/std
    img_tensor = img_tensor / 255.0
    img_tensor = (img_tensor - mean) / std
    
    return img_tensor

def get_mask_overlay(image, masks_tensor, obj_ids, alpha=0.5):
    """
    Overlays multiple binary masks (from masks_tensor) onto the original image.
    """
    if isinstance(masks_tensor, torch.Tensor):
        masks_np = masks_tensor.cpu().numpy()
    else:
        masks_np = masks_tensor

    if masks_np is None or len(masks_np) == 0:
        return image

    overlay = image.copy()
    
    # Iterate through masks (obj_ids)
    for i, obj_id in enumerate(obj_ids):
        if i >= len(masks_np):
            continue
        
        mask = masks_np[i]
        if mask.ndim == 3:
            mask = mask.squeeze()
        
        mask_bool = mask > 0
        if not mask_bool.any():
            continue
            
        color = COLORS[obj_id % len(COLORS)]
        
        # Draw Contour
        contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)

        # Fill mask area
        colored_roi = np.zeros_like(overlay)
        colored_roi[mask_bool] = color
        overlay[mask_bool] = cv2.addWeighted(image[mask_bool], 1 - alpha, colored_roi[mask_bool], alpha, 0)

    return overlay

def determine_target_obj_id(click_x, click_y, last_masks):
    if not last_masks:
        return None

    for obj_id in sorted(last_masks.keys(), reverse=True):
        mask = last_masks[obj_id]
        h, w = mask.shape
        cx, cy = int(click_x), int(click_y)
        
        if 0 <= cx < w and 0 <= cy < h:
            if mask[cy, cx] > 0:
                return obj_id
    
    return None

def mouse_callback(event, x, y, flags, param):
    global click_queue, display_params
    
    if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
        scale = display_params["scale"]
        pad_x = display_params["pad_x"]
        pad_y = display_params["pad_y"]
        
        img_x = x - pad_x
        img_y = y - pad_y
        
        if scale > 0:
            orig_x = int(img_x / scale)
            orig_y = int(img_y / scale)
        else:
            orig_x, orig_y = 0, 0
        
        orig_x = max(0, min(orig_x, 1280 - 1))
        orig_y = max(0, min(orig_y, 720 - 1))

        if event == cv2.EVENT_LBUTTONDOWN:
            click_queue.append((orig_x, orig_y, 1))
        elif event == cv2.EVENT_RBUTTONDOWN:
            click_queue.append((orig_x, orig_y, 0))

def main():
    global inference_state, is_tracking, frame_idx, click_queue, display_params, active_objects, last_masks_cache
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps_prop = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera Settings: {int(actual_w)}x{int(actual_h)} @ {actual_fps_prop} FPS")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 540)
    cv2.moveWindow(WINDOW_NAME, 100, 100)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    print("\n--- Instructions ---")
    print("Left Click (Background): Start Tracking New Object")
    print("Left Click (Object):     Refine Object (Include)")
    print("Right Click (Object):    Refine Object (Exclude)")
    print("R Key: Reset All Objects & Tracking")
    print("Q Key: Quit")
    print("-----------------------\n")

    # Prepare GPU constants for preprocessing
    # ImageNet Mean/Std: [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    gpu_mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(3, 1, 1)
    gpu_std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(3, 1, 1)

    if DEVICE == "cuda":
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = torch.no_grad()

    prev_time = 0

    with autocast_ctx:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame.")
                break

            frame_h, frame_w = frame.shape[:2]
            
            new_clicks_triggered_reinit = False 
            
            if click_queue:
                while click_queue:
                    cx, cy, clabel = click_queue.pop(0)
                    is_include = (clabel == 1)
                    
                    target_obj_id = determine_target_obj_id(cx, cy, last_masks_cache)
                    
                    if target_obj_id is None:
                        if is_include:
                            if active_objects:
                                target_obj_id = max(active_objects.keys()) + 1
                            else:
                                target_obj_id = 0
                            print(f"-> Creating NEW Object ID: {target_obj_id} at ({cx}, {cy})")
                            active_objects[target_obj_id] = {'points': [], 'labels': []}
                        else:
                            print("-> Ignored Right Click on background.")
                            continue
                    else:
                        print(f"-> Refining Object ID: {target_obj_id} at ({cx}, {cy})")

                    active_objects[target_obj_id]['points'].append([cx, cy])
                    active_objects[target_obj_id]['labels'].append(clabel)
                    new_clicks_triggered_reinit = True

            # --- Inference Logic ---
            current_display_masks_tensor = None
            current_display_obj_ids = []
            
            # OPTIMIZATION: Only preprocess if we are tracking or initializing
            # And perform preprocessing on GPU
            frame_tensor = None
            if is_tracking or new_clicks_triggered_reinit:
                frame_tensor = preprocess_frame_gpu(frame, DEVICE, gpu_mean, gpu_std)

            if new_clicks_triggered_reinit or (is_tracking == False and active_objects):
                if not active_objects:
                    is_tracking = False
                    inference_state = None
                    last_masks_cache = {}
                    print("No active objects to track.")
                else:
                    print(f"Re-initializing tracker with {len(active_objects)} objects...")
                    # Note: init_state might re-read frame info, so we pass current frame props
                    # However, EdgeTAM's init_state with images list expects just the tensors.
                    # The frame_h/w are for coordinate normalization inside the model.
                    inference_state = predictor.init_state(
                        images=[frame_tensor],
                        video_height=frame_h,
                        video_width=frame_w
                    )
                    is_tracking = True
                    frame_idx = 0
                    
                    temp_masks_list = []
                    temp_ids_list = []
                    
                    for obj_id in sorted(active_objects.keys()):
                        data = active_objects[obj_id]
                        pts = np.array(data['points'], dtype=np.float32)
                        lbls = np.array(data['labels'], dtype=np.int32)
                        
                        _, _, out_masks_for_obj = predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=0,
                            obj_id=obj_id,
                            points=pts,
                            labels=lbls,
                        )
                        temp_masks_list.append(out_masks_for_obj)
                        temp_ids_list.append(obj_id)
                    
                    if temp_masks_list:
                        current_display_masks_tensor = torch.cat(temp_masks_list, dim=0)
                        current_display_obj_ids = temp_ids_list
                        
                        last_masks_cache = {}
                        masks_np = current_display_masks_tensor.cpu().numpy()
                        for i, oid in enumerate(current_display_obj_ids):
                            if i < len(masks_np):
                                last_masks_cache[oid] = masks_np[i].squeeze()

            elif is_tracking and active_objects:
                predictor.append_frame(inference_state, frame_tensor)
                frame_idx += 1
                
                out_masks_from_track = predictor.track_new_frame(inference_state, frame_idx)
                
                current_display_masks_tensor = out_masks_from_track
                current_display_obj_ids = inference_state["obj_ids"]
                
                last_masks_cache = {}
                if current_display_masks_tensor is not None:
                    masks_np = current_display_masks_tensor.cpu().numpy()
                    for i, oid in enumerate(current_display_obj_ids):
                        if i < len(masks_np):
                            last_masks_cache[oid] = masks_np[i].squeeze()

            # --- Draw & Display ---
            display_frame = frame.copy()
            
            if current_display_masks_tensor is not None and len(current_display_obj_ids) > 0:
                display_frame = get_mask_overlay(display_frame, current_display_masks_tensor, current_display_obj_ids)

            curr_time = time.time()
            soft_fps = 0
            if prev_time != 0:
                dt = curr_time - prev_time
                if dt > 0: soft_fps = 1.0 / dt
            prev_time = curr_time
            
            cv2.putText(display_frame, f"FPS: {int(soft_fps)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Window Resizing
            try:
                win_rect = cv2.getWindowImageRect(WINDOW_NAME)
                win_w, win_h = win_rect[2], win_rect[3]
            except Exception:
                win_w, win_h = 960, 540
            
            if win_w > 0 and win_h > 0:
                img_h, img_w = display_frame.shape[:2]
                aspect_ratio = img_w / img_h
                win_aspect = win_w / win_h
                
                new_w, new_h, pad_x, pad_y = 0, 0, 0, 0
                if win_aspect > aspect_ratio:
                    new_h = win_h
                    new_w = int(win_h * aspect_ratio)
                    pad_x = (win_w - new_w) // 2
                else:
                    new_w = win_w
                    new_h = int(win_w / aspect_ratio)
                    pad_y = (win_h - new_h) // 2
                
                new_w = max(1, new_w); new_h = max(1, new_h)

                display_params["scale"] = new_w / img_w
                display_params["pad_x"] = pad_x
                display_params["pad_y"] = pad_y
                
                resized_display_frame = cv2.resize(display_frame, (new_w, new_h))
                canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
                canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized_display_frame
                
                cv2.imshow(WINDOW_NAME, canvas)
            else:
                cv2.imshow(WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("Resetting All Objects...")
                is_tracking = False
                inference_state = None
                frame_idx = 0
                click_queue = []
                active_objects = {}
                last_masks_cache = {}

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Critical Error: {e}")
        import traceback
        traceback.print_exc()