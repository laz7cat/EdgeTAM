import os
import sys
import cv2
import torch
import numpy as np
import time
import random # Not strictly used now, but good to have if we randomize colors more

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
# ImageNet Mean/Std for model normalization
IMG_MEAN = np.array([0.485, 0.456, 0.406])
IMG_STD = np.array([0.229, 0.224, 0.225])
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
def preprocess_frame(image_np):
    """
    Preprocess a raw numpy image frame for the EdgeTAM model.
    Resizes to model's IMAGE_SIZE, normalizes pixel values, and converts to a PyTorch tensor.
    """
    img_resized = cv2.resize(image_np, (IMAGE_SIZE, IMAGE_SIZE))
    img_float = img_resized.astype(np.float32) / 255.0
    img_float = (img_float - IMG_MEAN) / IMG_STD
    img_tensor = torch.from_numpy(img_float).permute(2, 0, 1).float()
    return img_tensor

def get_mask_overlay(image, masks_tensor, obj_ids, alpha=0.5):
    """
    Overlays multiple binary masks (from masks_tensor) onto the original image.
    Each mask is drawn with a unique color based on its obj_id, with a slight alpha blend
    and a visible contour.
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
        # Safety check if obj_ids might have more entries than masks_np
        if i >= len(masks_np):
            continue
        
        mask = masks_np[i]
        if mask.ndim == 3: # Mask might be (1, H, W), so squeeze to (H, W)
            mask = mask.squeeze()
        
        mask_bool = mask > 0 # Convert float mask to boolean
        if not mask_bool.any(): # Skip if mask is entirely empty
            continue
            
        color = COLORS[obj_id % len(COLORS)] # Pick color based on object ID
        
        # Draw Contour for better visibility
        contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2) # Draw 2-pixel thick contour

        # Fill mask area with alpha blending
        colored_roi = np.zeros_like(overlay)
        colored_roi[mask_bool] = color
        overlay[mask_bool] = cv2.addWeighted(image[mask_bool], 1 - alpha, colored_roi[mask_bool], alpha, 0)

    return overlay

def determine_target_obj_id(click_x, click_y, last_masks):
    """
    Determine if a click falls within an existing object's mask.
    Returns the obj_id if a hit is detected, otherwise None.
    """
    if not last_masks:
        return None

    # Iterate through masks in reverse order (to hit topmost drawn object if overlapping)
    for obj_id in sorted(last_masks.keys(), reverse=True):
        mask = last_masks[obj_id] # mask is already a numpy 2d array from cache
        h, w = mask.shape
        cx, cy = int(click_x), int(click_y)
        
        # Check bounds before accessing mask[cy, cx]
        if 0 <= cx < w and 0 <= cy < h:
            if mask[cy, cx] > 0: # If pixel in mask is foreground
                return obj_id
    
    return None # No object mask hit

def mouse_callback(event, x, y, flags, param):
    """
    OpenCV mouse callback function to handle click events.
    Maps window coordinates to original frame coordinates and queues clicks.
    """
    global click_queue, display_params
    
    # Only process Left or Right mouse button clicks
    if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
        scale = display_params["scale"]
        pad_x = display_params["pad_x"]
        pad_y = display_params["pad_y"]
        
        # Transform window coordinates to coordinates on the displayed image portion
        img_x = x - pad_x
        img_y = y - pad_y
        
        # Scale back to original frame resolution
        if scale > 0: # Avoid division by zero
            orig_x = int(img_x / scale)
            orig_y = int(img_y / scale)
        else: # Fallback if scale is zero (e.g. window minimized)
            orig_x, orig_y = 0, 0
        
        # Clamp coordinates to original frame dimensions (1280x720)
        # These are max dimensions, actual frame might be smaller if camera couldn't set 1280x720
        # The model's init_state takes actual frame_h, frame_w which is used for normalization
        # So clamping to 1280x720 is a reasonable default.
        orig_x = max(0, min(orig_x, 1280 - 1))
        orig_y = max(0, min(orig_y, 720 - 1))

        if event == cv2.EVENT_LBUTTONDOWN:
            click_queue.append((orig_x, orig_y, 1)) # Label 1 for Include
            # print(f"Click Left: Original({orig_x}, {orig_y})") # Debug print can be noisy
        elif event == cv2.EVENT_RBUTTONDOWN:
            click_queue.append((orig_x, orig_y, 0)) # Label 0 for Exclude
            # print(f"Click Right: ({orig_x}, {orig_y})") # Debug print can be noisy

def main():
    global inference_state, is_tracking, frame_idx, click_queue, display_params, active_objects, last_masks_cache
    
    cap = cv2.VideoCapture(0) # Open default camera
    if not cap.isOpened():
        print("Error: Could not open webcam. Please check if it's connected or used by another application.")
        return

    # --- Camera Optimization Setup (1280x720 @ 30fps MJPG) ---
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Verify camera settings
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps_prop = cap.get(cv2.CAP_PROP_FPS) # Get property FPS
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    try:
        fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
    except Exception:
        fourcc_str = "Unknown" # In case FourCC is invalid

    print(f"Camera Settings: {int(actual_w)}x{int(actual_h)} @ {actual_fps_prop} FPS (Format: {fourcc_str})")
    if int(actual_w) != 1280 or int(actual_h) != 720 or actual_fps_prop < 29:
        print("Warning: Could not set desired camera settings (1280x720 @ 30 FPS).")

    # --- Window Setup ---
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL) # Enable Resizable Window
    cv2.resizeWindow(WINDOW_NAME, 960, 540) # Default size (75% of 720p)
    cv2.moveWindow(WINDOW_NAME, 100, 100) # Force to visible position
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback) # Register mouse interaction

    print("\n--- Instructions ---")
    print("Left Click (Background): Start Tracking New Object")
    print("Left Click (Object):     Refine Object (Include)")
    print("Right Click (Object):    Refine Object (Exclude)")
    print("R Key: Reset All Objects & Tracking")
    print("Q Key: Quit")
    print("-----------------------\n")

    # --- Autocast Context for Inference ---
    if DEVICE == "cuda":
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = torch.no_grad() # Use no_grad as a fallback for non-CUDA

    prev_time = 0 # For FPS calculation (software FPS)

    with autocast_ctx:
        while True:
            # --- 1. Read Frame ---
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from camera. Exiting.")
                break

            frame_h, frame_w = frame.shape[:2] # Original frame dimensions
            
            # --- 2. Process Mouse Clicks & Update Object Registry ---
            # A flag to trigger re-initialization if any clicks modify active_objects
            new_clicks_triggered_reinit = False 
            
            if click_queue: # If there are pending mouse clicks
                while click_queue:
                    cx, cy, clabel = click_queue.pop(0) # Get next click
                    is_include = (clabel == 1)
                    
                    # Determine which object this click belongs to
                    target_obj_id = determine_target_obj_id(cx, cy, last_masks_cache)
                    
                    if target_obj_id is None: # Clicked on background
                        if is_include: # Left click on background -> New Object
                            if active_objects: # Get next available object ID
                                target_obj_id = max(active_objects.keys()) + 1
                            else: # First ever object starts with ID 0
                                target_obj_id = 0
                            print(f"-> Creating NEW Object ID: {target_obj_id} at ({cx}, {cy})")
                            active_objects[target_obj_id] = {'points': [], 'labels': []} # Initialize prompts for new object
                        else: # Right click on background -> Ignore
                            print("-> Ignored Right Click on background (cannot start new object with exclude).")
                            continue # Skip this click, doesn't modify registry
                    else: # Clicked on an existing object
                        print(f"-> Refining Object ID: {target_obj_id} at ({cx}, {cy})")

                    # Add point to the determined target object's prompt history
                    active_objects[target_obj_id]['points'].append([cx, cy])
                    active_objects[target_obj_id]['labels'].append(clabel)
                    new_clicks_triggered_reinit = True # A click always forces re-initialization

            # --- 3. Inference Logic ---
            current_display_masks_tensor = None
            current_display_obj_ids = []
            
            # Preprocess current frame for the model
            frame_tensor = preprocess_frame(frame).to(DEVICE)

            if new_clicks_triggered_reinit or (is_tracking == False and active_objects):
                # Scenario A: Re-initialize tracking if new clicks occurred OR if tracking is not active but there are objects in registry (first start)
                if not active_objects: # If no objects after processing clicks (e.g., only ignored clicks)
                    is_tracking = False # Ensure tracking is off
                    inference_state = None # Clear model state
                    last_masks_cache = {} # Clear cache
                    print("No active objects to track. Waiting for input.")
                else:
                    print(f"Re-initializing tracker with {len(active_objects)} objects...")
                    inference_state = predictor.init_state( # Re-initialize model state
                        images=[frame_tensor], # Current frame is treated as Frame 0 for this new session
                        video_height=frame_h,
                        video_width=frame_w
                    )
                    is_tracking = True
                    frame_idx = 0 # Reset frame counter for new tracking session
                    
                    temp_masks_list = []
                    temp_ids_list = []
                    
                    # Apply ALL accumulated prompt points for ALL active objects
                    for obj_id in sorted(active_objects.keys()):
                        data = active_objects[obj_id]
                        pts = np.array(data['points'], dtype=np.float32)
                        lbls = np.array(data['labels'], dtype=np.int32)
                        
                        # Call add_new_points for each object ID for Frame 0 of this new session
                        _, _, out_masks_for_obj = predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=0, # Apply points to the first frame of this new session
                            obj_id=obj_id,
                            points=pts,
                            labels=lbls,
                        )
                        temp_masks_list.append(out_masks_for_obj)
                        temp_ids_list.append(obj_id)
                    
                    if temp_masks_list:
                        current_display_masks_tensor = torch.cat(temp_masks_list, dim=0) # Concatenate all (N_obj, 1, H, W)
                        current_display_obj_ids = temp_ids_list
                        
                        # Update cache for next click's hit-testing
                        last_masks_cache = {}
                        masks_np = current_display_masks_tensor.cpu().numpy()
                        for i, oid in enumerate(current_display_obj_ids):
                            if i < len(masks_np): # Safety check
                                last_masks_cache[oid] = masks_np[i].squeeze()

            elif is_tracking and active_objects:
                # Scenario B: Continue normal tracking (no re-init needed, just propagate)
                predictor.append_frame(inference_state, frame_tensor) # Add current frame to model's sequence
                frame_idx += 1 # Increment frame counter
                
                # Standard tracking step: Get masks for all currently tracked objects
                out_masks_from_track = predictor.track_new_frame(inference_state, frame_idx)
                
                current_display_masks_tensor = out_masks_from_track
                current_display_obj_ids = inference_state["obj_ids"] # Get object IDs from predictor's state
                
                # Update cache for next click's hit-testing
                last_masks_cache = {}
                if current_display_masks_tensor is not None:
                    masks_np = current_display_masks_tensor.cpu().numpy()
                    for i, oid in enumerate(current_display_obj_ids):
                        if i < len(masks_np): # Safety check
                            last_masks_cache[oid] = masks_np[i].squeeze()

            # --- 4. Draw & Display ---
            display_frame = frame.copy() # Start with a clean copy of the original frame
            
            if current_display_masks_tensor is not None and len(current_display_obj_ids) > 0:
                display_frame = get_mask_overlay(display_frame, current_display_masks_tensor, current_display_obj_ids)

            # Draw FPS
            curr_time = time.time()
            soft_fps = 0
            if prev_time != 0:
                dt = curr_time - prev_time
                if dt > 0: soft_fps = 1.0 / dt
            prev_time = curr_time
            
            cv2.putText(display_frame, f"FPS: {int(soft_fps)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # --- 5. Window Resizing & Letterboxing ---
            try:
                win_rect = cv2.getWindowImageRect(WINDOW_NAME)
                win_w, win_h = win_rect[2], win_rect[3]
            except Exception:
                win_w, win_h = 960, 540 # Fallback default size
            
            if win_w > 0 and win_h > 0:
                img_h, img_w = display_frame.shape[:2]
                aspect_ratio = img_w / img_h
                win_aspect = win_w / win_h
                
                new_w, new_h, pad_x, pad_y = 0, 0, 0, 0
                if win_aspect > aspect_ratio: # Window is wider than image aspect ratio
                    new_h = win_h
                    new_w = int(win_h * aspect_ratio)
                    pad_x = (win_w - new_w) // 2
                else: # Window is taller than image aspect ratio
                    new_w = win_w
                    new_h = int(win_w / aspect_ratio)
                    pad_y = (win_h - new_h) // 2
                
                new_w = max(1, new_w); new_h = max(1, new_h)

                # Store params for mouse callback to translate coordinates
                display_params["scale"] = new_w / img_w
                display_params["pad_x"] = pad_x
                display_params["pad_y"] = pad_y
                
                resized_display_frame = cv2.resize(display_frame, (new_w, new_h))
                canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
                canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized_display_frame
                
                cv2.imshow(WINDOW_NAME, canvas)
            else: # If window is minimized or invalid, just show frame without resizing
                cv2.imshow(WINDOW_NAME, display_frame)

            # --- 6. Input Handling (Keyboard) ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("Resetting All Objects...")
                is_tracking = False
                inference_state = None
                frame_idx = 0
                click_queue = []
                active_objects = {} # Clear all object prompts
                last_masks_cache = {} # Clear mask cache

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Critical Error: {e}")
        import traceback
        traceback.print_exc()
