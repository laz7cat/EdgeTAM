import cv2
import gradio as gr
import time

def camera_stream():
    """
    Generator function that captures frames from the local camera,
    calculates FPS, draws it on the frame, and yields to Gradio.
    """
    # Open the default local camera (index 0)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        yield None
        return

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Calculate FPS
        current_time = time.time()
        fps = 0
        if prev_time != 0:
            # Avoid division by zero in edge cases
            time_diff = current_time - prev_time
            if time_diff > 0:
                fps = 1.0 / time_diff
        prev_time = current_time
        
        # Draw FPS on the frame (Top Left corner)
        # Position: (10, 30), Font: Hershey Simplex, Scale: 1, Color: Green (0, 255, 0), Thickness: 2
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert OpenCV BGR format to RGB for Gradio display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame_rgb

def reset_display():
    """
    Simple reset function to clear the image component.
    """
    return None

with gr.Blocks() as demo:
    gr.Markdown("# 本地摄像头实时画面 (FPS显示)")
    
    with gr.Row():
        # Display component for the camera stream
        camera_display = gr.Image(label="摄像头捕捉画面", streaming=True)
    
    with gr.Row():
        reset_btn = gr.Button("Reset (重置)")

    # Start the camera stream immediately upon load
    demo.load(camera_stream, inputs=None, outputs=camera_display)
    
    # Reset button logic
    reset_btn.click(reset_display, outputs=camera_display)

if __name__ == "__main__":
    demo.launch()