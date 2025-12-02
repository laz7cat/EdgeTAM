import argparse
import asyncio
import json
import logging
import os
import time
import cv2
import uvicorn
import threading
import queue
from fractions import Fraction
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from av import VideoFrame
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simple_webrtc")

class CameraTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self):
        super().__init__()
        # Use a queue to decouple capture from sending
        # maxsize=1 ensures we always work on the freshest frame and drop old ones if processing is slow
        self.frame_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        
        self.start_time = time.time()
        self.frame_count = 0
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _capture_loop(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Try DirectShow for better FPS on Windows
        if not cap.isOpened():
            # Fallback
            cap = cv2.VideoCapture(0)
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            logger.error("CRITICAL: Could not open camera.")
            return

        logger.info(f"Camera opened. Backend: {cap.getBackendName()}")

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Put in queue. If full, remove old item first (drop frame strategy)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.frame_queue.put(frame)
            
        cap.release()

    async def recv(self):
        # Manual PTS
        pts = int(time.time() * 90000)
        time_base = Fraction(1, 90000)
        
        # Get frame from queue (non-blocking or minimal blocking)
        # Run in executor to avoid blocking event loop if queue is empty
        loop = asyncio.get_event_loop()
        try:
            # This might block slightly if queue empty, but thread is filling it fast
            # Using executor makes it truly async-friendly
            frame = await loop.run_in_executor(None, self.frame_queue.get)
        except Exception:
            frame = np.zeros((720, 1280, 3), dtype='uint8')

        # Calculate FPS
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            fps = self.frame_count / elapsed
        else:
            fps = 0
            
        # Draw FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        # Convert BGR to RGB (aiortc expects RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        new_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        new_frame.pts = pts
        new_frame.time_base = time_base
        return new_frame

    def stop(self):
        self.stop_event.set()
        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        super().stop()

app = FastAPI()
pcs = set()

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
    <head>
        <title>Simple WebRTC</title>
    </head>
    <body>
        <h1>Simple WebRTC Stream (Threaded)</h1>
        <video id="video" autoplay playsinline muted style="width: 1280px; border: 1px solid black;"></video>
        <br>
        <button onclick="start()">Start</button>
        <div id="log"></div>
        
        <script>
            var pc = null;
            
            function log(msg) {
                document.getElementById('log').innerHTML += msg + "<br>";
            }

            async function start() {
                var config = {
                    sdpSemantics: 'unified-plan',
                    iceServers: [{ urls: ['stun:stun.l.google.com:19302'] }]
                };

                pc = new RTCPeerConnection(config);

                pc.addEventListener('track', function(evt) {
                    if (evt.track.kind == 'video') {
                        document.getElementById('video').srcObject = evt.streams[0];
                        log("Track received");
                    }
                });
                
                pc.addEventListener('iceconnectionstatechange', function() {
                    log("ICE State: " + pc.iceConnectionState);
                });

                var offer = await pc.createOffer({ offerToReceiveVideo: true });
                await pc.setLocalDescription(offer);

                log("Local description set. Gathering candidates...");
                await new Promise(r => setTimeout(r, 1000));
                log("Sending offer...");
                
                try {
                    var response = await fetch('/offer', {
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
                    log("Remote description set");
                } catch (e) {
                    log("Error sending offer: " + e);
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    video = CameraTrack()
    pc.addTrack(video)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return JSONResponse({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })

@app.on_event("shutdown")
async def on_shutdown():
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

if __name__ == "__main__":
    # Use 0.0.0.0 to bind all interfaces
    uvicorn.run(app, host="0.0.0.0", port=7860)