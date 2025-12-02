import argparse
import asyncio
import json
import logging
import os
import time
import cv2
import uvicorn
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
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            logger.error("CRITICAL: Could not open camera (index 0). Check if it's used by another app.")
        else:
            logger.info(f"Camera opened successfully. Backend: {self.cap.getBackendName()}")
            
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.start_time = time.time()
        self.frame_count = 0

    async def recv(self):
        # Manually calculate PTS to avoid AttributeError with next_timestamp
        # Video time base is typically 1/90000
        pts = int(time.time() * 90000)
        time_base = Fraction(1, 90000)
        
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            # Return black frame on failure
            frame = np.zeros((720, 1280, 3), dtype='uint8')
        else:
            # Log every 60 frames to confirm flow without spamming
            if self.frame_count % 60 == 0:
                logger.info(f"Frame {self.frame_count} captured and processing")
        
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
        self.cap.release()

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
        <h1>Simple WebRTC Stream</h1>
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
                    // STUN server is critical
                    iceServers: [{ urls: ['stun:stun.l.google.com:19302'] }]
                };

                pc = new RTCPeerConnection(config);

                // Handle incoming video track
                pc.addEventListener('track', function(evt) {
                    if (evt.track.kind == 'video') {
                        document.getElementById('video').srcObject = evt.streams[0];
                        log("Track received");
                    }
                });
                
                pc.addEventListener('iceconnectionstatechange', function() {
                    log("ICE State: " + pc.iceConnectionState);
                });

                // Create Offer
                var offer = await pc.createOffer({ offerToReceiveVideo: true });
                await pc.setLocalDescription(offer);

                log("Local description set. Gathering candidates...");

                // OPTIMIZATION: Don't wait indefinitely for ICE complete.
                // Wait for a short time (e.g., 500ms) or proceed immediately.
                // aiortc often works without full trickle ICE support in simple examples.
                await new Promise(r => setTimeout(r, 1000));

                log("Sending offer...");
                
                // Send to server
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

    # Add camera track
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
    # Close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
