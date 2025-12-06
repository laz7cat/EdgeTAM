"""
EdgeTAM Python 前端客户端
使用 OpenCV 窗口显示视频，直接连接后端 WebRTC 服务
"""
import asyncio
import json
import cv2
import numpy as np
import requests
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("edgetam_client")

BACKEND_URL = "http://localhost:7860"

# 全局变量
current_frame = None
pc = None
click_coords = []


class VideoReceiver:
    """接收并处理视频帧"""
    
    def __init__(self):
        self.frame = None
        
    async def receive_frames(self, track):
        """持续接收视频帧"""
        try:
            while True:
                frame = await track.recv()
                # 转换 av.VideoFrame 为 numpy array
                img = frame.to_ndarray(format="bgr24")
                self.frame = img
        except Exception as e:
            logger.error(f"接收帧错误: {e}")


async def setup_webrtc():
    """建立 WebRTC 连接"""
    global pc
    
    logger.info("创建 RTCPeerConnection...")
    config = {
        "sdpSemantics": "unified-plan",
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
    
    pc = RTCPeerConnection()
    
    # 用于接收视频帧
    receiver = VideoReceiver()
    
    @pc.on("track")
    async def on_track(track):
        logger.info(f"收到轨道: {track.kind}")
        if track.kind == "video":
            logger.info("开始接收视频流...")
            asyncio.create_task(receiver.receive_frames(track))
    
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"连接状态: {pc.connectionState}")
    
    # 添加接收视频的 transceiver
    pc.addTransceiver("video", direction="recvonly")
    
    # 创建 offer
    logger.info("创建 offer...")
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    
    # 等待 ICE 收集
    await asyncio.sleep(0.5)
    
    # 发送 offer 到后端
    logger.info("发送 offer 到后端...")
    response = requests.post(
        f"{BACKEND_URL}/offer",
        json={
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        },
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code != 200:
        raise Exception(f"后端响应错误: {response.status_code}")
    
    # 设置远端描述
    answer_data = response.json()
    answer = RTCSessionDescription(sdp=answer_data["sdp"], type=answer_data["type"])
    await pc.setRemoteDescription(answer)
    
    logger.info("WebRTC 连接已建立!")
    return receiver


def send_click(x, y):
    """发送点击坐标到后端"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/click",
            json={"x": x, "y": y, "label": 1},
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            logger.info(f"点击已发送: ({x}, {y})")
        else:
            logger.error(f"发送点击失败: {response.status_code}")
    except Exception as e:
        logger.error(f"发送点击错误: {e}")


def send_reset():
    """发送重置信号到后端"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/reset",
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            logger.info("重置信号已发送")
        else:
            logger.error(f"发送重置失败: {response.status_code}")
    except Exception as e:
        logger.error(f"发送重置错误: {e}")


def mouse_callback(event, x, y, flags, param):
    """鼠标回调函数"""
    if event == cv2.EVENT_LBUTTONDOWN:
        logger.info(f"检测到点击: ({x}, {y})")
        send_click(x, y)


async def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("EdgeTAM Python 前端客户端")
    logger.info("=" * 60)
    logger.info(f"连接到后端: {BACKEND_URL}")
    logger.info("=" * 60)
    
    # 建立 WebRTC 连接
    receiver = await setup_webrtc()
    
    # 等待第一帧
    logger.info("等待视频流...")
    while receiver.frame is None:
        await asyncio.sleep(0.1)
    
    logger.info("收到第一帧!")
    
    # 创建窗口
    window_name = "EdgeTAM 实时跟踪器"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("使用说明:")
    logger.info("  - 左键点击视频画面中的物体开始跟踪")
    logger.info("  - 按 'R' 键重置跟踪状态")
    logger.info("  - 按 'Q' 或 'ESC' 键退出")
    logger.info("=" * 60)
    
    # 显示循环
    try:
        while True:
            await asyncio.sleep(0.01)  # 允许异步任务运行
            
            if receiver.frame is not None:
                frame = receiver.frame.copy()
                
                # 添加提示文字
                # cv2.putText(frame, "Click to track | Press R to reset | Press Q to quit",
                #            (10, frame.shape[0] - 20),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow(window_name, frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q 或 ESC
                logger.info("退出程序...")
                break
            elif key == ord('r') or key == ord('R'):
                logger.info("重置跟踪状态...")
                send_reset()
                
    finally:
        cv2.destroyAllWindows()
        if pc:
            await pc.close()
        logger.info("已关闭连接")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n程序被用户中断")
    except Exception as e:
        logger.error(f"程序错误: {e}", exc_info=True)
