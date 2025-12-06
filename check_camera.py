"""
检查摄像头支持的分辨率和参数
"""
import cv2
import sys

def test_camera_resolution(camera_id=0):
    """测试摄像头支持的分辨率"""
    print("=" * 70)
    print(f"检查摄像头 {camera_id} 的分辨率支持情况")
    print("=" * 70)
    
    # 尝试打开摄像头
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"使用 CAP_DSHOW 打开失败，尝试默认后端...")
        cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头 {camera_id}")
        return
    
    print(f"✅ 摄像头 {camera_id} 打开成功\n")
    
    # 获取当前默认设置
    print("📋 当前默认设置:")
    print("-" * 70)
    default_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    default_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    default_fps = cap.get(cv2.CAP_PROP_FPS)
    default_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((default_fourcc >> 8 * i) & 0xFF) for i in range(4)])
    
    print(f"  分辨率: {int(default_width)} x {int(default_height)}")
    print(f"  FPS: {default_fps}")
    print(f"  编码格式: {fourcc_str}")
    print()
    
    # 常见的分辨率列表
    resolutions = [
        (320, 240, "QVGA"),
        (640, 480, "VGA"),
        (800, 600, "SVGA"),
        (1024, 768, "XGA"),
        (1280, 720, "HD 720p"),
        (1280, 960, "960p"),
        (1600, 1200, "UXGA"),
        (1920, 1080, "Full HD 1080p"),
        (2560, 1440, "2K QHD"),
        (3840, 2160, "4K UHD"),
    ]
    
    print("🔍 测试常见分辨率支持情况:")
    print("-" * 70)
    print(f"{'分辨率':<20} {'名称':<15} {'状态':<10} {'实际分辨率':<20} FPS")
    print("-" * 70)
    
    supported_resolutions = []
    
    for width, height, name in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # 读取实际设置的值
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 判断是否支持
        if actual_width == width and actual_height == height:
            status = "✅ 支持"
            supported_resolutions.append((width, height, name, actual_fps))
        else:
            status = "❌ 不支持"
        
        actual_res = f"{actual_width}x{actual_height}"
        print(f"{width}x{height:<15} {name:<15} {status:<10} {actual_res:<20} {actual_fps}")
    
    print()
    print("=" * 70)
    print(f"✅ 支持的分辨率总数: {len(supported_resolutions)}")
    print("=" * 70)
    
    if supported_resolutions:
        print("\n📊 推荐使用的分辨率:")
        print("-" * 70)
        for width, height, name, fps in supported_resolutions:
            print(f"  {width}x{height} ({name}) @ {fps} FPS")
    
    # 测试读取一帧
    print("\n🎥 测试帧读取:")
    print("-" * 70)
    # 恢复到一个支持的分辨率
    if supported_resolutions:
        test_w, test_h, test_name, _ = supported_resolutions[0]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, test_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, test_h)
    
    ret, frame = cap.read()
    if ret:
        print(f"✅ 成功读取帧")
        print(f"   帧大小: {frame.shape[1]}x{frame.shape[0]}")
        print(f"   帧格式: {frame.shape[2]} 通道, dtype={frame.dtype}")
    else:
        print(f"❌ 无法读取帧")
    
    # 测试不同的编码格式
    print("\n🎬 测试不同编码格式 (在 640x480):")
    print("-" * 70)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    codecs = [
        ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
        ('YUYV', cv2.VideoWriter_fourcc(*'YUYV')),
        ('YUY2', cv2.VideoWriter_fourcc(*'YUY2')),
        ('H264', cv2.VideoWriter_fourcc(*'H264')),
    ]
    
    for codec_name, fourcc in codecs:
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        actual_codec = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        ret, frame = cap.read()
        if ret and actual_codec.strip().upper() == codec_name:
            print(f"  {codec_name:<8} ✅ 支持")
        else:
            print(f"  {codec_name:<8} ❌ 不支持 (实际: {actual_codec})")
    
    cap.release()
    print("\n" + "=" * 70)
    print("检查完成！")
    print("=" * 70)


if __name__ == "__main__":
    camera_id = 0
    if len(sys.argv) > 1:
        try:
            camera_id = int(sys.argv[1])
        except ValueError:
            print(f"无效的摄像头 ID: {sys.argv[1]}")
            print("使用方法: python check_camera.py [camera_id]")
            sys.exit(1)
    
    test_camera_resolution(camera_id)
