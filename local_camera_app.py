import cv2
import time

# 全局变量，用于存储窗口名称，供鼠标回调函数使用
WINDOW_NAME = 'Local Camera (High FPS - Fixed Aspect Ratio)'

# 鼠标回调函数
def mouse_callback(event, x, y, flags, param):
    """
    当鼠标在窗口上发生事件时，OpenCV 会调用此函数。
    我们检测左键点击事件，并打印点击坐标。
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"鼠标在 ({x}, {y}) 处被点击。")

def main():
    # 打开本地摄像头（索引 0）
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头，请检查设备连接。")
        return

    print("已启动本地摄像头预览。")
    print("按 'q' 键退出程序。")
    print("按 'r' 键重置摄像头连接。")
    print("左键点击画面，将打印点击坐标。")

    prev_time = 0
    
    # 设置窗口名称，并使用 cv2.WINDOW_AUTOSIZE 固定画面比例
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    # 注册鼠标回调函数
    # OpenCV 将会把发生在 WINDOW_NAME 窗口上的鼠标事件，传递给 mouse_callback 函数处理
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取画面帧。")
            break

        # --- 计算 FPS ---
        current_time = time.time()
        fps = 0
        if prev_time != 0:
            time_diff = current_time - prev_time
            if time_diff > 0:
                fps = 1.0 / time_diff
        prev_time = current_time

        # --- 绘制 FPS 信息 (左上角) ---
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- 显示画面 ---
        cv2.imshow(WINDOW_NAME, frame)

        # --- 键盘交互 ---
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            # 按 'q' 退出
            print("退出程序...")
            break
        elif key == ord('r'):
            # 按 'r' 模拟 "Reset" 按钮：重新初始化摄像头
            print("正在重置摄像头...")
            cap.release()
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("重置失败：无法重新打开摄像头。")
                break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()