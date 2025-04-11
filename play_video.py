import cv2


def play_video(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 判断视频是否成功打开
    if not cap.isOpened():
        print("无法打开视频文件:", video_path)
        return

    # 循环读取视频帧
    while cap.isOpened():
        ret, frame = cap.read()  # ret 表示是否成功读取帧，frame 为读取的图像帧

        if ret:
            # 显示图像帧
            cv2.imshow("video", frame)
            # waitKey(25) 控制帧率，同时监听键盘输入。如果按下 'q' 键，退出播放
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
        else:
            # 如果没有读取到帧，则退出循环（通常视频播放完毕）
            break

    # 释放视频捕获对象并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 请将 'your_video.mp4' 替换成你的视频路径
    play_video("output/inpainted.mp4")
    # play_video("output/segmented.mp4")
    # play_video("output/segmented.mp4")

