import torch
import numpy as np
import cv2
import os
from tqdm import tqdm
from ultralytics import YOLO


def download_model():
    """下载YOLO11n-seg模型（如果尚未下载）"""
    # YOLO11n-seg会在首次使用时自动下载
    print("正在加载YOLO11n-seg模型...")
    model = YOLO("yolo11n-seg.pt")
    print("模型加载完成")
    return model


def process_video(video_path, new_bg_path, output_dir="output", conf_threshold=0.3):
    """
    使用YOLO11n-seg处理视频：分割篮球人物，替换背景，从原视频中移除人物

    Args:
        video_path: 输入视频路径
        new_bg_path: 新背景图片路径
        output_dir: 输出目录
        conf_threshold: 检测置信度阈值
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)

    # 加载YOLO11n-seg模型
    model = download_model()

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 加载新背景图片
    new_bg = cv2.imread(new_bg_path)
    if new_bg is None:
        raise ValueError(f"无法加载背景图片: {new_bg_path}")
    new_bg = cv2.resize(new_bg, (width, height))

    # 创建输出视频文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    segmented_video = cv2.VideoWriter(os.path.join(output_dir, "segmented.mp4"),
                                      fourcc, fps, (width, height))
    transplanted_video = cv2.VideoWriter(os.path.join(output_dir, "transplanted.mp4"),
                                         fourcc, fps, (width, height))
    inpainted_video = cv2.VideoWriter(os.path.join(output_dir, "inpainted.mp4"),
                                      fourcc, fps, (width, height))

    # 读取视频的所有帧
    print("读取视频帧...")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"共读取 {len(frames)} 帧")

    # 处理每一帧以获取人物掩码
    print("使用YOLO11n-seg进行人物分割...")
    all_masks = []
    person_found = False

    for i, frame in enumerate(tqdm(frames)):
        # 使用YOLO进行实例分割，仅检测类别0（person）
        results = model(frame, classes=0, conf=conf_threshold)

        # 创建一个空掩码
        mask = np.zeros((height, width), dtype=bool)

        # 如果检测到人，则更新掩码
        if len(results[0].boxes) > 0:
            conf_values = results[0].boxes.conf.cpu().numpy()
            if len(conf_values) > 0:
                person_found = True
                segmentation_masks = results[0].masks
                if segmentation_masks is not None:
                    # 选择置信度最高的人物掩码
                    best_idx = np.argmax(conf_values)
                    best_mask = segmentation_masks.data[best_idx].cpu().numpy()
                    mask = cv2.resize(best_mask, (width, height))
                    mask = mask > 0.5  # 二值化
        # 如果当前帧没有检测到人，但之前检测到过，则沿用上一帧的掩码
        elif person_found and all_masks:
            mask = all_masks[-1]

        all_masks.append(mask)

        # 每30帧保存一次掩码可视化
        if i % 30 == 0:
            mask_viz = mask.astype(np.uint8) * 255
            cv2.imwrite(os.path.join(output_dir, "frames", f"mask_{i:04d}.png"), mask_viz)

    # 对掩码进行简单时序平滑，使用投票机制减少闪烁
    print("对掩码进行时序平滑...")
    smoothed_masks = []
    window_size = 5  # 平滑窗口大小
    for i in range(len(all_masks)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(all_masks), i + window_size // 2 + 1)
        window_masks = all_masks[start_idx:end_idx]
        sum_mask = np.zeros_like(all_masks[0], dtype=np.float32)
        for m in window_masks:
            sum_mask += m.astype(np.float32)
        threshold = len(window_masks) / 2
        smoothed_mask = sum_mask > threshold
        smoothed_masks.append(smoothed_mask)

    # 进行视频处理，生成分割、移植和修复视频
    print("生成输出视频...")
    # 为了更智能地填充背景，我们在修复部分使用 OpenCV 的 inpaint 算法（基于 Telea 方法）
    for i, (frame, mask) in enumerate(tqdm(zip(frames, smoothed_masks), total=len(frames))):
        # 1. 分割视频：仅显示检测到的人物
        segmented = frame.copy()
        segmented[~mask] = 0
        segmented_video.write(segmented)

        # 2. 移植视频：将人物区域从原视频复制到新背景上
        transplanted = new_bg.copy()
        transplanted[mask] = frame[mask]
        transplanted_video.write(transplanted)

        # 3. 修复视频：移除人物并修复背景
        # 将布尔掩码转换为 0/255 的 uint8 掩码（inpaint 要求的格式）
        mask_uint8 = (mask.astype(np.uint8)) * 255
        # 使用 cv2.inpaint 进行背景修复，参数 inpaintRadius 可以根据实际效果调节
        inpainted = cv2.inpaint(frame, mask_uint8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        inpainted_video.write(inpainted)

    # 释放视频写入资源
    segmented_video.release()
    transplanted_video.release()
    inpainted_video.release()

    print(f"处理完成！输出文件保存在 {output_dir} 目录")
    print(f"- 分割视频: {os.path.join(output_dir, 'segmented.mp4')}")
    print(f"- 移植视频: {os.path.join(output_dir, 'transplanted.mp4')}")
    print(f"- 修复视频: {os.path.join(output_dir, 'inpainted.mp4')}")


def main():
    """主函数"""
    video_path = "gg_24_to_28.mp4"       # 输入视频路径（打篮球的视频）
    new_bg_path = "image/background.png"      # 新背景图片路径
    output_dir = "output"

    if not os.path.exists(video_path):
        print(f"错误: 找不到输入视频 {video_path}")
        return
    if not os.path.exists(new_bg_path):
        print(f"错误: 找不到背景图片 {new_bg_path}")
        return

    process_video(video_path, new_bg_path, output_dir)


if __name__ == "__main__":
    main()
