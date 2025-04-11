import os

from moviepy import VideoFileClip


def convert_mp4_to_gif(input_path, output_path=None, duration=5, fps=15):
    """
    将MP4视频转换为GIF格式

    参数:
        input_path (str): 输入MP4文件的路径
        output_path (str, 可选): 输出GIF文件的路径。如果未提供，将使用与输入相同的名称但扩展名为.gif
        duration (float, 可选): 要转换的视频持续时间（秒），默认为5秒
        fps (int, 可选): GIF的每秒帧数，默认为15

    返回:
        str: 输出GIF文件的路径
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    # 如果未提供输出路径，则基于输入路径生成
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + ".gif"

    try:
        # 加载视频
        clip = VideoFileClip(input_path)

        # 如果视频时长超过指定持续时间，则裁剪视频
        if clip.duration > duration:
            clip = clip.subclip(0, duration)

        # 转换为GIF
        clip.write_gif(output_path, fps=fps)

        print(f"成功将视频转换为GIF: {output_path}")
        return output_path

    except Exception as e:
        print(f"转换过程中出错: {str(e)}")
        raise
    finally:
        # 确保关闭资源
        if 'clip' in locals():
            clip.close()


if __name__ == "__main__":
    # 硬编码文件路径
    input_path = r"output/transplanted.mp4"  # 在这里修改输入MP4文件的路径
    output_path = "image/trans.gif"  # 在这里修改输出GIF文件的路径
    duration = 5  # 截取的视频持续时间（秒）
    fps = 5  # GIF的每秒帧数

    # 执行转换
    convert_mp4_to_gif(input_path, output_path, duration, fps)