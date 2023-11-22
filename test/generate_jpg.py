import cv2
import os

def video_to_frames(video_file, output_folder,fps=20):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    video_capture = cv2.VideoCapture(video_file)
    # fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    # 计算帧间隔
    frame_interval = int(round(1000/fps))

    # 逐帧读取视频并保存为图片
    success, frame = video_capture.read()
    while success:
        frame_count += 1
        # 转换为灰度图像
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 重采样为88*88
        resized_frame = cv2.resize(gray_frame, (88, 88))
        # 归一化
        normalized_frame = cv2.normalize(resized_frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        # 保存图片
        output_path = os.path.join(output_folder, '{}.jpg'.format(frame_count))
        cv2.imwrite(output_path, normalized_frame)

        # 读取下一帧
        video_capture.set(cv2.CAP_PROP_POS_MSEC, (frame_count*frame_interval))
        success, frame = video_capture.read()

    # 释放视频文件
    video_capture.release()

# 使用示例
video_file = 'input_videos/gfb.mp4'
output_folder = 'output_pics/gfb'
video_to_frames(video_file, output_folder)
