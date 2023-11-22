import cv2
import os
from torch.utils.data import Dataset
import cv2
import os
import glob
import numpy as np
import random
import torch
from collections import defaultdict
import sys
from torch.utils.data import DataLoader
from turbojpeg import TurboJPEG #, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE


jpeg = TurboJPEG()
class LRW1000_Dataset(Dataset):
    
    def __init__(self, infos, pic_dir):
        
        self.data = []
        self.data_root = pic_dir
        lines = infos
        self.padding = 40
        pinyins = sorted(np.unique([line[2] for line in lines]))
        self.data = [(line[0], int(float(line[3])*25)+1, int(float(line[4])*25)+1, pinyins.index(line[2])) for line in lines]
        max_len = max([data[2]-data[1] for data in self.data])
        print(f"before filter :{self.data}")
        data = list(filter(lambda data: data[2]-data[1] <= self.padding, self.data))                
        print(f"after filter: {data}")
        self.lengths = [data[2]-data[1] for data in self.data]
        self.pinyins = pinyins
        self.class_dict = defaultdict(list)
        for item in data:
            item = (item[0], "", item[1], item[2], item[3])                                
            self.class_dict[item[-1]].append(item)                

        self.data = []            
        self.unlabel_data = []
        for k, v in self.class_dict.items():
            n = len(v) 
            self.data.extend(v[:n])            
                      
    def __len__(self):
        return len(self.data)

    def load_video(self, item):
        #load video into a tensor
        (path, mfcc, op, ed, label) = item
        inputs, border = self.load_images(os.path.join(self.data_root, path), op, ed)        
                
        result = {}        
                
        result['video'] = inputs
        result['label'] = int(label)        
        result['duration'] = border.astype(np.bool_)
        
        savename = f'{path}_{op}_{ed}.pkl'
        torch.save(result, savename)
        return result

    def __getitem__(self, idx):

        r = self.load_video(self.data[idx])
        return r

    def load_images(self, path, op, ed):
        center = (op + ed) / 2
        length = (ed - op + 1)
        
        op = int(center - self.padding // 2)
        ed = int(op + self.padding)
        left_border = max(int(center - length / 2 - op), 0)
        right_border = min(int(center + length / 2 - op), self.padding)
        files =  [os.path.join(path, '{}.jpg'.format(i)) for i in range(op, ed)]
        files = filter(lambda path: os.path.exists(path), files)
        files = [cv2.imread(file) for file in files]
        files = [cv2.resize(file, (96, 96)) for file in files]        
        files = np.stack(files, 0)        
        t = files.shape[0]
        tensor = np.zeros((40, 96, 96, 3)).astype(files.dtype)
        border = np.zeros((40))
        tensor[:t,...] = files.copy()
        border[left_border:right_border] = 1.0
        tensor = [jpeg.encode(tensor[_]) for _ in range(40)]
        return tensor, border


def get_video_pkl(video_file,sid,fps=20,start=0,end=0):
    output_folder = f"tmp/pics/{sid}"
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
    infos = [
        [sid,"填充文本","tian chong wen ben",start,end]
    ]
    
    pic_dir = "tmp/pics/"
    loader = DataLoader(LRW1000_Dataset(infos, pic_dir),
            batch_size = 1, 
            num_workers = 1,   
            shuffle = False,         
            drop_last = False)
    for batch in loader:
        return batch

# 使用示例
# video_file = 'input_videos/gfb.mp4'
# output_folder = 'output_pics/gfb'
# video_to_frames(video_file, output_folder)
