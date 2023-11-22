import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import os
import sys
import numpy as np
import time
from model import *
import torch.optim as optim 
import random
import pdb
import shutil
from LSR import LSR
from torch.cuda.amp import autocast, GradScaler
from word_index import get_pinyin
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE
from utils.cvtransforms import *
from video_utils import *

jpeg = TurboJPEG()

torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=str, required=False,default="0")
parser.add_argument('--weights', type=str, required=False, default="checkpoints/lrw1000-border-se-mixup-label-smooth-cosine-lr-wd-1e-4-acc-0.56023.pt")
args = parser.parse_args()
args.se=True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

video_model = VideoModel(args).cuda()

def parallel_model(model):
    model = nn.DataParallel(model)
    return model        

def load_missing(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}                
    missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
    print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
    print('miss matched params:',missed_params)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model
    
if(args.weights is not None):
    print('load weights')
    weight = torch.load(args.weights, map_location=torch.device('cpu'))    
    load_missing(video_model, weight.get('video_model'))
         
video_model = parallel_model(video_model)
from flask import Flask, request

app = Flask(__name__)

@app.route('/lip-reading', methods=['POST'])
def lip_reading():
    raw_video_path = request.form['raw_video_path']
    start = float(request.form['start'])
    end = float(request.form['end'])
    sid = request.form['sid']

    with torch.no_grad():
        pkl = get_video_pkl(raw_video_path, sid=sid, start=start, end=end)
        video = pkl.get('video')
        video = [jpeg.decode(img[0], pixel_format=TJPF_GRAY) for img in video]
        video = np.stack(video, 0)
        video = video[:, :, :, 0]
        video = CenterCrop(video, (88, 88))
        pkl['video'] = torch.FloatTensor(video)[:, None, ...] / 255.0

        video_model.eval()
        video = pkl['video'].cuda(non_blocking=True)
        border = torch.Tensor(pkl['duration']).cuda(non_blocking=True).float()

        with autocast():
            video = video.unsqueeze(0)
            border = border
            y_v = video_model(video, border)
            pinyin = get_pinyin(y_v.argmax(-1))
    return pinyin
        
if(__name__ == '__main__'):
    app.run(debug=True)