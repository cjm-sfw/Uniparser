from mmdet.apis import init_detector, inference_detector_parsing, show_result_pyplot
import mmcv
import numpy as np
import cv2
import os
from PIL import Image

from visual import *

home_root = "/root/multi-parsing/"
data_root = "/root/data/LV-MHP-v2/"

config_file = './configs/repparsing/MHP_r50_fpn_half_gpu_1x_repparsing_DCN_fusion_metrics.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = './work_dirs/MHP_release_r50_fpn_8gpu_1x_repparsing_v0_DCN_fusion_metrics/epoch_12.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
img = data_root + 'val/images/1453.jpg'
gt_name = "7_02_02.png"

result = inference_detector_parsing(model, img)

seg_masks = result[0]
if type(seg_masks) == list:
    seg_masks = np.array(seg_masks)
offset_vis = result[1]
score_list = result[2]

img_ori = mmcv.imread(img)
h,w,_ = img_ori.shape