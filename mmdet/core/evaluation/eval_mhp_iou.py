import sys, os
import numpy as np
import cv2
import pickle, gzip
from tqdm import tqdm, trange
from mmdet.utils.voc_eval import voc_ap
import scipy.sparse
from PIL import Image
import mmcv
import copy

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    try:
        hist = np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)
    except Exception as e:
        import pdb;pdb.set_trace()
    return hist


def compute_hist(predict_list, im_dir, num_parsing):
    n_class = num_parsing
    hist = np.zeros((n_class, n_class))

    gt_root = im_dir

    for predict_png in tqdm(predict_list, desc='Calculating IoU ..'):
        gt_png = os.path.join(gt_root, predict_png.split('/')[-1])
        label = np.array(Image.open(gt_png))
        if len(label.shape)==3: label = label[:,:,0] # Make sure ann is a two dimensional np array. 
        tmp = np.array(Image.open(predict_png))
        if len(tmp.shape)==3: tmp =tmp[:,:,0] # Make sure ann is a two dimensional np array. 
        # label = cv2.imread(gt_png, 0)
        # tmp = cv2.imread(predict_png, 0)
        if np.max(label) > n_class:
            import pdb;pdb.set_trace()
        assert label.shape == tmp.shape, '{} VS {}'.format(str(label.shape), str(tmp.shape))
            
        gt = label.flatten()
        pre = tmp.flatten()
        try:
            hist += fast_hist(gt, pre, n_class)
        except Exception as e :
            print(e)
            import pdb;pdb.set_trace()
    return hist

def mean_IoU(overall_h):
    iu = np.diag(overall_h) / (overall_h.sum(1) + overall_h.sum(0) - np.diag(overall_h))
    return iu, np.nanmean(iu)


def per_class_acc(overall_h):
    acc = np.diag(overall_h) / overall_h.sum(1)
    return np.nanmean(acc)


def pixel_wise_acc(overall_h):
    return np.diag(overall_h).sum() / overall_h.sum()


def parsing_iou(predict_root, im_dir, num_parsing):
    predict_list = glob.glob(predict_root + '/*.png')

    hist = compute_hist(predict_list, im_dir, num_parsing)
    _iou, _miou = mean_IoU(hist)
    mean_acc = per_class_acc(hist)
    pixel_acc = pixel_wise_acc(hist)

    return _iou, _miou, mean_acc, pixel_acc

