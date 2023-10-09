import os.path as osp
import cv2
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
from PIL import Image
from ..builder import PIPELINES


@PIPELINES.register_module
class LoadImageFromFile_mhp(object):
    """
    input: result --- [{
                    'filepath' : img_add
                    'width' : image.size[1]
                    'height' : image.size[0]
                    'bboxes' : [{
                                        'class': 'person',
                                        'ann_path': ann_root + ann_add,
                                        'x1': x1,
                                        'y1': y1,
                                        'x2': x2,
                                        'y2': y2
                                        }]
                    }]
    return: result --- [{
                        'filename' : filename (str)
                        'img' : img (np.array)
                        'img_shape' : img.shape
                        'ori_shape' : img.shape 
    }]
    """
    def __init__(self, to_float32=False, color_type='color'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        filename = results['img_info']['filepath']

        img =  mmcv.imread(filename)
        # mmcv.imread(filename, self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)
        if img is None:
            try:
                import cv2
                img = cv2.imread(filename)
            except Exception as e:
                print(filename)
                print(e)
                import pdb;pdb.set_trace()
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape


        return results

    def __repr__(self):
        return '{} (to_float32={}, color_type={})'.format(
            self.__class__.__name__, self.to_float32, self.color_type)


@PIPELINES.register_module
class LoadAnnotations_mhp(object):
    """
    input: result --- [{
                    'filepath' : img_add
                    'width' : image.size[1]
                    'height' : image.size[0]
                    'bboxes' : [{
                                        'class': 'person',
                                        'ann_path': ann_root + ann_add,
                                        'x1': x1,
                                        'y1': y1,
                                        'x2': x2,
                                        'y2': y2
                                        }]
                    }]
    return: result --- [{
                        'filename' : filename (str)
                        'img' : img (np.array)
                        'img_shape' : img.shape
                        'ori_shape' : img.shape 
    }]
    """

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=True,
                 with_seg=False,
                 with_mhp_parsing=True,
                 with_keypoints=False,
                 ):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_seg = with_seg
        self.with_mask = with_mask
        self.with_mhp_parsing = with_mhp_parsing
        self.with_keypoints = with_keypoints

    def _load_bboxes(self, results):
        ann_info = results['ann_info']
        bboxes = ann_info['bboxes']
        results['gt_bboxes'] = np.array([[bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']] for bbox in bboxes])

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')
        return results

    def _load_labels(self, results):
        ann_info = results['ann_info']
        bboxes = ann_info['bboxes']
        results['gt_labels'] = [1 for bbox in bboxes]
        return results

    def _load_mhp_parsing(self, results):
        ann_info = results['ann_info']
        bboxes = ann_info['bboxes']
        results['gt_parsings_add'] = [bbox['ann_path'] for bbox in bboxes]
        results['gt_parsing'] = []
        for filename in results['gt_parsings_add']:
            ann =  np.array(Image.open(filename)) 
            if len(ann.shape) == 3:
                ann = ann[:, :, 0]  # Make sure ann is a two dimensional np array.
            results['gt_parsing'].append(ann.astype('float32'))
        results['parsing_fields'].append('gt_parsing')
        return results


    def _load_masks(self, results):
        results['gt_masks'] = []
        for filename in results['gt_parsings_add']:
            ann =  np.array(Image.open(filename)) 
            if len(ann.shape) == 3:
                ann = ann[:, :, 0]  # Make sure ann is a two dimensional np array.
            ann = ann.astype('bool').astype(int)
            results['gt_masks'].append(ann.astype('float32'))
        results['mask_fields'].append('gt_masks')
        return results

    def _load_semantic_seg(self, results):
        results['gt_semantic_seg'] = []
        cn = 0
        for filename in results['gt_parsings_add']:
            if cn == 0:
                temp = np.array(Image.open(filename)) 
                if len(temp.shape) == 3:
                    temp = temp[:, :, 0]  # Make sure ann is a two dimensional np array.
            else:
                ann = np.array(Image.open(filename)) 
                if len(ann.shape) == 3:
                    ann = ann[:, :, 0]  # Make sure ann is a two dimensional np array
                temp = temp + ann
            cn = cn + 1
        results['gt_semantic_seg'] = temp.astype('float32')
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def _load_keypoints(self, results):
        results['gt_keypoints'] = results['ann_info']['keypoints']
        results['keypoints_fields'].append('gt_keypoints')
        return results

    def __call__(self, results):

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mhp_parsing:
            results = self._load_mhp_parsing(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        if self.with_keypoints:
            results = self._load_keypoints(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(with_bbox={}, with_label={}, with_mask={}, with_parsing={}'
                     ' with_seg={})').format(self.with_bbox, self.with_label,
                                             self.with_mask, self.with_parsing, self.with_seg)
        return repr_str


