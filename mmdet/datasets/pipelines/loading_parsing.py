import os.path as osp
import cv2
import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from ..builder import PIPELINES


@PIPELINES.register_module
class LoadImageFromFile_Parsing(object):

    def __init__(self, to_float32=False, color_type='color'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        if results['img_prefix'] is not None:
            #import pdb;pdb.set_trace()
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            if 'filename' not in results['img_info']:
                filename = results['img_info']['filepath']
            else:
                filename = results['img_info']['filename']
        img = mmcv.imread(filename, self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

    def __repr__(self):
        return '{} (to_float32={}, color_type={})'.format(
            self.__class__.__name__, self.to_float32, self.color_type)


@PIPELINES.register_module
class LoadAnnotations_Parsing(object):

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 with_parsing=False,
                 with_mhp_parsing=False,
                 with_keypoints=False,
                 RLE2parsing=True,
                 poly2mask=True):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.with_parsing = with_parsing
        self.with_mhp_parsing = with_mhp_parsing
        self.with_keypoints = with_keypoints
        self.RLE2parsing = RLE2parsing

    def _load_bboxes(self, results):
        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes']

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')
        return results

    def _load_labels(self, results):
        results['gt_labels'] = results['ann_info']['labels']
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask
    
    def _poly2mask(self, mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask
    
    def _dp_mask_to_mask(self, polys):
        
        semantic_mask = np.zeros((256, 256), dtype=np.uint8)
        if len(polys) < 14:
            return semantic_mask
        for i in range(1, 15):
            if polys[i-1]:
                current_mask = maskUtils.decode(polys[i - 1])
                semantic_mask[current_mask > 0] = i

        return semantic_mask
    
    def _rle2parsing(self, mask_ann, h, w, bbox):
        # encoded dp_mask
        semantic_mask = np.zeros((h, w), dtype=np.uint8)
        if mask_ann == []:
            return semantic_mask
        mask = self._dp_mask_to_mask(mask_ann)
        bbr = np.array(bbox).astype(int)
        x1, y1, x2, y2 = bbr[0], bbr[1], bbr[2]+1, bbr[3]+1
        x2, y2 = min(x2, w), min(y2, h)
        if x1 < x2 and y1 < y2:
            mask = cv2.resize(mask, (int(x2 - x1), int(y2 - y1)),
                                      interpolation=cv2.INTER_NEAREST)
            mask_bool = np.where(mask > 0, 1, 0)
            semantic_mask[y1:y2, x1:x2][mask_bool > 0] = mask[mask_bool > 0]
            mask = semantic_mask
        return mask

    def _load_masks(self, results):
        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        if self.poly2mask:
            gt_masks = [self._poly2mask(mask, h, w) for mask in gt_masks]
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results
    
    def _load_parsing(self, results):
        
        h, w = results['img_info']['height'], results['img_info']['width']
        gt_parsing = results['ann_info']['parsing']
        
        if gt_parsing == None:
            return None
        if self.RLE2parsing:
            try:
                gt_parsing = [self._rle2parsing(gt_parsing[i], h, w, results['gt_bboxes'][i]) for i in range(len(results['gt_bboxes']))]
            except Exception as e:
                print(e)
                import pdb;pdb.set_trace()
        results['gt_parsing'] = gt_parsing
        results['parsing_fields'].append('gt_parsing')
        return results

    def _load_keypoints(self, results):
        
        h, w = results['img_info']['height'], results['img_info']['width']
        results['gt_keypoints'] = results['ann_info']['keypoints']
        results['keypoints_fields'].append('gt_keypoints')
        return results



    def _load_semantic_seg(self, results):
        results['gt_semantic_seg'] = mmcv.imread(
            osp.join(results['seg_prefix'], results['ann_info']['seg_map']),
            flag='unchanged').squeeze()
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __call__(self, results):
        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        if self.with_parsing:
            results = self._load_parsing(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(with_bbox={}, with_label={}, with_mask={}, with_parsing={}'
                     ' with_seg={})').format(self.with_bbox, self.with_label,
                                             self.with_mask, self.with_parsing, self.with_seg)
        return repr_str

