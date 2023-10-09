import torch
import torch.nn as nn

from .. import builder
from ..builder import DETECTORS
from .base import BaseDetector
import copy


@DETECTORS.register_module
class SingleStageInsParsingDetector(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 visual_data=False):

        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        super(SingleStageInsParsingDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        else:
            self.neck = None

        if bbox_head is not None:
            bbox_head.update(train_cfg=copy.deepcopy(train_cfg))
            bbox_head.update(test_cfg=copy.deepcopy(test_cfg))
            self.bbox_head = builder.build_head(bbox_head)
        else:
            self.bbox_head = None

        assert mask_head, f'`mask_head` must ' \
                    f'be implemented in {self.__class__.__name__}'

        mask_head.update(train_cfg=copy.deepcopy(train_cfg))
        mask_head.update(test_cfg=copy.deepcopy(test_cfg))
        self.mask_head = builder.build_head(mask_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        self.eval_file_count = 0
        self.visual_datas = visual_data


    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        x = self.extract_feat(img)
        outs = self.mask_head(x)
        return outs

    def forward_train(self,
                        img,
                        img_metas,
                        gt_bboxes,
                        gt_labels,
                        gt_parsing=None,
                        gt_bboxes_ignore=None,
                        gt_masks=None,
                        gt_semantic_seg=None,
                        gt_keypoints=None):
        x = self.extract_feat(img)
        outs = self.mask_head(x) 
        
        if self.visual_datas:
            self.visual_data(gt_parsing, img, img_metas)
        
        loss_inputs = outs + (gt_bboxes, gt_labels, gt_masks, gt_parsing, gt_semantic_seg, gt_keypoints, img_metas, self.train_cfg)
        losses = self.mask_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses
    
    def visual_data(self, gt_masks, img, img_metas):
        import mmcv
        import numpy as np
        for i in range(len(gt_masks)):
            for j in range(len(gt_masks[i])):
                if self.eval_file_count < 200:
                    img_g = np.zeros_like(img[i].detach().cpu().numpy())
                    img_g[:,:gt_masks[i][j].shape[0], :gt_masks[i][j].shape[1]] = gt_masks[i][j]
                    img_e = (img[i].detach().cpu().numpy()*30+122)*0.5
                    img_e[img_g.astype(bool)] = img_e[img_g.astype(bool)] + 122
                    mmcv.imwrite(img_e.astype(int).transpose((1,2,0)), "eval_file/densepose_parsing/" + img_metas[i]['filename'].split("/")[-1].split(".")[0] + "_" + str(j) + ".png")
                    self.eval_file_count = self.eval_file_count + 1
                else:
                    import pdb;pdb.set_trace()

    def simple_test(self, img, img_meta, rescale=False, **kwargs):
        x = self.extract_feat(img)
        outs = self.mask_head(x, eval=True) 
        seg_inputs = outs + (img_meta, self.test_cfg, rescale)
        seg_result = self.mask_head.get_seg(*seg_inputs)
        return seg_result

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
