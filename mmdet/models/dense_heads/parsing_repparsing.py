import os
import mmcv
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time

from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from mmcv.ops import RoIAlign
from mmdet.core import multi_apply
from mmdet.core import parsing_matrix_nms as matrix_nms
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.utils import AdaptBlock
# from mmdet.models.utils import ConvModule, bias_init_with_prob
# AdaptBlock

INF = 1e8
chn_per_part = 22


def center_of_mass(bitmasks, adapt_center=False):
    
    n, h, w = bitmasks.size()
    ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
    xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device)

    m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
    m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
    m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
    center_x = m10 / m00
    center_y = m01 / m00
    
    if adapt_center:
        # TODO
        # center_o_x = center_x
        # center_o_y = center_y
        dist_y = ((torch.ones_like(bitmasks)*ys[:, None]) - center_y.int().reshape(n,1,1))
        dist_x = ((torch.ones_like(bitmasks)*xs) - center_x.int().reshape(n,1,1))
        dist_maps = torch.sqrt(dist_y**2 + dist_x**2)*((~bitmasks.bool()).int()*(np.sqrt(h*h+w*w))+1)
        dists = dist_maps.reshape(n,-1)
        min_index = torch.argmin(dists,1)
        center_y = min_index // w
        center_x = min_index % w
        
        # if torch.tensor([True], device='cuda:0') in (center_o_x.int() != center_x):
        #     print("find adapt_center.")
        #     import pdb;pdb.set_trace()

    return center_x.float(), center_y.float()

def points_nms(heat, kernel=2):
    # kernel must be 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=1)
    keep = (hmax[:, :, :-1, :-1] == heat).float()
    return heat * keep

def dice_coef(input, target):
    a = torch.sum(input * target, 1)+ 0.001
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    return d

def dice_loss(input, target, pos_weight=False, gt_parsing=None):
    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1).float()
    
    input_index = input > 0
    input = input * input_index
    
    if pos_weight:
        import pdb;pdb.set_trace()
        for parsing in gt_parsing:
            gt_parsing[0].sum(0)
        index_pos = target>0
        #index_neg = target==0
        input_pos = input[index_pos]
        target_pos = target[index_pos]
        #input_neg = input[index_neg]
        #target_neg = target[index_neg]
        
        d_pos = dice_coef(input_pos, target_pos)
        d_norm = dice_coef(input, target)
        d = (1-d_pos) * pos_weight + (1-d_norm) * (1-pos_weight)
    else:
        d = dice_coef(input, target)
    return 1-d

@HEADS.register_module()
class Repparsing_Head(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 seg_feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 base_edge_list=(16, 32, 64, 128, 256),
                 scale_ranges=((8, 32), (16, 64), (32, 128), (64, 256), (128, 512)),
                 sigma=0.2,
                 num_grid=None,
                 ins_out_channels=64,
                 enable_center=True,
                 enable_kernel=True,
                 enable_instance=True,
                 enable_semantic=True,
                 enable_fusion=False,
                 enable_fusion_multi=False,
                 enable_metrics=False,
                 enable_moi=False,
                 enable_adapt_center=False,
                 enable_last_res=False,
                 enable_light_convert=False,
                 enable_diceloss_weight=False,
                 loss_ins=None,
                 loss_center=None,
                 loss_semantic=None,
                 loss_parsing=None,
                 loss_metrics=None,
                 loss_semantic_neg=None,
                 loss_semantic_neg_config=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 use_dcn_in_tower=False,
                 type_dcn=None,
                 parsing2mask=False,
                 fp16_enabled = False,
                 **kwargs):
        super(Repparsing_Head, self).__init__()
        self.num_classes = num_classes
        self.seg_num_grids = num_grid
        self.cate_out_channels = self.num_classes - 1
        self.ins_out_channels = ins_out_channels
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.sigma = sigma
        self.stacked_convs = stacked_convs
        self.kernel_out_channels = self.ins_out_channels * 1 * 1
        self.base_edge_list = base_edge_list
        self.scale_ranges = scale_ranges
        self.loss_center = build_loss(loss_center)
        self.ins_loss_weight = loss_ins['loss_weight']
        self.sem_loss = loss_semantic
        self.sem_loss_weight = loss_semantic['loss_weight']
        self.par_loss_weight = loss_parsing['loss_weight']
        self.loss_metrics = loss_metrics
        self.loss_semantic_neg = build_loss(loss_semantic_neg)
        self.loss_semantic_neg_weight = loss_semantic_neg['loss_weight']
        self.loss_semantic_neg_config = loss_semantic_neg_config
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.enable_center = enable_center
        self.enable_instance = enable_instance
        self.enable_semantic = enable_semantic
        self.enable_kernel = enable_kernel
        self.enable_fusion = enable_fusion
        self.enable_fusion_multi = enable_fusion_multi
        self.enable_metrics = enable_metrics
        self.enable_moi = enable_moi
        self.enable_adapt_center = enable_adapt_center
        self.enable_last_res = enable_last_res
        self.enable_light_convert = enable_light_convert
        self.enable_diceloss_weight = enable_diceloss_weight
        self.use_dcn_in_tower = use_dcn_in_tower
        self.type_dcn = type_dcn
        self.chn_per_part = chn_per_part
        self.parsing2mask = parsing2mask
        self.fp16_enabled = fp16_enabled
        self._init_layers()

    def _init_layers(self):
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        if self.enable_center == True:
            self.center_convs = nn.ModuleList()
        if self.enable_kernel == True:
            self.kernel_convs = nn.ModuleList()
            self.avgpooling = nn.AvgPool2d((int(self.seg_num_grids/8), int(self.seg_num_grids/8)), stride=1)
        if self.enable_semantic == True:
            self.seg_kernels = nn.Parameter(torch.empty(self.cate_out_channels, self.seg_feat_channels))
            self.semantic_convs = nn.ModuleList()
        if self.enable_instance == True:
            self.instance_convs = nn.ModuleList()
        if self.enable_fusion == True:
            self.fusion_convs = nn.ModuleList()

        for i in range(self.stacked_convs):
            if self.use_dcn_in_tower:
                cfg_conv = dict(type=self.type_dcn)
            else:
                cfg_conv = self.conv_cfg

            # branch for get segmantic kernels
            if self.enable_kernel:
                chn = self.in_channels + 2 if i == 0 else self.seg_feat_channels
                self.kernel_convs.append(
                    ConvModule(
                        chn,
                        self.seg_feat_channels,
                        3,
                        stride=2, # constract the features to a residual tensor
                        padding=1,
                        conv_cfg=cfg_conv,
                        norm_cfg=norm_cfg,
                        bias=norm_cfg is None))
                
            if self.enable_semantic == True:
                chn = self.in_channels + 2 if i == 0 else self.seg_feat_channels
                self.semantic_convs.append(
                    ConvModule(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=cfg_conv,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))

            if self.enable_instance == True:
                chn = self.in_channels + 2 if i == 0 else self.seg_feat_channels
                self.instance_convs.append(
                    ConvModule(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=cfg_conv,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))

            if self.enable_center == True:
                chn = self.in_channels + 2 if i == 0 else self.seg_feat_channels
                self.center_convs.append(
                    ConvModule(
                        chn,
                        self.seg_feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=cfg_conv,
                        norm_cfg=norm_cfg,
                        bias=norm_cfg is None))

            if self.enable_fusion == True:
                chn = self.seg_feat_channels * 2 if i == 0 else self.seg_feat_channels
                self.fusion_convs.append(
                    ConvModule(
                        chn,
                        self.seg_feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=cfg_conv,
                        norm_cfg=norm_cfg,
                        bias=norm_cfg is None))
            
        if self.enable_light_convert:
            convert_kernel = 1
        else:
            convert_kernel = 3
            
        if self.enable_last_res:
            in_channels = self.in_channels
        else:
            in_channels = self.in_channels * 5
            
        if self.enable_center == True:
            self.dekr_convs = nn.Conv2d(
                in_channels, self.in_channels, convert_kernel, padding=1)
            self.center_heatmap = nn.Conv2d(
                self.seg_feat_channels, 1, 3, padding=1)

        self.segment_converts = nn.Conv2d(
                in_channels, self.in_channels, convert_kernel, padding=1)

        self.ins_coord_conv = nn.Conv2d(
                self.seg_feat_channels, 2, 3, padding=1)

    def init_weights(self):
        if self.enable_kernel:
            for m in self.kernel_convs:
                normal_init(m.conv, std=0.01)
        bias_cate = bias_init_with_prob(0.01)
        
        if self.enable_semantic == True:
            for m in self.semantic_convs:
                normal_init(m.conv, std=0.01)
            # nn.init.orthogonal(self.seg_kernels)
            nn.init.normal_(self.seg_kernels, 0, std=0.01)
            #normal_init(self.seg_kernels, std=0.01)
            normal_init(self.segment_converts, std=0.01, bias=bias_cate)

        if self.enable_instance == True:
            for m in self.instance_convs:
                normal_init(m.conv, std=0.01)
            
        if self.enable_center == True:
            for m in self.center_convs:
                normal_init(m.conv, std=0.01)
            normal_init(self.center_heatmap, std=0.01, bias=bias_cate)
            normal_init(self.dekr_convs, std=0.01)

        if self.enable_fusion == True:
            for m in self.fusion_convs:
                normal_init(m.conv, std=0.01)

    @auto_fp16()
    def forward(self, feats, eval=False):

        # center branch & offset branch & segment branch
        new_feats = self.concate_feats(feats)
        mixed_feats = torch.cat(new_feats, dim=1)
        # mixed_feats.shape: (N, 5*C_in, H, W)
        if self.enable_center == True: 
            center_pred = self.forward_center(mixed_feats, eval)
        else:
            center_pred = None
        if self.enable_kernel:
            kernel_pred = self.forward_kernel(mixed_feats)
        else:
            kernel_pred = None

        if self.enable_semantic == True:
            semantic_pred = self.forward_semantic(mixed_feats)
        else:
            semantic_pred = None
        
        if self.enable_instance == True:
            instance_pred = self.forward_instance(mixed_feats)
        else:
            instance_pred = None

        return center_pred, semantic_pred, instance_pred

    def concate_feats(self, feats):
        return (feats[0],
                F.interpolate(feats[1], size=feats[0].shape[-2:], mode='bilinear',align_corners=True),
                F.interpolate(feats[2], size=feats[0].shape[-2:], mode='bilinear',align_corners=True),
                F.interpolate(feats[3], size=feats[0].shape[-2:], mode='bilinear',align_corners=True),
                F.interpolate(feats[4], size=feats[0].shape[-2:], mode='bilinear',align_corners=True))

    def generate_coordinate(self, feat):
        x_range = torch.linspace(-1, 1, feat.shape[-1], device=feat.device)
        y_range = torch.linspace(-1, 1, feat.shape[-2], device=feat.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([feat.shape[0], 1, -1, -1])
        x = x.expand([feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        return coord_feat

    def forward_center(self, x, eval=False):
        
        # reduce dimension by conv
        if self.enable_last_res:
            input_feat = x[:,:self.in_channels,:,:]
        else:
            input_feat = x
        center_feat = self.dekr_convs(input_feat)

        #concat coord
        coord_feat = self.generate_coordinate(center_feat)
        center_feat = torch.cat([center_feat, coord_feat], 1)

        # resize to grids
        seg_num_grid = self.seg_num_grids
        center_feat = F.interpolate(center_feat, size=seg_num_grid, mode='bilinear', align_corners=True)

        # get center heatmap
        center_feat = center_feat.contiguous()
        for i, center_layer in enumerate(self.center_convs):
            center_feat = center_layer(center_feat)
        center_pred = self.center_heatmap(center_feat)
        if eval:
            center_pred = points_nms(center_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1)

        return center_pred

    def forward_kernel(self, x):
        
        # reduce dimension by 1*1 conv
        if self.enable_last_res:
            input_feat = x[:,:self.in_channels,:,:]
        else:
            input_feat = x
        kernel_feat = self.segment_converts(input_feat)

        #concat coord
        coord_feat = self.generate_coordinate(kernel_feat)
        kernel_feat = torch.cat([kernel_feat, coord_feat], 1)

        # resize to grids
        seg_num_grid = self.seg_num_grids * 2
        kernel_feat = F.interpolate(kernel_feat, size=seg_num_grid, mode='bilinear', align_corners=True)

        # get kernel features
        kernel_feat = kernel_feat.contiguous()
        for i, kernel_layer in enumerate(self.kernel_convs):
            kernel_feat = kernel_layer(kernel_feat)

        kernel_pred = self.avgpooling(kernel_feat)

        return kernel_pred

    def forward_instance(self, x):
        
        if self.enable_last_res:
            input_feat = x[:,:self.in_channels,:,:]
        else:
            input_feat = x
        instance_feat = self.dekr_convs(input_feat)
        # ins branch
        # concat coord
        coord_feat = self.generate_coordinate(instance_feat)
        instance_feat = torch.cat([instance_feat, coord_feat], 1)

        # instance branch

        instance_feat = instance_feat.contiguous()
        for i, instance_layer in enumerate(self.instance_convs):
            instance_feat = instance_layer(instance_feat)

        return instance_feat

    def forward_semantic(self, x):
        
        if self.enable_last_res:
            input_feat = x[:,:self.in_channels,:,:]
        else:
            input_feat = x
        semantic_feat = self.segment_converts(input_feat)
        
        # semantic branch
        # concat coord
        coord_feat = self.generate_coordinate(semantic_feat)
        semantic_feat = torch.cat([semantic_feat, coord_feat], 1)

        semantic_feat = semantic_feat.contiguous()
        for i, semantic_layer in enumerate(self.semantic_convs):
            semantic_feat = semantic_layer(semantic_feat)

        return semantic_feat
    
    @force_fp32(apply_to=('center_preds', 'semantic_preds', 'instance_preds')) 
    def loss(self,
            center_preds, 
            # kernel_preds, 
            semantic_preds, 
            instance_preds,
            gt_bbox_list,
            gt_label_list,
            gt_mask_list,
            gt_parsing_list,
            gt_semantic_seg_list,
            gt_keypoints_list,
            img_metas,
            cfg,
            gt_bboxes_ignore=None):
        
        
        if self.enable_instance:
            mask_feat_size = instance_preds.size()[-2:]
        else:
            mask_feat_size = (int(img_metas[0]['pad_shape'][0] / 4), int(img_metas[0]['pad_shape'][1]/4))
        self.img_metas = img_metas

        # center loss
        if self.enable_center == True:
            # center
            ## Apply Focal Loss

            center_label_list, ins_label_list, ins_ind_label_list, grid_order_list  = multi_apply(
                self.parsing_center_target_single,
                gt_bbox_list, 
                gt_label_list, 
                gt_mask_list, 
                gt_parsing_list,
                img_metas,
                mask_feat_size = mask_feat_size,
                parsing2mask=self.parsing2mask)
        
            center_labels = [
                torch.cat([center_labels_level_img.flatten()
                        for center_labels_level_img in center_labels_level])
                for center_labels_level in zip(*center_label_list)
            ]
            flatten_center_labels = torch.cat(center_labels)
            
            num_ins = flatten_center_labels.sum()

            flatten_center_preds = center_preds.permute(0, 2, 3, 1).reshape(-1, 1)
            
            # Focal loss: the label of background should be k (1 here) in mmdet2.0.
            flatten_center_labels = flatten_center_labels - 1
            flatten_center_labels[flatten_center_labels<0] = 1

            loss_center = self.loss_center(flatten_center_preds, flatten_center_labels, avg_factor=num_ins + 1)
        else:
            loss_center = torch.tensor(0,dtype=torch.float32, device=center_preds.device)
        
        # instance loss
        if self.enable_instance:
            ins_labels = [torch.cat([ins_labels_level_img
                                for ins_labels_level_img in ins_labels_level], 0)
                      for ins_labels_level in zip(*ins_label_list)]
            
            #import pdb;pdb.set_trace()
            ins_kernel_map = F.interpolate(instance_preds, size=(self.seg_num_grids, self.seg_num_grids), mode='bilinear').reshape(instance_preds.shape[0],instance_preds.shape[1], -1)
            # ins_kernel_map = instance_preds.reshape(instance_preds.shape[0],instance_preds.shape[1], -1)

            ins_kernels = []
            ins_segments = []
            for kernel_preds_level, grid_orders_level, instance_pred in zip(ins_kernel_map, grid_order_list, instance_preds):
                ins_kernel = kernel_preds_level[:,grid_orders_level[0]].permute(1,0)
                N, C = ins_kernel.shape
                ins_kernel = ins_kernel.view(N, -1, 1, 1)
                ins_kernels.append(ins_kernel)
                ins_segments.append(F.conv2d(F.normalize(instance_pred,p=2,dim=0).unsqueeze(0), F.normalize(ins_kernel,p=2,dim=1), stride=1).squeeze(0))
                # ins_segments.append(F.conv2d(instance_pred.unsqueeze(0), ins_kernel, stride=1).squeeze(0))
            
            img_segments = ins_segments
            ins_segments = torch.cat(ins_segments, 0)
            loss_ins = dice_loss(ins_segments, ins_labels[0])
            loss_ins = (loss_ins * self.ins_loss_weight).mean()
        else:
            loss_ins = torch.tensor(0,dtype=torch.float32, device=center_preds.device)

        # semantic loss
        if self.enable_semantic:
            gt_parsings = multi_apply(
                self.parsing_semantic_target_single,
                gt_bbox_list, 
                gt_label_list, 
                gt_mask_list, 
                gt_parsing_list,
                mask_feat_size=mask_feat_size,
                adapt_center=self.enable_adapt_center)[0]
            
            
            gt_parsings = torch.cat(gt_parsings, 0).view(-1, mask_feat_size[0], mask_feat_size[1])
            
            parsing_index = (gt_parsings.sum(1).sum(1)>0)
            # gt_parsings = gt_parsings[parsing_index]

            Nclass, C = self.seg_kernels.shape
            semantic_kernels = self.seg_kernels.view(Nclass, C, 1, 1)
            sem_preds = semantic_preds
            
            semantic_preds = F.conv2d(F.normalize(semantic_preds,p=2,dim=1), F.normalize(semantic_kernels,p=2,dim=1), stride=1)
            # semantic_preds = F.conv2d(semantic_preds, semantic_kernels, stride=1)

            semantic_preds = semantic_preds.view(-1, mask_feat_size[0], mask_feat_size[1])
            
            # TODO: add adaptive weights for dice loss
            
            loss_semantic_pos = dice_loss(semantic_preds[parsing_index], gt_parsings[parsing_index], self.enable_diceloss_weight, gt_parsing_list)
            if self.loss_semantic_neg_config['loss_semantic_pos_square']:
                loss_semantic_pos = (loss_semantic_pos ** 2 * self.sem_loss_weight).mean()
            else:
                loss_semantic_pos = (loss_semantic_pos * self.sem_loss_weight).mean()
                
            # TODO: modify the neg samples loss function
            if self.loss_semantic_neg_config['enable']:
                if self.loss_semantic_neg_config['loss_semantic_neg_l2']:
                    semantic_preds_neg = semantic_preds[~parsing_index]
                    semantic_neg_index = semantic_preds_neg > 0.1
                    if semantic_neg_index.sum() > 0:
                        loss_semantic_neg = (semantic_preds_neg[semantic_neg_index] ** 2).mean() * self.loss_semantic_neg_weight
                    else:
                        loss_semantic_neg = torch.tensor(0,dtype=torch.float32, device=center_preds.device)
                else:
                    semantic_preds_neg = semantic_preds[~parsing_index]
                    semantic_neg_index = semantic_preds_neg > 0.1
                    if semantic_neg_index.sum() > 0:
                        loss_semantic_neg = (semantic_preds_neg[semantic_neg_index]).mean() * self.loss_semantic_neg_weight
                    else:
                        loss_semantic_neg = torch.tensor(0,dtype=torch.float32, device=center_preds.device)
            else:
                loss_semantic_neg = torch.tensor(0,dtype=torch.float32, device=center_preds.device)
        else:
            loss_semantic_pos = torch.tensor(0,dtype=torch.float32, device=center_preds.device)
            loss_semantic_neg = torch.tensor(0,dtype=torch.float32, device=center_preds.device)

        # parsing loss
        # ins_label_list: Batch * 1 * ins_p * H * W
        # parsing_label_list: Batch * 1 * ins_p
        # mask_center: Batch * 2 * ins_h
        ins_label_list, parsing_label_list, coords_list, mask_center = multi_apply(
            self.parsing_category_target_single,
            gt_bbox_list, 
            gt_label_list, 
            gt_mask_list, 
            gt_parsing_list,
            mask_feat_size=mask_feat_size,
            adapt_center=self.enable_adapt_center,
            parsing2mask=self.parsing2mask)
        
        if self.enable_fusion_multi:
            #TODO
            # sem_preds: (N,C,H,W)
            # img_segments: [(N,H,W),...]
            
            semantic_preds = F.conv2d(F.normalize(sem_preds,p=2,dim=1), F.normalize(semantic_kernels,p=2,dim=1), stride=1)
            par_segments_pos = []
            par_segments_neg = []
            for img_segment, sem_pred, parsing_label, ins_ind_label in zip(img_segments, semantic_preds, parsing_label_list, ins_ind_label_list):
                parsing_label = parsing_label[0]
                ins_ind_label = np.array(ins_ind_label[0])
                img_segment = self.get_instance_seg(img_segment, ins_ind_label)
                par_segments_pos = par_segments_pos + self.get_parsing_seg_pos(parsing_label, img_segment, sem_pred)
                
            par_segments_pos = torch.cat(par_segments_pos, 0)
            parsing_gt = torch.cat([ins_label[0] for ins_label in ins_label_list], 0)
            loss_parsing_pos = dice_loss(par_segments_pos, parsing_gt)
            loss_parsing_pos = (loss_parsing_pos * self.par_loss_weight).mean()
        
            
        elif self.enable_fusion:
            ins_kernels = []
            ins_indexes = []
            cate_indexes = []
            for img_i in range(ins_kernel_map.shape[0]):
                parsing_kernels = []
                ins_index = []
                cate_index = []
                mask_x, mask_y = mask_center[img_i]
                mask_index = mask_y*self.seg_num_grids+mask_x
                ins_kernel = ins_kernel_map[img_i,:,mask_index.long()].permute(1,0)
                cate_list = parsing_label_list[img_i][0]
                former_cate = 0
                ins_i = 0
                for cate_i in cate_list:
                    if cate_i == former_cate:
                        ins_i = ins_i + 1
                    else:
                        ins_i = 0
                    ins_index.append(ins_i)
                    cate_index.append(cate_i)
                    parsing_kernels.append(torch.cat([F.normalize(ins_kernel[ins_i],p=2,dim=0), F.normalize(self.seg_kernels[cate_i-1],p=2,dim=0)], 0).unsqueeze(0))
                    former_cate = cate_i
                ins_kernels.append(torch.cat(parsing_kernels, 0).reshape(-1, 2*self.seg_feat_channels, 1, 1))
                ins_indexes.append(ins_index)
                cate_indexes.append(cate_index)
            
            fusion_preds = torch.cat([F.normalize(instance_preds,p=2,dim=1), F.normalize(sem_preds,p=2,dim=1)], 1)
            for i, fusion_layer in enumerate(self.fusion_convs):
                fusion_preds = fusion_layer(fusion_preds)
                for img_i in range(ins_kernel_map.shape[0]):
                    ins_kernels[img_i] = fusion_layer(ins_kernels[img_i])
                    
            parsings = []
            for img_i in range(ins_kernel_map.shape[0]):
                parsings.append(F.conv2d(F.normalize(fusion_preds[img_i:img_i+1],p=2,dim=1), F.normalize(ins_kernels[img_i],p=2,dim=1), stride=1).squeeze(0))
            
            parsings = torch.cat(parsings, 0)
            parsing_gt = torch.cat([ins_label[0] for ins_label in ins_label_list], 0)
            loss_parsing_pos = dice_loss(parsings, parsing_gt)
            loss_parsing_pos = (loss_parsing_pos * self.par_loss_weight).mean()
        else:
            loss_parsing_pos = torch.tensor(0,dtype=torch.float32, device=center_preds.device)

        if self.enable_metrics:
            loss_metric_list = []
            
            for contend in self.loss_metrics['contend']:
                if contend == "semantic_kernels":
                    Nclass, C = self.seg_kernels.shape
                    metric_mask = torch.tril(torch.ones(Nclass, Nclass), diagonal=-1)
                    metric_mask = (metric_mask > 0).bool()
                    sem_metric = (torch.mm(F.normalize(self.seg_kernels,p=2,dim=1), F.normalize(self.seg_kernels,p=2,dim=1).permute(1,0)))[metric_mask]
                    sem_metric_index = sem_metric > self.loss_metrics['margin']
                    if sem_metric_index.sum() > 0:
                        loss_metric_list.append(sem_metric[sem_metric_index]**2)
                elif contend == "instance_kernels":
                    for img_i in range(ins_kernel_map.shape[0]):
                        mask_x, mask_y = mask_center[img_i]
                        mask_index = mask_y*self.seg_num_grids+mask_x
                        ins_kernel = ins_kernel_map[img_i,:,mask_index.long()].permute(1,0)
                        Nins, C = ins_kernel.shape
                        metric_mask = torch.tril(torch.ones(Nins, Nins), diagonal=-1)
                        metric_mask = (metric_mask > 0).bool()
                        ins_corr = (torch.mm(F.normalize(ins_kernel,p=2,dim=1), F.normalize(ins_kernel,p=2,dim=1).permute(1,0)))[metric_mask]
                        ins_corr_index = ins_corr > self.loss_metrics['margin']
                        if ins_corr_index.sum() > 0:
                            loss_metric_list.append(ins_corr[ins_corr_index]**2)
                elif contend == "parsing_kernels":
                    for img_i in range(ins_kernel_map.shape[0]):
                        ins_kernel = ins_kernels[img_i].squeeze(2).squeeze(2)
                        Nins, C = ins_kernel.shape
                        metric_mask = torch.tril(torch.ones(Nins, Nins), diagonal=-1)
                        metric_mask = (metric_mask > 0).bool()
                        par_corr = (torch.mm(F.normalize(ins_kernel,p=2,dim=1), F.normalize(ins_kernel,p=2,dim=1).permute(1,0)))[metric_mask]
                        par_corr_index = par_corr > self.loss_metrics['margin']
                        if par_corr_index.sum() > 0:
                            loss_metric_list.append(par_corr[par_corr_index]**2)
                            
                elif contend == "parsing_features":
                    part_scores = self.get_sem_center_maps(coords_list, sem_preds, parsing_gt, parsing_label_list)
                    part_corr_index = (1 - part_scores) > self.loss_metrics['margin']
                    if part_corr_index.sum() > 0:
                        loss_metric_list.append((1 - part_scores)[part_corr_index]**2)
                    
            if len(loss_metric_list) > 0:
                
                loss_metric_list = torch.cat(loss_metric_list, 0)
                loss_metrics = loss_metric_list.mean()
            else:
                loss_metrics = torch.tensor(0,dtype=torch.float32, device=center_preds.device)
        else:
            loss_metrics = torch.tensor(0,dtype=torch.float32, device=center_preds.device)

        return dict(
            loss_ins=loss_ins,
            loss_center=loss_center,
            loss_semantic_pos=loss_semantic_pos,
            loss_semantic_neg=loss_semantic_neg,
            loss_parsing_pos=loss_parsing_pos,
            loss_metrics=loss_metrics
            )

    def parsing_semantic_target_single(self,
                               gt_bboxes_raw,
                               gt_labels_raw,
                               gt_masks_raw,
                               gt_parsing_raw,
                               mask_feat_size,
                               adapt_center):

        device = gt_labels_raw[0].device
        upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4)
        # ins 
        gt_semantics = torch.zeros([mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8,
                                                device=device)
        for i in range(gt_parsing_raw.shape[0]):
            gt_parsing_i = gt_parsing_raw[i]
            gt_parsing_i = mmcv.imrescale(gt_parsing_i, scale=1. / 4, interpolation='nearest')
            gt_parsing_i = torch.from_numpy(gt_parsing_i).to(device=device)
            gt_semantic = torch.zeros([mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8,
                                        device=device)
            gt_semantic[:gt_parsing_i.shape[0], :gt_parsing_i.shape[1]] = gt_parsing_i
            gt_semantics += gt_semantic
        #import pdb;pdb.set_trace()
        gt_parsings = []
        for i in range(self.cate_out_channels):
            gt_parsing = gt_semantics == i+1
            gt_parsings.append(gt_parsing.unsqueeze(0))
        gt_parsings = torch.cat(gt_parsings, 0).int()

        return [gt_parsings.unsqueeze(0)]
    
    # TODO: add fusion parsing branch
    def parsing_category_target_single(self,
                               gt_bboxes_raw,
                               gt_labels_raw,
                               gt_masks_raw,
                               gt_parsing_raw,
                               mask_feat_size,
                               adapt_center,
                               parsing2mask):
        

        
        device = gt_labels_raw[0].device
        upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4)
        # ins
        gt_parsings = []
        gt_labels_parsing = []
        gt_bboxes_parsing = []
        parsing_label_list = []
        ins_label_list = []
        cate_label_list = []
        classify_label_list = []
        ins_ind_label_list = []
        grid_order_list = []
        
        if parsing2mask:
            gt_masks_raw = (gt_parsing_raw > 0).astype(float)
        
        for i in range(self.cate_out_channels):
            gt_parsing = gt_parsing_raw == i+1
            if gt_parsing.max() == 0:
                continue
            else:
                try:
                    for ins in range(gt_parsing.shape[0]):
                        ys, xs = np.where(gt_parsing[ins] > 0)
                        if len(xs) == 0 or len(ys) == 0:
                            continue
                        else:
                            x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
                            gt_bboxes_parsing.append(np.array([x1,y1,x2,y2]))
                            gt_parsings.append(gt_parsing[ins])
                            gt_labels_parsing.append(i+1)
                except Exception as e:
                    print(e)
                    import pdb;pdb.set_trace()
                    
        num_grid = self.seg_num_grids
        mask_x, mask_y = center_of_mass(torch.from_numpy(gt_masks_raw.astype('uint8')), adapt_center)
        mask_x = (mask_x / upsampled_size[1]) // (1. / num_grid)
        mask_y = (mask_y / upsampled_size[0]) // (1. / num_grid)
                    
        gt_bboxes_parsing = torch.tensor(np.array(gt_bboxes_parsing)).float().to(device)
        gt_parsings = np.array(gt_parsings)
        gt_labels_parsing = torch.tensor(np.array(gt_labels_parsing)).to(device)
        
        if len(gt_bboxes_parsing) == 0:
            print("train a empty parsing image.")
            for num_grid in self.seg_num_grids:
                parsing_label = torch.tensor([], dtype=torch.int64, device=device)
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
                cate_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device)
                classify_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device)
                ins_ind_label = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device)
                cate_label_list.append(cate_label)
                parsing_label_list.append(parsing_label)
                classify_label_list.append(classify_label)
                ins_label_list.append(ins_label)
                ins_ind_label_list.append(ins_ind_label)
                grid_order_list.append([])
            return ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list, parsing_label_list

        gt_areas = torch.sqrt((gt_bboxes_parsing[:, 2] - gt_bboxes_parsing[:, 0]) * (
                gt_bboxes_parsing[:, 3] - gt_bboxes_parsing[:, 1]))
        
        (lower_bound, upper_bound), stride, num_grid = self.scale_ranges, self.strides[0], self.seg_num_grids
        
        hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
        num_ins = len(hit_indices)

        parsing_label = []
        ins_label = []
        grid_order = []
        cate_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device)
        classify_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device)
        ins_ind_label = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device)

        if num_ins == 0:
            ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
            parsing_label = torch.tensor([], dtype=torch.int64, device=device)
            parsing_label_list.append(parsing_label)
            ins_label_list.append(ins_label)
            cate_label_list.append(cate_label)
            classify_label_list.append(classify_label)
            ins_ind_label_list.append(ins_ind_label)
            grid_order_list.append([])
            return ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list, classify_label_list, parsing_label_list

        gt_bboxes = gt_bboxes_parsing[hit_indices]
        gt_labels = gt_labels_parsing[hit_indices]
        gt_masks = gt_parsings[hit_indices.cpu().numpy(), ...].astype('uint8')

        half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
        half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

        # mass center
        gt_masks_pt = torch.from_numpy(gt_masks).to(device=device)
        try:
            center_ws, center_hs = center_of_mass(gt_masks_pt, adapt_center)
            valid_mask_flags = gt_masks_pt.sum(dim=-1).sum(dim=-1) > 0
        except Exception as e:
            print(e)
            import pdb;pdb.set_trace()
            
        coords = []
        coords_list = []
        output_stride = 4
        for seg_mask, gt_label, half_h, half_w, center_h, center_w, valid_mask_flag in zip(gt_masks, gt_labels, half_hs, half_ws, center_hs, center_ws, valid_mask_flags):
            if not valid_mask_flag:
                continue
            coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))
            coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))
            #coords.append((coord_w, coord_h))
            coords.append((int(center_w), int(center_h)))
            seg_mask_tmp = mmcv.imrescale(seg_mask, scale=1. / output_stride)
            if seg_mask_tmp.max()==0:
                seg_mask = (mmcv.imrescale(seg_mask.astype(float), scale=1. / output_stride, interpolation='area')>0).astype('uint8')
                seg_mask = torch.from_numpy(seg_mask).to(device=device)
            else:
                try:
                    seg_mask = seg_mask_tmp
                    seg_mask = torch.from_numpy(seg_mask).to(device=device)
                except Exception as e :
                    print(e)
                    import pdb; pdb.set_trace()
            
            cur_ins_label = torch.zeros([mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8,device=device)
            cur_ins_label[:seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask
            ins_label.append(cur_ins_label)
            if cur_ins_label.max() == 0:
                import pdb;pdb.set_trace()
            parsing_label.append(gt_label.item())

        if len(ins_label) == 0:
            ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
        else:
            ins_label = torch.stack(ins_label, 0)

        ins_label_list.append(ins_label)
        parsing_label_list.append(parsing_label)
        coords_list.append(coords)

        return ins_label_list, parsing_label_list, coords_list, (mask_x, mask_y)

    def parsing_center_target_single(self,
                               gt_bboxes_raw,
                               gt_labels_raw,
                               gt_masks_raw,
                               gt_parsing_raw,
                               img_metas = None,
                               mask_feat_size = None,
                               parsing2mask=False):
        
        device = gt_labels_raw[0].device
        num_grid = self.seg_num_grids
        upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4)

        # ins
        ins_label_list = []
        cate_label_list = []
        ins_ind_label_list = []
        grid_order_list = []
        gt_bboxes_parsing = []
        gt_masks = []
        grid_order_list = []
        
        if parsing2mask:
            gt_masks_raw = (gt_parsing_raw > 0).astype(float)
        
        for ins in range(gt_masks_raw.shape[0]):
            ys, xs = np.where(gt_masks_raw[ins] > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue
            else:
                x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
                gt_bboxes_parsing.append(np.array([x1,y1,x2,y2]))
                gt_masks.append(gt_masks_raw[ins])
                
        if len(gt_bboxes_parsing) == 0:
            ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
            ins_label_list.append(ins_label)
            cate_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device)
            cate_label_list.append(cate_label)
            return cate_label_list, ins_label_list
        
        gt_bboxes = np.array(gt_bboxes_parsing)
        gt_labels = gt_labels_raw
        gt_masks = np.array(gt_masks)

        cate_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device)
        parsing_label = []
        ins_label = []
        grid_order = []
        ins_ind_label = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device)

        half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
        half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

        # mass center
        gt_masks_pt = torch.from_numpy(gt_masks).to(device=device)
        center_ws, center_hs = center_of_mass(gt_masks_pt, self.enable_adapt_center)
        valid_mask_flags = gt_masks_pt.sum(dim=-1).sum(dim=-1) > 0

        output_stride = 4
        ins_index = 0
        ins_index_list = []

        for seg_mask, gt_label, half_h, half_w, center_h, center_w, valid_mask_flag in zip(gt_masks, gt_labels, half_hs, half_ws, center_hs, center_ws, valid_mask_flags):
            if not valid_mask_flag:
                continue
            
            coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))
            coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))

            # left, top, right, down
            top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))
            down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))
            left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))
            right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))

            top = max(top_box, coord_h-1)
            down = min(down_box, coord_h+1)
            left = max(coord_w-1, left_box)
            right = min(right_box, coord_w+1)

            cate_label[top:(down+1), left:(right+1)] = gt_label

            try:
                seg_mask = mmcv.imrescale(seg_mask, scale=1. / output_stride)
                seg_mask = torch.from_numpy(seg_mask).to(device=device)
            except Exception as e :
                print(e)
                import pdb; pdb.set_trace()
            for i in range(top, down+1):
                for j in range(left, right+1):
                    label = int(i * num_grid + j)

                    cur_ins_label = torch.zeros([mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8,
                                                device=device)
                    
                    cur_ins_label[:seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask
                    
                    ins_label.append(cur_ins_label)
                    ins_ind_label[label] = True
                    grid_order.append(label)
                    ins_index_list.append(ins_index)
            ins_index = ins_index + 1
        if len(ins_label) == 0:
            ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
        else:
            ins_label = torch.stack(ins_label, 0)
        
        ins_label_list.append(ins_label)
        #parsing_label_list.append(parsing_label)
        ins_ind_label_list.append(ins_index_list)
        grid_order_list.append(grid_order)
        cate_label_list.append(cate_label)
        
        return cate_label_list, ins_label_list, ins_ind_label_list, grid_order_list
    
    @force_fp32(apply_to=('center_preds', 'semantic_preds', 'instance_preds')) 
    def get_seg(self, 
            center_preds, 
            semantic_preds, 
            instance_preds, 
            img_metas, 
            cfg, 
            rescale=None, 
            parsing_gt=None):

        # center_preds.shape: [imgs,S,S,1]
        # semantic_preds.shape: [imgs, 512, H, w]
        # instance_preds.shape: [imgs, 512, H, w]
        if self.seg_kernels.requires_grad:
            self.seg_kernels.requires_grad=False

        if not isinstance(img_metas, list):
            img_metas = img_metas.data[0]
        result_list = []
        for img_id in range(len(img_metas)):
            # process single images. 
            center_pred_list = center_preds[img_id].detach()
            instance_pred_list = instance_preds[img_id].detach()
            semantic_pred_list = semantic_preds[img_id].detach()

            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            ori_shape = img_metas[img_id]['ori_shape']

            if self.enable_fusion:
                result = self.get_parsing_single_fusion(center_pred_list, instance_pred_list, semantic_pred_list, \
                                                    img_shape, ori_shape, scale_factor, cfg, img_metas, \
                                                    infer_by_pixel=False)
            else:
                result = self.get_parsing_single(center_pred_list, instance_pred_list, semantic_pred_list, \
                                                    img_shape, ori_shape, scale_factor, cfg, img_metas, \
                                                    infer_by_pixel=False)
            result_list.append(result)
        return result_list

    def get_parsing_single(self,
                        center_preds, 
                        instance_pred, 
                        semantic_pred, 
                        img_shape, 
                        ori_shape, 
                        scale_factor, 
                        cfg, 
                        img_metas,
                        infer_by_pixel=False):
        # overall info.
        h, w, _ = img_shape
        featmap_size = instance_pred.shape[-2:]
        C_ins = instance_pred.shape[0]
        C_sem = semantic_pred.shape[0]
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)
        cate_mask = np.zeros(ori_shape[:2], dtype=np.int8)

        # Screen the center of human.
        # center_preds.shape: [S,S,1]
        center_preds = center_preds.unsqueeze(0).permute(0,3,1,2)
        center_nms = nn.functional.max_pool2d(center_preds, (5,5), stride=1, padding=2)
        center_nms = torch.eq(center_nms, center_preds)
        center_preds = (center_preds*center_nms).squeeze(0).permute(1,2,0)

        inds = (center_preds > cfg.ctr_score)
        center_scores = center_preds[inds]
        center_scores, sorted_inds = torch.sort(center_scores)
        if len(center_scores) == 0:
            filename = img_metas[0]['filename'].split('/')[-1]
            print("Can't detect any parsing of " + filename + " because of center.")
            mmcv.imwrite(cate_mask, "./eval_file/pred_cate/"+img_metas[0]['filename'].split('/')[-1].split('/')[-1].split('.')[0] + '.png')
            return None

        t0 = time.time()
        # Process each instance
        ys, xs = torch.where(inds.squeeze(2))
        ys, xs = ((ys/self.seg_num_grids)*featmap_size[0]).int(), ((xs/self.seg_num_grids)*featmap_size[1]).int()
        indexs = ys*featmap_size[1] + xs
        indexs = indexs[sorted_inds]
        ins_kernels = instance_pred.permute(1,2,0).reshape(-1, C_ins)[indexs.long()].reshape(-1,C_ins,1,1)
        ins_conf = F.conv2d(F.normalize(instance_pred,p=2,dim=0).unsqueeze(0), F.normalize(ins_kernels,p=2,dim=1), stride=1)
        ins_masks = ins_conf > cfg.mask_thr
        
        # print("ins_time: ", time.time()-t0)
        # t0 = time.time()

        # Generate semantic Maps
        sem_kernels = self.seg_kernels.reshape(self.num_classes-1, C_sem, 1, 1)
        sem_conf = F.conv2d(F.normalize(semantic_pred.unsqueeze(0),p=2,dim=1), F.normalize(sem_kernels,p=2,dim=1), stride=1)
        sem_index = sem_conf > cfg.mask_thr
        #print("sem_time: ", time.time()-t0)
        
        
        if not infer_by_pixel:
            #TODO: ADD Matrix NMS
            # Get human parsing
            ## Matrix NMS
            ins_parsings = []
            
            for i_ins in range(len(ins_masks[0])):
                # t0 = time.time()
                ins_pars = (ins_masks[0][i_ins] * sem_conf[0]).float()
                sem_scores = self.maskness(ins_pars, cfg.mask_thr)
                cate_labels = torch.from_numpy(np.array([i for i in range(self.cate_out_channels)])).cuda()
                # _, labels, _, keep_inds = matrix_nms((ins_pars > cfg.mask_thr).float(), cate_labels, sem_scores, filter_thr=0.3)
                labels = torch.where(sem_scores>cfg.cate_score_thr)[0]
                cate_labes = self.get_elements_not_in_b(cate_labels, labels)
                ins_pars = self.zero_out_masks(ins_pars, cate_labes)
                ins_pars = F.interpolate(ins_pars.unsqueeze(0), size=upsampled_size_out, mode='bilinear')[:,:,:h,:w]
                if h*w > ori_shape[0]*ori_shape[1]:
                    ins_pars = F.interpolate(ins_pars, size=ori_shape[:2], mode='bilinear')
                ins_pars_index = ins_pars > cfg.mask_thr
                # print("pre_~: ", time.time()-t0)
                # t0 = time.time()
                pos_index = ins_pars_index.sum(1).bool()
                ins_pars = (torch.argmax(ins_pars,1) + 1)
                ins_pars = ins_pars * pos_index
                
                # import pdb;pdb.set_trace()
                if h*w < ori_shape[0]*ori_shape[1]:
                    ins_pars = F.interpolate(ins_pars.unsqueeze(0).float(), size=ori_shape[:2], mode='nearest')[0].int()
                ins_parsings.append(ins_pars)
                # print("finish_~: ", time.time()-t0)
            ins_parsings = torch.concat(ins_parsings,0)
            #import pdb;pdb.set_trace()
            
            
            # Save semantic outputs for evaluation. (Can be omitted during inference only.)
            t0 = time.time()
            semantic_masks = F.interpolate(sem_conf.float(), size=upsampled_size_out, mode='bilinear')[:,:,:h,:w]
            if h*w > ori_shape[0]*ori_shape[1]:
                semantic_masks = F.interpolate(semantic_masks, size=ori_shape[:2], mode='bilinear')
            semantic_index = semantic_masks > cfg.mask_thr
            neg_index = ~semantic_index.sum(1).bool()[0]
            semantic_masks = (torch.argmax(semantic_masks,1) + 1)
            semantic_masks[:,neg_index] = 0
            if h*w < ori_shape[0]*ori_shape[1]:
                semantic_masks = F.interpolate(semantic_masks.unsqueeze(0).float(), size=ori_shape[:2], mode='nearest')[0].int()
            mmcv.imwrite(semantic_masks[0].int().cpu().numpy(), "eval_file/pred_cate/"+img_metas[0]['filename'].split('/')[-1].split('.')[0] + '.png')
            print("save smantic masks: ", time.time()-t0)
            
            # Output the intermedia results for visualization. (Can be omitted during inference only.)
            vis_results = (center_preds.cpu().numpy(), ins_conf.cpu().numpy(), sem_conf.cpu().numpy())
            
            return ins_parsings.cpu().numpy(), vis_results, center_scores
                
        ## Interpolate the instance masks to original shape
        ins_conf = F.interpolate(ins_conf, size=upsampled_size_out, mode='bilinear')[:,:,:h,:w]
        ins_conf = F.interpolate(ins_conf, size=ori_shape[:2], mode='bilinear').squeeze(0)
        ins_masks = ins_conf > cfg.mask_thr

        # Save semantic outputs for evaluation. (Can be omitted during inference only.)
        semantic_masks = F.interpolate(sem_conf.float(), size=upsampled_size_out, mode='bilinear')[:,:,:h,:w]
        semantic_masks = F.interpolate(semantic_masks, size=ori_shape[:2], mode='bilinear')
        semantic_index = semantic_masks > cfg.mask_thr
        neg_index = ~semantic_index.sum(1).bool()[0]
        semantic_masks = (torch.argmax(semantic_masks,1) + 1)
        semantic_masks[:,neg_index] = 0
        mmcv.imwrite(semantic_masks[0].int().cpu().numpy(), "eval_file/pred_cate/"+img_metas[0]['filename'].split('/')[-1].split('.')[0] + '.png')

        # Generate Parsing
        ins_parsing = (ins_masks.int() * semantic_masks[0])

        # Output the intermedia results for visualization. (Can be omitted during inference only.)
        vis_results = (center_preds.cpu().numpy(), ins_masks.cpu().numpy(), semantic_masks.cpu().numpy())

        return ins_parsing.cpu().numpy(), vis_results, center_scores

    def get_parsing_single_fusion(self,
                        center_preds, 
                        instance_pred, 
                        semantic_pred, 
                        img_shape, 
                        ori_shape, 
                        scale_factor, 
                        cfg, 
                        img_metas,
                        infer_by_pixel=False):
        # overall info.
        h, w, _ = img_shape
        featmap_size = instance_pred.shape[-2:]
        C_ins = instance_pred.shape[0]
        C_sem = semantic_pred.shape[0]
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)
        cate_mask = np.zeros(ori_shape[:2], dtype=np.int8)

        # Screen the center of human.
        # center_preds.shape: [S,S,1]
        center_preds = center_preds.unsqueeze(0).permute(0,3,1,2)
        center_nms = nn.functional.max_pool2d(center_preds, (5,5), stride=1, padding=2)
        center_nms = torch.eq(center_nms, center_preds)
        center_preds = (center_preds*center_nms).squeeze(0).permute(1,2,0)

        inds = (center_preds > cfg.ctr_score)
        center_scores = center_preds[inds]
        center_scores, sorted_inds = torch.sort(center_scores)
        if len(center_scores) == 0:
            filename = img_metas[0]['filename'].split('/')[-1]
            print("Can't detect any parsing of " + filename + " because of center.")
            mmcv.imwrite(cate_mask, "./eval_file/pred_cate/"+img_metas[0]['filename'].split('/')[-1].split('/')[-1].split('.')[0] + '.png')
            return None

        # Generate kernels
        ys, xs = torch.where(inds.squeeze(2))
        ys, xs = ((ys/self.seg_num_grids)*featmap_size[0]).int(), ((xs/self.seg_num_grids)*featmap_size[1]).int()
        indexs = ys*featmap_size[1] + xs
        indexs = indexs[sorted_inds]
        ins_kernels = instance_pred.permute(1,2,0).reshape(-1, C_ins)[indexs.long()].reshape(-1,C_ins)
        ins_index = []
        cate_index = []
        parsing_kernels = []
        for ins_i in range(ins_kernels.shape[0]):
            for cate_i in range(self.cate_out_channels):
                ins_index.append(ins_i)
                parsing_kernels.append(torch.cat([ins_kernels[ins_i:ins_i+1], self.seg_kernels[cate_i:cate_i+1]], 0))
        parsing_kernels = torch.cat(parsing_kernels, 0).reshape(-1, 2*self.seg_feat_channels, 1, 1)
        
        # Generate fusion Maps
        fusion_preds = torch.cat([instance_pred, semantic_pred], 0).unsqueeze(0)
        for i, fusion_layer in enumerate(self.fusion_convs):
            fusion_preds = fusion_layer(fusion_preds)
            parsing_kernels = fusion_layer(parsing_kernels)
        
        # Generate parsing masks
        par_conf = F.conv2d(F.normalize(fusion_preds,p=2,dim=1), F.normalize(parsing_kernels,p=2,dim=1), stride=1)
        par_ind = (par_conf > 0.5)
        par_sum = par_ind.sum(2).sum(2)
        par_sum_index = par_sum > 32
        par_sum = par_sum * par_sum_index
        par_scores = (par_conf * par_ind).sum(2).sum(2) * par_sum_index / (par_sum+1)
        parsing_index = par_scores > 0.7
   
        ins_parsings = []
        ins_index = torch.from_numpy(np.array(ins_index))
        for ins_i in range(ins_kernels.shape[0]):
            ins_inds = (ins_index == ins_i).cuda() * parsing_index
            if ins_inds.sum() == 0:
                continue
            cate_list = torch.where(ins_inds)[1]%58 + 1
            N_ins = cate_list.shape[0]
            pars_conf = par_conf[ins_inds]
            try:
                pars_conf = F.interpolate(pars_conf.unsqueeze(0), size=upsampled_size_out, mode='bilinear')[:,:,:h,:w]
            except Exception as e:
                import pdb;pdb.set_trace()
            pars_conf = F.interpolate(pars_conf, size=ori_shape[:2], mode='bilinear').squeeze(0)
            par_max = pars_conf.max(0)[1] + 1
            back = pars_conf.max(0)[0] < 0.5
            par_max = par_max * ~back
            result = torch.zeros_like(par_max)
            for value in range(N_ins):
                result[par_max == value+1] = cate_list[value]
            ins_parsings.append(result.unsqueeze(0))
        if len(ins_parsings) == 0:
            print("Can't detect any parsing of " + filename + " because of kernel.")
            mmcv.imwrite(cate_mask, "./eval_file/pred_cate/"+img_metas[0]['filename'].split('/')[-1].split('/')[-1].split('.')[0] + '.png')
            return None
        else:
            ins_parsings = torch.cat(ins_parsings,0)

        # Generate instance Maps
        ins_conf = F.conv2d(F.normalize(instance_pred,p=2,dim=0).unsqueeze(0), F.normalize(ins_kernels.reshape(-1,C_ins,1,1),p=2,dim=1), stride=1)
        ins_masks = ins_conf > cfg.mask_thr

        ## Generate semantic Maps
        # sem_kernels = self.seg_kernels.reshape(self.num_classes-1, C_sem, 1, 1)
        # sem_conf = F.conv2d(F.normalize(semantic_pred.unsqueeze(0),p=2,dim=1), F.normalize(sem_kernels,p=2,dim=1), stride=1)
        # sem_index = sem_conf > cfg.mask_thr
                
        ## Interpolate the instance masks to original shape
        #ins_conf = F.interpolate(ins_conf, size=upsampled_size_out, mode='bilinear')[:,:,:h,:w]
        #ins_conf = F.interpolate(ins_conf, size=ori_shape[:2], mode='bilinear').squeeze(0)
        #ins_masks = ins_conf > cfg.mask_thr

        # Save semantic outputs for evaluation. (Can be omitted during inference only.)
        semantic_masks = torch.sum(ins_parsings, 0)
        mmcv.imwrite(semantic_masks.int().cpu().numpy(), "eval_file/pred_cate/"+img_metas[0]['filename'].split('/')[-1].split('.')[0] + '.png')

        # Output the intermedia results for visualization. (Can be omitted during inference only.)
        vis_results = (center_preds.cpu().numpy(), ins_masks.cpu().numpy(), semantic_masks.cpu().numpy())

        return ins_parsings.cpu().numpy(), vis_results, center_scores
    
    # define maskness
    def maskness(self, ins, mask_thr):
        sem_conf = ins
        sem_index = sem_conf > mask_thr
        sem_pixels = sem_conf > 0.1
        sem_scores = (sem_conf * sem_index).sum(1).sum(1)
        sem_pixels = sem_pixels.sum(1).sum(1) + 0.00001
        sem_scores = sem_scores / sem_pixels
        return sem_scores

    def get_elements_not_in_b(self, a, b):
        """
        Returns a tensor containing the elements in a that are not in b.

        Arguments:
            a (torch.Tensor): The input tensor a.
            b (torch.Tensor): The input tensor b.

        Returns:
            torch.Tensor: A tensor containing the elements in a that are not in b.
        """
        # Convert the tensors to sets
        a_set = set(a.tolist())
        b_set = set(b.tolist())

        # Compute the set difference between a and b
        difference_set = a_set - b_set

        # Convert the set difference back to a tensor
        return list(difference_set)

    def zero_out_masks(self, masks, index):
        """
        Zeros out the masks at the specified indices.

        Arguments:
            masks (torch.Tensor): A tensor of shape (C, H, W) containing the masks.
            index (torch.Tensor): A 1D tensor containing the indices of the masks to be zeroed out.

        Returns:
            torch.Tensor: A tensor of shape (C, H, W) containing the modified masks.
        """
        # Convert the index tensor to a list
        index_list = index
        # Iterate over the indices and set the corresponding masks to 0
        for i in index_list:
            masks[i, :, :] = 0
        return masks
    
    def get_instance_seg(self, img_segment, ins_ind_label):
        ins_segments = []
        ins_ind = np.unique(ins_ind_label)
        for ind in ins_ind:
            ins_segments.append(torch.max(img_segment[ins_ind_label==ind], 0)[0].unsqueeze(0))
        return torch.cat(ins_segments,0)
    
    def get_parsing_seg_pos(self, parsing_label, img_segment, sem_pred):
        parsing_segments_pos = []
        par_ins_ind = 0
        pre_par_label = 0
        for par_label in parsing_label:
            if par_label == pre_par_label:
                par_ins_ind = par_ins_ind + 1
            else:
                par_ins_ind = 0
            parsing_segments_pos.append((img_segment[par_ins_ind] * sem_pred[par_label - 1]).unsqueeze(0))
            pre_par_label = par_label
        return parsing_segments_pos
    
    def get_sem_center_maps(self, center_coords, feats, gt_masks, parsing_labels):
        ins_maps = []
        feats = F.normalize(feats, p=2,dim=1)
        for center_coord, feat, parsing_label in zip(center_coords, feats, parsing_labels):
            part_kernels = []
            center_coord = center_coord[0]
            #parsing_label = parsing_label[0]
            for center in center_coord:
                part_kernels.append(feat[:,center[1]//4, center[0]//4].reshape(1,feat.shape[0],1,1))
            
            part_kernels = torch.cat(part_kernels, 0)
            
            ins_maps.append(F.conv2d(feat.unsqueeze(0), part_kernels, stride=1).squeeze(0))
        
        ins_maps = torch.cat(ins_maps, 0)
        
        part_scores = []
        for ins_map, gt_mask in zip(ins_maps, gt_masks):
            part_scores.append((ins_map[gt_mask>0]).mean())
        
        return torch.stack(part_scores)
        