_base_ = ['../_base_/default_runtime.py']
model = dict(
    type='POIT',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(
        type='POITHead',
        num_query=300,
        num_classes=1,
        in_channels=2048,
        sync_cls_avg_factor=True,
        dynamic_params_dims=441,
        dynamic_encoder_heads=4,
        mask_positional_encoding_cfg=dict(
            type='RelSinePositionalEncoding',
            num_feats=4,
            normalize=True),
        dice_mask_loss_weight=8.0,
        bce_mask_loss_weight=2.0,
        with_box_refine=True,
        as_two_stage=True,
        transformer=dict(
            type='SOITTransformer',
            mask_channels=8*58,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            seg_encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=1,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_heads=1,
                        num_levels=1),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DeformableDetrTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
    test_cfg=dict(max_per_img=100))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
# dataset settings
dataset_type = 'MHP'
data_root = '/home/notebook/code/personal/S9043252/Parsing-R-CNN/data/LV-MHP-v2/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile_mhp'),
    dict(type='LoadAnnotations_mhp', with_mask=True, with_seg=True ,with_mhp_parsing=True, with_keypoints=True),
    dict(type='Resize_Parsing',
         img_scale=[(1333, 800), (1333, 768), (1333, 736),
                    (1333, 704), (1333, 672), (1333, 640)],
         multiscale_mode='value',
         keep_ratio=True),
    dict(type='RandomFlip_Parsing', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad_Parsing', size_divisor=32),
    dict(type='DefaultFormatBundle_Parsing'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_parsing', 'gt_semantic_seg', 'gt_keypoints'],
                         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                            'scale_factor', 'flip', 'img_norm_cfg')),
]
test_pipeline = [
    dict(type='LoadImageFromFile_mhp'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize_Parsing', keep_ratio=True),
            dict(type='RandomFlip_Parsing'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad_Parsing', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/parsing_annos/',
        img_prefix=data_root + 'train/images/',
        pipeline=train_pipeline,
        data_root=data_root),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val/parsing_annos/',
        img_prefix=data_root + 'val/images/',
        test_mode = True,
        pipeline=test_pipeline,
        data_root=data_root),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/parsing_annos/',
        img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline,
        data_root=data_root
        ))
#evaluation = dict(interval=1, metric=['bbox', 'segm'])
# optimizer
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[40])
runner = dict(type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(interval=10)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# runtime settings
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/MHP_release_r101_fpn_8gpu_1x_POIT/ori/'
load_from = None
resume_from = None
workflow = [('train', 1)]