#!/usr/bin/env sh

#cp resnet101-5d3b4d8f.pth /root/.cache/torch/checkpoints/resnet101-5d3b4d8f.pth

CUDA_VISIBLE_DEVICES=1 python tools/test_densepose_mhp.py configs/repparsing/densepose_r101_fpn_8gpu_3x_offset_parsing_v1_ori_grid_unified_DCN.py work_dirs/densepose_release_r101_fpn_8gpu_3x_offset_parsing_v1/large/epoch_30.pth --json_out visual/test_segment_json.json --eval 'parsing'

#python train.py configs/denseposeparsing/densepose_r101_fpn_8gpu_3x.py --gpus 2

#./tools/dist_train.sh configs/denseposeparsing/densepose_r101_fpn_8gpu_3x.py 2