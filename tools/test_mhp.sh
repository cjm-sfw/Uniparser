#!/usr/bin/env sh

#cp resnet101-5d3b4d8f.pth /root/.cache/torch/checkpoints/resnet101-5d3b4d8f.pth

# python tools/test_mhp.py /home/notebook/code/personal/S9043252/SMP2.0/multi-parsing/configs/smp/MHP_r101_fpn_8gpu_1x_offset_parsing_v1_ori_grid_unified_DCN_large.py /home/notebook/code/personal/S9043252/SMP2.0/multi-parsing/work_dirs/MHP_release_r101_fpn_8gpu_1x_offset_parsing_v1/ori_gird_DCN_1x__large/epoch_12.pth --show --eval 'parsing'

# python tools/test_mhp.py /home/notebook/code/personal/S9043252/SMP2.0/multi-parsing/configs/repparsing/MHP_r101_fpn_8gpu_1x_repparsing_DCN_light128.py /home/notebook/code/personal/S9043252/SMP2.0/multi-parsing/work_dirs/MHP_release_r101_fpn_8gpu_1x_repparsing_v0/light128/epoch_12.pth --show --eval 'parsing'

# python tools/test_mhp.py /home/notebook/code/personal/S9043252/SMP2.0/multi-parsing/configs/smp/MHP_r101_fpn_8gpu_1x_offset_parsing_v1_ori_grid_unified_DCN_large.py /home/notebook/code/personal/S9043252/SMP2.0/multi-parsing/work_dirs/MHP_release_r101_fpn_8gpu_1x_offset_parsing_v1/ori_gird_DCN_1x_large_half_sigma/epoch_12.pth --show --eval 'parsing'

# python tools/test_mhp.py /home/notebook/code/personal/S9043252/SMP2.0/multi-parsing/configs/smp/MHP_r101_fpn_1gpu_1x_offset_parsing_v1_ori_grid_unified_DCN_little.py /home/notebook/code/personal/S9043252/SMP2.0/multi-parsing/work_dirs/MHP_release_r101_fpn_8gpu_1x_offset_parsing_v1/ori_gird_DCN_1x_little/epoch_12.pth --show --eval 'parsing'

CUDA_VISIBLE_DEVICES=0 python tools/test_mhp.py configs/repparsing/MHP_r101_fpn_half_gpu_3x_repparsing_DCN_fusion_metrics_light_amp_cluster.py work_dirs/MHP_release_r101_fpn_8gpu_3x_repparsing_v0_DCN_fusion_metrics/multi_light_amp_cluster/epoch_36.pth --show --eval 'parsing'

# python tools/test_mhp.py configs/smp/MHP_r50_fpn_2gpu_1x_offset_parsing_v1_ori_grid_unified_DCN.py work_dirs/MHP_release_r50_fpn_2gpu_1x_offset_parsing_v1/DCN/epoch_12.pth --show --eval 'parsing'

#python train.py configs/denseposeparsing/densepose_r101_fpn_8gpu_3x.py --gpus 2

#./tools/dist_train.sh configs/denseposeparsing/densepose_r101_fpn_8gpu_3x.py 2