import argparse
import os
import os.path as osp
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import shutil
import tempfile
from PIL import Image
import mmcv
import torch
import torch.nn.functional as F
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import init_dist, get_dist_info, wrap_fp16_model, load_checkpoint

from mmdet.core import parsing_matrix_nms, eval_parsing_ap, eval_seg_ap, eval_parsing_scale_ap
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
import time
import numpy as np
import pycocotools.mask as mask_util
from mmdet.datasets.pipelines.loading_mhp import LoadAnnotations_mhp
from mmdet.datasets.mhp_data import get_data
from mmdet.apis import set_random_seed
import glob
from tqdm import tqdm
import cv2

sum_grids = [3728, 3472, 2896, 1600, 0]
grids = [12,16,24,36,40]

parsing_name = ['background','cap/hat', ' helmet', ' face', ' hair', ' left-arm', ' right-arm', ' left-hand', ' right-hand', 'protector', ' bikini/bra', ' jacket/windbreaker/hoodie', ' t-shirt', 'polo-shirt', ' sweater', ' singlet', ' torso-skin', ' pants', ' shorts/swimshorts', ' skirt', ' stockings', ' socks', ' left-boot', ' right-boot', ' leftshoe', ' right-shoe', ' left-highheel', ' right-highheel', ' left-sandal', 'right-sandal', ' left-leg', ' right-leg', ' left-foot', ' right-foot', ' coat', 'dress', ' robe', ' jumpsuits', ' other-full-body-clothes', ' headwear', 'backpack', ' ball', ' bats', ' belt', ' bottle', ' carrybag', ' cases', ' sunglasses', ' eyewear', ' gloves', ' scarf', ' umbrella', ' wallet/purse', 'watch', ' wristband', ' tie', ' other-accessaries', ' other-upper-bodyclothes', ' other-lower-body-clothes']

def get_gt_cate(list_dat, nb_class=59, ovthresh_seg=0.5):
    class_recs = {}
    npos = 0

    for dat in tqdm(list_dat, desc='Loading gt..'):
        imagename = dat['filepath'].split('/')[-1].replace('.jpg','')
        if len(dat['bboxes']) == 0:
            gt_box=np.array([])
            det = []
            anno_adds = []
        else:
            gt_box = []
            anno_adds = []
            masks = np.zeros((dat['height'],dat['width']),dtype='uint8')
            for bbox in dat['bboxes']:
                mask_gt = np.array(Image.open(bbox['ann_path']))
                if len(mask_gt.shape)==3: mask_gt = mask_gt[:,:,0] # Make sure ann is a two dimensional np array. 
                if np.sum(mask_gt>0)==0: continue
                masks_idx = mask_gt > 0
                masks[masks_idx] = mask_gt[masks_idx]
                anno_adds.append(bbox['ann_path'])
                gt_box.append((bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']))
                npos = npos + 1 
            det = [False] * len(anno_adds)
        if np.max(masks)>59:
            import pdb;pdb.set_trace()
        class_recs[imagename] = {'gt_box': np.array(gt_box),
                                 'anno_adds': anno_adds, 
                                 'masks': masks,
                                 'det': det}
        mmcv.imwrite(masks, "eval_file/gt_cate/"+dat['filepath'].split('/')[-1].split('.')[0] + '.png')
        

def single_gpu_test(model, data_loader, eval_types, cfg, show=False, verbose=True):
    model.eval()
    results_all = {}
    dataset = data_loader.dataset
    num_classes = 58

    prog_bar = mmcv.ProgressBar(len(dataset))
    empty_result_list = []
    num_empty_result = 0
    all_time = 0
    import time

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # seg_result: [imgs, (segs, labels, scores), ins, (H,W)]
            t0 = time.time()
            seg_result = model(return_loss=False, rescale=not show, **data)
            print('inference: ', time.time() - t0)
            all_time += time.time() - t0
        for img_i in range(len(seg_result)):
            img_info = data['img_metas'][0].data[0][img_i]
            h, w = img_info['ori_shape'][0], img_info['ori_shape'][1]
            try:
                if seg_result[img_i] == None:
                    result = {'MASKS': [],'DETS':[]}
                    num_empty_result = num_empty_result + 1
                    print('number of the empty result: ', num_empty_result)
                    print(img_info['filename'])
                    empty_result_list.append(img_info['filename']) 
                else:
                    all_parsings, all_scores = seg_result[img_i][0], seg_result[img_i][2]
                    result = {'MASKS': all_parsings,'DETS': [[i,d.cpu()] for i,d in enumerate(all_scores)]}
                
            except Exception as e:
                print(e)
                import pdb;pdb.set_trace()
        # results_all {dict}
        # {
        #   '3':
        #       'MASKS': [mask0, mask1]
        #       'DETS': [ score0, score1]
        # }
        
        #import pdb;pdb.set_trace()
            key = img_info['filename'].split('/')[-1].replace('.jpg', '')
            results_all[key] = result
        for _ in range(data_loader.batch_size):
            prog_bar.update()
    print(all_time)
    return results_all


def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    num_classes = len(dataset.CLASSES)

    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            seg_result = model(return_loss=False, rescale=True, **data)
        #import pdb;pdb.set_trace()
        #result = get_masks(seg_result, num_classes=num_classes)
        results.append(seg_result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--json_out',
        help='output result file name without extension',
        type=str)
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints', 'parsing', 'scale'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    try:
        hist = np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)
    except Exception as e:
        import pdb;pdb.set_trace()
    return hist

def compute_hist(predict_list, im_dir, num_parsing, scale_test=False):
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




def main():
    args = parse_args()

    assert args.out or args.show or args.json_out, \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "--out" or "--show" or "--json_out"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    if args.json_out is not None and args.json_out.endswith('.json'):
        args.json_out = args.json_out[:-5]
    if args.seed is not None:
        print('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)
    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)

    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=None)
    #import pdb;pdb.set_trace()
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    while not osp.isfile(args.checkpoint):
        print('Waiting for {} to exist...'.format(args.checkpoint))
        time.sleep(60)
    #import pdb;pdb.set_trace()
    #cp_add = args.checkpoint.split('epoch_')[:-1][0]
    ap_result = {}

    for epoch_num in range(12,13):
        # args.checkpoint = cp_add + 'epoch_'+str(epoch_num) +'.pth'
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES
        
        distributed = False
        eval_types = args.eval
        print('eval_types:', eval_types)
        data_root = cfg.data['test']['data_root']
        root = os.getcwd()
        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            outputs = single_gpu_test(model, data_loader, eval_types, cfg)
        else:
            model = MMDistributedDataParallel(model.cuda())
            outputs = multi_gpu_test(model, data_loader, args.tmpdir)

        
        rank, _ = get_dist_info()
        if eval_types == ['parsing']:
            dat_list = get_data(data_root, 'val')
            #import pdb;pdb.set_trace()
            # pcp, ap,miou = eval_parsing_ap(outputs, dat_list, nb_class=59, ovthresh_seg=0.5, From_pkl=False, Sparse=False)

            #get_gt_cate(dat_list, nb_class=59, ovthresh_seg=0.5)

            _iou, _miou, mean_acc, pixel_acc = parsing_iou('eval_file/pred_cate/', 'eval_file/gt_cate', 59)

            parsing_result = {'mIoU': _miou, 'pixel_acc': pixel_acc, 'mean_acc': mean_acc}

            assert len(parsing_name) == len(_iou), '{} VS {}'.format(str(len(parsing_name)), str(len(_iou)))

            for i, iou in enumerate(_iou):
                print(' {:<30}:  {:.2f}'.format(parsing_name[i], 100 * iou))

            print('----------------------------------------')
            print(' {:<30}:  {:.2f}'.format('mean IoU', 100 * _miou))
            print(' {:<30}:  {:.2f}'.format('pixel acc', 100 * pixel_acc))
            print(' {:<30}:  {:.2f}'.format('mean acc', 100 * mean_acc))

            all_ap_p, all_pcp, miou = eval_parsing_ap(outputs, dat_list, nb_class=59, ovthresh_seg=0.5,
                                                        From_pkl=False, Sparse=False, 
                                                        data_root=data_root, root=root, 
                                                        visual_bad_case=False)
            ap_p_vol = np.mean(all_ap_p)

            print('~~~~ Summary metrics ~~~~')
            print(' Average Precision based on part (APp)               @[mIoU=0.10:0.90 ] = {:.3f}'.format(ap_p_vol))
            print(' Average Precision based on part (APp)               @[mIoU=0.10      ] = {:.3f}'.format(all_ap_p[0]))
            print(' Average Precision based on part (APp)               @[mIoU=0.20      ] = {:.3f}'.format(all_ap_p[1]))
            print(' Average Precision based on part (APp)               @[mIoU=0.30      ] = {:.3f}'.format(all_ap_p[2]))
            print(' Average Precision based on part (APp)               @[mIoU=0.40      ] = {:.3f}'.format(all_ap_p[3]))
            print(' Average Precision based on part (APp)               @[mIoU=0.50      ] = {:.3f}'.format(all_ap_p[4]))
            print(' Average Precision based on part (APp)               @[mIoU=0.60      ] = {:.3f}'.format(all_ap_p[5]))
            print(' Average Precision based on part (APp)               @[mIoU=0.70      ] = {:.3f}'.format(all_ap_p[6]))
            print(' Average Precision based on part (APp)               @[mIoU=0.80      ] = {:.3f}'.format(all_ap_p[7]))
            print(' Average Precision based on part (APp)               @[mIoU=0.90      ] = {:.3f}'.format(all_ap_p[8]))
            print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.10      ] = {:.3f}'.format(all_pcp[0]))
            print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.20      ] = {:.3f}'.format(all_pcp[1]))
            print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.30      ] = {:.3f}'.format(all_pcp[2]))
            print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.40      ] = {:.3f}'.format(all_pcp[3]))
            print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.50      ] = {:.3f}'.format(all_pcp[4]))
            print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.60      ] = {:.3f}'.format(all_pcp[5]))
            print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.70      ] = {:.3f}'.format(all_pcp[6]))
            print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.80      ] = {:.3f}'.format(all_pcp[7]))
            print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.90      ] = {:.3f}'.format(all_pcp[8]))
            import pdb;pdb.set_trace()
        elif eval_types == ['segm']:
            dat_list = get_data(data_root, 'val')
            all_ap_p, all_pcp, miou = eval_seg_ap(outputs, dat_list, nb_class=2, ovthresh_seg=0.5,
                                                        From_pkl=False, Sparse=False, 
                                                        data_root=data_root, root=root, 
                                                        visual_bad_case=False)
            ap_p_vol = np.mean(all_ap_p)
            
            print('~~~~ Summary metrics ~~~~')
            print(' Average Precision based on part (APp)               @[mIoU=0.10:0.90 ] = {:.3f}'.format(ap_p_vol))
            print(' Average Precision based on part (APp)               @[mIoU=0.10      ] = {:.3f}'.format(all_ap_p[0]))
            print(' Average Precision based on part (APp)               @[mIoU=0.20      ] = {:.3f}'.format(all_ap_p[1]))
            print(' Average Precision based on part (APp)               @[mIoU=0.30      ] = {:.3f}'.format(all_ap_p[2]))
            print(' Average Precision based on part (APp)               @[mIoU=0.40      ] = {:.3f}'.format(all_ap_p[3]))
            print(' Average Precision based on part (APp)               @[mIoU=0.50      ] = {:.3f}'.format(all_ap_p[4]))
            print(' Average Precision based on part (APp)               @[mIoU=0.60      ] = {:.3f}'.format(all_ap_p[5]))
            print(' Average Precision based on part (APp)               @[mIoU=0.70      ] = {:.3f}'.format(all_ap_p[6]))
            print(' Average Precision based on part (APp)               @[mIoU=0.80      ] = {:.3f}'.format(all_ap_p[7]))
            print(' Average Precision based on part (APp)               @[mIoU=0.90      ] = {:.3f}'.format(all_ap_p[8]))
            print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.10      ] = {:.3f}'.format(all_pcp[0]))
            print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.20      ] = {:.3f}'.format(all_pcp[1]))
            print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.30      ] = {:.3f}'.format(all_pcp[2]))
            print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.40      ] = {:.3f}'.format(all_pcp[3]))
            print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.50      ] = {:.3f}'.format(all_pcp[4]))
            print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.60      ] = {:.3f}'.format(all_pcp[5]))
            print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.70      ] = {:.3f}'.format(all_pcp[6]))
            print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.80      ] = {:.3f}'.format(all_pcp[7]))
            print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.90      ] = {:.3f}'.format(all_pcp[8]))
        
        elif eval_types == ['scale']:
            dat_list = get_data(data_root, 'val')
            all_ap_p, all_pcp, miou = eval_parsing_scale_ap(outputs, dat_list, nb_class=2, ovthresh_seg=0.5,
                                                        From_pkl=False, Sparse=False, 
                                                        data_root=data_root, root=root, 
                                                        visual_bad_case=False)
            ap_p_vol = np.mean(all_ap_p)
            
            print('~~~~ Summary metrics ~~~~')
            print(' Average Precision based on part (APp)               @[mIoU=0.10:0.90 ] = {:.3f}'.format(ap_p_vol))
            print(' Average Precision based on part (APp)               @[mIoU=0.10      ] = {:.3f}'.format(all_ap_p[0]))
            print(' Average Precision based on part (APp)               @[mIoU=0.20      ] = {:.3f}'.format(all_ap_p[1]))
            print(' Average Precision based on part (APp)               @[mIoU=0.30      ] = {:.3f}'.format(all_ap_p[2]))
            print(' Average Precision based on part (APp)               @[mIoU=0.40      ] = {:.3f}'.format(all_ap_p[3]))
            print(' Average Precision based on part (APp)               @[mIoU=0.50      ] = {:.3f}'.format(all_ap_p[4]))
            print(' Average Precision based on part (APp)               @[mIoU=0.60      ] = {:.3f}'.format(all_ap_p[5]))
            print(' Average Precision based on part (APp)               @[mIoU=0.70      ] = {:.3f}'.format(all_ap_p[6]))
            print(' Average Precision based on part (APp)               @[mIoU=0.80      ] = {:.3f}'.format(all_ap_p[7]))
            print(' Average Precision based on part (APp)               @[mIoU=0.90      ] = {:.3f}'.format(all_ap_p[8]))
            print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.10      ] = {:.3f}'.format(all_pcp[0]))
            print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.20      ] = {:.3f}'.format(all_pcp[1]))
            print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.30      ] = {:.3f}'.format(all_pcp[2]))
            print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.40      ] = {:.3f}'.format(all_pcp[3]))
            print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.50      ] = {:.3f}'.format(all_pcp[4]))
            print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.60      ] = {:.3f}'.format(all_pcp[5]))
            print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.70      ] = {:.3f}'.format(all_pcp[6]))
            print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.80      ] = {:.3f}'.format(all_pcp[7]))
            print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.90      ] = {:.3f}'.format(all_pcp[8]))
        
        ap_result[epoch_num] = {'miou':miou,'pcp':all_pcp,'ap':all_ap_p}  


if __name__ == '__main__':
    main()

