import os.path as osp

import mmcv
import numpy as np
from torch.utils.data import Dataset

from .builder import DATASETS
from .pipelines import Compose

from .mhp_data import get_data
import json
from tqdm import tqdm
from mmdet.core import parsing_iou, eval_parsing_ap

@DATASETS.register_module
class MHP(Dataset):
    """MHP dataset for parsing.

    Annotation format:
    [
        {
            
        },
        ...
    ]

    The `ann` field is optional for testing.
    """

    CLASSES = None

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True):
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self._set = 'train' if test_mode == False else 'val'
        self.ann_file = ann_file
        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        # load annotations (and proposals)
        self.img_infos = json.load(open("cache/dat_list_{}.json".format(self._set), 'r'))
        self.ann_infos = json.load(open("cache/data_list_{}.json".format(self._set), 'r'))
        for i, img_info in enumerate(self.img_infos):
            img_info['filepath'] = data_root + img_info['filepath'].split('/home/notebook/code/personal/S9043252/Parsing-R-CNN/data/LV-MHP-v2/')[1]
        for i, ann_info in enumerate(self.ann_infos):
            ann_info['filepath'] = data_root + ann_info['filepath'].split('/home/notebook/code/personal/S9043252/Parsing-R-CNN/data/LV-MHP-v2/')[1]
            for bbox in ann_info['bboxes']:
                bbox['ann_path'] = data_root + bbox['ann_path'].split('/home/notebook/code/personal/S9043252/Parsing-R-CNN/data/LV-MHP-v2/')[1]
        
        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None
        # filter images too small
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.img_infos = [self.img_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # processing pipeline
        self.pipeline = Compose(pipeline)

    def __len__(self):
        return len(self.img_infos)

    def load_annotations(self, ann_file):
        return mmcv.load(ann_file)

    def load_proposals(self, proposal_file):
        return mmcv.load(proposal_file)

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['parsing_fields'] = []
        results['keypoints_fields'] = []

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            #import pdb;pdb.set_trace()
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):
        """ 
        return
        result :
            {'img_info' :{
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
                }
            'anno_info': {
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
                }
            'img_prefix' = self.img_prefix
            'seg_prefix' = self.seg_prefix
            'proposal_file' = self.proposal_file
            'bbox_fields' = []
            'mask_fields' = []
            'seg_fields' = []
            'parsing_fields' = []
        """
        img_info = self.img_infos[idx]
        ann_info = self.ann_infos[idx]
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.ann_infos[idx]
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """

        dat_list = get_data(self.data_root, 'val')
        _iou, _miou, mean_acc, pixel_acc = parsing_iou('work_dirs/pred_cate', 'work_dirs/gt_cate', 59)
        parsing_result = {'mIoU': _miou, 'pixel_acc': pixel_acc, 'mean_acc': mean_acc}

        for i, iou in enumerate(_iou):
            print(' {:<30}:  {:.2f}'.format(parsing_name[i], 100 * iou))

        print('----------------------------------------')
        print(' {:<30}:  {:.2f}'.format('mean IoU', 100 * _miou))
        print(' {:<30}:  {:.2f}'.format('pixel acc', 100 * pixel_acc))
        print(' {:<30}:  {:.2f}'.format('mean acc', 100 * mean_acc))

        all_ap_p, all_pcp, miou = eval_parsing_ap(results, dat_list, nb_class=59, ovthresh_seg=0.5, From_pkl=False, Sparse=False)
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

        eval_results = parsing_result
        eval_results['AP0.5'] = all_ap_p[4]
        eval_results['APvol'] = ap_p_vol
        eval_results['PCP0.5'] = all_pcp[4]

        return eval_results
