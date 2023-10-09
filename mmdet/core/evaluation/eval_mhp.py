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
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def cal_one_mean_iou(image_array, label_array, NUM_CLASSES):
    hist = fast_hist(label_array, image_array, NUM_CLASSES).astype(np.float)
    num_cor_pix = np.diag(hist)
    num_gt_pix = hist.sum(1)
    union = num_gt_pix + hist.sum(0) - num_cor_pix
    iu = num_cor_pix / (num_gt_pix + hist.sum(0) - num_cor_pix)
    return iu

def get_gt(list_dat, task_id=None):
    if task_id is not None:
        cached = pickle.load(open('cache/gt_record_{}.pkl'.format(task_id)))
        if os.path.isfile(cached):
            return cached['class_recs'], cached['npos']
         
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
            for bbox in dat['bboxes']:
                mask_gt = np.array(Image.open(bbox['ann_path']))
                if len(mask_gt.shape)==3: mask_gt = mask_gt[:,:,0] # Make sure ann is a two dimensional np array. 
                if np.sum(mask_gt>0)==0: continue
                anno_adds.append(bbox['ann_path'])
                gt_box.append((bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']))
                npos = npos + 1 

            det = [False] * len(anno_adds)
        class_recs[imagename] = {'gt_box': np.array(gt_box),
                                 'anno_adds': anno_adds, 
                                 'det': det}
    return class_recs, npos


def eval_parsing_ap(results_all, dat_list, nb_class=59, ovthresh_seg=0.5, Sparse=False, From_pkl=False, task_id=None, data_root="/home/notebook/code/personal/S9043252/Parsing-R-CNN/data/LV-MHP-v2/", root="/home/notebook/code/personal/S9043252/multi-parsing/", visual_bad_case=False):
    '''
    results_all:
        {
            '3':
            'MASKS': [mask0,mask1...]
            'DETS' : [[center0, confidence0], [center0, confidence0]...]
        }
    From_pkl: load results from pickle files 
    Sparse: Indicate that the masks in the results are sparse matrices
    '''
    confidence = []
    image_ids  = []
    BB = []
    Local_segs_ptr = []
    ovthresh_seg = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for imagename in tqdm(results_all.keys(), desc='Loading results ..'):
        if From_pkl:
            results = pickle.load(gzip.open(results_all[imagename]))
        else:
            results = results_all[imagename]

        det_rects = results['DETS']
        for idx, rect in enumerate(det_rects):
            image_ids.append(imagename)
            confidence.append(rect[-1])
            BB.append(rect[:1])
            Local_segs_ptr.append(idx)

    confidence = np.array(confidence)
    BB = np.array(BB)
    Local_segs_ptr = np.array(Local_segs_ptr)

    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    Local_segs_ptr = Local_segs_ptr[sorted_ind]
    image_ids =  [image_ids[x]  for x in sorted_ind]


    class_recs_temp, npos = get_gt(dat_list, task_id=task_id)
    class_recs = [copy.deepcopy(class_recs_temp) for _ in range(len(ovthresh_seg))]
    nd = len(image_ids)
    tp_seg = [np.zeros(nd) for _ in range(len(ovthresh_seg))]
    fp_seg = [np.zeros(nd) for _ in range(len(ovthresh_seg))]
    pcp_list = [[] for _ in range(len(ovthresh_seg))]
    iou_list = []
    redundancy = 0
    low_quality = 0
    
    
    for d in trange(nd, desc='Finding AP^P at thres..'):
        try:
            #R = class_recs[image_ids[d]]
            R = []
            for j in range(len(ovthresh_seg)):
                R.append(class_recs[j][image_ids[d]])
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            jmax = -1
            if From_pkl:
                results = pickle.load(gzip.open(results_all[image_ids[d]]))
            else:
                results = results_all[image_ids[d]]

            mask0 = results['MASKS'][Local_segs_ptr[d]]
            if Sparse:
                mask_pred = mask0.toarray().astype(np.int) # decode sparse array if it is one
            else:
                mask_pred = mask0.astype(np.int)

            for i in range(len(R[0]['anno_adds'])):
                mask_gt = np.array(Image.open(R[0]['anno_adds'][i]))
                if len(mask_gt.shape)==3: mask_gt = mask_gt[:,:,0] # Make sure ann is a two dimensional np array. 

                seg_iou= cal_one_mean_iou(mask_pred.astype(np.uint8), mask_gt, nb_class)

                mean_seg_iou = np.nanmean(seg_iou)
                #import pdb;pdb.set_trace()
                #if d < 20:
                #    print(mean_seg_iou)

                if mean_seg_iou > ovmax:
                    ovmax =  mean_seg_iou
                    seg_iou_max = seg_iou 
                    jmax = i
                    mask_gt_u = np.unique(mask_gt)
            
            iou_list.append(ovmax)
        except Exception as e:
            print(e)
            print("anno_adds:",R[0]['anno_adds'][i])
            print(Local_segs_ptr[d])
            import pdb;pdb.set_trace()
            continue
            
        for j in range(len(ovthresh_seg)):
            if ovmax > ovthresh_seg[j]:
                if not R[j]['det'][jmax]:
                    tp_seg[j][d] = 1.
                    R[j]['det'][jmax] = 1
                    pcp_d = len(mask_gt_u[np.logical_and(mask_gt_u>0, mask_gt_u<nb_class)])
                    pcp_n = float(np.sum(seg_iou_max[1:]>ovthresh_seg[j]))
                    if pcp_d > 0:
                        pcp_list[j].append(pcp_n/pcp_d)
                    else:
                        pcp_list[j].append(0.0)
                else:
                    if j == 4:
                        redundancy = redundancy + 1
                    fp_seg[j][d] =  1.
            else:
                if j == 4:
                    low_quality = low_quality + 1
                fp_seg[j][d] =  1.
                if low_quality < 2000 and j==4:
                    if visual_bad_case:
                        filename = R[j]['anno_adds'][jmax].split("/")[-1].split(".")[0]
                        img = mmcv.imread(data_root + "val/images/" + filename.split("_")[0] + ".jpg")
                        img = img * 0.5
                        gt = np.array(Image.open(R[j]['anno_adds'][jmax]))
                        if len(gt.shape)==3: gt = gt[:,:,0]
                        img[gt>0] = img[gt>0] + np.array([0,0,200])
                        img[gt!=mask_pred] = img[gt!=mask_pred] + np.array([255,255,255])
                        mmcv.imwrite(img, root + "visual/offset/bad_case/"  + filename + "_" + str(ovmax) + ".jpg")


    print("redundancy: ",redundancy)
    print("low_quality: ",low_quality)
    # compute precision recall
    all_ap_seg = []
    all_pcp = []
    for j in range(len(ovthresh_seg)):
        fp_seg[j] = np.cumsum(fp_seg[j])
        tp_seg[j] = np.cumsum(tp_seg[j])
        rec_seg = tp_seg[j] / float(npos)
        prec_seg = tp_seg[j] / np.maximum(tp_seg[j] + fp_seg[j], np.finfo(np.float64).eps)

        ap_seg = voc_ap(rec_seg, prec_seg)
        all_ap_seg.append(ap_seg)

        assert (np.max(tp_seg[j]) == len(pcp_list[j])), "%d vs %d" % (np.max(tp_seg[j]), len(pcp_list[j]))
        pcp_list[j].extend([0.0] * (npos - len(pcp_list[j])))
        pcp = np.mean(pcp_list[j])
        all_pcp.append(pcp)

    miou = np.mean(iou_list)
    print('instance miou: ', miou)
    iou_list = np.array(iou_list)

    # fp_seg = np.cumsum(fp_seg)
    # tp_seg = np.cumsum(tp_seg)
    # rec_seg = tp_seg / float(npos)
    # prec_seg = tp_seg / (tp_seg + fp_seg)

    # ap_seg = voc_ap(rec_seg, prec_seg)

    # assert(np.max(tp_seg) == len(pcp_list)), "%d vs %d"%(np.max(tp_seg),len(pcp_list))
    # pcp_list.extend([0.0]*(npos - len(pcp_list)))
    # pcp = np.mean(pcp_list)
    # print('AP_seg, PCP:', ap_seg, pcp)
    # total_number = len(iou_list)
    # ap_v = []
    # for thres in [0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    #     ap_i = (iou_list>thres).sum()/total_number
    #     print('AP^',str(thres), ap_i)
    #     ap_v.append(ap_i)
    # print("ap_v: ", np.mean(ap_v))
    #import pdb;pdb.set_trace()

    return all_ap_seg, all_pcp, miou

def eval_parsing_scale_ap(results_all, dat_list, nb_class=59, ovthresh_seg=0.5, Sparse=False, From_pkl=False, task_id=None, data_root="/home/notebook/code/personal/S9043252/Parsing-R-CNN/data/LV-MHP-v2/", root="/home/notebook/code/personal/S9043252/multi-parsing/", visual_bad_case=False):
    '''
    results_all:
        {
            '3':
            'MASKS': [mask0,mask1...]
            'DETS' : [[center0, confidence0], [center0, confidence0]...]
        }
    From_pkl: load results from pickle files 
    Sparse: Indicate that the masks in the results are sparse matrices
    
    
    small: <32*32
    middle: 32*32 < <96*96
    large: >96*96
    '''
    
    confidence = []
    image_ids  = []
    BB = []
    Local_segs_ptr = []
    ovthresh_seg = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for imagename in tqdm(results_all.keys(), desc='Loading results ..'):
        if From_pkl:
            results = pickle.load(gzip.open(results_all[imagename]))
        else:
            results = results_all[imagename]

        det_rects = results['DETS']
        for idx, rect in enumerate(det_rects):
            image_ids.append(imagename)
            confidence.append(rect[-1])
            BB.append(rect[:1])
            Local_segs_ptr.append(idx)

    confidence = np.array(confidence)
    BB = np.array(BB)
    Local_segs_ptr = np.array(Local_segs_ptr)

    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    Local_segs_ptr = Local_segs_ptr[sorted_ind]
    image_ids =  [image_ids[x]  for x in sorted_ind]


    class_recs_temp, npos = get_gt(dat_list, task_id=task_id)
    class_recs = [copy.deepcopy(class_recs_temp) for _ in range(len(ovthresh_seg))]
    nd = len(image_ids)
    tp_seg = [np.zeros(nd) for _ in range(len(ovthresh_seg))]
    fp_seg = [np.zeros(nd) for _ in range(len(ovthresh_seg))]
    pcp_list = [[] for _ in range(len(ovthresh_seg))]
    iou_list = []
    redundancy = 0
    low_quality = 0
    large_seg_tp = [[] for _ in range(len(ovthresh_seg))]
    middle_seg_tp = [[] for _ in range(len(ovthresh_seg))]
    small_seg_tp = [[] for _ in range(len(ovthresh_seg))]
    large_seg_fp = [[] for _ in range(len(ovthresh_seg))]
    middle_seg_fp = [[] for _ in range(len(ovthresh_seg))]
    small_seg_fp = [[] for _ in range(len(ovthresh_seg))]
    large_num = 0
    middle_num = 0
    small_num = 0
    
    calculated_ids = []
    for d in trange(nd, desc='Finding AP^P at thres..'):
        try:
            #R = class_recs[image_ids[d]]
            R = []
            for j in range(len(ovthresh_seg)):
                R.append(class_recs[j][image_ids[d]])
    
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            jmax = -1
            if From_pkl:
                results = pickle.load(gzip.open(results_all[image_ids[d]]))
            else:
                results = results_all[image_ids[d]]

            mask0 = results['MASKS'][Local_segs_ptr[d]]
            if Sparse:
                mask_pred = mask0.toarray().astype(np.int) # decode sparse array if it is one
            else:
                mask_pred = mask0.astype(np.int)

            
            if image_ids[d] not in calculated_ids:
                add_nums = True
            else:
                add_nums = False
                
            for i in range(len(R[0]['anno_adds'])):
                part_nums_img = 0
                mask_gt = np.array(Image.open(R[0]['anno_adds'][i]))
                if len(mask_gt.shape)==3: mask_gt = mask_gt[:,:,0] # Make sure ann is a two dimensional np array. 
                
                gt_part_label = np.unique(mask_gt)
                
                if add_nums:
                    part_nums_img += len(gt_part_label)-1
                
                for p in gt_part_label:
                    
                    if p == 0:
                        continue
                    
                    part_mask_gt = mask_gt == p
                    part_mask_pred = mask_pred == p
                    
                    size_p = part_mask_gt.astype(np.uint8).sum()
                    if size_p < 32*32:
                        part_size = 's'
                        if add_nums:
                            small_num += 1
                    elif 32*32 <= size_p <= 96*96:
                        part_size = 'm'
                        if add_nums:
                            middle_num += 1
                    elif size_p > 96*96:
                        part_size = 'l'
                        if add_nums:
                            large_num += 1
                    
                    part_seg_iou = cal_one_mean_iou(part_mask_pred.astype(np.uint8), part_mask_gt.astype(np.uint8), nb_class)[1]
                    
                    for j in range(len(ovthresh_seg)):
                        if part_seg_iou > ovthresh_seg[j]:
                            if part_size == 's':
                                small_seg_tp[j].append(1)
                            elif part_size == 'm':
                                middle_seg_tp[j].append(1)
                            elif part_size == 'l':
                                large_seg_tp[j].append(1)
            calculated_ids.append(image_ids[d])

        except Exception as e:
            print(e)
            print("anno_adds:",R[0]['anno_adds'][i])
            print(Local_segs_ptr[d])
            import pdb;pdb.set_trace()
            continue

    print("large_num: ",large_num)
    print("middle_num: ",middle_num)
    print("small_num: ",small_num)
    # compute precision recall
    all_ap_seg_large = []
    all_ap_seg_middle = []
    all_ap_seg_small = []
    def cal(seg_tp, num):
        acc_list = []
        for j in range(len(ovthresh_seg)):
            acc_list.append(np.sum(seg_tp[j])/ num)
            print("AP threshold of " + str(ovthresh_seg[j]) + " is : " + str(np.sum(seg_tp[j])/ num))
    cal(large_seg_tp, large_num)
    cal(middle_seg_tp, middle_num)
    cal(small_seg_tp, small_num)
    import pdb;pdb.set_trace()

    return all_ap_seg_large, all_ap_seg_middle, all_ap_seg_small


def eval_seg_ap(results_all, dat_list, nb_class=2, ovthresh_seg=0.5, Sparse=False, From_pkl=False, task_id=None, data_root="data/LV-MHP-v2/", root="multi-parsing/", visual_bad_case=False):
    '''
    results_all:
        {
            '3':
            'MASKS': [mask0,mask1...]
            'DETS' : [[center0, confidence0], [center0, confidence0]...]
        }
    From_pkl: load results from pickle files 
    Sparse: Indicate that the masks in the results are sparse matrices
    '''
    confidence = []
    image_ids  = []
    BB = []
    Local_segs_ptr = []
    ovthresh_seg = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for imagename in tqdm(results_all.keys(), desc='Loading results ..'):
        if From_pkl:
            results = pickle.load(gzip.open(results_all[imagename]))
        else:
            results = results_all[imagename]

        det_rects = results['DETS']
        for idx, rect in enumerate(det_rects):
            image_ids.append(imagename)
            confidence.append(rect[-1])
            BB.append(rect[:1])
            Local_segs_ptr.append(idx)

    confidence = np.array(confidence)
    BB = np.array(BB)
    Local_segs_ptr = np.array(Local_segs_ptr)

    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    Local_segs_ptr = Local_segs_ptr[sorted_ind]
    image_ids =  [image_ids[x]  for x in sorted_ind]


    class_recs_temp, npos = get_gt(dat_list, task_id=task_id)
    class_recs = [copy.deepcopy(class_recs_temp) for _ in range(len(ovthresh_seg))]
    nd = len(image_ids)
    tp_seg = [np.zeros(nd) for _ in range(len(ovthresh_seg))]
    fp_seg = [np.zeros(nd) for _ in range(len(ovthresh_seg))]
    pcp_list = [[] for _ in range(len(ovthresh_seg))]
    iou_list = []
    redundancy = 0
    low_quality = 0
    
    
    for d in trange(nd, desc='Finding AP^P at thres..'):
        try:
            #R = class_recs[image_ids[d]]
            R = []
            for j in range(len(ovthresh_seg)):
                R.append(class_recs[j][image_ids[d]])
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            jmax = -1
            if From_pkl:
                results = pickle.load(gzip.open(results_all[image_ids[d]]))
            else:
                results = results_all[image_ids[d]]

            mask0 = results['MASKS'][Local_segs_ptr[d]]
            if Sparse:
                mask_pred = mask0.toarray().astype(np.int) # decode sparse array if it is one
            else:
                mask_pred = mask0.astype(np.int)

            for i in range(len(R[0]['anno_adds'])):
                mask_gt = np.array(Image.open(R[0]['anno_adds'][i]))
                if len(mask_gt.shape)==3: mask_gt = mask_gt[:,:,0] # Make sure ann is a two dimensional np array. 
                
                mask_pred = mask_pred>0
                mask_gt = (mask_gt>0).astype(np.uint8)
                seg_iou= cal_one_mean_iou(mask_pred.astype(np.uint8), mask_gt, nb_class)

                mean_seg_iou = np.nanmean(seg_iou)
                #import pdb;pdb.set_trace()
                #if d < 20:
                #    print(mean_seg_iou)

                if mean_seg_iou > ovmax:
                    ovmax =  mean_seg_iou
                    seg_iou_max = seg_iou 
                    jmax = i
                    mask_gt_u = np.unique(mask_gt)
            
            iou_list.append(ovmax)
        except Exception as e:
            print(e)
            print("anno_adds:",R[0]['anno_adds'][i])
            print(Local_segs_ptr[d])
            import pdb;pdb.set_trace()
            continue
            
        for j in range(len(ovthresh_seg)):
            if ovmax > ovthresh_seg[j]:
                if not R[j]['det'][jmax]:
                    tp_seg[j][d] = 1.
                    R[j]['det'][jmax] = 1
                    pcp_d = len(mask_gt_u[np.logical_and(mask_gt_u>0, mask_gt_u<nb_class)])
                    pcp_n = float(np.sum(seg_iou_max[1:]>ovthresh_seg[j]))
                    if pcp_d > 0:
                        pcp_list[j].append(pcp_n/pcp_d)
                    else:
                        pcp_list[j].append(0.0)
                else:
                    if j == 4:
                        redundancy = redundancy + 1
                    fp_seg[j][d] =  1.
            else:
                if j == 4:
                    low_quality = low_quality + 1
                fp_seg[j][d] =  1.
                if low_quality < 2000 and j==4:
                    if visual_bad_case:
                        filename = R[j]['anno_adds'][jmax].split("/")[-1].split(".")[0]
                        img = mmcv.imread(data_root + "val/images/" + filename.split("_")[0] + ".jpg")
                        img = img * 0.5
                        gt = np.array(Image.open(R[j]['anno_adds'][jmax]))
                        if len(gt.shape)==3: gt = gt[:,:,0]
                        img[gt>0] = img[gt>0] + np.array([0,0,200])
                        img[gt!=mask_pred] = img[gt!=mask_pred] + np.array([255,255,255])
                        mmcv.imwrite(img, root + "visual/offset/bad_case/"  + filename + "_" + str(ovmax) + ".jpg")


    print("redundancy: ",redundancy)
    print("low_quality: ",low_quality)
    # compute precision recall
    all_ap_seg = []
    all_pcp = []
    for j in range(len(ovthresh_seg)):
        fp_seg[j] = np.cumsum(fp_seg[j])
        tp_seg[j] = np.cumsum(tp_seg[j])
        rec_seg = tp_seg[j] / float(npos)
        prec_seg = tp_seg[j] / np.maximum(tp_seg[j] + fp_seg[j], np.finfo(np.float64).eps)

        ap_seg = voc_ap(rec_seg, prec_seg)
        all_ap_seg.append(ap_seg)

        assert (np.max(tp_seg[j]) == len(pcp_list[j])), "%d vs %d" % (np.max(tp_seg[j]), len(pcp_list[j]))
        pcp_list[j].extend([0.0] * (npos - len(pcp_list[j])))
        pcp = np.mean(pcp_list[j])
        all_pcp.append(pcp)

    miou = np.mean(iou_list)
    print('instance miou (not semantic): ', miou)
    iou_list = np.array(iou_list)

    # fp_seg = np.cumsum(fp_seg)
    # tp_seg = np.cumsum(tp_seg)
    # rec_seg = tp_seg / float(npos)
    # prec_seg = tp_seg / (tp_seg + fp_seg)

    # ap_seg = voc_ap(rec_seg, prec_seg)

    # assert(np.max(tp_seg) == len(pcp_list)), "%d vs %d"%(np.max(tp_seg),len(pcp_list))
    # pcp_list.extend([0.0]*(npos - len(pcp_list)))
    # pcp = np.mean(pcp_list)
    # print('AP_seg, PCP:', ap_seg, pcp)
    # total_number = len(iou_list)
    # ap_v = []
    # for thres in [0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    #     ap_i = (iou_list>thres).sum()/total_number
    #     print('AP^',str(thres), ap_i)
    #     ap_v.append(ap_i)
    # print("ap_v: ", np.mean(ap_v))
    #import pdb;pdb.set_trace()

    return all_ap_seg, all_pcp, miou

def get_prediction_from_gt(dat_list, NUM_CLASSES, cache_pkl=False, cache_pkl_path='tmp/', Sparse=False):
    '''
    cache_pkl: if the memory can't hold all the results, set cache_pkl to be true to pickle down the results 
    Sparse: Sparsify the masks to save memory
    '''
    results_all = {}
    for dat in tqdm(dat_list, desc='Generating predictions ..'):
        results = {} 

        dets, masks = [], []
        for bbox in dat['bboxes']:
            mask_gt = np.array(Image.open(bbox['ann_path']))
            if len(mask_gt.shape)==3: mask_gt = mask_gt[:,:,0] # Make sure ann is a two dimensional np array. 
            if np.sum(mask_gt)==0: continue
            ys, xs = np.where(mask_gt>0)
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            dets.append((x1, y1, x2, y2, 1.0))
            masks.append(mask_gt)            

        # Reuiqred Field of each result: a list of masks, each is a multi-class masks for one person.
            # It can also be sparsified to [scipy.sparse.csr_matrix(mask) for mask in masks] to save memory cost
        results['MASKS']= masks if not Sparse else [scipy.sparse.csr_matrix(mask) for mask in masks]
        # Reuiqred Field of each result, a list of detections corresponding to results['MASKS']. 
        results['DETS'] = dets    

        key = dat['filepath'].split('/')[-1].replace('.jpg', '')
        if cache_pkl:
            results_cache_add = cache_pkl_path + key + '.pklz'
            pickle.dump(results, gzip.open(results_cache_add, 'w'))
            results_all[key] = results_cache_add
        else:
            results_all[key]=results
    return results_all

def cache_gt_record():
    for set_ in ['val', 'test_all', 'test_inter_top20', 'test_inter_top10']:
        dat_list = pickle.load(open('cache/dat_list_{}.pkl'.format(set_)))
        class_recs, npos = get_gt(dat_list)
        pickle.dump({'class_recs':class_recs, 'npos': npos}, open('cache/gt_record_{}.pkl'.format(set_), 'w'))
        
if __name__ == '__main__':
    import mhp_data
    data_root = '/home/lijianshu/MultiPerson/data/LV-MHP-v2/'
    # set_ in ['train', 'val', 'test_all', 'test_inter_top20', 'test_inter_top10'])
    set_ = 'val'
    dat_list = mhp_data.get_data(data_root, set_)
    #dat_list = pickle.load(open('cache/dat_list_val.pkl'))
    
    NUM_CLASSES = 59
    results_all = get_prediction_from_gt(dat_list, NUM_CLASSES, cache_pkl=False, Sparse=False)
    eval_seg_ap(results_all, dat_list, nb_class=NUM_CLASSES,ovthresh_seg=0.5, From_pkl=False, Sparse=False)

