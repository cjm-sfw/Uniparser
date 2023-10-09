from mmdet.apis import init_detector, inference_detector_parsing, show_result_pyplot
import mmcv
import numpy as np
import cv2
import os
from PIL import Image

root = os.getcwd().split('demo')[0]
data_root = "/root/data/LV-MHP-v2/"

def offset_visual(offset_vis, img, filename, S=40):
    save_dir = root + "visual/offset/offset_map/test/" + filename.split('/')[-1].split('.')[0] + '/'
    os.makedirs(save_dir, exist_ok=True)
    img_center = mmcv.imresize(img, (S,S))
    img_center = img_center * 0.75

    for ind in offset_vis:
        img_ind = mmcv.imresize(img, (S,S))
        img_ind = img_ind * 0.75

        p_is, x, y, (ys, xs, _), scores = ind

        # p = ind[0]
        # p_i, x, y, x_p, y_p, score = p
        img_center[y][x] = np.array([255,255,255])
        img_center[y-1][x] = np.array([255,255,255])
        img_center[y+1][x] = np.array([255,255,255])
        img_center[y][x-1] = np.array([255,255,255])
        img_center[y][x+1] = np.array([255,255,255])
        
        for i in range(len(ys)):
            img_ori = mmcv.imresize(img, (S,S))
            img_ori = img_ori * 0.75
            p_i, x_p, y_p, score = p_is[i], xs[i], ys[i], scores[i]
            # x,y = int(x*w/80), int(y*h/80)
            # x_p, y_p = int(x_p*w/80), int(y_p*h/80)
            img_ori[y][x] = np.array([255,255,255])
            img_ori[y-1][x] = np.array([255,255,255])
            img_ori[y+1][x] = np.array([255,255,255])
            img_ori[y][x-1] = np.array([255,255,255])
            img_ori[y][x+1] = np.array([255,255,255])
            img_ori[y_p][x_p] = np.array([0,255,255])
            img_ori[y_p][min(x_p+1,S-1)] = np.array([0,255,255])
            img_ori[min(y_p+1,S-1)][x_p] = np.array([0,255,255])
            img_ori[y_p][min(x_p-1,S-1)] = np.array([0,255,255])
            img_ori[min(y_p-1,S-1)][x_p] = np.array([0,255,255])
            mmcv.imwrite(img_ori, save_dir + str(x.item())+ "_" + str(y.item()) + "_" + str(p_i.item()) + "_" + str(score.item()) + ".jpg")

            if score>0.3:
                img_ind[y][x] = np.array([255,255,255])
                img_ind[y-1][x] = np.array([255,255,255])
                img_ind[y+1][x] = np.array([255,255,255])
                img_ind[y][x-1] = np.array([255,255,255])
                img_ind[y][x+1] = np.array([255,255,255])
                img_ind[y_p][x_p] = np.array([0,255,255])
                img_ind[y_p][min(x_p+1,S-1)] = np.array([0,255,255])
                img_ind[min(y_p+1,S-1)][x_p] = np.array([0,255,255])
                img_ind[y_p][min(x_p-1,S-1)] = np.array([0,255,255])
                img_ind[min(y_p-1,S-1)][x_p] = np.array([0,255,255])

        mmcv.imwrite(img_ind, save_dir + str(x.item())+ "_" + str(y.item())  + ".jpg")
    mmcv.imwrite(img_center, save_dir + "center.jpg")

def heatmap_visual(offset_vis, img, filename, S=40):
    save_dir = root + "visual/offset/offset_map/test/" + filename.split('/')[-1].split('.')[0] + '/'
    os.makedirs(save_dir, exist_ok=True)
    for ind in offset_vis:
        img_ind = mmcv.imresize(img, (S,S))
        img_ind = img_ind * 0.75

        for p in ind:
            img_ori = mmcv.imresize(img, (S,S))
            img_ori = img_ori * 0.75
            p_i, x, y, x_p, y_p, score = p
            # x,y = int(x*w/80), int(y*h/80)
            # x_p, y_p = int(x_p*w/80), int(y_p*h/80)
            img_ori[y][x] = np.array([255,255,255])
            img_ori[y-1][x] = np.array([255,255,255])
            img_ori[y+1][x] = np.array([255,255,255])
            img_ori[y][x-1] = np.array([255,255,255])
            img_ori[y][x+1] = np.array([255,255,255])
            img_ori[y_p][x_p] = np.array([0,0,255])
            img_ori[y_p][min(x_p+1,39)] = np.array([0,0,255])
            img_ori[min(y_p+1,39)][x_p] = np.array([0,0,255])
            img_ori[y_p][max(x_p-1,39)] = np.array([0,0,255])
            img_ori[max(y_p-1,39)][x_p] = np.array([0,0,255])
            mmcv.imwrite(img_ori, save_dir + str(x.item())+ "_" + str(y.item()) + "_" + str(p_i) + "_" + str(score.item()) + ".jpg")

            img_ind[y][x] = np.array([255,255,255])
            img_ind[y-1][x] = np.array([255,255,255])
            img_ind[y+1][x] = np.array([255,255,255])
            img_ind[y][x-1] = np.array([255,255,255])
            img_ind[y][x+1] = np.array([255,255,255])
            img_ind[y_p][x_p] = np.array([0,0,255])
            img_ind[y_p][min(x_p+1,39)] = np.array([0,0,255])
            img_ind[min(y_p+1,39)][x_p] = np.array([0,0,255])
            img_ind[y_p][max(x_p-1,39)] = np.array([0,0,255])
            img_ind[max(y_p-1,39)][x_p] = np.array([0,0,255])

        mmcv.imwrite(img_ind, save_dir + str(x.item())+ "_" + str(y.item())  + ".jpg")


import matplotlib.pyplot as plt
def seg_visual(seg_masks, img):
    for i in range(len(seg_masks)):
        seg_mask = seg_masks[i]
        plt.figure(figsize=(8,5))
        plt.imshow(seg_mask)

def seg_visual_in_one(seg_masks, img):
    parsing = np.zeros_like(seg_masks[0])
    for i in range(len(seg_masks)):
        seg_mask = seg_masks[i]
        p_ind = seg_mask>0
        parsing[p_ind] =  seg_mask[p_ind]
    plt.figure(figsize=(16,10))
    plt.imshow(parsing)

def seg_visual_human_in_one(seg_masks, img):
    parsing = np.zeros_like(seg_masks[0])
    for i in range(len(seg_masks)):
        seg_mask = seg_masks[i]
        p_ind = seg_mask>0
        parsing[p_ind] = get_color(i,data_root).mean()*255
    plt.figure(figsize=(16,10))
    plt.imshow(parsing)

def center_of_mass(bitmasks):
    _, h, w = bitmasks.size()
    ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
    xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device)

    m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
    m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
    m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
    center_x = m10 / m00
    center_y = m01 / m00
    return center_x, center_y

def gt_visual(gtname, data_root):
    add = data_root + "val/parsing_annos/"
    gt = np.array(Image.open(add+gtname))
    if len(gt.shape)==3: gt = gt[:,:,0]
    plt.figure(figsize=(8,5))
    plt.imshow(gt)

def gt_visual_in_one(gtname_list, data_root):
    add = data_root + "val/parsing_annos/"
    gt = np.array(Image.open(add+gtname_list[0]))
    if len(gt.shape)==3: gt = gt[:,:,0]
    parsing = np.zeros_like(gt)
    for i in range(len(gtname_list)):
        gtname = gtname_list[i]
        gt = np.array(Image.open(add+gtname))
        if len(gt.shape)==3: gt = gt[:,:,0]
        gt_ind = gt>0
        parsing[gt_ind] = gt[gt_ind]
    plt.figure(figsize=(16,10))
    plt.imshow(parsing)

def gt_visual_human_in_one(gtname_list, data_root):
    add =  data_root + "val/parsing_annos/"
    gt = np.array(Image.open(add+gtname_list[0]))
    if len(gt.shape)==3: gt = gt[:,:,0]
    parsing = np.zeros_like(gt)
    for i in range(len(gtname_list)):
        gtname = gtname_list[i]
        gt = np.array(Image.open(add+gtname))
        if len(gt.shape)==3: gt = gt[:,:,0]
        gt_ind = gt>0
        parsing[gt_ind] = get_color(i, data_root).mean()*255
    plt.figure(figsize=(16,10))
    plt.imshow(parsing)

def gt_visual_human_in_one_densepose(gtname):
    gt = np.array(Image.open(add+gtname))
    if len(gt.shape)==3: gt = gt[:,:,0]
    parsing = np.zeros_like(gt)
    for i in range(len(gtname_list)):
        gtname = gtname_list[i]
        gt = np.array(Image.open(add+gtname))
        if len(gt.shape)==3: gt = gt[:,:,0]
        gt_ind = gt>0
        parsing[gt_ind] = get_color(i, data_root).mean()*255
    plt.figure(figsize=(16,10))
    plt.imshow(parsing)

def center_visual(gtname, data_root):
    add =  data_root + "val/parsing_annos/"
    gt = np.array(Image.open(add+gtname))
    if len(gt.shape)==3: gt = gt[:,:,0]
    human_center = center_of_mass(torch.from_numpy(gt>0).unsqueeze(0))
    part_list = np.unique(gt)
    part_centers = []
    for i in part_list:
        bit_mask = torch.from_numpy((gt == i)>0).unsqueeze(0)
        part_centers.append(center_of_mass(bit_mask))
    return human_center,part_centers

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def cal_one_mean_iou(image_array, label_array, NUM_CLASSES, detail=False):
    hist = fast_hist(label_array, image_array, NUM_CLASSES).astype(np.float)
    num_cor_pix = np.diag(hist)
    num_gt_pix = hist.sum(1)
    union = num_gt_pix + hist.sum(0) - num_cor_pix
    if detail == True:
        print("inter: ", num_cor_pix)
        print("union: ", (num_gt_pix + hist.sum(0) - num_cor_pix))
    iu = num_cor_pix / (num_gt_pix + hist.sum(0) - num_cor_pix)
    return iu

def gt_part_eval(gt_mask_name, p, data_root):
    add =  data_root + "val/parsing_annos/"
    gt = np.array(Image.open(add+gt_mask_name))
    if len(gt.shape)==3: gt = gt[:,:,0]
    print("gt pixels:",((gt==p)>0).sum())
    plt.figure(figsize=(8,5))
    plt.imshow((gt==p))

def pred_part_eval(seg_mask, p):
    print("pred pixels:",((seg_mask==p)>0).sum())
    plt.figure(figsize=(8,5))
    plt.imshow((seg_mask==p))

def eval_single(seg_mask, gt_mask_name, data_root, detail=False):
    add =  data_root + "val/parsing_annos/"
    gt = np.array(Image.open(add+gt_mask_name))
    if len(gt.shape)==3: gt = gt[:,:,0]

    seg_iou = cal_one_mean_iou(gt, seg_mask, 59, detail)
    mean_seg_iou = np.nanmean(seg_iou)

    print("gt unique: ", np.unique(gt))
    print("pred unique: ", np.unique(seg_mask))
    print(seg_iou)
    print(mean_seg_iou)

def compare_part(i,p,gt_name,seg_masks):
    part = ['cap/hat', ' helmet', ' face', ' hair', ' left-arm', ' right-arm', ' left-hand', ' right-hand', 'protector', ' bikini/bra', ' jacket/windbreaker/hoodie', ' t-shirt', 'polo-shirt', ' sweater', ' singlet', ' torso-skin', ' pants', ' shorts/swimshorts', ' skirt', ' stockings', ' socks', ' left-boot', ' right-boot', ' leftshoe', ' right-shoe', ' left-highheel', ' right-highheel', ' left-sandal', 'right-sandal', ' left-leg', ' right-leg', ' left-foot', ' right-foot', ' coat', 'dress', ' robe', ' jumpsuits', ' other-full-body-clothes', ' headwear', 'backpack', ' ball', ' bats', ' belt', ' bottle', ' carrybag', ' cases', ' sunglasses', ' eyewear', ' gloves', ' scarf', ' umbrella', ' wallet/purse', 'watch', ' wristband', ' tie', ' other-accessaries', ' other-upper-bodyclothes', ' other-lower-body-clothes']
    print(part[p-1])
    gt_part_eval(gt_name, p)
    pred_part_eval(seg_masks[i], p)

def cobmine_img_and_gt(gtname, img_name, data_root):
    gt_add =  data_root + "val/parsing_annos/"
    gt = np.array(Image.open(gt_add+gtname))
    img = np.array(Image.open(img_name))
    if len(gt.shape)==3: gt = gt[:,:,0]
    print(np.unique(gt))
    for i in np.unique(gt):
        ind = gt==i
        if i == 0:
            continue
        else:
            img[ind] = img[ind]*0.5 + np.array([125-i*2,i*2,125-i*2])
    #comb = (img*0.5+gt).astype('uint8')
    #comb = img
    plt.figure(figsize=(8,5))
    plt.imshow(img)

def cobmine_img_and_seg(seg_masks, img_name, data_root):
    gt_add =  data_root + "val/parsing_annos/"
    img = mmcv.bgr2rgb(mmcv.imread(img_name))
    for ins in seg_masks:
        for i in np.unique(ins):
            ind = ins==i
            if i ==0:
                continue
            else:
                try:
                    img[ind] = img[ind]*0.6 + (get_color(i,data_root))*255*0.4
                except Exception as e:
                    print(e)
                    import pdb;pdb.set_trace()
        #comb = (img*0.5*np.expand_dims(gt, axis=2)).astype('uint8')
    plt.figure(figsize=(16,10))
    plt.imshow(img)
    

def show_interact(i,p, gt_name, data_root):
    add =  data_root + "val/parsing_annos/"
    gt = np.array(Image.open(add+gt_name))
    if len(gt.shape)==3: gt = gt[:,:,0]
    print("gt pixels: ",((gt==p)>0).sum())
    ys,xs = np.where((gt==p)>0)
    print(xs.min(),ys.min())
    print(xs.max(),ys.max())

    seg_mask = seg_masks[i]
    print("pred pixels: ",((seg_mask==p)>0).sum())
    ys,xs = np.where((seg_mask==p)>0)
    print(xs.min(),ys.min())
    print(xs.max(),ys.max())

    part_gt = gt==p
    part_pred = seg_mask==p
    inter = part_gt*part_pred
    print("inter: ", (inter>0).sum())

    img_fusion = inter.astype("uint8")
    ind_inter = inter>0
    ind_gt = (gt==p)>0
    ind_pred = (seg_mask==p)>0
    img_fusion[ind_gt] = 2
    img_fusion[ind_pred] = 3
    img_fusion[ind_inter] = 1
    plt.figure(figsize=(8,5))
    plt.imshow(img_fusion)

def get_color(index, data_root):
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import scipy.io as scio
    path =  data_root + 'LV-MHP-v2_colormap.mat'
    colormap = scio.loadmat(path)
    return colormap['MHP_colormap'][index]