import cv2
import cv2.xfeatures2d as f2d
import numpy as np
import matplotlib.pyplot as PIL
from PIL import Image
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from utils import load_img, convert_eval, get_h, get_img_pad, get_warp_img, dice_score
from utils import test_size_w, test_size_h, num_grid

from utils import transform_fusion, model_RM, border_mask_erode, ransac_torch_sim



def stitch_CRP(temp_dir_1, temp_dir_2, pad_size, flag=True):
    temp_1_src = load_img(temp_dir_1)
    temp_2_src = load_img(temp_dir_2)
    temp_mask = cv2.imread('./border_mask.png', 0)
    h_ori, w_ori = temp_1_src.shape[:2]

    # temp_1_src = convert_downsample(temp_1_src, ratio=1/16)
    # temp_2_src = convert_downsample(temp_2_src, ratio=1/16)

    temp_1_src = convert_eval(cv2.resize(temp_1_src, (test_size_w, test_size_h)))
    temp_2_src = convert_eval(cv2.resize(temp_2_src, (test_size_w, test_size_h)))
    temp_mask = cv2.resize(temp_mask, (test_size_w, test_size_h))

    # temp_1 = cv2.cvtColor(temp_1_src, cv2.COLOR_BGR2GRAY)
    # temp_2 = cv2.cvtColor(temp_2_src, cv2.COLOR_BGR2GRAY)
    temp_1 = temp_1_src.copy()
    temp_2 = temp_2_src.copy()

    mask = cv2.threshold(temp_mask, 5, 255, cv2.THRESH_BINARY)[-1]
    if flag:
        
        if h_ori < 1000:
            mask_roi = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)),iterations=2)
        else:
            mask_roi = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55,55)), iterations=2)
        temp_1_src = cv2.bitwise_and(temp_1_src, temp_1_src, mask=mask)
        temp_2_src = cv2.bitwise_and(temp_2_src, temp_2_src, mask=mask)
    else:
        mask_roi = mask.copy()

    # if h_ori < 1000:
    #     sift = f2d.SURF_create(5, extended=True, upright=True)
    # else:
    #     sift = f2d.SURF_create(20, extended=True, upright=True)
    sift = f2d.SURF_create(50, extended=True, upright=True)
    # sift = f2d.SIFT_create(10000, 8)
    temp_p1, des_1 = sift.detectAndCompute(temp_1, mask_roi)
    # temp_out_1 = temp_1
    temp_p2, des_2 = sift.detectAndCompute(temp_2, mask_roi)
    # temp_out_2 = temp_2

    # temp_out_1 = cv2.drawKeypoints(temp_1, temp_p1, None, (0,0,255))

    # temp_out_2 = cv2.drawKeypoints(temp_2, temp_p2, None, (0,0,255))

    bf = cv2.BFMatcher_create(normType=cv2.NORM_L1, crossCheck=True)
    matches = bf.knnMatch(des_1, des_2, 1)

    matches_bf = []
    for match in matches:
        if len(match) > 0:
            matches_bf.append(match[0])
    good = cv2.xfeatures2d.matchGMS(temp_1.shape[:2], temp_2.shape[:2], temp_p1, temp_p2, matches1to2=matches_bf, withRotation=True, withScale=False, thresholdFactor=0)

    # img_out = cv2.drawMatches(temp_1, temp_p1, temp_2, temp_p2, good, None, matchColor=(0,255,0), singlePointColor=(0,0,255))
    # cv2.imshow('out', img_out)


    p1 = np.float32([temp_p1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    p2 = np.float32([temp_p2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    # h,_ = cv2.estimateAffinePartial2D(p2, p1, inliers=mask, method=cv2.RANSAC, ransacReprojThreshold=2.0, confidence=0.96)
    h_inv,_ = cv2.estimateAffinePartial2D(p1, p2, inliers=mask, method=cv2.RANSAC, ransacReprojThreshold=2.0, confidence=0.96)
    p1 += pad_size
    p2 += pad_size
    h_pad,_ = cv2.estimateAffinePartial2D(p2, p1, inliers=mask, method=cv2.RANSAC, ransacReprojThreshold=2.0, confidence=0.96)
    

    #### filter data
    _, inline_mask = cv2.estimateAffinePartial2D(p1, p2, inliers=mask, method=cv2.RANSAC, ransacReprojThreshold=2.0, confidence=0.96)
    inline_mask = inline_mask.reshape(-1).tolist()
    good_ransac = []
    for id, temp_inline in enumerate(inline_mask):
        if temp_inline > 0:
            good_ransac.append(good[id])
            

    good_filter = cv2.xfeatures2d.matchGMS(temp_1.shape[:2], temp_2.shape[:2], temp_p1, temp_p2, matches1to2=good_ransac, withRotation=False, withScale=False, thresholdFactor=0)



    return h_pad, h_inv, len(good_filter)

def get_p1s_with_mask(h, w, num_grid=36, mask=None):
    p1_list = []
    step_w = w//num_grid
    step_h = h//num_grid
    for i in range(step_w, w, step_w):
        for j in range(step_h, h, step_h):
            if len(p1_list)==1024:
                break
            if type(mask) != type(None):
                if i<num_grid*step_w and j<num_grid*step_h and mask[j, i]>0:
                    p1_list.append([i,j])
            else:
                if i<num_grid*step_w and j<num_grid*step_h:
                    p1_list.append([i,j])
    p1_np = np.float32(p1_list).reshape(-1,1,2)
    return p1_np

temp_p1 = get_p1s_with_mask(test_size_h, test_size_w, num_grid=num_grid, mask=border_mask_erode)
temp_p1 = torch.from_numpy(temp_p1).to(device)

def stitch_RM(img1_dir, img2_dir, pad_size):
    temp_1_src = load_img(img1_dir)
    temp_2_src = load_img(img2_dir)
    temp_1_cv = convert_eval(cv2.resize(temp_1_src, (test_size_w, test_size_h)))
    temp_2_cv = convert_eval(cv2.resize(temp_2_src, (test_size_w, test_size_h)))
    
    temp_1_pil = Image.fromarray(cv2.cvtColor(temp_1_cv, cv2.COLOR_BGR2RGB))
    temp_2_pil = Image.fromarray(cv2.cvtColor(temp_2_cv, cv2.COLOR_BGR2RGB))
    temp_1_tensor = torch.stack([transform_fusion(temp_1_pil)], 0).to(device)
    temp_2_tensor = torch.stack([transform_fusion(temp_2_pil)], 0).to(device)
    model_RM.eval()
    with torch.no_grad():
        pred_diff, _ = model_RM.predict_H(temp_1_tensor, temp_2_tensor)
        # _, H_nopad = model.predict_H(temp_2_tensor, temp_1_tensor)

        # pred_diff = pred_diff.view(-1, 1024, 2)
        # pred_diff = pred_diff.view(-1, 32, 32, 2)
        # pred_diff = pred_diff.permute(0,3,1,2)
        # pred_diff = torch.nn.functional.avg_pool2d(pred_diff, 3, 1, 1)

    pred_diff = pred_diff[0].view(-1,1,2)

    pred_diff_ori = pred_diff
    pred_diff_ori[:,:,:1] = pred_diff[:,:,:1]*test_size_w
    pred_diff_ori[:,:,1:] = pred_diff[:,:,1:]*test_size_h
    temp_p2 = (temp_p1 + pred_diff_ori)

    temp_p1_pad = temp_p1 + pad_size
    temp_p2_pad = temp_p2 + pad_size

    pred_H_inv,_ = ransac_torch_sim(temp_p1, temp_p2, 128)
    pred_H_pad,_ = ransac_torch_sim(temp_p2_pad, temp_p1_pad, 128)

    return pred_H_inv, pred_H_pad

def stitch_RM_Plus(img1_dir, img2_dir, pad_size):
    p1s_init = get_p1s_with_mask(test_size_h, test_size_w, num_grid=num_grid, mask=border_mask_erode)
    p1s_init = torch.from_numpy(p1s_init).to(device)
    ######## stage 1, get the init registration relationship with deep learning
    temp_1_src = cv2.resize(load_img(img1_dir), (test_size_w, test_size_h))
    temp_2_src = cv2.resize(load_img(img2_dir), (test_size_w, test_size_h))


    temp_1_cv = convert_eval(temp_1_src)
    temp_2_cv = convert_eval(temp_2_src)

    temp_1_pil = Image.fromarray(temp_1_cv)
    temp_2_pil = Image.fromarray(temp_2_cv)
    temp_1_tensor = torch.stack([transform_fusion(temp_1_pil)], 0).to(device)
    temp_2_tensor = torch.stack([transform_fusion(temp_2_pil)], 0).to(device)

    model_RM.eval()
    with torch.no_grad():
        pred_diff, _ = model_RM.predict_H(temp_1_tensor, temp_2_tensor)
    
    pred_diff = pred_diff[0].view(-1,1,2)

    pred_diff_ori = pred_diff
    pred_diff_ori[:,:,:1] = pred_diff[:,:,:1]*test_size_w
    pred_diff_ori[:,:,1:] = pred_diff[:,:,1:]*test_size_h
    p2s_preds_1 = p1s_init + pred_diff_ori


    p1s_init_pad = p1s_init + pad_size
    p2s_preds_1_pad = p2s_preds_1 + pad_size

    H_init_a2t_pad, _ = ransac_torch_sim(p2s_preds_1_pad, p1s_init_pad)
    
    mask_t = border_mask_erode.copy()
    mask_a = border_mask_erode.copy()
    mask_t_pad = get_img_pad(mask_t, pad_size)
    mask_a_pad = get_img_pad(mask_a, pad_size)
    img_1_pad = get_img_pad(temp_1_cv, pad_size)
    img_2_pad = get_img_pad(temp_2_cv, pad_size)
    mask_a2t_pad = get_warp_img(mask_a_pad, H_init_a2t_pad[:2,:])
    mask_roi_pad = cv2.bitwise_and(mask_t_pad, mask_t_pad, mask=mask_a2t_pad)
    img_a2t_pad = get_warp_img(img_2_pad, H_init_a2t_pad)


    
    # img_stage1 = cv2.addWeighted(img_1_pad, 0.5, img_a2t_pad, 0.5, 0)
    # cv2.imshow('stage1', img_stage1[600:-600,600:-600,:])


    ######## stage 2 local registration with surf with roi mask
    sf = f2d.SURF_create(50, extended=True, upright=True)
    kps_1, des_1 = sf.detectAndCompute(img_1_pad, mask_roi_pad)
    kps_2, des_2 = sf.detectAndCompute(img_a2t_pad, mask_roi_pad)
    bf = cv2.BFMatcher_create(normType=cv2.NORM_L1, crossCheck=True)
    matches = bf.match(des_1,des_2)
    matches = sorted(matches, key = lambda x:x.distance)
    matches_bf = []
    for match in matches:
        matches_bf.append(match)
        if len(matches_bf) > 512:
            break
    p1 = np.float32([kps_1[m.queryIdx].pt for m in matches_bf]).reshape(-1,1,2)
    p2 = np.float32([kps_2[m.trainIdx].pt for m in matches_bf]).reshape(-1,1,2)

    H_s2_a2t_pad = get_h(p2, p1)
    H_s2_t2a_pad = get_h(p1, p2)

    ### analysis the second prediction result #####

    mask_s2_a2t_pad = get_warp_img(mask_a2t_pad, H_s2_a2t_pad)
    temp_num_1 = np.sum(mask_a2t_pad)/255
    temp_num_2 = np.sum(mask_s2_a2t_pad)/255
    num_ratio_1 = float(temp_num_2)/float(temp_num_1)
    test_mask = np.zeros_like(mask_a2t_pad)
    # test_mask = cv2.circle(test_mask, (320,240), 300, 255, -1)
    pts = np.array([[640+pad_size,240+pad_size],[100+pad_size,10+pad_size],[10+pad_size,480+pad_size]], dtype=np.int32)
    test_mask = cv2.drawContours(test_mask, [pts], -1, 255, -1)
    test_mask_s2 = get_warp_img(test_mask, H_s2_a2t_pad)
    iou_ratio_1 = dice_score(test_mask, test_mask_s2)


    # img_s2_a2t = get_warp_img(img_a2t_pad, H_s2_a2t_pad)
    # img_stage2 = cv2.addWeighted(img_1_pad, 0.5, img_s2_a2t, 0.5, 0)
    # cv2.imshow('stage2', img_stage2[600:-600,600:-600,:])
    # print(num_ratio_1)
    # cv2.waitKey()


    ## if result is not suitable, just return the dl result
    if num_ratio_1 >= 1.2 or num_ratio_1 <= 0.8 or iou_ratio_1 <=0.25:
        pred_H_inv_init, _ = ransac_torch_sim(p1s_init, p2s_preds_1)
        return pred_H_inv_init, H_init_a2t_pad
    else:
        p1s_init_pad_np = p1s_init_pad.cpu().numpy()
        temp_pt_ex = np.ones_like(p1s_init_pad_np[:,:,:1])
        p1s_pad_ex = np.concatenate([p1s_init_pad_np, temp_pt_ex], axis=-1)
        temp_h = H_s2_t2a_pad.T
        p2s_temp = np.matmul(p1s_pad_ex, temp_h)
        p2s_pad_ex = np.concatenate([p2s_temp, temp_pt_ex], axis=-1)
        H_init_t2a_pad, _ = ransac_torch_sim(p1s_init_pad, p2s_preds_1_pad)
        temp_h = H_init_t2a_pad.T
        p2s_pad = np.matmul(p2s_pad_ex, temp_h)

        
        # temp_pt_ex = torch.ones_like(p1s_init_pad[:,:,1])
        # p1s_pad_ex = torch.cat([p1s_init_pad, temp_pt_ex], dim=-1)
        # temp_h = torch.from_numpy(H_s2_t2a_pad).to(device).T
        # p2s_temp = torch.matmul(p1s_pad_ex, temp_h)
        # p2s_pad_ex = torch.cat([p2s_temp, temp_pt_ex], dim=-1)
        # H_init_t2a_pad, _ = ransac_torch_sim(p1s_init_pad, p2s_preds_1_pad)
        # temp = torch.from_numpy(H_init_t2a_pad).to(device).T
        # p2s_pad = torch.matmul(p2s_pad_ex, temp_h).cpu().numpy()
        # p1s_init_pad_np = p1s_init_pad.cpu().numpy()

        
        pred_H_inv = get_h(p1s_init.cpu().numpy(), p2s_pad-pad_size)
        pred_H_pad = get_h(p2s_pad, p1s_init_pad_np)
        return pred_H_inv, pred_H_pad





