import cv2
import numpy as np
import os
import json
import math
from tqdm import tqdm
import torch
import torchvision.transforms as transform 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
## image process size
test_size_w = 640
test_size_h = 480
pad_size_global = 400
clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))

## registration model init
num_grid=36
transform_fusion = transform.Compose([
            transform.ToTensor(),
        ])
from models import EfficientNet_cycle_ransac
model_RM = EfficientNet_cycle_ransac.from_name('efficientnet-b1', None, pad_size_global, 1024)
try:
    temp_ck = torch.load('D:/data_ROP/pretrain/model_l.pth', map_location='cpu')
    model_RM.load_state_dict(temp_ck, strict=False)
except Exception as e:
    print(e)
    pass
border_mask = cv2.imread('./border_mask.png', 0)
border_mask = cv2.resize(border_mask, (test_size_w, test_size_h))
temp_k = np.ones((14,14),np.uint8)
border_mask_erode = cv2.erode(border_mask, temp_k, iterations=1)
border_mask_erode[border_mask_erode<200] = 0
border_mask_erode[border_mask_erode>0] = 255


def get_img_pad(img, pad):
    img = cv2.resize(img, (test_size_w,test_size_h))
    img_pad = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, dst=None, value=0)
    return img_pad

def get_h(p1, p2):
    H, _ = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=2.0, confidence=0.96)
    # H, _ = cv2.findHomography(p1, p2, cv2.RANSAC,5.0)
    return H

def get_warp_img(img, H):
    t_h, t_w = img.shape[:2]
    img_warp = cv2.warpAffine(img, H, (t_w, t_h)).astype(np.uint8)
    return img_warp

def convert(img):
    b, g, r = cv2.split(img)
    # z = np.zeros_like(b)
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)

    img_new_rgb = cv2.merge([b, g, r])
    return img_new_rgb

def convert_eval(img):
    b, g, r = cv2.split(img)
    g = clahe.apply(g)
    img_new_rgb = cv2.merge([g,g,g])
    return img_new_rgb

def load_img(dir):
    img = cv2.imdecode(np.fromfile(dir, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img

def dice_score(pred, target):
    pred = pred.copy()
    target = target.copy()
    smooth = 1e-6
    pred[pred>0] = 1
    target[target>0] = 1
    intersection = np.sum(pred*target)
    scores = (2.0 * intersection + smooth)/(np.sum(pred)+np.sum(target)+smooth)
    return scores

## pytorch RANSAC

def estimate_H_combine(src, dst, num_combine_ids):
    M = torch.zeros((num_combine_ids, 3,3)).to(device)
    x1s,y1s = src[:,0,0],src[:,0,1]
    x2s,y2s = src[:,1,0],src[:,1,1]
    X1s,Y1s = dst[:,0,0],dst[:,0,1]
    X2s,Y2s = dst[:,1,0],dst[:,1,1]

    smooth = 1e-16
    dxs = x1s-x2s
    dys = y1s-y2s
    dXs = X1s-X2s
    dYs = Y1s-Y2s
    x1y2s = x1s*y2s
    x2y1s = x2s*y1s

    # if dxs==0 and dys==0:
    #     d = 1.0/(dxs*dxs+dys*dys+smooth)
    # else:
    #     d = 1.0/(dxs*dxs+dys*dys)
    d = 1.0/(dxs*dxs+dys*dys)
    S0 = d*(dXs*dxs+dYs*dys)
    S1 = d*(dYs*dxs-dXs*dys)
    S2 = d*( dYs*(x1y2s-x2y1s)-(X1s*y2s-X2s*y1s)*dys - (X1s*x2s-X2s*x1s)*dxs)
    S3 = d*(-dXs*(x1y2s-x2y1s)-(Y1s*x2s-Y2s*x1s)*dxs - (Y1s*y2s-Y2s*y1s)*dys)

    ## S0  -S1  S2 
    ## S1  S0   S3
    ## 0   0    1
    M[:,0,0] = S0
    M[:,0,1] = -S1
    M[:,0,2] = S2

    M[:,1,0] = S1
    M[:,1,1] = S0
    M[:,1,2] = S3

    M[:,2,2] += 1

    return M

def compute_err_combine(src, dst, H):
    src_ex = torch.cat([src, torch.ones_like(src[:,:1])], axis=-1).float()
    # src_ex = torch.unsqueeze(src_ex, dim=0)
    H_t = H[:,:2,:].permute(0,2,1)
    src_dst = torch.matmul(src_ex, H_t)
    err=torch.linalg.norm(src_dst-dst, axis=-1)

    # err = torch.sqrt(torch.sum())
    return err

def ransac_torch_sim(data_1, data_2, max_trials=512, err_threshold=5):
    data_1 = data_1.view(-1,2)
    data_2 = data_2.view(-1,2)
    err_threshold = math.sqrt(err_threshold)

    best_inliers = []

    num_samples = data_1.shape[0]
    if num_samples > 1000:
        num_samples = 1000
    if num_samples >= max_trials:
        spl_idxs = torch.multinomial(torch.ones(num_samples), max_trials, replacement=False)
    else:
        spl_idxs = torch.multinomial(torch.ones(num_samples), num_samples, replacement=False)
    # spl_idxs = torch.multinomial(torch.ones(num_samples), num_samples, replacement=False)

    combine_ids = torch.combinations(spl_idxs, 2)
    data_1_combine = data_1[combine_ids]
    data_2_combine = data_2[combine_ids]
    num_combine_ids = len(combine_ids)

    H_combine = estimate_H_combine(data_1_combine, data_2_combine, num_combine_ids)
    err_combine = compute_err_combine(data_1, data_2, H_combine)

    inliers_combine = err_combine < err_threshold
    inliers_count_combine = torch.count_nonzero(inliers_combine, dim=-1)
    errs_sum_combine = torch.sum(err_combine, dim=-1)

    best_id = torch.argmax(inliers_count_combine+(1-errs_sum_combine/torch.max(errs_sum_combine)), dim=-1)
    # best_id = torch.argmax(inliers_count_combine, dim=-1)

    best_H = H_combine[best_id]
    best_inliers = inliers_combine[best_id]
    return best_H.cpu().numpy()[:2,:], best_inliers



def get_CI_sim(list_scores):
    avg = np.mean(list_scores)
    std = np.std(list_scores)
    num = len(list_scores)
    sted = std/math.sqrt(num)
    temp_l = avg-1.96*sted
    temp_u = avg+1.96*sted
    return temp_l, temp_u, avg