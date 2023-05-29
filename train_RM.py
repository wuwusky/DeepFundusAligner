import os
import shutil
import sys
if len(sys.argv) > 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
    print('ai01 mul gpu')
else:
    # os.environ['CUDA_VISIBLE_DEVICES'] = '7,6,5,4'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print('cu single gpu')
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import sys

import gc

from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transform
# from model_new import *
from models import *

from PIL import Image, ImageFilter
import cv2
from tqdm import tqdm

import random
import platform
sysp = platform.system()


import warnings
warnings.filterwarnings('ignore')
flag_amp = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if len(sys.argv) > 1:
    device_ids = [0,1,2,3]
else:
    device_ids = [0]
# device_ids = [0,1]

import json

test_size_w = 640
test_size_h = 480
pad_size = 400
num_grid=36
grid_h = 36
grid_w = 36
max_pts = 1024


max_epoch = 13
num_train = 1000
batch_size = 12*len(device_ids)
learn_rate = 1e-4
weight_diff = 1.0
weight_value_diff = 10.0


model_save_dir = './pretrain_resnet50/'
temp_save_dir = model_save_dir


# temp_jsons_dir = 'stitch_infos_640_480/good_10/' #222w
# temp_jsons_dir = 'stitch_infos_640_480/good_20/' #134w
temp_jsons_dir = 'stitch_infos_640_480/good_30_sim/' #98w
# temp_jsons_dir = 'stitch_infos_640_480/good_50/' #61w







if os.path.exists(model_save_dir) is False:
    os.makedirs(model_save_dir)

if os.path.exists(temp_save_dir) is False:
    os.makedirs(temp_save_dir)

shutil.copy('./main_cycle_ransac.py', model_save_dir+'main.py')
shutil.copy('./model_eff.py', model_save_dir+'model.py')

def mask_bin(mask):
    mask[mask<100] = 0
    mask[mask > 0] = 1
    return mask.astype(np.int)

class randomGaussBlur(object):
    def __init__(self, radius=None, ratio=0.5):
        if radius is not None:
            self.radius = random.uniform(radius[0], radius[1])
        else:
            self.radius = 0.0
        self.ratio = ratio
    def __call__(self, img):
        if random.random() < self.ratio:
            return img.filter(ImageFilter.GaussianBlur(radius=self.radius))
        else:
            return img

def H_cv2torch(temp_Hm, w, h):
    temp_Hm_ex = np.concatenate([temp_Hm, np.array([[0,0,1]])], axis=0)
    temp_Hm_ex = np.linalg.inv(temp_Hm_ex)
    theta = np.zeros_like(temp_Hm)
    theta[0,0] = temp_Hm_ex[0,0]
    theta[0,1] = temp_Hm_ex[0,1]*h/w
    theta[0,2] = temp_Hm_ex[0,2]*2/w + temp_Hm_ex[0,0] + temp_Hm_ex[0,1] - 1
    theta[1,0] = temp_Hm_ex[1,0]*w/h
    theta[1,1] = temp_Hm_ex[1,1]
    theta[1,2] = temp_Hm_ex[1,2]*2/h + temp_Hm_ex[1,0] + temp_Hm_ex[1,1] - 1
    return theta


def H2diff_new(temp_Hm, h, w, grid_h, grid_w, mask=None): 
    p1_list = []
    step_w = w//grid_w
    step_h = h//grid_h
    for i in range(step_w, w, step_w):
        for j in range(step_h, h, step_h):
            if len(p1_list) == max_pts:
                break
            if type(mask) != type(None):
                if i<grid_w*step_w and j<grid_h*step_h and mask[j, i]>0:
                    p1_list.append([i, j, 1])
            else:
                if i<grid_w*step_w and j<grid_h*step_h:
                    p1_list.append([i, j, 1])
            
    p1_np = np.float32(p1_list).reshape(-1,1,3)
    pt_est = np.matmul(p1_np, temp_Hm.T)
    pt_diff = pt_est - p1_np[:,:,:2]
    pt_diff[:,:,:1] = pt_diff[:,:,:1]/w
    pt_diff[:,:,1:] = pt_diff[:,:,1:]/h
    pt_diff = pt_diff.reshape(-1)
    pt_diff[pt_diff<-1] = -1
    pt_diff[pt_diff>1] = 1
    return pt_diff, p1_np[:,:,:2], pt_est

def norm_one(a):
    a[a<0.5] = 0.0
    a[a>0.0] = 1.0
    return a

clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))

def convert(img, flag_train=False):
    # b, g, r = cv2.split(img)
    # z = np.zeros_like(b)
    # if flag_train:
    #     if random.random() < 0.5:
    #         g = clahe.apply(g)
    # img_new_rgb = cv2.merge([g, g, g])
    # return img_new_rgb
    return img

class randomCLAHE(object):
    def __init__(self, cliplimits=[2,8], gridsizes=[4,4], ratio=0.5):
        self.cliplimits = cliplimits
        self.gridsizes = gridsizes
        self.ratio = ratio
    
    def init_clahe(self):
        temp_cliplimit = random.uniform(self.cliplimits[0], self.cliplimits[1])
        temp_grid = random.randint(self.gridsizes[0], self.gridsizes[1])
        self.clahe=cv2.createCLAHE(temp_cliplimit, (temp_grid, temp_grid))

    def __call__(self, img):
        # print(img.shape)
        if random.random() < self.ratio:
            b, g, r = cv2.split(img)
            g = self.clahe.apply(g)
            img_new_rgb = cv2.merge([g, g, g])
            return img_new_rgb
        else:
            return img

def convert_random_clahe(img, cliplimits=[2,8], gridsizes=[4,12], ratio=0.5):
    # print(img.shape)
    if random.random() < ratio:
        temp_cliplimit = random.uniform(cliplimits[0], cliplimits[1])
        temp_grid = random.randint(gridsizes[0], gridsizes[1])
        clahe=cv2.createCLAHE(temp_cliplimit, (temp_grid, temp_grid))
        b, g, r = cv2.split(img)
        g = clahe.apply(g)
        img_dst = cv2.merge([g, g, g])
    else:
        img_dst = img
    # print(img.shape)
    # time.sleep(1)
    return img_dst


class Dataset_stitch_cycle(Dataset):
    def __init__(self, data_dir, list_data, status='train'):
        super(Dataset_stitch_cycle, self).__init__()
        self.data_dir = data_dir
        self.list_data = list_data
        self.transform_img = transform.Compose([
            transform.ColorJitter(brightness=0.5, contrast=0.5),
            randomGaussBlur(radius=[0,7], ratio=0.25),
            transform.ToTensor(),
            # transform.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ])
        self.transform_img_valid = transform.Compose([
            # transform.ColorJitter(brightness=0.25, contrast=0.5, saturation=0.0, hue=0.0),
            # randomGaussBlur(radius=[0,5]),
            # transform.Resize((output_size, output_size)),
            transform.ToTensor(),
            # transform.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ])
        self.transform_label = transform.Compose([
            # transform.Resize((output_size, output_size)),
            transform.ToTensor(),
        ])
        self.transform_fusion = transform.Compose([
            # transform.Resize((output_size, output_size)),
            transform.ToTensor(),
            # transform.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ])
        
        border_mask = cv2.imread('./border_mask.png', 0)
        self.border_mask = cv2.resize(border_mask, dsize=(test_size_w,test_size_h))
        temp_k = np.ones((14,14),np.uint8)
        self.border_mask = cv2.erode(self.border_mask, temp_k, iterations=1)
        self.border_mask[self.border_mask<200] = 0
        self.border_mask[self.border_mask>0] = 255

        self.status = status
        # self.random_clahe = randomCLAHE([2,6], [4,12], 0.5)

    def __getitem__(self, index):
        # self.random_clahe.init_clahe()

        temp_dir_info = self.data_dir + temp_jsons_dir + self.list_data[index]
        with open(temp_dir_info, mode='r', encoding='utf-8') as F:
            temp_info = json.load(F)
            temp_1_dir = temp_info['t_dir']
            temp_2_dir = temp_info['a_dir']
            temp_img_t_dir = self.data_dir + 'rop_data_images_all/' + temp_1_dir.split('/')[-1]
            temp_img_a_dir = self.data_dir + 'rop_data_images_all/' + temp_2_dir.split('/')[-1]


        # temp_img_s_dir, temp_img_t_dir, temp_info = self.data_mem[index]
        if self.status == 'train':
            temp_random_value = random.random()
            if temp_random_value<0.5:
                t_dir = temp_img_t_dir
                a_dir = temp_img_a_dir
                p1_sim = np.array(temp_info['pts_sim_1'])
                p2_sim = np.array(temp_info['pts_sim_2'])
            elif temp_random_value<0.65:
                t_dir = temp_img_t_dir
                a_dir = temp_img_t_dir
                p1_sim = np.array(temp_info['pts_sim_1'])
                p2_sim = np.array(temp_info['pts_sim_1'])
            else:
                t_dir = temp_img_a_dir
                a_dir = temp_img_t_dir
                p2_sim = np.array(temp_info['pts_sim_1'])
                p1_sim = np.array(temp_info['pts_sim_2'])

        temp_img_t_o = cv2.imread(t_dir)
        temp_img_a_o = cv2.imread(a_dir)
        
        img_t = cv2.resize(temp_img_t_o, dsize=(test_size_w, test_size_h))
        img_a = cv2.resize(temp_img_a_o, dsize=(test_size_w, test_size_h))

        img_t = convert(img_t, True)
        img_a = convert(img_a, True)
        # img_t = convert_random_clahe(img_t)
        # img_a = convert_random_clahe(img_a)

        # img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB)
        # img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)

        img_t_ex = cv2.copyMakeBorder(img_t, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, dst=None, value=0)
        img_a_ex = cv2.copyMakeBorder(img_a, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, dst=None, value=0)
        mask_t = self.border_mask
        mask_a = self.border_mask
        mask_t_ex = cv2.copyMakeBorder(mask_t, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, dst=None, value=0)
        mask_a_ex = cv2.copyMakeBorder(mask_a, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, dst=None, value=0)

        h, w = img_t.shape[:2]
        h_ex, w_ex = img_t_ex.shape[:2]
        num_p = p1_sim.shape[0]
        p1_sim_ex = p1_sim + pad_size
        p2_sim_ex = p2_sim + pad_size

        if self.status == 'train':
            if random.random() < 0.5:
                #### random data argument, (random rotate\scale\move)
                temp_rotate = random.randint(-45, 45)
                temp_dx = random.uniform(-0.25, 0.25)*(test_size_w)
                temp_dy = random.uniform(-0.25, 0.25)*(test_size_h)
                # temp_dx = random.randint(-test_size_w//4, test_size_w//4)
                # temp_dy = random.randint(-test_size_h//4, test_size_h//4)
                # temp_scale = random.uniform(0.95,1.0)
                temp_rotate_matirx = cv2.getRotationMatrix2D((w//2, h//2), temp_rotate, scale=1.0)
                temp_rotate_matirx[0,2] += temp_dx
                temp_rotate_matirx[1,2] += temp_dy
                p2_sim_new = np.concatenate([p2_sim, np.ones((num_p,1,1))], axis=2).reshape(num_p, 1, 3)
                p2_sim_random = np.matmul(p2_sim_new, temp_rotate_matirx.T)[:,:,:2]
                ## from p2(no pad) to p2_random(no pad)
                # H_2_random,_ = cv2.estimateAffinePartial2D(p2_sim, p2_sim_random, method=cv2.RANSAC, ransacReprojThreshold=2.0, confidence=0.96)
                ## from p2(with pad) to p2_random(with pad)
                H_2_random_ex,_ = cv2.estimateAffinePartial2D(p2_sim_ex, p2_sim_random+pad_size, method=cv2.RANSAC, ransacReprojThreshold=2.0, confidence=0.96)
                ## get the random moved image(no pad)
                img_a_random = cv2.warpAffine(img_a, temp_rotate_matirx, dsize=(w, h))
                ## get the random moved image(with pad)
                img_a_ex_random = cv2.warpAffine(img_a_ex, H_2_random_ex, dsize=(w_ex, h_ex))
                ## get the random moved mask(with pad)
                mask_a_ex_random = cv2.warpAffine(mask_a_ex, H_2_random_ex, dsize=(w_ex, h_ex))
                ## get the trans matirx from p2(random moved) to p1 (with pad)
                H_random_ex_a2t, _ = cv2.estimateAffinePartial2D(p2_sim_random+pad_size, p1_sim_ex, method=cv2.RANSAC, ransacReprojThreshold=2.0, confidence=0.96)
                
                H_a2t_ex = H_random_ex_a2t
                img_a_ex = img_a_ex_random
                img_a = img_a_random
                mask_a_ex = mask_a_ex_random
                p2_sim = p2_sim_random
                p2_sim_ex = p2_sim_random + pad_size
            else:
                H_a2t_ex, _ = cv2.estimateAffinePartial2D(p2_sim_ex, p1_sim_ex, method=cv2.RANSAC, ransacReprojThreshold=2.0, confidence=0.96)
        else:
            H_a2t_ex, _ = cv2.estimateAffinePartial2D(p2_sim_ex, p1_sim_ex, method=cv2.RANSAC, ransacReprojThreshold=2.0, confidence=0.96)

        ## from activate to target, transform matrix (pad)
        H_a2t_ex_torch_np = H_cv2torch(H_a2t_ex, w_ex, h_ex)
        ## from activate to target, transform matrix (without pad)
        H_a2t, _ = cv2.estimateAffinePartial2D(p2_sim, p1_sim, method=cv2.RANSAC, ransacReprojThreshold=2.0, confidence=0.96)
        H_a2t_torch_np = H_cv2torch(H_a2t, w, h)
        ## from target to activate, transform matrix inverse (without pad)
        H_t2a, _ = cv2.estimateAffinePartial2D(p1_sim, p2_sim, method=cv2.RANSAC, ransacReprojThreshold=2.0, confidence=0.96)
        H_t2a_torch_np = H_cv2torch(H_t2a, w, h)
        ## from target to activate, transform matrix inverse (pad)
        H_t2a_ex, _ = cv2.estimateAffinePartial2D(p1_sim_ex, p2_sim_ex, method=cv2.RANSAC, ransacReprojThreshold=2.0, confidence=0.96)
        H_t2a_ex_torch_np = H_cv2torch(H_t2a_ex, w_ex, h_ex)


        ## get the diff from activate to target (without pad)
        diff_center, pts_target, _ = H2diff_new(H_t2a, h, w, grid_h, grid_w, mask_t)
        ## get the diff from activate to target (pad)
        # diff_center_ex, _, _ = H2diff_new(H_t2a_ex, h_ex, w_ex, grid_h, grid_w, mask_t_ex)

        ## get the diff from target to activate (without pad)
        diff_center_inverse, _, _ = H2diff_new(H_a2t, h, w, grid_h, grid_w, mask_t)
        ## get the diff from target to activate (pad)
        # diff_center_ex_inverse, _, _ = H2diff_new(H_a2t_ex, h_ex, w_ex, grid_h, grid_w, mask_t_ex)
        
        
        mask_a_result = cv2.warpAffine(mask_a, H_a2t, (w, h))
        mask_t_result = cv2.warpAffine(mask_t, H_t2a, (w, h))
        mask_a_ex_result = cv2.warpAffine(mask_a_ex, H_a2t_ex, (w_ex, h_ex))
        mask_t_ex_result = cv2.warpAffine(mask_t_ex, H_t2a_ex, (w_ex, h_ex))



        img_t_roi = Image.fromarray(img_t)
        img_a_roi = Image.fromarray(img_a)
        
        
        mask_a_ex = Image.fromarray(mask_a_ex)
        mask_t_ex = Image.fromarray(mask_t_ex)

        mask_a_ex_result = Image.fromarray(mask_a_ex_result)
        mask_t_ex_result = Image.fromarray(mask_t_ex_result)
        mask_a_result = Image.fromarray(mask_a_result)
        mask_t_result = Image.fromarray(mask_t_result)

        

        if self.status == 'train':
            img_t_roi_tensor = self.transform_img(img_t_roi)
            img_a_roi_tensor = self.transform_img(img_a_roi)
        else:
            img_t_roi_tensor = self.transform_img_valid(img_t_roi)
            img_a_roi_tensor = self.transform_img_valid(img_a_roi)

        # mask_a_tensor = norm_one(self.transform_label(mask_a))
        # mask_t_tensor = norm_one(self.transform_label(mask_t))
        # mask_a_result_tensor = norm_one(self.transform_label(mask_a_result))
        # mask_t_result_tensor = norm_one(self.transform_label(mask_t_result))
        
        mask_a_ex_tensor = norm_one(self.transform_label(mask_a_ex))
        mask_t_ex_tensor = norm_one(self.transform_label(mask_t_ex))
        mask_a_ex_result_tensor = norm_one(self.transform_label(mask_a_ex_result))
        mask_t_ex_result_tensor = norm_one(self.transform_label(mask_t_ex_result))


        H_a2t_ex_tensor = torch.from_numpy(H_a2t_ex_torch_np).float()
        H_a2t_tensor = torch.from_numpy(H_a2t_torch_np).float()
        H_t2a_tensor = torch.from_numpy(H_t2a_torch_np).float()
        H_t2a_ex_tesnor = torch.from_numpy(H_t2a_ex_torch_np).float()

        diff_loc_tensor = torch.from_numpy(diff_center).float()
        diff_loc_inverse_tesnor = torch.from_numpy(diff_center_inverse).float()

        pts_tensor = torch.from_numpy(pts_target).float()



        return img_t_roi_tensor, img_a_roi_tensor, \
                mask_a_ex_tensor, mask_a_ex_result_tensor, mask_t_ex_tensor, mask_t_ex_result_tensor, \
                H_a2t_tensor, H_a2t_ex_tensor, H_t2a_tensor, H_t2a_ex_tesnor, \
                diff_loc_tensor, diff_loc_inverse_tesnor, \
                pts_tensor

    
    def __len__(self):
        return len(self.list_data)


import base64
def base642img(img_base64):
    image_data = base64.b64decode(img_base64)
    image_array = np.fromstring(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.COLOR_RGB2BGR)
    return image

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, input, target):
        # torch.flatten(start_dim=1)
        n, c, h, w = input.shape
        input_flat = input.clone().contiguous().flatten(start_dim=1).float()
        target_flat = target.clone().contiguous().flatten(start_dim=1).float()

        intersection = torch.sum(torch.mul(input_flat, target_flat), dim=1)
        scores = (2.0 * intersection + self.smooth)/(torch.sum(input_flat, dim=1)+torch.sum(target_flat, dim=1)+self.smooth)
        loss_d = 1.0 - scores

        if self.reduction == 'mean':
            loss_d = loss_d.mean()
        elif self.reduction == 'sum':
            loss_d = loss_d.sum()
        else:
            loss_d = loss_d
        
        return loss_d


def main_train():
    if sysp == 'Windows':
        num_worker = 0
        data_dir = 'D:/data_ROP/'
        data_valid_dir = 'D:/data_ROP/'
    else:
        num_worker = 6
        data_dir = '/data1/data_zh/image_stitch/'
        data_valid_dir_py = '/data1/data_zh/image_stitch/image_stitch_valid_panyu/'
        data_valid_dir_zoc = '/data1/data_zh/image_stitch/image_stitch_valid_zoc/'


    list_images_skip = os.listdir(data_dir+temp_jsons_dir)

    dataset_train = Dataset_stitch_cycle(data_dir, list_images_skip[:], 'train')

    border_mask_erode = dataset_train.border_mask
    transform_fusion = dataset_train.transform_fusion

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_worker, pin_memory=True, drop_last=True, prefetch_factor=2)
    model = EfficientNet_cycle_ransac.from_name('efficientnet-b1', None, pad_size, max_pts)
    model_valid = EfficientNet_cycle_ransac.from_name('efficientnet-b1', None, pad_size, max_pts)

    # model = resnet50(pad_size, max_pts)
    # model_valid = resnet50(pad_size, max_pts)

    try:
        # model.load_state_dict(torch.load('./pretrain_resnet50/model_temp_15.pth', map_location='cpu'), strict=True)
        pass
    except Exception as e:
        print(e)
    
    model = model.to(device)
    
    if len(sys.argv) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate, weight_decay=5e-4, amsgrad=True)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=1e-5, amsgrad=False)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate, weight_decay=5e-4, momentum=0.9)
    # lr_sch = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learn_rate, epochs=max_epoch, steps_per_epoch=len(loader_train))

    lr_sch = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.1)
    # lr_sch = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=2e-5, cycle_momentum=False, step_size_down=5000, step_size_up=5000)
    loss_fun_mse = nn.MSELoss()
    loss_fun_l1 = nn.L1Loss()
    loss_fun_bce = nn.BCELoss()
    loss_fun_dice = DiceLoss()
    loss_fun_l1_s = nn.SmoothL1Loss()
    loss_fun_l1_s_sum = nn.SmoothL1Loss(reduction='sum')
    # loss_fun_l1_s = nn.MSELoss()

    cur_epoch = 0
    cur_iter = 0
    
    best_dist = 500
    for epoch in range(cur_epoch, max_epoch):
        time_epoch_start = time.time()
        model.train()
        start_time_data = time.time()
        for i, data in enumerate(loader_train, start=cur_iter):
            cost_time_data = time.time() - start_time_data
            start_time = time.time()


            img_t_roi_tensor, img_a_roi_tensor, \
            mask_a_ex_tensor, mask_a_ex_result_tensor, \
            mask_t_ex_tensor, mask_t_ex_result_tensor, \
            H_a2t_tensor, H_a2t_ex_tensor, H_t2a_tensor, H_t2a_ex_tesnor, \
            diff_loc_tensor, diff_loc_inverse_tesnor, \
            pts_tensor\
                 = data

            
            
            img_t, img_a = img_t_roi_tensor.to(device), img_a_roi_tensor.to(device)
            m_a_ex, m_a_ex_dst = mask_a_ex_tensor.to(device), mask_a_ex_result_tensor.to(device)
            m_t_ex, m_t_ex_dst = mask_t_ex_tensor.to(device), mask_t_ex_result_tensor.to(device)

            H_a2t, H_a2t_ex = H_a2t_tensor.to(device), H_a2t_ex_tensor.to(device)
            H_t2a, H_t2a_ex = H_t2a_tensor.to(device), H_t2a_ex_tesnor.to(device)

            diff = diff_loc_tensor.to(device)
            diff_inv = diff_loc_inverse_tesnor.to(device)

            pts_t = pts_tensor.to(device)


            optimizer.zero_grad()

            if epoch+1 < max_epoch*10:
                out = model(img_t, img_a, m_a_ex, m_t_ex, pts_t, train_flag='w')
            else:
                out = model(img_t, img_a, m_a_ex, m_t_ex, pts_t, train_flag='l')
            pred_diff, pred_diff_inv, \
            H_theta_pad, H_theta_nopad, H_theta_pad_inv, H_theta_nopad_inv, \
            pred_m_a,  pred_m_t = out
            
            # loss_d = loss_fun_l1_s_sum(pred_diff*weight_value_diff, diff*weight_value_diff)*weight_diff/(batch_size*len(device_ids)*100)
            # loss_d_inv = loss_fun_l1_s_sum(pred_diff_inv*weight_value_diff, diff_inv*weight_value_diff)*weight_diff/(batch_size*len(device_ids)*100)

            loss_d = loss_fun_l1_s(pred_diff*weight_value_diff, diff*weight_value_diff)*weight_diff
            loss_d_inv = loss_fun_l1_s(pred_diff_inv*weight_value_diff, diff_inv*weight_value_diff)*weight_diff


            loss_h = loss_fun_l1_s(H_theta_nopad*weight_value_diff, H_a2t*weight_value_diff)*weight_diff
            loss_h_ex = loss_fun_l1_s(H_theta_pad*weight_value_diff, H_a2t_ex*weight_value_diff)*weight_diff
            loss_h_inv = loss_fun_l1_s(H_theta_nopad_inv*weight_value_diff, H_t2a*weight_value_diff)*weight_diff
            loss_h_ex_inv = loss_fun_l1_s(H_theta_pad_inv*weight_value_diff, H_t2a_ex*weight_value_diff)*weight_diff

            loss_ma = loss_fun_dice(pred_m_a, m_a_ex_dst)
            loss_mt = loss_fun_dice(pred_m_t, m_t_ex_dst)

            # temp_loss_t = loss_ma.clone().detach()
            # loss_scores_good = loss_fun_l1_s(pred_diff_score, torch.ones_like(pred_diff_score)-temp_loss_t)
            # loss_scores_pool = loss_fun_l1_s(pred_diff_fake_score, torch.zeros_like(pred_diff_score)+temp_loss_t)
            # loss_score = loss_scores_good + loss_scores_pool
            
            

            loss_diff = loss_d + loss_d_inv
            loss_t = loss_ma + loss_mt
            loss_h_all = loss_h + loss_h_ex + loss_h_inv + loss_h_ex_inv
            

            
            loss = loss_diff + loss_t + loss_h_all
            loss.backward()
            optimizer.step()
            

            cost_time = time.time() - start_time

            if i < 4 :
                print('Epoch:{}/{}, Iter:{}/{}, loss_diff: {:.4f}, loss_h: {:.4f}, loss_t: {:.4f}, loss_score: {:.4f}, lr:{:.8f}, cost_time_model:{:.2f}s, cost_time_data:{:.4f}s'
                .format(epoch+1, max_epoch, i+1, len(loader_train), loss_diff.item(), loss_h_all.item(), loss_t.item(), loss_t.item(), optimizer.param_groups[0]['lr'], cost_time, cost_time_data))
            else:
                print('Epoch:{}/{}, Iter:{}/{}, loss_diff: {:.4f}, loss_h: {:.4f}, loss_t: {:.4f}, loss_score: {:.4f}, lr:{:.8f}, cost_time_model:{:.2f}s, cost_time_data:{:.4f}s'
            .format(epoch+1, max_epoch, i+1, len(loader_train), loss_diff.item(), loss_h_all.item(), loss_t.item(), loss_t.item(), optimizer.param_groups[0]['lr'], cost_time, cost_time_data), end='\r')
            # if i % 2000 == 0:
            #     train_state_dict = {'model':model.module.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch, 'iter':i}
            #     torch.save(train_state_dict, model_save_dir + '/model_iter.pth')
            start_time_data = time.time()
        lr_sch.step()
        model.eval()
        try:
            torch.save(model.module.state_dict(), model_save_dir + '/model_temp_'+ str(epoch+1) +'.pth')
        except Exception as e:
            torch.save(model.state_dict(), model_save_dir + '/model_temp_'+ str(epoch+1) +'.pth')



if __name__ == '__main__':
    main_train()






