import cv2
import numpy as np
import cv2.xfeatures2d as f2d
# import cv2.Feature2D as f2d
# import math
import os
import json
from tqdm import tqdm
import p_tqdm


ex_size_LR = 0
ex_size_TB = 0
resize_w = 640
resize_h = 480
stitch_mask = cv2.imread('./border_mask.png', 0)
clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))

def clahe_transform(img):
    img_b, img_g, img_r = cv2.split(img)
    z = np.zeros_like(img_b)
    # img_b = clahe.apply(img_b)
    img_g = clahe.apply(img_g)
    img_r = clahe.apply(img_r)
    img_dst = cv2.merge([z, img_g, img_r])
    return img_dst

def stitch(temp_dir_1, temp_dir_2, pad_size):
    
    temp_1_src = cv2.imread(temp_dir_1)
    temp_2_src = cv2.imread(temp_dir_2)
    temp_mask = stitch_mask
    h_ori, w_ori = temp_1_src.shape[:2]
    # temp_1_src = cv2.copyMakeBorder(temp_1_src, ex_size_TB, ex_size_TB, ex_size_LR, ex_size_LR, cv2.BORDER_CONSTANT, None, value=(0,0,0))
    # temp_2_src = cv2.copyMakeBorder(temp_2_src, ex_size_TB, ex_size_TB, ex_size_LR, ex_size_LR, cv2.BORDER_CONSTANT, None, value=(0,0,0))
    # temp_mask = cv2.copyMakeBorder(temp_mask, ex_size_TB, ex_size_TB, ex_size_LR, ex_size_LR, cv2.BORDER_CONSTANT, None, value=(0,0,0))
    temp_1_src = cv2.resize(temp_1_src, (resize_w, resize_h))
    temp_2_src = cv2.resize(temp_2_src, (resize_w, resize_h))
    temp_mask = cv2.resize(temp_mask, (resize_w, resize_h))

    # temp_1 = cv2.cvtColor(temp_1_src, cv2.COLOR_BGR2GRAY)
    # temp_2 = cv2.cvtColor(temp_2_src, cv2.COLOR_BGR2GRAY)
    temp_1 = temp_1_src
    temp_2 = temp_2_src


    mask = cv2.threshold(temp_mask, 5, 255, cv2.THRESH_BINARY)[-1]
    if h_ori < 1000:
        mask_roi = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)),iterations=2)
    else:
        mask_roi = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55,55)), iterations=2)
    temp_1_src = cv2.bitwise_and(temp_1_src, temp_1_src, mask=mask)
    temp_2_src = cv2.bitwise_and(temp_2_src, temp_2_src, mask=mask)
    # temp_1 = clahe_transform(temp_1)
    # temp_2 = clahe_transform(temp_2)

    # temp_1 = cv2.blur(temp_1, (3,3))
    # temp_2 = cv2.blur(temp_2, (3,3))

    # cv2.imshow('clahe1', temp_1)
    # cv2.imshow('clahe2', temp_2)

    # sift = f2d.SURF_create(5, extended=True, upright=True)
    if h_ori < 1000:
        sift = f2d.SURF_create(5, extended=True, upright=True)
    else:
        sift = f2d.SURF_create(50, extended=True, upright=True)
    # sift = f2d.SIFT_create(1024,5)
    temp_p1, des_1 = sift.detectAndCompute(temp_1, mask_roi)
    temp_out_1 = temp_1
    temp_p2, des_2 = sift.detectAndCompute(temp_2, mask_roi)
    temp_out_2 = temp_2

    temp_out_1 = cv2.drawKeypoints(temp_1, temp_p1, None, (0,0,255))
    # cv2.imshow('keypoints1', temp_out_1)
    temp_out_2 = cv2.drawKeypoints(temp_2, temp_p2, None, (0,0,255))
    # cv2.imshow('keypoints2', temp_out_2)


    bf = cv2.BFMatcher_create(normType=cv2.NORM_L1, crossCheck=True)
    matches = bf.knnMatch(des_1, des_2, 1)

    matches_bf = []
    for match in matches:
        if len(match) > 0:
            matches_bf.append(match[0])
    good = cv2.xfeatures2d.matchGMS(temp_1.shape[:2], temp_2.shape[:2], temp_p1, temp_p2, matches1to2=matches_bf, withRotation=True, withScale=False, thresholdFactor=0)

    img_out = cv2.drawMatches(temp_1, temp_p1, temp_2, temp_p2, good, None, matchColor=(0,255,0), singlePointColor=(0,0,255))
    # cv2.imshow('out', img_out)


    p1 = np.float32([temp_p1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    p2 = np.float32([temp_p2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    
    p1 += pad_size
    p2 += pad_size

    h, m_points = cv2.estimateAffinePartial2D(p2, p1, inliers=mask, method=cv2.RANSAC, ransacReprojThreshold=2.0, confidence=0.96)

    m_points = m_points.reshape(1, -1)
    m_points = m_points.ravel().tolist()
    p1_sim = []
    p2_sim = []
    for i, temp_mask in enumerate(m_points):
        if temp_mask == 1:
            p1_sim.append(p1[i][0].tolist())
            p2_sim.append(p2[i][0].tolist())
    

    p1_sim = np.float32(p1_sim).reshape(-1, 1, 2)
    p2_sim = np.float32(p2_sim).reshape(-1, 1, 2)
    diff_sim = p2_sim-p1_sim
    img_out = cv2.drawMatches(temp_1, temp_p1, temp_2, temp_p2, good, None, matchColor=(0,255,0), singlePointColor=(0,0,255), matchesMask=m_points, flags=2)
    # cv2.imshow('out ransac', img_out)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    #### filter data
    _, inline_mask = cv2.estimateAffinePartial2D(p1, p2, inliers=mask, method=cv2.RANSAC, ransacReprojThreshold=2.0, confidence=0.96)
    inline_mask = inline_mask.reshape(-1).tolist()
    good_ransac = []
    for id, temp_inline in enumerate(inline_mask):
        if temp_inline > 0:
            good_ransac.append(good[id])
    good_filter = cv2.xfeatures2d.matchGMS(temp_1.shape[:2], temp_2.shape[:2], temp_p1, temp_p2, matches1to2=good_ransac, withRotation=False, withScale=False, thresholdFactor=0)


    return img_out, h, temp_1, temp_2, p1_sim, p2_sim, diff_sim, len(good_filter)


# import torch
# import torch.nn.functional as F
# from PIL import Image
# import torchvision.transforms as transforms

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

def H_torch2cv(temp_Hm, w, h):
    H = temp_Hm.copy()
    H[0,2] = H[0,2]*w
    H[1,2] = H[1,2]*h
    return H

def get_N(W, H):
    """N that maps from unnormalized to normalized coordinates"""
    N = np.zeros((3, 3), dtype=np.float64)
    N[0, 0] = 2.0 / W
    N[0, 1] = 0
    N[1, 1] = 2.0 / H
    N[1, 0] = 0
    N[0, -1] = -1.0
    N[1, -1] = -1.0
    N[-1, -1] = 1.0
    return N

def get_N_inv(W, H):
    """N that maps from normalized to unnormalized coordinates"""
    # TODO: do this analytically maybe?
    N = get_N(W, H)
    return np.linalg.inv(N)

def cvt_MToTheta(M, w, h):
    """convert affine warp matrix `M` compatible with `opencv.warpAffine` to `theta` matrix
    compatible with `torch.F.affine_grid`

    Parameters
    ----------
    M : np.ndarray
        affine warp matrix shaped [2, 3]
    w : int
        width of image
    h : int
        height of image

    Returns
    -------
    np.ndarray
        theta tensor for `torch.F.affine_grid`, shaped [2, 3]
    """
    M_aug = np.concatenate([M, np.zeros((1, 3))], axis=0)
    M_aug[-1, -1] = 1.0
    N = get_N(w, h)
    N_inv = get_N_inv(w, h)
    theta = N @ M_aug @ N_inv
    theta = np.linalg.inv(theta)
    return theta[:2, :]

def cvt_ThetaToM(theta, w, h, return_inv=False):
    """convert theta matrix compatible with `torch.F.affine_grid` to affine warp matrix `M`
    compatible with `opencv.warpAffine`.

    Note:
    M works with `opencv.warpAffine`.
    To transform a set of bounding box corner points using `opencv.perspectiveTransform`, M^-1 is required

    Parameters
    ----------
    theta : np.ndarray
        theta tensor for `torch.F.affine_grid`, shaped [2, 3]
    w : int
        width of image
    h : int
        height of image
    return_inv : False
        return M^-1 instead of M.

    Returns
    -------
    np.ndarray
        affine warp matrix `M` shaped [2, 3]
    """
    theta_aug = np.concatenate([theta, np.zeros((1, 3))], axis=0)
    theta_aug[-1, -1] = 1.0
    N = get_N(w, h)
    N_inv = get_N_inv(w, h)
    M = np.linalg.inv(theta_aug)
    M = N_inv @ M @ N
    if return_inv:
        M_inv = np.linalg.inv(M)
        return M_inv[:2, :]
    return M[:2, :]

import shutil
##########============================================================================================================================================
##########============================================================================================================================================
##########============================================================================================================================================

def extract_files_per_ID_img():
    data_root_dir = 'D:/proj/ROP/data_ROP/test_sim/'
    data_save_dir = 'D:/data_per_id_sim/'
    list_img_names = os.listdir(data_root_dir)

    # list_img_id = []
    # for temp_img_name in list_img_names:
    #     temp_img_id = temp_img_name.split('.')[0]
    #     list_img_id.append(temp_img_id)
    # list_img_id = set(list_img_id)

    for i,temp_img_name in enumerate(list_img_names):
        temp_img_id = temp_img_name.split('.')[0]
        temp_save_dir = data_save_dir + temp_img_id + '/'
        if os.path.exists(temp_save_dir) is False:
            os.makedirs(temp_save_dir)
        
        temp_dir_old = data_root_dir+temp_img_name
        temp_dir_new = temp_save_dir+temp_img_name[:-4] + '.jpg'

        temp_img = cv2.imread(temp_dir_old)
        cv2.imwrite(temp_dir_new, temp_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        # shutil.copy(temp_dir_old, temp_dir_new)
        print('process:{}/{}'.format(i+1, len(list_img_names)))
    print('test ok')

def generate_stitch_data_ml(flag_save_per_ID=True):

    data_root_dir = 'D:/data_per_id_sim/'
    list_img_groups = os.listdir(data_root_dir)[:]
    result_dir = 'D:/data_stitch_info_new/'
    if os.path.exists(result_dir) is False:
        os.makedirs(result_dir)
    
    def get_info_per_group(info_data):
        list_imgs_per_group, group_name, data_root_dir = info_data
        num_pair = 0
        list_imgs_a = list_imgs_per_group
        list_imgs_b = list_imgs_per_group
        # print('num of imgs in this group:', len(list_imgs_a))
        for i in range(len(list_imgs_a)):
            for j in range(len(list_imgs_b)):
                if i != j:
                    temp_1_dir = data_root_dir + group_name + '/' + list_imgs_a[i]
                    temp_2_dir = data_root_dir + group_name + '/' + list_imgs_b[j]
                    temp_ex = 0

                    try:
                        temp_out, H, temp_1, temp_2, p1_sim, p2_sim, diff_sim, num_filter \
                        = stitch(temp_1_dir, temp_2_dir, pad_size=temp_ex)
                        temp_info = {}
                        # temp_info['fusion_out'] = temp_out
                        temp_info['t_dir'] = temp_1_dir
                        temp_info['a_dir'] = temp_2_dir
                        temp_info['H_cv2'] = H.tolist()
                        temp_info['pts_sim_1'] = p1_sim.tolist()
                        temp_info['pts_sim_2'] = p2_sim.tolist()
                        temp_info['diff_sim'] = diff_sim.tolist()
                        temp_info['num_filter'] = num_filter


                        h, w = temp_1.shape[:2]
                        w_ex = w + temp_ex * 2
                        temp_1_ex = cv2.copyMakeBorder(temp_1, temp_ex,temp_ex,temp_ex,temp_ex, cv2.BORDER_CONSTANT, None, 0)
                        temp_2_ex = cv2.copyMakeBorder(temp_2, temp_ex,temp_ex,temp_ex,temp_ex, cv2.BORDER_CONSTANT, None, 0)
                        temp_2_ex_H = cv2.warpAffine(temp_2_ex, H, dsize=(w+temp_ex*2,h+temp_ex*2))

                        
                        # temp_2_ex_tensor = transforms.ToTensor()(Image.fromarray(cv2.cvtColor(temp_2_ex, cv2.COLOR_BGR2RGB)))
                        # temp_2_ex_tensor = torch.stack([temp_2_ex_tensor], 0)
                        # temp_Hm = H.copy()
                        # temp_Hm = H_cv2torch(temp_Hm, w_ex, w_ex)
                        # temp_Hm_H = cvt_ThetaToM(temp_Hm, w_ex, w_ex)


                        
                        # temp_Hm = temp_Hm.reshape(1,2,3)
                        # temp_Hm = torch.tensor(temp_Hm, dtype=torch.float)
                        # grid = F.affine_grid(temp_Hm, (1, 3, w_ex, w_ex), align_corners=True)
                        # output = F.grid_sample(temp_2_ex_tensor, grid, align_corners=True)
                        # out_show = output[0].numpy()
                        # out_show = np.uint8(np.transpose(out_show, (1,2,0))*255) 
                        # out_show = cv2.cvtColor(out_show, cv2.COLOR_RGB2BGR)

                        temp_fusion_cv = (temp_1_ex//2 + temp_2_ex_H//2)
                        # temp_fusion_torch = (temp_1_ex//2 + out_show//2)


                        # diff = temp_fusion_cv - temp_fusion_torch
                        temp_out_concat = np.hstack([temp_out, temp_fusion_cv])
                        # cv2.imshow('img_out_concat', temp_out_concat)
                        # cv2.imshow('diff', diff)
                        # cv2.waitKey(2)
                        num_pair += 1

                        
                        if flag_save_per_ID:
                            if os.path.exists(result_dir + group_name + '/') is False:
                                os.makedirs(result_dir + group_name + '/')
                            temp_save_dir = result_dir + group_name + '/' + group_name + '_' + str(i) + '_' + str(j) + '.json'
                            # cv2.imwrite(result_dir + group_name + '/' + group_name + '_' + str(i) + '_' + str(j) + '.jpg', temp_out_concat, [cv2.IMWRITE_JPEG_QUALITY, 100])
                        else:
                            temp_save_dir = result_dir + group_name + '_' + str(i) + '_' + str(j) + '.json'
                            # cv2.imwrite(result_dir + group_name + '_' + str(i) + '_' + str(j) + '.jpg', temp_out_concat, [cv2.IMWRITE_JPEG_QUALITY, 100])
                        with open(temp_save_dir,'w',encoding='utf-8') as json_file:
                            json.dump(temp_info, json_file, ensure_ascii=False)
                            json_file.close()
                        
                    except Exception as e:
                        # print(e)
                        continue
        return num_pair
    num_all_pair = 0
    list_info_data = []
    for i, temp_img_group in enumerate(tqdm(list_img_groups)):
        list_imgs_per_group = os.listdir(data_root_dir + temp_img_group + '/')
        list_info_data.append([list_imgs_per_group, temp_img_group, data_root_dir])
    
    iterator_ml =  p_tqdm.p_imap(get_info_per_group, list_info_data, num_cpus=16)
    for result in iterator_ml:
        num_all_pair += result
    print('all num of pairs of dataset', num_all_pair)

##########============================================================================================================================================
##########============================================================================================================================================

def analysis_data_quality_ml():
    data_dir = 'd:/data_stitch_info_new/'
    list_json_imgs = os.listdir(data_dir)
    list_jsons = []
    for temp_json_img in list_json_imgs:
        if temp_json_img[-4:] == 'json':
            list_jsons.append(temp_json_img)

    data_save_dir = 'd:/stitch_infos/'
    data_good_save_dir = data_save_dir + '/good/'
    data_bad_save_dir = data_save_dir + '/bad/'
    if os.path.exists(data_good_save_dir) is False:
        os.makedirs(data_good_save_dir)
    if os.path.exists(data_bad_save_dir) is False:
        os.makedirs(data_bad_save_dir)

    def get_info_json(temp_json):
        temp_json_dir = data_dir + temp_json
        temp_json_info = json.load(open(temp_json_dir, mode='r', encoding='utf-8'))
        num_pts = temp_json_info['num_filter']
        return num_pts, temp_json_dir, temp_json

    iterator_ml = p_tqdm.p_imap(get_info_json, list_jsons)
    
    import shutil
    # num_10 = 0
    # num_20 = 0
    # num_8 = 0
    for result in iterator_ml:
        num_pts, temp_json_dir, temp_json = result
        if num_pts > 50:
            temp_save_dir = data_good_save_dir.strip('/') + '_50/'
            if os.path.exists(temp_save_dir) is False:
                os.makedirs(temp_save_dir)
            shutil.copy(temp_json_dir, temp_save_dir+temp_json)
        if num_pts > 30:
            temp_save_dir = data_good_save_dir.strip('/') + '_30/'
            if os.path.exists(temp_save_dir) is False:
                os.makedirs(temp_save_dir)
            shutil.copy(temp_json_dir, temp_save_dir+temp_json)
        if num_pts > 20:
            temp_save_dir = data_good_save_dir.strip('/') + '_20/'
            if os.path.exists(temp_save_dir) is False:
                os.makedirs(temp_save_dir)
            shutil.copy(temp_json_dir, temp_save_dir+temp_json)
        if num_pts > 10:
            temp_save_dir = data_good_save_dir.strip('/') + '_10/'
            if os.path.exists(temp_save_dir) is False:
                os.makedirs(temp_save_dir)
            shutil.copy(temp_json_dir, temp_save_dir+temp_json)
        else:
            # shutil.copy(temp_json_dir, data_bad_save_dir+temp_json)
            pass
    #     if num_pts > 20:
    #         num_20 += 1
    #     if num_pts > 8:
    #         num_8 += 1
    # print('num > 8', num_8)
    # print('num > 10', num_10)
    # print('num > 20', num_20)
    # print('num all', len(list_jsons)) 
    print('test ')

def show(win_name, img):
    cv2.imshow(win_name, img)
    cv2.waitKey()

import base64
def img2base64(img):
    img = img.astype(np.uint8)
    img_b = cv2.imencode('.jpg', img)[1]
    img_base64 = str(base64.b64encode(img_b))[2:-1]
    return img_base64
def base642img(img_base64):
    image_data = base64.b64decode(img_base64)
    image_array = np.fromstring(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.COLOR_RGB2BGR)
    return image

def generate_dataset():
    infos_dir = 'D:/stitch_infos_640_480/good_30/'
    imgs_dir = 'D:/data_per_id_sim/'
    temp_save_dir = 'd:/dataset_30.npy'


    
    list_infos = []
    list_info_names = os.listdir(infos_dir)[:]
    list_img_dirs = []

    def get_infos(info_name):
        info_dir = infos_dir + info_name
        f = open(info_dir, mode='r', encoding='utf-8')
        temp_info = json.load(f)
        f.close()
        # temp_info_new = {}
        # temp_info_new['pts_sim_1'] = temp_info['pts_sim_1']
        # temp_info_new['pts_sim_2'] = temp_info['pts_sim_2']

        # temp_info_new['t'] = temp_info['t_dir'].split('/')[-1]
        # temp_info_new['a'] = temp_info['a_dir'].split('/')[-1]

        temp_info_new = [temp_info['pts_sim_1'], temp_info['pts_sim_2'], temp_info['t_dir'].split('/')[-1],temp_info['a_dir'].split('/')[-1]]
        return temp_info_new, temp_info['t_dir'], temp_info['a_dir']


    iterator_ml =  p_tqdm.p_imap(get_infos, list_info_names[:], num_cpus=16)
    for result in iterator_ml:
        temp_info_new, t_dir, a_dir = result
        list_infos.append(temp_info_new)
        list_img_dirs.append(t_dir)
        list_img_dirs.append(a_dir)
    

    temp_save_dir = 'd:/dataset_30_info.json'
    with open(temp_save_dir,'w',encoding='utf-8') as json_file:
        json.dump(list_infos, json_file, ensure_ascii=False)
        json_file.close()


    list_img_dirs_sim = set(list_img_dirs)
    image_pool = {}
    def get_imgs(dir):
        temp_img_str = img2base64(cv2.imread(dir))
        temp_img_name = dir.split('/')[-1]
        return temp_img_str, temp_img_name

    iterator_ml_img = p_tqdm.p_imap(get_imgs, list_img_dirs_sim, num_cpus=16)
    for result in iterator_ml_img:
        temp_img_str, temp_img_name = result
        image_pool[temp_img_name] = temp_img_str

    temp_save_dir = 'd:/dataset_30_images.json'
    with open(temp_save_dir,'w',encoding='utf-8') as json_file:
        json.dump(image_pool, json_file, ensure_ascii=False)
        json_file.close()


    # dataset_info = {}
    # # dataset_info['infos'] = list_infos
    # temp_save_dir = 'd:/dataset_30_info.npy'
    # np.save(temp_save_dir, list_infos)
    # del dataset_info
    # del list_infos

    # # dataset_info = {}
    # # # dataset_info['images'] = image_pool
    # # temp_save_dir = 'd:/dataset_30_images.npy'
    # # np.save(temp_save_dir, image_pool)

def load_dataset():
    temp_save_dir = 'd:/dataset_30_images.json'
    with open(temp_save_dir, mode='r', encoding='utf-8') as f:
        temp_info = json.load(f)
        f.close()
    print('test')

 # with open(temp_save_dir,'w',encoding='utf-8') as json_file:
    #     json.dump(dataset_info, json_file, ensure_ascii=False)
    #     json_file.close()

if __name__ == '__main__':
    ## 
    extract_files_per_ID_img()
    ##
    generate_stitch_data_ml(flag_save_per_ID=False)
    ##
    analysis_data_quality_ml()
    ##
    generate_dataset()
    ##
    load_dataset()
