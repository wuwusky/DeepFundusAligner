import os
import cv2
import json
import pandas
from tqdm import tqdm
import numpy as np

from utils import test_size_w, test_size_h, pad_size_global
from utils import border_mask_erode
from utils import get_CI_sim, dice_score, get_h, get_img_pad, get_warp_img
from Registration_Module import stitch_CRP, stitch_RM, stitch_RM_Plus



def pts2mask(p1_sim, p2_sim, border_mask, test_size_h, test_size_w, pad_size):
    mask_a = border_mask.copy()
    mask_a_ex = cv2.copyMakeBorder(mask_a, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, dst=None, value=0)
    h_ex, w_ex = test_size_h+2*pad_size, test_size_w+2*pad_size
    p1_sim_ex = p1_sim + pad_size
    p2_sim_ex = p2_sim + pad_size

    H_a2t_ex, _ = cv2.estimateAffinePartial2D(p2_sim_ex, p1_sim_ex, method=cv2.RANSAC, ransacReprojThreshold=2.0, confidence=0.96)
    mask_a_result = cv2.warpAffine(mask_a_ex, H_a2t_ex, (w_ex, h_ex))
    return mask_a_result

def H2mask(H, border_mask, test_size_h, test_size_w, pad_size):
    mask_a = border_mask.copy()
    mask_a_ex = cv2.copyMakeBorder(mask_a, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, dst=None, value=0)
    h_ex, w_ex = test_size_h+2*pad_size, test_size_w+2*pad_size
    mask_a_result = cv2.warpAffine(mask_a_ex, H, (w_ex, h_ex))
    return mask_a_result

def H2pts(temp_Hm, h, w, flag_init=False):
    p1 = [w/2, h/2, 1]


    p10 = [w/2, h*3/8, 1]
    p11 = [w*5/8, h/2, 1]
    p12 = [w/2, h*5/8, 1]
    p13 = [w*3/8, h/2, 1]

    p2 = [w/4, h/2, 1]
    p3 = [w*3/4, h/2, 1]
    p4 = [w/2, h/4, 1]
    p5 = [w/2, h*3/4, 1]

    p6 = [w/4, h/4, 1]
    p7 = [w*0.75, h*0.75, 1]
    p8 = [w*0.25, h*0.75, 1]
    p9 = [w*0.75, h*0.25, 1]

    

    # p1_list = [p1, p2, p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13]
    # p1_list = [p1, p10, p11, p12, p13]
    # p1_list = [p2, p3, p4, p5, p6, p7, p8, p9]
    p1_list = [p1, p2, p3, p4, p5, p6, p7, p8, p9]


    p1_np = np.float32(p1_list).reshape(-1,1,3)
    pt_est = np.matmul(p1_np, temp_Hm.T)

    if flag_init:
        return pt_est, p1_np
    else:
        return pt_est

def cal_pts_diff(pts_est, pts_label):
    temp_dis_mean = np.mean(np.linalg.norm(pts_est-pts_label, axis=-1))
    return temp_dis_mean

def metric_pts_diff(p2_sim, p1_sim, pred_H_inv, test_size_h, test_size_w):
    H_t2a_label, _ = cv2.estimateAffinePartial2D(p1_sim, p2_sim, method=cv2.RANSAC, ransacReprojThreshold=2.0, confidence=0.96)
    pts_label = H2pts(H_t2a_label, test_size_h, test_size_w)
    
    pts_est = H2pts(pred_H_inv, test_size_h, test_size_w)
    temp_distance = cal_pts_diff(pts_est, pts_label)
    return temp_distance

def metric_dice(p2_sim, p1_sim, pred_H, test_size_h, test_size_w, border_mask):
    temp_mask_label = pts2mask(p1_sim, p2_sim, border_mask, test_size_h, test_size_w, pad_size_global)
    temp_mask_pred = H2mask(pred_H, border_mask, test_size_h, test_size_w, pad_size_global)
    temp_score = dice_score(temp_mask_pred, temp_mask_label)
    return temp_score

def eval_metric_CRP_dice_dist(data='zoc',flag=True):
    if data == 'zoc':
        root_stitch_info_dir = 'E:/data_stitch/demo_test_info_zoc/'
    elif data == 'py':
        root_stitch_info_dir = 'E:/data_stitch/test_data_py/'
    else:
        root_stitch_info_dir = 'D:/data_ROP/ROP_cases_stitch_info/0cb7aecc-f5b6-481c-8701-de85c1a5c79b/OS/'

    temp_results_save_dir = './demo_data/results_CRP_'+ data + '_' + str(flag) +'.json'
    list_stitch_info_dir = os.listdir(root_stitch_info_dir)

    list_scores = []
    list_pts_dist = []
    list_rights = []
    list_infos_none = []
    list_pair_name = []
    list_num_robust = []
    for temp_info_name in tqdm(list_stitch_info_dir[:], ncols=100):
        try:
            temp_info_dir = root_stitch_info_dir + temp_info_name
            f = open(temp_info_dir, mode='r', encoding='utf-8')
            temp_info = json.load(f)
            f.close()

            border_mask = cv2.imread('./border_mask.png', 0)
            border_mask = cv2.resize(border_mask, dsize=(test_size_w,test_size_h))
            
            p1_sim = temp_info['pts_sim_1']
            p2_sim = temp_info['pts_sim_2']
            temp_len1 = len(p1_sim)
            temp_len2 = len(p2_sim)
            if temp_len1 < temp_len2:
                p2_sim = p2_sim[:temp_len1]
            else:
                p1_sim = p1_sim[:temp_len2]
            p1_sim = np.array(p1_sim)
            p2_sim = np.array(p2_sim)

            if temp_len1 == 0:
                list_infos_none.append(temp_info_dir)
                continue
            try:
                img1_dir = temp_info['s_dir']
                img2_dir = temp_info['t_dir']
            except Exception as e:
                img1_dir = temp_info['dir_t']
                img2_dir = temp_info['dir_a']
            pred_H_pad, pred_H_inv, num_robust = stitch_CRP(img1_dir, img2_dir, pad_size=pad_size_global, flag=flag)

            temp_score = metric_dice(p2_sim, p1_sim, pred_H_pad, test_size_h, test_size_w, border_mask_erode)
            temp_distance = metric_pts_diff(p2_sim, p1_sim, pred_H_inv, test_size_h, test_size_w)
            

            list_scores.append(temp_score)
            list_pts_dist.append(temp_distance)
            list_pair_name.append(temp_info_name)
            if num_robust > 30:
                list_num_robust.append(num_robust)
            else:
                list_num_robust.append(num_robust)
            if temp_distance < 50:
                list_rights.append(1)
            
        
        except Exception as e:
            # list_scores.append(0.0)
            # list_pts_dist.append(1000)
            # list_rights.append(0)
            continue
    
    print('total checked number pairs:', len(list_scores))
    print('None number pairs:         ', len(list_infos_none))

    temp_l, temp_u = get_CI_sim(list_scores)[:2]
    print('dice  score  avg: {:.4f}, 95% CI:[{:.4f}, {:.4f}]'.format(np.mean(list_scores), temp_l, temp_u))
    temp_l, temp_u = get_CI_sim(list_pts_dist)[:2]
    print('pts distance avg: {:.4f}, 95% CI:[{:.4f}, {:.4f}]'.format(np.mean(list_pts_dist), temp_l, temp_u))
    print('stitch acc:       {:.4f}'.format(np.sum(list_rights)/len(list_pts_dist)))

    p1 = pandas.Series(list_scores)
    p2 = pandas.Series(list_num_robust)
    coor = p1.corr(p2, 'pearson')
    print('R Corr: {:.4f}'.format(coor))

    score_info = {}
    score_info['dice_score'] = list_scores
    score_info['dist_score'] = list_pts_dist
    score_info['pair_name'] = list_pair_name
    score_info['num_robust'] = list_num_robust

    with open(temp_results_save_dir, mode='w', encoding='utf-8') as json_file:
        json.dump(score_info, json_file, ensure_ascii=False)
    
def eval_metric_RM_dice_dist(data='zoc', ref_flag=True):
    if data == 'zoc':
        root_stitch_info_dir = 'E:/data_stitch/demo_test_info_zoc/'
    elif data == 'py':
        root_stitch_info_dir = 'E:/data_stitch/test_data_py/'
    elif data == 'temp':
        root_stitch_info_dir = 'D:/data_ROP/ROP_cases_stitch_info/0cb7aecc-f5b6-481c-8701-de85c1a5c79b/OS/'
    else:
        return 0

    if ref_flag:
        temp_results_save_dir = './demo_data/results_RM_Plus_'+ data +'.json'
    else:
        temp_results_save_dir = './demo_data/results_RM_'+ data +'.json'
    list_stitch_info_dir = os.listdir(root_stitch_info_dir)

    
    
    list_scores = []
    list_pts_dist = []
    list_rights = []
    list_pair_name = []

    # temp_data_exp = {}
    # list_temp_datas = []
    # temp_save_dir = './temp_results_dl_stage2/'+ data + '/'
    # if os.path.exists(temp_save_dir) is False:
    #     os.makedirs(temp_save_dir)

    for temp_info_name in tqdm(list_stitch_info_dir[:], ncols=100):
        try:
            temp_info_dir = root_stitch_info_dir + temp_info_name
            f = open(temp_info_dir, mode='r', encoding='utf-8')
            temp_info = json.load(f)
            f.close()

            p1_sim = temp_info['pts_sim_1']
            p2_sim = temp_info['pts_sim_2']
            temp_len1 = len(p1_sim)
            temp_len2 = len(p2_sim)
            if temp_len1 < temp_len2:
                p2_sim = p2_sim[:temp_len1]
            else:
                p1_sim = p1_sim[:temp_len2]
            p1_sim = np.array(p1_sim)
            p2_sim = np.array(p2_sim)

            if temp_len1 == 0:
                continue
            try:
                img1_dir = temp_info['s_dir']
                img2_dir = temp_info['t_dir']
            except Exception as e:
                img1_dir = temp_info['dir_t']
                img2_dir = temp_info['dir_a']

            # pred_H_inv, pred_H_pad = stitch_dl_new_test(img1_dir, img2_dir, 0.75)

            if ref_flag:
                pred_H_inv, pred_H_pad = stitch_RM_Plus(img1_dir, img2_dir, pad_size_global)
            else:
                pred_H_inv, pred_H_pad = stitch_RM(img1_dir, img2_dir, pad_size_global)


            temp_score = metric_dice(p2_sim, p1_sim, pred_H_pad[:2,:], test_size_h, test_size_w, border_mask_erode)
            temp_distance = metric_pts_diff(p2_sim, p1_sim, pred_H_inv[:2,:], test_size_h, test_size_w)

            # temp_save_result_dir = temp_save_dir + temp_info_name[:-5] + '---' + str(int(temp_distance)) +'.png'
            # if temp_distance > 30:
            #     save_result(img1_dir, img2_dir, pred_H_pad, temp_save_result_dir)
            # if temp_distance > 100:
            #     shutil.copy(temp_info_dir, temp_save_dir+temp_info_name)

            list_scores.append(temp_score)
            list_pts_dist.append(temp_distance)
            list_pair_name.append(temp_info_name)
            if temp_distance < 50:
                list_rights.append(1)
        except Exception as e:
            print(e)
            continue
    
    print('total checked number pairs:', len(list_scores))
    temp_l, temp_u = get_CI_sim(list_scores)[:2]
    print('dice  score  avg: {:.4f}, 95% CI:[{:.4f}, {:.4f}]'.format(np.mean(list_scores), temp_l, temp_u))
    temp_l, temp_u = get_CI_sim(list_pts_dist)[:2]
    print('pts distance avg: {:.4f}, 95% CI:[{:.4f}, {:.4f}]'.format(np.mean(list_pts_dist), temp_l, temp_u))
    print('stitch acc:       {:.4f}'.format(np.sum(list_rights)/len(list_pts_dist)))

    score_info = {}
    score_info['dice_score'] = list_scores
    score_info['dist_score'] = list_pts_dist
    score_info['pair_name'] = list_pair_name

    with open(temp_results_save_dir, mode='w', encoding='utf-8') as json_file:
        json.dump(score_info, json_file, ensure_ascii=False)

##### ========================================================= panoramic view ==================================================================

def cal_dice_panoramic(p_dir, show_pad, flag_show=False):
    # p_dir = 'D:/data_ROP/ROP_cases_stitch_info/790326a0-35ba-4c30-85ba-137757020f1b/OD/'

    list_json_names = os.listdir(p_dir)
    list_imgs = []
    list_imgs_pred = []
    list_dists = []
    list_dices = []
    for i, json_name in enumerate(list_json_names):
        with open(p_dir + json_name, mode='r', encoding='utf-8') as f:   
            temp_json_info = json.load(f)
        if i == 0:
            temp_mask_t_pad = get_img_pad(border_mask_erode, show_pad)
            list_imgs.append(temp_mask_t_pad)
            list_imgs_pred.append(temp_mask_t_pad)
        temp_mask_a_pad = get_img_pad(border_mask_erode, show_pad)

        try:
            p1 = temp_json_info['pts_sim_1']
            p2 = temp_json_info['pts_sim_2']
            p1_pad = np.array(p1) + show_pad
            p2_pad = np.array(p2) + show_pad
            h = get_h(p2_pad, p1_pad)
            temp_a_dst = get_warp_img(temp_mask_a_pad, h)
            list_imgs.append(temp_a_dst)

            try:
                img_t_dir = temp_json_info['dir_t']
                img_a_dir = temp_json_info['dir_a']
            except Exception as e:
                img_t_dir = temp_json_info['s_dir']
                img_a_dir = temp_json_info['t_dir']
            h_pred, h_inv,_ = stitch_CRP(img_t_dir, img_a_dir, show_pad, True)
            temp_a_pred = get_warp_img(temp_mask_a_pad, h_pred)
            list_imgs_pred.append(temp_a_pred)

            temp_distance = metric_pts_diff(np.array(p2), np.array(p1), h_inv[:2,:], test_size_h, test_size_w)
            temp_dice = metric_dice(np.array(p2), np.array(p1), h_pred, test_size_h, test_size_w, border_mask_erode)
            list_dists.append(temp_distance)
            list_dices.append(temp_dice)


        except Exception as e:
            continue
    
        

    
    label = np.zeros_like(list_imgs[0])
    for temp_img in list_imgs:
        label = np.where(label[:,:]>temp_img[:,:], label, temp_img)
    
    preds = np.zeros_like(list_imgs_pred[0])
    for temp_img in list_imgs_pred:
        preds = np.where(preds[:,:]>temp_img[:,:], preds, temp_img)
    

    temp_dice = dice_score(preds, label)
    # print('surf dice:', temp_dice)


    

    if flag_show:
        cv2.imshow('label', label)
        cv2.imshow('stitch result', preds)
        
        cv2.waitKey()
    return temp_dice, np.mean(list_dists), list_dices, list_dists

def cal_dice_panoramic_dl(p_dir, show_pad, flag_show=False, ref_flag=False):
    # p_dir = 'D:/data_ROP/ROP_cases_stitch_info/790326a0-35ba-4c30-85ba-137757020f1b/OD/'
    list_json_names = os.listdir(p_dir)
    list_imgs = []
    list_imgs_pred = []

    list_masks = []
    list_masks_pred = []

    list_dists = []
    list_dices = []
    for i, json_name in enumerate(list_json_names):
        with open(p_dir + json_name, mode='r', encoding='utf-8') as f:   
            temp_json_info = json.load(f)
        
        try:
            img_t_dir = temp_json_info['dir_t']
            img_a_dir = temp_json_info['dir_a']
        except Exception as e:
            img_t_dir = temp_json_info['s_dir']
            img_a_dir = temp_json_info['t_dir']



        if i == 0:
            temp_mask_t_pad = get_img_pad(border_mask_erode, show_pad)
            list_masks.append(temp_mask_t_pad)
            list_masks_pred.append(temp_mask_t_pad)

            temp_img_t = cv2.imread(img_t_dir)
            temp_img_t_pad = get_img_pad(temp_img_t, show_pad)
            list_imgs_pred.append(temp_img_t_pad)

        temp_mask_a_pad = get_img_pad(border_mask_erode, show_pad)

        temp_img_a = cv2.imread(img_a_dir)
        temp_img_a_pad = get_img_pad(temp_img_a, show_pad)

        try:
            p1 = temp_json_info['pts_sim_1']
            p2 = temp_json_info['pts_sim_2']
            p1_pad = np.array(p1) + show_pad
            p2_pad = np.array(p2) + show_pad
            h = get_h(p2_pad, p1_pad)
            temp_a_dst = get_warp_img(temp_mask_a_pad, h)
            list_masks.append(temp_a_dst)


            if ref_flag:
                h_inv, pred_H_pad = stitch_RM_Plus(img_t_dir, img_a_dir, pad_size_global)
            else:
                h_inv, pred_H_pad = stitch_RM(img_t_dir, img_a_dir, pad_size_global)
            temp_mask_a_pred = get_warp_img(temp_mask_a_pad, pred_H_pad[:2,:])
            list_masks_pred.append(temp_mask_a_pred)

            temp_img_a_pred = get_warp_img(temp_img_a_pad, pred_H_pad[:2,:])
            list_imgs_pred.append(temp_img_a_pred)

            temp_distance = metric_pts_diff(np.array(p2), np.array(p1), h_inv[:2,:], test_size_h, test_size_w)
            temp_dice = metric_dice(np.array(p2), np.array(p1), pred_H_pad, test_size_h, test_size_w, border_mask_erode)
            list_dists.append(temp_distance)
            list_dices.append(temp_dice)


        except Exception as e:
            continue
    
        

    label = np.zeros_like(list_masks[0])
    for temp_img in list_masks:
        label = np.where(label[:,:]>temp_img[:,:], label, temp_img)
    
    preds = np.zeros_like(list_masks_pred[0])
    for temp_img in list_masks_pred:
        preds = np.where(preds[:,:]>temp_img[:,:], preds, temp_img)
    

    temp_dice = dice_score(preds, label)

    pred_img = list_imgs_pred[0]
    for temp_img in list_imgs_pred:
        pred_img = np.where(pred_img[:,:]>temp_img[:,:], pred_img, temp_img)
    


    # print('dl dice:', temp_dice)
        

    # cv2.imshow('stitch result', preds)
    if flag_show:
        temp_mask_gt_show = np.zeros_like(label)
        temp_mask_gt_show = cv2.merge([temp_mask_gt_show+255, temp_mask_gt_show+255, temp_mask_gt_show+255])
        temp_mask_gt_show = cv2.bitwise_and(temp_mask_gt_show, temp_mask_gt_show, mask=label)

        temp_mask_dl_show = np.zeros_like(preds)
        temp_mask_dl_show = cv2.merge([temp_mask_dl_show, temp_mask_dl_show, temp_mask_dl_show+255])
        temp_mask_dl_show = cv2.bitwise_and(temp_mask_dl_show, temp_mask_dl_show, mask=preds)

        temp_mask_fusion = cv2.addWeighted(temp_mask_gt_show, 1.0, temp_mask_dl_show, 1.0, 0)
        cv2.imshow('mask_fusion', temp_mask_fusion)

        image = show_generation_stitch(p_dir, show_pad)
        cv2.imshow('label', image)
        cv2.imshow('pred_img', pred_img)


        contours_pred = cv2.findContours(preds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1][0].reshape(-1,2)

        # for temp_pt in contours_pred:
        #     temp_x, temp_y = temp_pt[0], temp_pt[1]
        #     temp_mask_gt_show = cv2.circle(temp_mask_gt_show, (temp_x, temp_y), 2, (0,0,255), -1)
        temp_mask_gt_show = cv2.drawContours(temp_mask_gt_show, [contours_pred], -1, (0,0,255), 8, lineType=cv2.LINE_AA)
        # cv2.drawContours(original_contours_img, contours, -1, (0,0,255), 2, lineType=cv2.LINE_AA)
        
        cv2.imshow('prediction with gt', temp_mask_gt_show)



        # cv2.imshow('predict_dl', preds)
        cv2.imwrite('./panoramic_results/mask_fusion.png', temp_mask_fusion)
        cv2.imwrite('./panoramic_results/result_label.png', image)
        cv2.imwrite('./panoramic_results/result_pred.png', pred_img)
        cv2.imwrite('./panoramic_results/preds_with_gt.png', temp_mask_gt_show)
        cv2.waitKey()
    return temp_dice, np.mean(list_dists), list_dices, list_dists

def show_generation_stitch(jsons_dir, show_pad):
    list_json_names = os.listdir(jsons_dir)
    list_imgs = []
    for i, json_name in enumerate(list_json_names):
        with open(jsons_dir + json_name, mode='r', encoding='utf-8') as f:   
            temp_json_info = json.load(f)
        
        try:
            img_t_dir = temp_json_info['dir_t']
            img_a_dir = temp_json_info['dir_a']
        except Exception as e:
            img_t_dir = temp_json_info['s_dir']
            img_a_dir = temp_json_info['t_dir']


        if i == 0:
            temp_t = cv2.imread(img_t_dir)
            temp_t_pad = get_img_pad(temp_t, show_pad)
            list_imgs.append(temp_t_pad)
        temp_a = cv2.imread(img_a_dir)
        temp_a_pad = get_img_pad(temp_a, show_pad)

        try:
            p1 = temp_json_info['pts_sim_1']
            p2 = temp_json_info['pts_sim_2']
            p1_pad = np.array(p1) + show_pad
            p2_pad = np.array(p2) + show_pad
            h = get_h(p2_pad, p1_pad)
            temp_a_dst = get_warp_img(temp_a_pad, h)
            list_imgs.append(temp_a_dst)
        except Exception as e:
            continue

    temp_current_img = np.zeros_like(list_imgs[0])
    for temp_img in list_imgs:
        temp_current_img = np.where(temp_current_img[:,:,:]>temp_img[:,:,:], temp_current_img, temp_img)
        
    # cv2.imshow('stitch result', temp_current_img)
    # # cv2.waitKey()
    # cv2.imwrite('./result.png', temp_current_img)

    # # tp_opt = [1015,830]
    # # tp_mac = [890,840]
    # # R = int(math.sqrt((tp_opt[0]-tp_mac[0])**2+(tp_opt[1]-tp_mac[1])**2)) * 2
    # # temp_current_img = cv2.circle(temp_current_img, tuple(tp_opt), R, (100,100,100), 2)
    # # temp_current_img = cv2.circle(temp_current_img, tuple(tp_opt), R+200, (100,100,100), 2)

    # cv2.imshow('stitch result', temp_current_img)
    # cv2.waitKey()

    # cv2.destroyAllWindows()
    return temp_current_img

def eval_metric_dice_panoramic_view(data='zoc', flag='sf', show_pad=600, ref_flag=True):
    if data == 'zoc':
        root_stitch_info_dir = 'D:/data_ROP/ROP_cases_stitch_info_zoc_statistical/'
    elif data == 'py':
        root_stitch_info_dir = 'D:/data_ROP/ROP_cases_stitch_info_py_statistical/'
    else:
        root_stitch_info_dir = 'D:/data_ROP/ROP_cases_stitch_info/'

    list_sample_ids = os.listdir(root_stitch_info_dir)[:4]
    list_dices = []
    list_dists = []

    list_dices_exam = []
    list_dists_exam = []

    for temp_sample_id in tqdm(list_sample_ids, ncols=100):
        temp_sample_dir = root_stitch_info_dir + temp_sample_id
        temp_sample_dir_OD = temp_sample_dir + '/OD/'
        temp_sample_dir_OS = temp_sample_dir + '/OS/'
        if flag == 'sf':
            try:
                temp_dice_OD, temp_dist, temp_dices, temp_dists = cal_dice_panoramic(temp_sample_dir_OD, show_pad)
                # cal_dice_panoramic_show(temp_sample_dir_OD, show_pad, 'OD')
                list_dices.append(temp_dice_OD)
                list_dists.append(temp_dist)
                list_dices_exam.append(temp_dices)
                list_dists_exam.append(temp_dists)
            except Exception as e:
                # print(e)
                pass
            try:
                temp_dice_OS, temp_dist, temp_dices, temp_dists = cal_dice_panoramic(temp_sample_dir_OS, show_pad)
                # cal_dice_panoramic_show(temp_sample_dir_OS, show_pad, 'OS')
                list_dices.append(temp_dice_OS)
                list_dists.append(temp_dist)
                list_dices_exam.append(temp_dices)
                list_dists_exam.append(temp_dists)
            except Exception as e:
                # print(e)
                pass
        else:
            try:
                temp_dice_OD, temp_dist, temp_dices, temp_dists = cal_dice_panoramic_dl(temp_sample_dir_OD, show_pad, False, ref_flag)
                # cal_dice_panoramic_dl_show(temp_sample_dir_OD, show_pad, 'OD', ref_flag, data=data)
                list_dices.append(temp_dice_OD)
                list_dists.append(temp_dist)
                list_dices_exam.append(temp_dices)
                list_dists_exam.append(temp_dists)
            except Exception as e:
                # print(e)
                pass
            try:
                temp_dice_OS, temp_dist, temp_dices, temp_dists = cal_dice_panoramic_dl(temp_sample_dir_OS, show_pad, False, ref_flag)
                # cal_dice_panoramic_dl_show(temp_sample_dir_OS, show_pad, 'OS', ref_flag, data=data)
                list_dices.append(temp_dice_OS)
                list_dists.append(temp_dist)
                list_dices_exam.append(temp_dices)
                list_dists_exam.append(temp_dists)
            except Exception as e:
                # print(e)
                pass

    print('num of eyes:', len(list_dices))
    temp_l, temp_u = get_CI_sim(list_dices)[:2]
    print('dice  score  avg: {:.4f}, 95% CI:[{:.4f}, {:.4f}]'.format(np.mean(list_dices), temp_l, temp_u))
    temp_l, temp_u = get_CI_sim(list_dists)[:2]
    print('pts distance avg: {:.4f}, 95% CI:[{:.4f}, {:.4f}]'.format(np.mean(list_dists), temp_l, temp_u))


    score_info = {}
    score_info['dice_score'] = list_dices
    score_info['dist_score'] = list_dists
    score_info['dices'] = list_dices_exam
    score_info['dists'] = list_dists_exam

    if flag == 'sf':
        temp_results_save_dir = './demo_data/results_panoramic_' + flag + '_'+ data +'.json'
    elif flag == 'dl' and ref_flag:
        temp_results_save_dir = './demo_data/results_panoramic_ref_'+ data +'.json'
    else:
        temp_results_save_dir = './demo_data/results_panoramic_dl_'+ data +'.json'
        
    with open(temp_results_save_dir, mode='w', encoding='utf-8') as json_file:
        json.dump(score_info, json_file, ensure_ascii=False)


if __name__ == '__main__':
    # eval_metric_CRP_dice_dist('py')
    # eval_metric_RM_dice_dist('py', False)
    # eval_metric_RM_dice_dist('py', True)

    eval_metric_dice_panoramic_view('rop', 'dl', pad_size_global, True)
