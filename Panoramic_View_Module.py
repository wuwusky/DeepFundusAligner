import json
import cv2
import os

import numpy as np
from tqdm import tqdm
seed = 1
import random
import torch
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


from utils import get_img_pad, get_warp_img, get_h
from Registration_Module import stitch_CRP, stitch_RM, stitch_RM_Plus
from utils import pad_size_global


def generate_avg_fundus():
    samples_dir = 'D:/data_ROP/ROP_cases_optic_mac_info/results_filter/1600-1200/OS/'
    infos_dir = 'D:/data_ROP/ROP_cases_optic_mac_info/infos/'
    list_sample_names = os.listdir(samples_dir)

    pix = samples_dir.split('/')[-3]
    loc = samples_dir.split('/')[-2]

    list_imgs = []
    for i, temp_sample_name in enumerate(tqdm(list_sample_names, ncols=100)):
        img_dir = samples_dir + temp_sample_name
        temp_img = cv2.imread(img_dir)
        temp_info = json.load(open(infos_dir + temp_sample_name[:-4]+'.json'))
        if i == 0:
            target_img = temp_img
            target_info = temp_info
            list_pts_target = sorted(target_info['pts_optic_mac'])
            list_imgs.append(target_img)
            continue
        else:
            act_img = temp_img
            act_info = temp_info
            act_pts = sorted(act_info['pts_optic_mac'])
            temp_h = get_h(np.array(act_pts), np.array(list_pts_target))
            temp_img_warp = get_warp_img(act_img, temp_h)
            list_imgs.append(temp_img_warp)
        
    
    avg_img = np.zeros_like(list_imgs[0], dtype=np.float)
    for temp_img in list_imgs:
        avg_img += temp_img.astype(np.float)
    avg_img /= len(list_imgs)
    cv2.imshow('avg funds', avg_img.astype(np.uint8))
    cv2.imwrite('./avg_funds_'+pix+'_'+ loc +'.png', avg_img.astype(np.uint8))
    cv2.waitKey()

def get_generation_panoramic(target_H, jsons_dir, show_pad, scale):
    list_json_names = os.listdir(jsons_dir)
    list_imgs = []
    for i, json_name in enumerate(list_json_names):
        with open(jsons_dir + json_name, mode='r', encoding='utf-8') as f:   
            temp_json_info = json.load(f)
        if i == 0:
            temp_t = cv2.imread(temp_json_info['dir_t'])
            temp_t = cv2.resize(temp_t, (640,480))
            temp_t_pad = get_img_pad(temp_t, show_pad)
            if type(target_H) == type(None):
                list_imgs.append(temp_t_pad)
            else:
                list_imgs.append(get_warp_img(temp_t_pad, target_H))
            
        temp_a = cv2.imread(temp_json_info['dir_a'])
        temp_a = cv2.resize(temp_a, (640,480))
        temp_a_pad = get_img_pad(temp_a, show_pad)

        try:
            p1 = temp_json_info['pts_sim_1']
            p2 = temp_json_info['pts_sim_2']
            p1_pad = np.array(p1) + show_pad
            p2_pad = np.array(p2) + show_pad
            h = get_h(p2_pad, p1_pad)
            temp_a_dst = get_warp_img(temp_a_pad, h)
            if type(target_H) == type(None):
                list_imgs.append(temp_a_dst)
            else:
                list_imgs.append(get_warp_img(temp_a_dst, target_H))
        except Exception as e:
            continue

    temp_current_img = np.zeros_like(list_imgs[0])
    for temp_img in list_imgs:
        temp_current_img = np.where(temp_current_img[:,:,:]>temp_img[:,:,:], temp_current_img, temp_img)
    return temp_current_img

def generate_avg_panoramic_fundus(show_pad):
    samples_dir = 'D:/data_ROP/ROP_cases_optic_mac_info/results_OD_OS/640-480/OS/'
    infos_dir = 'D:/data_ROP/ROP_cases_optic_mac_info/infos/'
    stitch_info_dir = 'D:/data_ROP/ROP_cases_stitch_info/'
    results_save_dir = 'D:/data_ROP/ROP_cases_optic_mac_info/'
    temp_location_lf = samples_dir.split('/')[-2]
    list_sample_names = os.listdir(samples_dir)[:]

    pix = samples_dir.split('/')[-3]
    loc = samples_dir.split('/')[-2]

    list_imgs = []
    for i, temp_sample_name in enumerate(tqdm(list_sample_names, ncols=100)):
        img_dir = samples_dir + temp_sample_name
        temp_img = cv2.imread(img_dir)
        temp_height, temp_width = temp_img.shape[:2]
        temp_scale = 640/temp_width
        temp_info = json.load(open(infos_dir + temp_sample_name[:-4]+'.json'))
        
        # act_img = temp_img
        if samples_dir.split('/')[-2] == 'OD':
            list_pts_target = [[465,240],[320,240]]
            act_info = temp_info
            act_pts = sorted(act_info['pts_optic_mac'], reverse=True)
        else:
            list_pts_target = [[175,240],[320,240]]
            act_info = temp_info
            act_pts = sorted(act_info['pts_optic_mac'], reverse=False)
        temp_h = get_h(np.array(act_pts)*temp_scale+show_pad, np.array(list_pts_target)+show_pad)
        # temp_img_warp = get_warp_img(act_img, temp_h)
        # cv2.imshow('temp_1', temp_img_warp)
        temp_json_dir = temp_sample_name.split('.')[0]
        try:
            temp_panoramic_img = get_generation_panoramic(temp_h, stitch_info_dir+temp_json_dir+'/'+temp_location_lf+'/', show_pad, temp_scale)
            list_imgs.append(temp_panoramic_img)
        except Exception as e:
            continue
        # cv2.imshow('temp_2', temp_panoramic_img)
        # print(temp_sample_name)
        temp_step_save_dir = './panoramic_results/atlas/'

        cv2.imwrite(temp_step_save_dir +temp_json_dir+ '_' + loc + '.png' , temp_panoramic_img)


class label_atlas(object):
    def __init__(self, img, mask, pts_mask):
        self.image_ori = img.copy()
        self.image_show = img.copy()
        cv2.namedWindow('label_image')
        cv2.setMouseCallback('label_image', self.label_callback)
        self.img = img
        self.list_pts = []
        self.save_flag = True
        self.label_flag = 'None'
        self.width = img.shape[1]//2

        self.mask = mask
        self.list_pts_mask = pts_mask

    
    def label_callback(self, event, x, y, flag, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(self.image_show, (x,y), 3, (0,255,0), -1)
            cv2.circle(self.image_show, (x,y), 145, (0,0,255), 2)
            cv2.imshow('label_image', self.image_show)
            self.list_pts.append([x,y])
        elif  event == cv2.EVENT_RBUTTONDOWN:
            cv2.circle(self.image_show, (x,y), 3, (0,255,255), -1)
            cv2.imshow('label_image', self.image_show)
            self.list_pts.append([x,y])

        elif event == cv2.EVENT_MBUTTONDBLCLK:
            self.image_show = self.image_ori.copy()
            cv2.imshow('label_image', self.image_show)
            self.list_pts = []

    def show_current_label(self, mask, img, list_pts, list_pts_mask):
        p1_sim = np.array(list_pts_mask)
        p2_sim = np.array(list_pts)
        t_h, t_w = mask.shape[:2]
        h_a2t, _ = cv2.estimateAffinePartial2D(p2_sim, p1_sim, method=cv2.RANSAC, ransacReprojThreshold=2.0, confidence=0.96)
        
        image_a2t = cv2.warpAffine(img, h_a2t, (t_w, t_h)).astype(np.uint8)
        image_fusion = (mask/2 + image_a2t/2).astype(np.uint8)
        # cv2.imshow('current label', image_fusion)
        show_img(image_fusion, 'current mapping', 0.5)
        self.h_t2atlas = h_a2t

    def show(self):
        while(1):
            cv2.imshow('label_image', self.image_show)
            # self.show_current_label(self.image_ori, self.label_json['pts_sim_1'].copy(), self.label_json['pts_sim_2'].copy())
            key = cv2.waitKeyEx()
            key = key % 255
            if key == 39:##right
                # self.label_flag = 'skip and next'
                # # print('next')
                # break
                pass
            elif key == 40:##down
                # self.label_flag = 'save and next'
                # # print('save')
                # break
                pass
            elif key == 38:##up
                self.label_flag = 'delete and next'
                # print('delete')
                break
                pass
            elif key == 37:##left
                self.label_flag == 'show label info'
                try:
                    self.show_current_label(self.mask, self.img, self.list_pts, self.list_pts_mask)
                except Exception as e:
                    print('标注配准点数量不够！！！')
            # elif key == 27:
            #     self.label_flag == 'exit'
            #     break
        cv2.destroyAllWindows()
        return self.label_flag

    def save_result(self, json_info, result_dir):
        if self.label_flag == 'skip and next':
            with open(result_dir,'w',encoding='utf-8') as json_file:
                json.dump(json_info, json_file, ensure_ascii=False)
        elif self.label_flag == 'save and next':
            json_info['pts_sim_1'] = self.list_pts_t
            json_info['pts_sim_2'] = self.list_pts_a
            with open(result_dir, 'w', encoding='utf-8') as json_file:
                json.dump(json_info, json_file, ensure_ascii=False)

def show_generation_atlas(jsons_dir, show_pad):
    list_json_names = os.listdir(jsons_dir)
    list_imgs = []
    flag_location = jsons_dir.split('/')[-2]
    if flag_location == 'OD':
        temp_mask_dir = './atlas/OD-atlas-clock.png'
        list_pts = [[465,240],[320,240]]
    else:
        temp_mask_dir = './atlas/OS-atlas-clock.png'
        list_pts = [[175,240],[320,240]]
    
    list_pts = np.array(list_pts) + show_pad
    list_pts = list_pts.tolist()

    for i, json_name in enumerate(list_json_names):
        with open(jsons_dir + json_name, mode='r', encoding='utf-8') as f:
            temp_json_info = json.load(f)
        if i == 0:
            temp_t = cv2.imread(temp_json_info['dir_t'])
            temp_t_pad = get_img_pad(temp_t, show_pad)
            list_imgs.append(temp_t_pad)
            temp_mask = cv2.imread(temp_mask_dir)


            atlas_detect = label_atlas(temp_t_pad, temp_mask, list_pts)
            atlas_detect.show()
            H_t2atlas = atlas_detect.h_t2atlas
        
        temp_a = cv2.imread(temp_json_info['dir_a'])
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
            # print(e)
            continue
            
    temp_current_img = np.zeros_like(list_imgs[0])
    for temp_img in list_imgs:
        temp_img = get_warp_img(temp_img, H_t2atlas)
        temp_current_img = np.where(temp_current_img[:,:,:]>temp_img[:,:,:], temp_current_img, temp_img)
        
    
    cv2.imshow('stitch result', temp_current_img)
    cv2.imwrite('stitch_result.png', temp_current_img)
    temp_result_with_mask = cv2.addWeighted(temp_current_img, 0.9, temp_mask, 0.1, 0)
    cv2.imshow('stitch result with atlas mask', temp_result_with_mask)
    cv2.waitKey()
    cv2.imwrite('temp_atlas.png', temp_result_with_mask)

def demo_generation_atlas(files_dir, show_pad, target_id):
    list_filenames = os.listdir(files_dir)
    list_imgs = []
    flag_location = files_dir.split('/')[-2]
    if 'OD' in flag_location:
        temp_mask_dir = './atlas/OD-atlas-clock.png'
        list_pts = [[465,240],[320,240]]
    else:
        temp_mask_dir = './atlas/OS-atlas-clock.png'
        list_pts = [[175,240],[320,240]]
    
    list_pts = np.array(list_pts) + show_pad
    list_pts = list_pts.tolist()

    temp_mask = cv2.imread(temp_mask_dir)

    for img_name in (list_filenames):
        if int(img_name.split('.')[-2]) == target_id:
            target_dir = files_dir + img_name
            temp_t = cv2.imread(target_dir)
            temp_t_pad = get_img_pad(temp_t, show_pad)
            list_imgs.append(temp_t_pad)

            atlas_detect = label_atlas(temp_t_pad, temp_mask, list_pts)
            atlas_detect.show()
            H_t2atlas = atlas_detect.h_t2atlas

    
    for img_name in tqdm(list_filenames, ncols=100):
        if int(img_name.split('.')[-2]) == target_id:
            continue
        activate_dir = files_dir + img_name
        temp_a = cv2.imread(activate_dir)
        temp_a_pad = get_img_pad(temp_a, show_pad)

        _, pred_h_pad = stitch_RM_Plus(target_dir, activate_dir, show_pad)
        temp_a_dst = get_warp_img(temp_a_pad, pred_h_pad)
        list_imgs.append(temp_a_dst)
    
    temp_path_list = files_dir.split('/')[:-2]
    save_dir = ''
    for temp_path in temp_path_list:
        save_dir += temp_path + '/'

    save_dir += '/results_' + flag_location + '/'


    # save_dir = 'D:/data_ROP/fig cases/' + files_dir.split('/')[-3] + '/results_' + flag_location + '/'
    os.makedirs(save_dir, exist_ok=True)

    temp_current_img = np.zeros_like(list_imgs[0])
    for i, temp_img in enumerate(list_imgs):
        temp_img = get_warp_img(temp_img, H_t2atlas)
        temp_img = cv2.normalize(temp_img, dst=None, alpha=220, beta=-20, norm_type=cv2.NORM_MINMAX)

        
        cv2.imwrite(save_dir + str(i)+'_map.png', temp_img)

        temp_current_img = np.where(temp_current_img[:,:,:]>temp_img[:,:,:], temp_current_img, temp_img)
        cv2.imwrite(save_dir + str(i)+'_pair.png', temp_current_img)
        cv2.imshow('temp result', temp_current_img)
        cv2.waitKey()
        
    
    
    cv2.imshow('stitch result', temp_current_img)
    cv2.imwrite(save_dir + 'panormaic.png', temp_current_img)
    # temp_file_name = files_dir.split('/')[-3] + '_' + flag_location + '.png'
    # cv2.imwrite(save_dir + temp_file_name, temp_current_img)


    # cv2.imshow('stitch result with atlas mask', temp_result_with_mask)
    # temp_file_name = files_dir.split('/')[-3] + '_' + flag_location + '_atlas.png'
    # cv2.imwrite(save_dir + temp_file_name, temp_result_with_mask)
    cv2.waitKey()



    # cv2.imshow('stitch result with atlas mask', temp_result_with_mask)
    # temp_file_name = files_dir.split('/')[-3] + '_' + flag_location + '_atlas.png'
    # cv2.imwrite(save_dir + temp_file_name, temp_result_with_mask)
    # cv2.waitKey()


def get_generation_panoramic_dl(target_H, jsons_dir, show_pad):
    list_json_names = os.listdir(jsons_dir)
    list_imgs = []
    list_imgnames = []
    list_h_a2t = []
    for i, json_name in enumerate(list_json_names):
        with open(jsons_dir + json_name, mode='r', encoding='utf-8') as f:   
            temp_json_info = json.load(f)
        if i == 0:
            temp_t = cv2.imread(temp_json_info['dir_t'])
            temp_t = cv2.resize(temp_t, (640,480))
            temp_t_pad = get_img_pad(temp_t, show_pad)
            if type(target_H) == type(None):
                list_imgs.append(temp_t_pad)
            else:
                list_imgs.append(get_warp_img(temp_t_pad, target_H))
            list_imgnames.append(temp_json_info['dir_t'])
            list_h_a2t.append(target_H)
            
        temp_a = cv2.imread(temp_json_info['dir_a'])
        temp_a = cv2.resize(temp_a, (640,480))
        temp_a_pad = get_img_pad(temp_a, show_pad)

        try:
            # p1 = temp_json_info['pts_sim_1']
            # p2 = temp_json_info['pts_sim_2']
            # p1_pad = np.array(p1) + show_pad
            # p2_pad = np.array(p2) + show_pad
            # h = get_h(p2_pad, p1_pad)
            dir_t = temp_json_info['dir_t']
            dir_a = temp_json_info['dir_a']
            _, h = stitch_RM_Plus(dir_t, dir_a, show_pad)

            temp_a_dst = get_warp_img(temp_a_pad, h)
            if type(target_H) == type(None):
                list_imgs.append(temp_a_dst)
            else:
                list_imgs.append(get_warp_img(temp_a_dst, target_H))
            list_imgnames.append(dir_a)
            list_h_a2t.append(h)
        except Exception as e:
            continue

    temp_current_img = np.zeros_like(list_imgs[0])
    for temp_img in list_imgs:
        temp_current_img = np.where(temp_current_img[:,:,:]>temp_img[:,:,:], temp_current_img, temp_img)
    return temp_current_img, list_imgs, list_imgnames, list_h_a2t


def get_img_name(root_dir, sample_name):
    list_filenames = os.listdir(root_dir)
    for temp_filename in list_filenames:
        if sample_name in temp_filename:
            return temp_filename
    return 'none'


def demo_generation_without_atlas(files_dir, show_pad, target_id):
    list_filenames = os.listdir(files_dir)
    list_imgs = []
    flag_location = files_dir.split('/')[-2]


    for img_name in (list_filenames):
        if int(img_name.split('.')[-2]) == target_id:
            target_dir = files_dir + img_name
            temp_t = cv2.imread(target_dir)
            temp_t_pad = get_img_pad(temp_t, show_pad)
            list_imgs.append(temp_t_pad)

    
    for img_name in tqdm(list_filenames, ncols=100):
        if int(img_name.split('.')[-2]) == target_id:
            continue
        activate_dir = files_dir + img_name
        temp_a = cv2.imread(activate_dir)
        temp_a_pad = get_img_pad(temp_a, show_pad)

        _, pred_h_pad = stitch_RM_Plus(target_dir, activate_dir, show_pad)
        temp_a_dst = get_warp_img(temp_a_pad, pred_h_pad)
        list_imgs.append(temp_a_dst)
        
    
    save_dir = 'D:/data_ROP/2cases/results_cases_without_atlas/'
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    
    temp_current_img = np.zeros_like(list_imgs[0])
    id = 0
    for temp_img in list_imgs:
        temp_current_img = np.where(temp_current_img[:,:,:]>temp_img[:,:,:], temp_current_img, temp_img)
        cv2.imshow('current result', temp_current_img)
        os.makedirs(save_dir + '/temp/', exist_ok=True)
        cv2.waitKey()
        cv2.imwrite(save_dir+'/temp/'+str(id)+'.png', temp_current_img)
        id += 1
        
    
    
    cv2.imshow('stitch result', temp_current_img)
    cv2.waitKey()
    temp_file_name = files_dir.split('/')[-3] + '_' + flag_location + '.png'
    cv2.imwrite(save_dir + temp_file_name, temp_current_img)


def show_img(img, img_name, scale=1.0):
    img_show = cv2.resize(img, None, fx=scale, fy=scale)
    cv2.imshow(img_name, img_show)


def demo_generation_with_center_json(jsons_dir, show_pad):
    list_filenames = os.listdir(jsons_dir)
    flag_location = jsons_dir.split('/')[-2]
    examination_id = jsons_dir.split('/')[-3]
    infos_dir = 'D:/data_ROP/ROP_cases_optic_mac_info/infos/'
    temp_save_dir = 'D:/proj_stitch/panoramic_results/results_center_mapping/'
    

    ### target image and target location
    temp_json_dir = jsons_dir + list_filenames[0]
    temp_json_info = json.load(open(temp_json_dir, mode='r', encoding='utf-8'))
    temp_img_t_dir = temp_json_info['dir_t']
    temp_img_target = cv2.imread(temp_img_t_dir)
    temp_width = temp_img_target.shape[1]
    temp_scale = 640/temp_width

    location_info_dir = infos_dir + temp_img_t_dir.split('/')[-1][:-4] + '.json'
    
    if flag_location=='OD':
        location_pts = sorted(json.load(open(location_info_dir, mode='r', encoding='utf-8'))['pts_optic_mac'], reverse=True)
        temp_dist = np.linalg.norm(np.array(location_pts[0])-np.array(location_pts[1]))*temp_scale
        list_pts_target = [[320+temp_dist,240],[320,240]]
    else:
        location_pts = sorted(json.load(open(location_info_dir, mode='r', encoding='utf-8'))['pts_optic_mac'], reverse=False)
        temp_dist = np.linalg.norm(np.array(location_pts[0])-np.array(location_pts[1]))*temp_scale
        list_pts_target = [[320-temp_dist,240],[320,240]]
    
    H_target2center = get_h(np.array(location_pts)*temp_scale+show_pad, np.array(list_pts_target)+show_pad)

    
    ### generate panoramic image by mapping center location
    temp_panoramic_img, list_imgs, list_imgnames, list_h_a2t = get_generation_panoramic_dl(H_target2center, jsons_dir, show_pad)
    center_target = np.zeros_like(temp_panoramic_img)
    center_target = cv2.circle(center_target, (320+show_pad, 240+show_pad), 30, (0,255,0), -1)

    
    # show_img(center_target, 'center target', 0.5)
    # show_img(temp_panoramic_img, 'panoramic view', 0.5)

    
    # cv2.waitKey()

    results_save_dir = temp_save_dir + examination_id + '/' + flag_location +'/'
    if os.path.exists(results_save_dir) is False:
        os.makedirs(results_save_dir)
    
    cv2.imwrite(results_save_dir + 'center_target.png', center_target)
    cv2.imwrite(results_save_dir + 'panoramic_view.png', temp_panoramic_img)
    for i in range(len(list_imgs)):
        cv2.imwrite(results_save_dir+'map_' + str(i) + '.png', list_imgs[i])
    

if __name__ == '__main__':
    # generate the average fundus image for atlas
    generate_avg_fundus()
    # generate the average panoramic fundus image
    generate_avg_panoramic_fundus(show_pad=pad_size_global)


    

    ### case 1:follow-up
    temp_dir = 'D:/data_ROP/2cases/case1/lv yiyan 18565186249 20211013/OD_sim/'
    target_id = 4

    demo_generation_atlas(temp_dir, pad_size_global, target_id)
    demo_generation_without_atlas(temp_dir, pad_size_global, target_id)














