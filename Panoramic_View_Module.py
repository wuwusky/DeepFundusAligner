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

def get_img_pad(img, pad):
    img = cv2.resize(img, (640,480))
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



from metric_new import stitch_dl_sf_new


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

        _, pred_h_pad = stitch_dl_sf_new(target_dir, activate_dir)
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


def demo_generation_atlas_json(jsons_dir, show_pad):
    print('start panoramic:', jsons_dir)
    temp_sample_name = jsons_dir.split('/')[-3]



    list_imgs = []
    flag_location = jsons_dir.split('/')[-2]
    if 'OD' in flag_location:
        temp_mask_dir = './atlas/OD_black_clock.bmp'
        list_pts = [[465,240],[320,240]]
    else:
        temp_mask_dir = './atlas/OS_black_clock.bmp'
        list_pts = [[175,240],[320,240]]
    
    list_pts = np.array(list_pts) + show_pad
    list_pts = list_pts.tolist()
    temp_mask = cv2.imread(temp_mask_dir)


    list_json_names = os.listdir(jsons_dir)
    list_imgs = []
    with open(jsons_dir + list_json_names[0], mode='r', encoding='utf-8') as f:   
            temp_json_info = json.load(f)

    temp_t = cv2.imread(temp_json_info['dir_t'])
    temp_t = cv2.resize(temp_t, (640,480))
    temp_t_pad = get_img_pad(temp_t, show_pad)
    list_imgs.append(temp_t_pad)


    temp_t_name = temp_json_info['dir_t'].split('/')[-1]
    temp_t_opt_mac_dir = r'D:\data_ROP\ROP_cases_optic_mac_info\results/' + temp_t_name
    temp_img_opt_mac = cv2.imread(temp_t_opt_mac_dir)
    try:
        show_img(temp_img_opt_mac, 'opt_mac_info', 0.5)
    except Exception as e:
        pass



    atlas_detect = label_atlas(temp_t_pad, temp_mask, list_pts)
    atlas_detect.show()
    H_t2atlas = atlas_detect.h_t2atlas

    temp_panoramic_img, list_imgs,_,_ = get_generation_panoramic_dl(H_t2atlas, stitch_info_dir+temp_sample_name+'/'+flag_location+'/', show_pad)
    temp_panoramic_img_gt, _ = get_generation_panoramic_gt(H_t2atlas, stitch_info_dir+temp_sample_name+'/'+flag_location+'/', show_pad)

    show_img(temp_panoramic_img,'pred',0.5)
    show_img(temp_panoramic_img_gt, 'gt', 0.5)
    cv2.waitKey()
    cv2.destroyAllWindows()


    temp_atlas_img = cv2.imread(temp_mask_dir)
    temp_panoramic_img_with_atlas = cv2.addWeighted(temp_panoramic_img, 0.7, temp_atlas_img, 0.7, 0)

    sample_save_dir = samples_save_dir + temp_sample_name + '/'+ flag_location +'/'
    if os.path.exists(sample_save_dir) is False:
        os.makedirs(sample_save_dir)
    cv2.imwrite(sample_save_dir+'0_panoramic.png', temp_panoramic_img)
    cv2.imwrite(sample_save_dir+'0_panoramic_atlas.png', temp_panoramic_img_with_atlas)
    for n, temp_patch_img in enumerate(list_imgs):
        cv2.imwrite(sample_save_dir + str(n+1) +'.png', temp_patch_img)
        temp_patch_img_atlas = cv2.addWeighted(temp_patch_img, 0.7, temp_atlas_img, 0.7, 0)
        cv2.imwrite(sample_save_dir + str(n+1) + '_atlas.png', temp_patch_img_atlas)


    
    
    cv2.imwrite(temp_save_dir+temp_sample_name+'_'+flag_location+'_pred.png', temp_panoramic_img)
    cv2.imwrite(temp_save_dir+temp_sample_name+'_'+flag_location+'_gt.png', temp_panoramic_img_gt)




    
        
    

    





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
            _, h = stitch_dl_sf_new(dir_t, dir_a)

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


def get_generation_panoramic_gt(target_H, jsons_dir, show_pad):
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
    return temp_current_img, list_imgs


def get_img_name(root_dir, sample_name):
    list_filenames = os.listdir(root_dir)
    for temp_filename in list_filenames:
        if sample_name in temp_filename:
            return temp_filename
    return 'none'

def generate_panoramic_fundus_with_atlas(show_pad):
    
    temp_save_dir = 'D:/data_ROP/ROP_cases_optic_mac_info/ROP_cases_104_source_new/temp_results/'
    if os.path.exists(temp_save_dir) is False:
        os.makedirs(temp_save_dir)

    list_sample_names = os.listdir(samples_dir)[:]

    list_names_error = []


    list_imgs = []
    for i, temp_sample_name in enumerate(tqdm(list_sample_names, ncols=100)):
        ## OD
        sample_img_dir = sample_images_dir +'/OD/'
        temp_img_dir = get_img_name(sample_img_dir, temp_sample_name)
        if temp_img_dir == 'none':
            # print('')
            # print(temp_sample_name+'/OD')
            list_names_error.append(temp_sample_name+'/OD')
        else:
            temp_img = cv2.imread(sample_img_dir + temp_img_dir)
            temp_width = temp_img.shape[1]
            temp_scale = 640/temp_width
            temp_info = json.load(open(infos_dir+temp_img_dir[:-4]+'.json'))
            list_pts_target = [[465,240],[320,240]]
            act_info = temp_info
            act_pts = sorted(act_info['pts_optic_mac'], reverse=True)
            H_t2atlas = get_h(np.array(act_pts)*temp_scale+show_pad, np.array(list_pts_target)+show_pad)
            temp_panoramic_img, list_imgs,_,_ = get_generation_panoramic_dl(H_t2atlas, stitch_info_dir+temp_sample_name+'/OD/', show_pad)
            temp_panoramic_img_gt, _ = get_generation_panoramic_gt(H_t2atlas, stitch_info_dir+temp_sample_name+'/OD/', show_pad)

            temp_mask_dir = './atlas/OD_black_clock.bmp'
            temp_atlas_img = cv2.imread(temp_mask_dir)
            temp_panoramic_img_with_atlas = cv2.addWeighted(temp_panoramic_img, 0.7, temp_atlas_img, 0.7, 0)

            sample_save_dir = samples_save_dir + temp_sample_name + '/OD/'
            if os.path.exists(sample_save_dir) is False:
                os.makedirs(sample_save_dir)
            cv2.imwrite(sample_save_dir+'0_panoramic.png', temp_panoramic_img)
            cv2.imwrite(sample_save_dir+'0_panoramic_atlas.png', temp_panoramic_img_with_atlas)
            for n, temp_patch_img in enumerate(list_imgs):
                cv2.imwrite(sample_save_dir + str(n+1) +'.png', temp_patch_img)
                temp_patch_img_atlas = cv2.addWeighted(temp_patch_img, 0.7, temp_atlas_img, 0.7, 0)
                cv2.imwrite(sample_save_dir + str(n+1) + '_atlas.png', temp_patch_img_atlas)


            cv2.imwrite(temp_save_dir+temp_sample_name+'_OD_pred.png', temp_panoramic_img)
            cv2.imwrite(temp_save_dir+temp_sample_name+'_OD_gt.png', temp_panoramic_img_gt)

        ##OS
        sample_img_dir = sample_images_dir +'/OS/'
        temp_img_dir = get_img_name(sample_img_dir, temp_sample_name)
        if temp_img_dir == 'none':
            # print('')
            # print(temp_sample_name+'/OS')
            list_names_error.append(temp_sample_name+'/OS')
        else:
            temp_img = cv2.imread(sample_img_dir + temp_img_dir)
            temp_width = temp_img.shape[1]
            temp_scale = 640/temp_width
            temp_info = json.load(open(infos_dir+temp_img_dir[:-4]+'.json'))
            list_pts_target = [[175,240],[320,240]]
            act_info = temp_info
            act_pts = sorted(act_info['pts_optic_mac'], reverse=False)
            H_t2atlas = get_h(np.array(act_pts)*temp_scale+show_pad, np.array(list_pts_target)+show_pad)
            temp_panoramic_img, list_imgs,_,_ = get_generation_panoramic_dl(H_t2atlas, stitch_info_dir+temp_sample_name+'/OS/', show_pad)
            temp_panoramic_img_gt, _ = get_generation_panoramic_gt(H_t2atlas, stitch_info_dir+temp_sample_name+'/OS/', show_pad)

            temp_mask_dir = './atlas/OS_black_clock.bmp'
            temp_atlas_img = cv2.imread(temp_mask_dir)
            temp_panoramic_img_with_atlas = cv2.addWeighted(temp_panoramic_img, 0.7, temp_atlas_img, 0.7, 0)

            sample_save_dir = samples_save_dir + temp_sample_name + '/OS/'
            if os.path.exists(sample_save_dir) is False:
                os.makedirs(sample_save_dir)
            cv2.imwrite(sample_save_dir+'0_panoramic.png', temp_panoramic_img)
            cv2.imwrite(sample_save_dir+'0_panoramic_atlas.png', temp_panoramic_img_with_atlas)
            for n, temp_patch_img in enumerate(list_imgs):
                cv2.imwrite(sample_save_dir + str(n+1) +'.png', temp_patch_img)
                temp_patch_img_atlas = cv2.addWeighted(temp_patch_img, 0.7, temp_atlas_img, 0.7, 0)
                cv2.imwrite(sample_save_dir + str(n+1) + '_atlas.png', temp_patch_img_atlas)
            

            cv2.imwrite(temp_save_dir+temp_sample_name+'_OS_pred.png', temp_panoramic_img)
            cv2.imwrite(temp_save_dir+temp_sample_name+'_OS_gt.png', temp_panoramic_img_gt)

    print(list_names_error)


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

        _, pred_h_pad = stitch_dl_sf_new(target_dir, activate_dir)
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
    

tags =  [
"视盘",
"黄斑",
"分界线",
"嵴",
"异常血管分支",
"血管扩张",
"新生血管",
"出血",
"渗出",
"不全视网膜脱离",
"全视网膜脱离",
"皱襞",
"新生血管+出血",
"无血管区"
]
classes = (
"ShiPan",
"HuangBan",
"FenJieXian",
"Ji",
"YiChangXueGuanFenZhi",
"XueGuanKuoZhang",
"XinShengXueGuan",
"ChuXue",
"ShenChu",
"BuQuanShiWangMoTuoLi",
"QuanShiWangMoTuoLi",
"ZhouBi",
"XinShengXueGuan+ChuXue",
"WuXueGuanQu"
)

dict_cls2id = {}
dict_id2cls = {}
for i in range(len(classes)):
    temp_cls = classes[i]
    dict_cls2id[temp_cls] = i
    dict_id2cls[str(i)] = temp_cls


def NMS(dets, thresh):
    #x1、y1、x2、y2、以及score赋值
    # （x1、y1）（x2、y2）为box的左上和右下角标
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]


    #每一个候选框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #order是按照score降序排序的，得到的是排序的本来的索引，不是排完序的原数组
    order = scores.argsort()[::-1]
    # ::-1表示逆序


    temp = []
    while order.size > 0:
        i = order[0]
        temp.append(i)
        #计算当前概率最大矩形框与其他矩形框的相交框的坐标
        # 由于numpy的broadcast机制，得到的是向量
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.minimum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.maximum(y2[i], y2[order[1:]])

        #计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，需要用0代替
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        #计算重叠度IoU
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        #找到重叠度不高于阈值的矩形框索引
        inds = np.where(ovr <= thresh)[0]
        #将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[inds + 1]
    return temp


def transform_pts(src, h):
    src_ex = np.concatenate([src, np.ones_like(src[:,:1])], axis=-1)
    dst = np.matmul(src_ex, h[:2,:].T)
    return dst[:,:2]


def demo_generation_center_align_json_with_lesion(jsons_dir, show_pad, nms=False):
    list_filenames = os.listdir(jsons_dir)
    flag_location = jsons_dir.split('/')[-2]
    examination_id = jsons_dir.split('/')[-3]
    infos_dir = 'D:/data_ROP/ROP_cases_optic_mac_info/infos/'
    temp_save_dir = 'D:/proj_stitch/panoramic_results/results_center_alignment_lesion/'
    infos_lesion_dir = 'D:/data_ROP/lesion_targets/'
    

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
    temp_panoramic_img_lesion = temp_panoramic_img.copy()


    ### screen lesion annotations and mapping lesion locations to panoramic view
    list_lesion_bboxs = []
    for i in range(len(dict_cls2id.keys())):
        list_lesion_bboxs.append([''])
    
    for i, (temp_img_name, temp_h_stitch) in enumerate(zip(list_imgnames, list_h_a2t)):
        temp_lesion_name = temp_img_name.split('/')[-1][:-4]
        if flag_location == 'OD':
            temp_lesion_name = 'R_' + temp_lesion_name + '.json'
        else:
            temp_lesion_name = 'L_' + temp_lesion_name + '.json'

        temp_lesion_dir = infos_lesion_dir + temp_lesion_name
        if os.path.exists(temp_lesion_dir):
            temp_info = json.load(open(temp_lesion_dir, mode='r', encoding='utf-8'))['info']

            if i==0:
                temp_h_target = temp_h_stitch
            for temp_bbox_info in temp_info:
                temp_x = temp_bbox_info['boundingBox']['left']
                temp_y = temp_bbox_info['boundingBox']['top']
                temp_w = temp_bbox_info['boundingBox']['width']
                temp_h = temp_bbox_info['boundingBox']['height']
                temp_label = dict_cls2id[temp_bbox_info['tags']]
                temp_x1, temp_y1 = temp_x, temp_y
                temp_x2, temp_y2 = temp_x1+temp_w, temp_y1
                temp_x3, temp_y3 = temp_x1, temp_y1+temp_h
                temp_x4, temp_y4 = temp_x1+temp_w, temp_y1+temp_h
                
                pts = np.array([[temp_x1, temp_y1],[temp_x2, temp_y2],[temp_x3,temp_y3],[temp_x4,temp_y4]])+show_pad
                if i==0:
                    pts_target = transform_pts(pts, temp_h_target)
                else:
                    pts_stitch = transform_pts(pts, temp_h_stitch)
                    pts_target = transform_pts(pts_stitch, temp_h_target)
                temp_bbox = pts_target.reshape(-1).tolist()+[temp_label]

                list_lesion_bboxs[temp_label].append(temp_bbox)


    # if len(list_lesion_bboxs[3])>1 and len(list_lesion_bboxs[2])>1:
    #     list_lesion_bboxs[2] = ['']


    list_lesion_maps = []
    for temp_lesion_bbox_cls in list_lesion_bboxs:
        list_bboxs = []
        for temp_lesion_bbox in temp_lesion_bbox_cls:
            if temp_lesion_bbox != '':
                pts = np.array(temp_lesion_bbox[:-1]).reshape(-1,1,2).astype(np.int32)
                temp_label = temp_lesion_bbox[-1]
                x,y,w,h = cv2.boundingRect(pts)
                list_bboxs.append([x,y,x+w,y+h,temp_label])

        if len(list_bboxs)<1:
            temp_map = np.zeros_like(temp_panoramic_img_lesion[:,:,0])
            list_lesion_maps.append(temp_map)
            continue
            
        else:
            if nms:
                indexs= NMS(np.array(list_bboxs), 0.15)
                for index in indexs:
                    lesion_bbox = list_bboxs[index]
                    x1,y1,x2,y2,label= lesion_bbox
                    temp_panoramic_img_lesion = cv2.rectangle(temp_panoramic_img_lesion, (x1,y1), (x2, y2), (0,0,255), 2)

                    temp_map = np.zeros_like(temp_panoramic_img_lesion[:,:,0])
                    temp_map = cv2.rectangle(temp_map, (x1,y1), (x2,y2), 255, -1)
                    list_lesion_maps.append(temp_map)
            else:
                temp_map = np.zeros_like(temp_panoramic_img_lesion[:,:,0])
                for lesion_bbox in list_bboxs:
                    x1,y1,x2,y2,label= lesion_bbox
                    temp_panoramic_img_lesion = cv2.rectangle(temp_panoramic_img_lesion, (x1,y1), (x2, y2), (0,0,255), 2)
                    temp_map = cv2.rectangle(temp_map, (x1,y1), (x2,y2), 255, -1)
                list_lesion_maps.append(temp_map)

            # temp_panoramic_img = cv2.drawContours(temp_panoramic_img, [pts], -1, (0,0,255), 5)

    # show_img(center_target, 'center target', 0.5)
    # show_img(temp_panoramic_img, 'panoramic view', 0.5)
    show_img(temp_panoramic_img_lesion, 'panoramic view with lesion', 0.5)

    
    cv2.waitKey()

    results_save_dir = temp_save_dir + examination_id + '/' + flag_location +'/'
    if os.path.exists(results_save_dir) is False:
        os.makedirs(results_save_dir)
    
    cv2.imwrite(results_save_dir + 'center_target.png', center_target)
    cv2.imwrite(results_save_dir + 'panoramic_view.png', temp_panoramic_img)
    cv2.imwrite(results_save_dir + 'panoramic_view_lesion.png', temp_panoramic_img_lesion)
    for i in range(len(list_imgs)):
        cv2.imwrite(results_save_dir+'map_' + str(i) + '.png', list_imgs[i])
    for i in range(len(list_lesion_maps)):
        cv2.imwrite(results_save_dir+'lesion_map_'+str(i)+'.png', list_lesion_maps[i])
    
    return temp_panoramic_img, temp_panoramic_img_lesion, np.stack(list_lesion_maps, axis=0)/255

def norm_one(img):
    temp_max = img.max()
    temp_min = img.min()
    img  = (img-temp_min)/(temp_max-temp_min)
    return img


if __name__ == '__main__':
    ## generate the average fundus image for atlas
    # generate_avg_fundus()
    ## generate the average panoramic fundus image
    # generate_avg_panoramic_fundus(600)

    # stitch_info_dir = 'D:/data_ROP/ROP_cases_stitch_info/'
    # samples_dir = 'D:/data_ROP/ROP_cases_optic_mac_info/ROP_cases_104_source_new/1/'
    # sample_images_dir = 'D:/data_ROP/ROP_cases_optic_mac_info/results_OD_OS/all/'
    # infos_dir = 'D:/data_ROP/ROP_cases_optic_mac_info/infos/'
    # samples_save_dir = samples_dir[:-1] + '_atlas/'
    # temp_save_dir = 'D:/data_ROP/ROP_cases_optic_mac_info/ROP_cases_104_source_new/temp_results/'
    # if os.path.exists(temp_save_dir) is False:
    #     os.makedirs(temp_save_dir)
    # ## generate the panoramic fundus image with atlas
    # # generate_panoramic_fundus_with_atlas(600)

    # ### clinical data(no optic mac annotation info)
    # temp_list = os.listdir(samples_dir)
    # temp_list_t = os.listdir(samples_save_dir)


    # for temp_ss in temp_list:
    #     if temp_ss in temp_list_t:
    #         temp_dir = samples_save_dir + temp_ss + '/OD/'
    #         if os.path.exists(temp_dir):
    #             pass
    #         else:
    #             try:
    #                 temp_dir = 'D:/data_ROP/ROP_cases_stitch_info/'+temp_ss+'/OD/'    
    #                 demo_generation_atlas_json(temp_dir, show_pad=600)
    #             except Exception as e:
    #                 print(e)
            
    #         temp_dir = samples_save_dir + temp_ss + '/OS/'
    #         if os.path.exists(temp_dir):
    #             pass
    #         else:
    #             try:
    #                 temp_dir = 'D:/data_ROP/ROP_cases_stitch_info/'+temp_ss+'/OS/'    
    #                 demo_generation_atlas_json(temp_dir, show_pad=600)
    #             except Exception as e:
    #                 print(e)

    #     else:
    #         temp_dir = 'D:/data_ROP/ROP_cases_stitch_info/'+temp_ss+'/OD/'
    #         try:    
    #             demo_generation_atlas_json(temp_dir, show_pad=600)
    #         except Exception as e:
    #             print(e)
    #         temp_dir = 'D:/data_ROP/ROP_cases_stitch_info/'+temp_ss+'/OS/'
    #         try:    
    #             demo_generation_atlas_json(temp_dir, show_pad=600)
    #         except Exception as e:
    #             print(e)






    # # temp_dir = 'D:/data_ROP/ROP_cases_stitch_info/0cb7aecc-f5b6-481c-8701-de85c1a5c79b/OD/'
    # # temp_dir = 'D:/data_ROP/ROP_cases_stitch_info/0cb7aecc-f5b6-481c-8701-de85c1a5c79b/OS/'

    # temp_dir = 'D:/data_ROP/ROP_cases_stitch_info/9cd27cf7-7b5d-4bf0-9c83-4903304c33f9/OD/'
    # # temp_dir = 'D:/data_ROP/ROP_cases_stitch_info/9cd27cf7-7b5d-4bf0-9c83-4903304c33f9/OS/'

    # # temp_dir = 'D:/data_ROP/ROP_cases_stitch_info/0e9b7859-7373-472a-8b5d-46ae992be8cb/OD/'
    # # temp_dir = 'D:/data_ROP/ROP_cases_stitch_info/0e9b7859-7373-472a-8b5d-46ae992be8cb/OS/'

    # # temp_dir = 'D:/data_ROP/ROP_cases_stitch_info/1d1b3540-e569-4484-9c23-3e1db773b25c/OD/'
    # # temp_dir = 'D:/data_ROP/ROP_cases_stitch_info/1d1b3540-e569-4484-9c23-3e1db773b25c/OS/'

    # # temp_dir = 'D:/data_ROP/ROP_cases_stitch_info/2b0729f7-f840-4e87-bd2f-9d203e83961f/OD/'
    # # temp_dir = 'D:/data_ROP/ROP_cases_stitch_info/2b0729f7-f840-4e87-bd2f-9d203e83961f/OS/'

    # # temp_dir = 'D:/data_ROP/ROP_cases_stitch_info/3e14fde9-4b44-4ff7-bb17-2a352454ab4a/OD/'
    # # temp_dir = 'D:/data_ROP/ROP_cases_stitch_info/3e14fde9-4b44-4ff7-bb17-2a352454ab4a/OS/'

    # temp_dir = 'D:/data_ROP/ROP_cases_stitch_info/05ff05d0-8485-4e7c-863b-1e48f608bc14/OD/'
    # temp_dir = 'D:/data_ROP/ROP_cases_stitch_info/05ff05d0-8485-4e7c-863b-1e48f608bc14/OS/'

    # temp_dir = 'D:/data_ROP/ROP_cases_stitch_info/68e3544d-a189-4447-a2eb-bdcf56f882e4/OS/'
    # show_generation_atlas(temp_dir, 600)


    

    ### case 1:follow-up
    # temp_dir = 'D:/data_ROP/2cases/case1/lv yiyan 18565186249 20211013/OD_sim/'
    # target_id = 4
    # temp_dir = 'D:/data_ROP/2cases/case1/lv yiyan 18565186249 20211013/OS/'
    # target_id = 53

    # temp_dir = 'D:/data_ROP/2cases/case1/lv yiyan 18565186249 20211018/OD_sim/'
    # target_id = 18
    # temp_dir = 'D:/data_ROP/2cases/case1/lv yiyan 18565186249 20211018/OS_sim/'
    # target_id = 52


    # temp_dir = 'D:/data_ROP/2cases/case1/lv yiyan 18565186249 20211025/OD/'
    # target_id = 2
    # temp_dir = 'D:/data_ROP/2cases/case1/lv yiyan 18565186249 20211025/OS_sim/'
    # target_id = 49


    # temp_dir = 'D:/data_ROP/2cases/case1/lv yiyan 18565186249 20211104/OD_sim/'
    # target_id = 4
    # temp_dir = 'D:/data_ROP/2cases/case1/lv yiyan 18565186249 20211104/OS_sim/'
    # target_id = 39


    # temp_dir = 'D:/data_ROP/2cases/case1/lv yiyan 18565186249 20211123/OD_sim/'
    # target_id = 30
    # temp_dir = 'D:/data_ROP/2cases/case1/lv yiyan 18565186249 20211123/OS_sim/'
    # target_id = 46


    # temp_dir = 'D:/data_ROP/2cases/case1/lv yiyan 18565186249 20211202/OD_sim/'
    # target_id = 11
    # temp_dir = 'D:/data_ROP/2cases/case1/lv yiyan 18565186249 20211202/OS_sim/'
    # target_id = 64

    # temp_dir = 'D:/data_ROP/2cases/case1/lv yiyan L81141551 20211213/OD_sim/'
    # target_id = 18
    # temp_dir = 'D:/data_ROP/2cases/case1/lv yiyan L81141551 20211213/OS_sim/'
    # target_id = 157




    # ### case 2:follow-up
    # temp_dir = 'D:/data_ROP/2cases/case2/zhang yuwei L81082196 20210817/OD_sim/'
    # target_id = 18
    # temp_dir = 'D:/data_ROP/2cases/case2/zhang yuwei L81082196 20210817/OS_sim/'
    # target_id = 21


    # temp_dir = 'D:/data_ROP/2cases/case2/zhang yuwei L81082196 20210824/OD_sim/'
    # target_id = 10
    # temp_dir = 'D:/data_ROP/2cases/case2/zhang yuwei L81082196 20210824/OS_sim/'
    # target_id = 46


    # temp_dir = 'D:/data_ROP/2cases/case2/zhang yuwei L81082196 20210826/OD_sim/'
    # target_id = 11
    # temp_dir = 'D:/data_ROP/2cases/case2/zhang yuwei L81082196 20210826/OS_sim/'
    # target_id = 37


    # temp_dir = 'D:/data_ROP/2cases/case2/zhang yuwei L81082196 20210901/OD_sim/'
    # target_id = 3
    # temp_dir = 'D:/data_ROP/2cases/case2/zhang yuwei L81082196 20210901/OS_sim/'
    # target_id = 62


    # temp_dir = 'D:/data_ROP/2cases/case2/zhang yuwei L81082196 20210906/OD_sim/'
    # target_id = 20
    # temp_dir = 'D:/data_ROP/2cases/case2/zhang yuwei L81082196 20210906/OS_sim/'
    # target_id = 50


    # temp_dir = 'D:/data_ROP/2cases/case2/zhang yuwei L81082196 20210913/OD_sim/'
    # target_id = 39
    # temp_dir = 'D:/data_ROP/2cases/case2/zhang yuwei L81082196 20210913/OS_sim/'
    # target_id = 50


    # temp_dir = 'D:/data_ROP/2cases/case2/zhang yuwei L81082196 20210927/OD_sim/'
    # target_id = 8
    # temp_dir = 'D:/data_ROP/2cases/case2/zhang yuwei L81082196 20210927/OS/'
    # target_id = 44


    # temp_dir = 'D:/data_ROP/2cases/case2/zhang yuwei L81082196 20211009/OD_sim/'
    # target_id = 36
    # temp_dir = 'D:/data_ROP/2cases/case2/zhang yuwei L81082196 20211009/OS_sim/'
    # target_id = 12


    # temp_dir = 'D:/data_ROP/2cases/case2/zhang yuwei L81082196 20211021/OD/'
    # target_id = 11
    # temp_dir = 'D:/data_ROP/2cases/case2/zhang yuwei L81082196 20211021/OS_sim/'
    # target_id = 23


    # temp_dir = 'D:/data_ROP/2cases/case2/zhang yuwei L81082196 20211108/OD_sim/'
    # target_id = 12
    # temp_dir = 'D:/data_ROP/2cases/case2/zhang yuwei L81082196 20211108/OS_sim/'
    # target_id = 32


    # temp_dir = 'D:/data_ROP/2cases/case2/zhang yuwei L81082196 20211115/OD_sim/'
    # target_id = 1
    # temp_dir = 'D:/data_ROP/2cases/case2/zhang yuwei L81082196 20211115/OS_sim/'
    # target_id = 23


    # temp_dir = 'D:/data_ROP/2cases/case2/zhang yuwei L81082196 20211206/OD_sim/'
    # target_id = 15
    # temp_dir = 'D:/data_ROP/2cases/case2/zhang yuwei L81082196 20211206/OS_sim/'
    # target_id = 27


    ### case 3: follow-up
    # temp_dir = 'D:/data_ROP/cases_follow_up/1/jiang hao L81555053 20220929/OD_sim/'
    # target_id = 24
    # temp_dir = 'D:/data_ROP/cases_follow_up/1/jiang hao L81555053 20220929/OS_sim/'
    # target_id = 38

    # temp_dir = 'D:/data_ROP/cases_follow_up/1/jiang hao L81555053 20221008/OD_sim/'
    # target_id = 2
    # temp_dir = 'D:/data_ROP/cases_follow_up/1/jiang hao L81555053 20221008/OS_sim/'
    # target_id = 40

    # temp_dir = 'D:/data_ROP/cases_follow_up/1/jiang hao L81555053 20221020/OD_sim/'
    # target_id = 5
    # temp_dir = 'D:/data_ROP/cases_follow_up/1/jiang hao L81555053 20221020/OS_sim/'
    # target_id = 42


    ### case 4: follow-up
    # temp_dir = 'D:/data_ROP/cases_follow_up/2/he chengqian L81221507 20211224/OD_sim/'
    # target_id = 53
    # temp_dir = 'D:/data_ROP/cases_follow_up/2/he chengqian L81221507 20211224/OS_sim/'
    # target_id = 119

    # temp_dir = 'D:/data_ROP/cases_follow_up/2/he chengqian L81221507 20211227/OD_sim/'
    # target_id = 1
    # temp_dir = 'D:/data_ROP/cases_follow_up/2/he chengqian L81221507 20211227/OS_sim/'
    # target_id = 47

    # temp_dir = 'D:/data_ROP/cases_follow_up/2/he chengqian L81221507 20220104/OD_sim/'
    # target_id = 40
    # temp_dir = 'D:/data_ROP/cases_follow_up/2/he chengqian L81221507 20220104/OS_sim/'
    # target_id = 27


    # temp_dir = 'D:/data_ROP/cases_follow_up/2/he chengqian L81221507 20220111/OD_sim/'
    # target_id = 16
    # temp_dir = 'D:/data_ROP/cases_follow_up/2/he chengqian L81221507 20220111/OS_sim/'
    # target_id = 65

    # temp_dir = 'D:/data_ROP/cases_follow_up/2/he chengqian L81221507 20220125/OD_sim/'
    # target_id = 32
    # temp_dir = 'D:/data_ROP/cases_follow_up/2/he chengqian L81221507 20220125/OS_sim/'
    # target_id = 59

    # temp_dir = 'D:/data_ROP/cases_follow_up/2/he chengqian L81221507 20220208/OD_sim/'
    # target_id = 134
    # temp_dir = 'D:/data_ROP/cases_follow_up/2/he chengqian L81221507 20220208/OS_sim/'
    # target_id = 135

    # temp_dir = 'D:/data_ROP/cases_follow_up/2/he chengqian L81221507 20220304/OD_sim/'
    # target_id = 20
    # temp_dir = 'D:/data_ROP/cases_follow_up/2/he chengqian L81221507 20220304/OS_sim/'
    # target_id = 62

    # temp_dir = 'D:/data_ROP/cases_follow_up/2/he chengqian L81221507 20220426/OD_sim/'
    # target_id = 29
    # temp_dir = 'D:/data_ROP/cases_follow_up/2/he chengqian L81221507 20220426/OS_sim/'
    # target_id = 38

    # temp_dir = 'D:/data_ROP/cases_follow_up/2/he chengqian L81221507 20220628/OD_sim/'
    # target_id = 42
    # temp_dir = 'D:/data_ROP/cases_follow_up/2/he chengqian L81221507 20220628/OS_sim/'
    # target_id = 67

    # temp_dir = 'D:/data_ROP/cases_follow_up/2/he chengqian L81221507 20220927/OD_sim/'
    # target_id = 24
    # temp_dir = 'D:/data_ROP/cases_follow_up/2/he chengqian L81221507 20220927/OS_sim/'
    # target_id = 53


    ### case 5: follow-up
    # temp_dir = 'D:/data_ROP/cases_follow_up/3/che shunxi L81393899 20220620/OD_sim/'
    # target_id = 27
    # temp_dir = 'D:/data_ROP/cases_follow_up/3/che shunxi L81393899 20220620/OS_sim_2/'
    # target_id = 62

    # temp_dir = 'D:/data_ROP/cases_follow_up/3/che shunxi L81393899 20220704/OD_sim/'
    # target_id = 13
    # temp_dir = 'D:/data_ROP/cases_follow_up/3/che shunxi L81393899 20220704/OS_sim/'
    # target_id = 43

    # temp_dir = 'D:/data_ROP/cases_follow_up/3/che shunxi L81393899 20220718/OD_sim/'
    # target_id = 4
    # temp_dir = 'D:/data_ROP/cases_follow_up/3/che shunxi L81393899 20220718/OS_sim/'
    # target_id = 66

    # temp_dir = 'D:/data_ROP/cases_follow_up/3/che shunxi L81393899 20220815/OD_sim/'
    # target_id = 2
    # temp_dir = 'D:/data_ROP/cases_follow_up/3/che shunxi L81393899 20220815/OS_sim/'
    # target_id = 51

    # temp_dir = 'D:/data_ROP/cases_follow_up/3/che shunxi L81393899 20221010/OD_sim/'
    # target_id = 17
    # temp_dir = 'D:/data_ROP/cases_follow_up/3/che shunxi L81393899 20221010/OS_sim/'
    # target_id = 56


    ### case 6: follow-up
    # temp_dir = 'D:/data_ROP/cases_follow_up/4/chen wanzu L81549229 20220928/OD_sim/'
    # target_id = 8
    # temp_dir = 'D:/data_ROP/cases_follow_up/4/chen wanzu L81549229 20220928/OS_sim/'
    # target_id = 46


    ### case 7: follow-up
    # temp_dir = 'D:/data_ROP/cases_follow_up/5/chen yisheng l81551285 20220928/OD_sim/'
    # target_id = 4
    # temp_dir = 'D:/data_ROP/cases_follow_up/5/chen yisheng l81551285 20220928/OS_sim/'
    # target_id = 33

    ### case 8: follow-up
    temp_dir = 'D:/data_ROP/cases_follow_up/dengyibing-3/deng yibing L81119132 20210922/OD_sim/'
    target_id = 1
    temp_dir = 'D:/data_ROP/cases_follow_up/dengyibing-3/deng yibing L81119132 20210922/OS_sim/'
    target_id = 31

    
    temp_dir = 'D:/data_ROP/cases_follow_up/dengyibing-3/deng yibing L81119132 20210927/OD_sim/'
    target_id = 14
    temp_dir = 'D:/data_ROP/cases_follow_up/dengyibing-3/deng yibing L81119132 20210927/OS_sim/'
    target_id = 18


    temp_dir = 'D:/data_ROP/cases_follow_up/dengyibing-3/deng yibing L81119132 20211011/OD_sim/'
    target_id = 59
    temp_dir = 'D:/data_ROP/cases_follow_up/dengyibing-3/deng yibing L81119132 20211011/OS_sim/'
    target_id = 54

    temp_dir = 'D:/data_ROP/cases_follow_up/dengyibing-3/deng yibing L81119132 20211018/OD_sim/'
    target_id = 5
    temp_dir = 'D:/data_ROP/cases_follow_up/dengyibing-3/deng yibing L81119132 20211018/OS_sim/'
    target_id = 54



















    # ## case 3:(retcam vs slo)
    # temp_dir = 'D:/data_ROP/retcam-slo/weijian peng L80837134 20201214 vd4/OD_sim/'
    # target_id = 9
    # temp_dir = 'D:/data_ROP/retcam-slo/weijian peng L80837134 20201214 vd4/OS_sim/'
    # target_id = 1


    # ## case 4(retcam vs slo)
    # temp_dir = 'D:/data_ROP/retcam-slo/wen bowen L80838894 20201127 vd4/OD_sim/'
    # target_id = 45
    # temp_dir = 'D:/data_ROP/retcam-slo/wen bowen L80838894 20201127 vd4/OS_sim/'
    # target_id = 57


    # ## case 5(retcam vs slo)
    # temp_dir = 'D:/data_ROP/retcam-slo/mantang chen L80844661 20201204 vd4/OD_sim/'
    # target_id = 3
    # temp_dir = 'D:/data_ROP/retcam-slo/mantang chen L80844661 20201204 vd4/OS_sim/'
    # target_id = 35




    # ## case fig 1
    # temp_dir = 'E:/data_stitch/data_panyu_per_id/78e2080c-9435-47cc-a2a6-0be77ffa6e34/OD/'
    # target_id = 1

    
    
    ## case pano
    # temp_dir = 'D:/data_ROP/panocam/case1/L/'
    # target_id = 3
    # temp_dir = 'D:/data_ROP/panocam/case1/R/'
    # target_id = 9
    # temp_dir = 'D:/data_ROP/panocam/case2/L/'
    # target_id = 1
    # temp_dir = 'D:/data_ROP/panocam/case2/R/'
    # target_id = 1
    # demo_generation_atlas(temp_dir, 600, target_id)
    demo_generation_without_atlas(temp_dir, 600, target_id)




    # #### case fig 4
    # temp_dir = r'D:/data_ROP/fig cases/040b27ca-f697-47ba-81a0-e39c5e6052e2/OS_sim/'
    # target_id = 97
    # # temp_dir = r'D:/data_ROP/fig cases/040b27ca-f697-47ba-81a0-e39c5e6052e2/OD_sim/'
    # # target_id = 54

    # # temp_dir = r'D:/data_ROP/fig cases/2b0729f7-f840-4e87-bd2f-9d203e83961f/OS_sim/'
    # # target_id = 31
    # # temp_dir = r'D:/data_ROP/fig cases/2b0729f7-f840-4e87-bd2f-9d203e83961f/OD_sim/'
    # # target_id = 23

    # demo_generation_atlas(temp_dir, 600, target_id)



    # ### case demo
    # ### paranomic view with mapping center location
    # jsons_dir = 'D:/data_ROP/ROP_cases_stitch_info/0cb7aecc-f5b6-481c-8701-de85c1a5c79b/OS/'
    # demo_generation_with_center_json(jsons_dir, show_pad=600)



    # ## panoramic view with center alignment and lesion
    # jsons_dir = 'D:/data_ROP/ROP_cases_stitch_info/0cb7aecc-f5b6-481c-8701-de85c1a5c79b/OS/'
    # # jsons_dir = 'D:/data_ROP/ROP_cases_stitch_info/3080410f-74d9-4f4a-9376-e247179c1d2c/OS/'
    # # jsons_dir = 'D:/data_ROP/ROP_cases_stitch_info/f911b0fc-b7be-42d0-8c17-a2245f625f2b/OD/'
    # demo_generation_center_align_json_with_lesion(jsons_dir, show_pad=600, nms=True)

    

    # list_examination_jsons = []
    # root_dir = 'D:/data_ROP/ROP_cases_stitch_info/'
    # flag_loc = 'OD'
    # list_examination_ids = os.listdir(root_dir)
    # for temp_examination_id in list_examination_ids:
    #     if flag_loc == 'OD':
    #         temp_dir = root_dir + temp_examination_id + '/' + 'OD/'
    #         if os.path.exists(temp_dir):
    #             list_examination_jsons.append(temp_dir)
    #     else:
    #         temp_dir = root_dir + temp_examination_id + '/' + 'OS/'
    #         if os.path.exists(temp_dir):
    #             list_examination_jsons.append(temp_dir)
    
    # lesion_map_avg = None
    # for temp_json_dir in tqdm(list_examination_jsons[:], ncols=100):
    #     try:
    #         temp_panoramic_img, temp_panoramic_img_lesion, list_lesion_maps = demo_generation_center_align_json_with_lesion(temp_json_dir, show_pad=600)
    #         if type(lesion_map_avg) == type(None):
    #             lesion_map_avg = list_lesion_maps
    #         else:
    #             lesion_map_avg += list_lesion_maps
    #     except Exception as e:
    #         print(e)
    #         pass
    

    # results_map_dir = 'D:/proj_stitch/panoramic_results/results_center_alignment_lesion/lesion_avg/' + flag_loc + '/'
    # if os.path.exists(results_map_dir) is False:
    #     os.makedirs(results_map_dir)
    # for i in range(len(lesion_map_avg)):
    #     lesion_map_avg_cls = lesion_map_avg[i]
    #     lesion_map_avg_cls = norm_one(lesion_map_avg_cls)*255
    #     lesion_map_avg_cls = lesion_map_avg_cls.astype(np.uint8)
    #     lesion_map_avg_cls = cv2.applyColorMap(255-lesion_map_avg_cls, cv2.COLORMAP_RAINBOW)
    #     cv2.imwrite(results_map_dir + 'cls_' +str(i) + '.png', lesion_map_avg_cls)
    # print('sss')










