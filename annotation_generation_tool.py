import json
import cv2
import os

import numpy as np
from tqdm import tqdm

from utils import convert, test_size_w, test_size_h

## opencv 可以检测的鼠标和键盘事件
['EVENT_FLAG_ALTKEY', 'EVENT_FLAG_CTRLKEY',  ## 检测到按键 ALT、CTRL 
'EVENT_FLAG_LBUTTON', 'EVENT_FLAG_MBUTTON', 'EVENT_FLAG_RBUTTON',    ##检测到鼠标的左键、中键、右键
'EVENT_FLAG_SHIFTKEY', 'EVENT_LBUTTONDBLCLK', 'EVENT_LBUTTONDOWN',   ##检测到Shift按键，左键双击，左键按下
'EVENT_LBUTTONUP', 'EVENT_MBUTTONDBLCLK', 'EVENT_MBUTTONDOWN',       ##检测到左键弹起、中键双击，中键按下
'EVENT_MBUTTONUP', 'EVENT_MOUSEHWHEEL', 'EVENT_MOUSEMOVE',           ##检测到中键弹起、滚轮活动，鼠标活动
'EVENT_MOUSEWHEEL', 'EVENT_RBUTTONDBLCLK',                           ##检测到滚轮活动、右键双击
'EVENT_RBUTTONDOWN', 'EVENT_RBUTTONUP']                              ##检测到右键按下、右键弹起



class label_function_generation(object):
    def __init__(self, target_dir, activate_dir):
        img_1 = cv2.resize(cv2.imread(target_dir), (640,480))
        img_2 = cv2.resize(cv2.imread(activate_dir), (640,480))
        img_1 = convert(img_1)
        img_2 = convert(img_2)

        image = np.hstack([img_1, img_2])
        self.image_ori = image.copy()
        self.image_show = image.copy()
        cv2.namedWindow('label_image')
        cv2.setMouseCallback('label_image', self.label_callback)
        self.list_pts_t = []
        self.list_pts_a = []
        self.save_flag = True
        self.label_flag = 'None'
        self.width = image.shape[1]//2
    
    def label_callback(self, event, x, y, flag, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if x < 640:
                cv2.circle(self.image_show, (x,y), 3, (0,255,0), -1)
                cv2.imshow('label_image', self.image_show)
                self.list_pts_t.append([x,y])
            else:
                cv2.circle(self.image_show, (x,y), 3, (0,0,255), -1)
                cv2.imshow('label_image', self.image_show)
                self.list_pts_a.append([x-self.width,y])
        elif event == cv2.EVENT_RBUTTONDOWN:
            if x > 640:
                cv2.circle(self.image_show, (x,y), 3, (0,0,255), -1)
                cv2.imshow('label_image', self.image_show)
                self.list_pts_a.append([x-self.width,y])
            else:
                cv2.circle(self.image_show, (x,y), 3, (0,255,0), -1)
                cv2.imshow('label_image', self.image_show)
                self.list_pts_t.append([x,y])
        elif event == cv2.EVENT_MBUTTONDBLCLK:
            self.image_show = self.image_ori.copy()
            cv2.imshow('label_image', self.image_show)
            self.list_pts_t = []
            self.list_pts_a = []

    def show_current_label(self, image, list_pts_t, list_pts_a):
        image_t = image.copy()[:,:640,:]
        image_a = image.copy()[:,640:,:]
        p1_sim = np.array(list_pts_t)
        p2_sim = np.array(list_pts_a)
        pad_size = 200
        t_h, t_w = image_t.shape[:2]
        p1s_np_ex = p1_sim + pad_size
        p2s_np_ex = p2_sim + pad_size

        h_a2t, _ = cv2.estimateAffinePartial2D(p2s_np_ex, p1s_np_ex, method=cv2.RANSAC, ransacReprojThreshold=2.0, confidence=0.96)
        img_t_ex = cv2.copyMakeBorder(image_t, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, dst=None, value=0)
        img_a_ex = cv2.copyMakeBorder(image_a, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, dst=None, value=0)
        image_a2t = cv2.warpAffine(img_a_ex, h_a2t, (t_w+pad_size*2, t_h+pad_size*2)).astype(np.uint8)
        image_fusion = (img_t_ex/2 + image_a2t/2).astype(np.uint8)
        cv2.imshow('current label', image_fusion)

    def show(self):
        while(1):
            cv2.imshow('label_image', self.image_show)
            # self.show_current_label(self.image_ori, self.label_json['pts_sim_1'].copy(), self.label_json['pts_sim_2'].copy())
            key = cv2.waitKeyEx()
            key = key % 255
            if key == 39:##right
                self.label_flag = 'skip and next'
                # print('next')
                break
            elif key == 40:##down
                self.label_flag = 'save and next'
                # print('save')
                break
            elif key == 38:##up
                self.label_flag = 'delete and next'
                # print('delete')
                break
            elif key == 37:##left
                self.label_flag == 'show label info'
                try:
                    self.show_current_label(self.image_ori, self.list_pts_t, self.list_pts_a)
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


def demo_annotation(temp_target_dir, temp_activate_dir, temp_result_dir):
    # temp_target_dir = ''
    # temp_activate_dir = ''
    # temp_result_dir = ''
    annotation_function = label_function_generation(temp_target_dir, temp_activate_dir)
    annotation_function.label_flag = annotation_function.show()
    temp_result_json = {}
    temp_result_json['dir_t'] = temp_target_dir
    temp_result_json['dir_a'] = temp_activate_dir
    annotation_function.save_result(temp_result_json, temp_result_dir)

def show_annotation_json(temp_json_dir, pad_size=400):
    with open(temp_json_dir, mode='r', encoding='utf-8') as F:
        temp_info = json.load(F)
        temp_1_dir = temp_info['dir_t']
        temp_2_dir = temp_info['dir_a']
        temp_img_s_dir = temp_1_dir
        temp_img_t_dir = temp_2_dir


    #### image_target, image_activate
    border_mask = cv2.imread('./border_mask.png', 0)
    border_mask = cv2.resize(border_mask, dsize=(test_size_w,test_size_h))
    temp_img_t = convert(cv2.imread(temp_img_s_dir))
    temp_img_a = convert(cv2.imread(temp_img_t_dir))
    p1_sim = np.array(temp_info['pts_sim_1'])
    p2_sim = np.array(temp_info['pts_sim_2'])
    if len(p1_sim)<1:
        return None
    if len(p1_sim.shape) < 3:
        p1_sim = np.expand_dims(p1_sim, 2)
        p2_sim = np.expand_dims(p2_sim, 2)
    x_scale = test_size_w/640
    y_scale = test_size_h/480
    p1_sim[:,:,:1] = p1_sim[:,:,:1]*x_scale
    p1_sim[:,:,1:] = p1_sim[:,:,1:]*y_scale
    p2_sim[:,:,:1] = p2_sim[:,:,:1]*x_scale
    p2_sim[:,:,1:] = p2_sim[:,:,1:]*y_scale

    img_t = cv2.resize(temp_img_t, dsize=(test_size_w, test_size_h))
    img_a = cv2.resize(temp_img_a, dsize=(test_size_w, test_size_h))

    # img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB)
    # img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)

    img_t_ex = cv2.copyMakeBorder(img_t, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, dst=None, value=0)
    img_a_ex = cv2.copyMakeBorder(img_a, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, dst=None, value=0)


    h, w = img_t.shape[:2]
    h_ex, w_ex = img_t_ex.shape[:2]
    p1_sim_ex = p1_sim + pad_size
    p2_sim_ex = p2_sim + pad_size


    
    
    H_a2t_ex, _ = cv2.estimateAffinePartial2D(p2_sim_ex, p1_sim_ex, method=cv2.RANSAC, ransacReprojThreshold=5.0, confidence=0.96)
    img_a_ex_result = cv2.warpAffine(img_a_ex, H_a2t_ex, (w_ex, h_ex))


    # cv2.imshow('src_1', img_t)
    # cv2.imshow('src_2', img_a)
    cv2.imshow('src_1_ex', img_t_ex)
    cv2.imshow('src_2_ex', img_a_ex)

    # cv2.imshow('result_img', img_a_ex_result)

    temp_fusion_random = (img_t_ex//2 + img_a_ex_result//2)
    cv2.imshow('fusion', temp_fusion_random)
    cv2.waitKey()


if __name__ == '__main__':
    ## annotation stitch pair manually
    demo_annotation('./demo_data/1_t.png', './demo_data/1_a.png', './demo_data/1.json')

    ## show the annotation result
    show_annotation_json('./demo_data/1.json', 400)
