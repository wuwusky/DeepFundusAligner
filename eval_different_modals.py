import cv2
import os
import numpy as np
from utils import load_img, get_warp_img, get_img_pad, pad_size_global
from Registration_Module import stitch_CRP, stitch_RM, stitch_RM_Plus

def demo_evaluate_modals_MPipeline(img1_dir, img2_dir, pad_size):
    img_t = load_img(img1_dir)
    img_a = load_img(img2_dir)

    # img1_dir = './temp_img_1.png'
    # img2_dir = './temp_img_2.png'
    # cv2.imwrite(img1_dir, img_t)
    # cv2.imwrite(img2_dir, img_a)
    img_t = cv2.resize(img_t, (640,480))
    img_a = cv2.resize(img_a, (640,480))

    img_pair = np.hstack([img_t, img_a])
    # img_pair = cv2.resize(img_pair, None, fx=0.25, fy=0.25)
    img_t_pad = get_img_pad(img_t, pad_size)
    img_a_pad = get_img_pad(img_a, pad_size)

    pred_H_pad, pred_H_inv, num_robust = stitch_CRP(img1_dir, img2_dir, pad_size=pad_size, flag=True)
    img_a_result = get_warp_img(img_a_pad, pred_H_pad)
    img_at_sf =  cv2.addWeighted(img_t_pad, 0.5, img_a_result, 0.5, 0)


    pred_H_inv, pred_H_pad = stitch_RM(img1_dir, img2_dir, pad_size)
    img_a_pad = get_img_pad(img_a, pad_size)
    img_a_result = get_warp_img(img_a_pad, pred_H_pad)
    img_at_dl =  cv2.addWeighted(img_t_pad, 0.5, img_a_result, 0.5, 0)


    # cv2.waitKey()
    # get_json_result_dl_stage2_pair(img1_dir, img2_dir)
    pred_H_inv, pred_H_pad = stitch_RM_Plus(img1_dir, img2_dir, pad_size)
    img_a_pad = get_img_pad(img_a, pad_size)
    img_a_result = get_warp_img(img_a_pad, pred_H_pad)
    img_at_ref =  cv2.addWeighted(img_t_pad, 0.5, img_a_result, 0.5, 0)


    cv2.imshow('pair', img_pair)
    cv2.imshow('CRP', img_at_sf)
    cv2.imshow('RM', img_at_dl)
    cv2.imshow('RM_Plus', img_at_ref)
    temp_save_dir = './demo_data/'
    os.makedirs(temp_save_dir, exist_ok=True)
    cv2.imwrite(temp_save_dir+'pair.png', img_pair)
    cv2.imwrite(temp_save_dir+'CRP.png', img_at_sf)
    cv2.imwrite(temp_save_dir+'RM.png', img_at_dl)
    cv2.imwrite(temp_save_dir+'RM_Rlus.png', img_at_ref)
    cv2.waitKey()


if __name__ == '__main__':
    test_dir1 = './demo_data/1_t.png'
    test_dir2 = './demo_data/1_a.png'

    demo_evaluate_modals_MPipeline(test_dir1, test_dir2, pad_size_global)