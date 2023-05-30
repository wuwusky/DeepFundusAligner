import os
import cv2
import time
import numpy as np
from utils import get_img_pad, get_warp_img
from Registration_Module import stitch_RM, stitch_RM_Plus, stitch_RM_class


show_pad = 600
stitch_fun = stitch_RM_class()
##==========================================demo for folder watcher======================================================
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

list_imgs_dir = []
list_imgs = []

def show_img(win_name, img):
    img_show = cv2.resize(img, None, fx=0.5, fy=0.5)
    cv2.imshow(win_name, img_show)

class test_handler(FileSystemEventHandler):
    def on_created(self, event):
        time.sleep(0.5)
        print('new file has been created:{}'.format(event.src_path))
        list_imgs_dir.append(event.src_path)
        if len(list_imgs_dir) == 1:
            self.img_target = cv2.imread(list_imgs_dir[-1])
            try:
                show_img('First cap image', self.img_target)
            except Exception as e:
                self.img_target = cv2.imread(list_imgs_dir[-1])
                show_img('First cap image', self.img_target)
            cv2.waitKey(1)
            temp_t_pad = get_img_pad(self.img_target, show_pad)
            list_imgs.append(temp_t_pad)
        
        elif len(list_imgs_dir) >= 2:
            temp_img = cv2.imread(list_imgs_dir[-1])
            try:
                show_img('New cap image', temp_img)
            except Exception as e:
                temp_img = cv2.imread(list_imgs_dir[-1])
                show_img('New cap image', temp_img)
            cv2.waitKey(1)
            temp_a_pad = get_img_pad(temp_img, show_pad)
            _, pred_h_pad = stitch_fun.stitch_RM_Plus(list_imgs_dir[0], list_imgs_dir[-1], pad_size=show_pad)
            temp_a_dst = get_warp_img(temp_a_pad, pred_h_pad)
            list_imgs.append(temp_a_dst)
        
        temp_current_img = np.zeros_like(list_imgs[0])
        for temp_img in list_imgs:
            temp_current_img = np.where(temp_current_img[:,:,:]>temp_img[:,:,:], temp_current_img, temp_img)
        

        temp_current_img = cv2.circle(temp_current_img, (320+show_pad, 240+show_pad), 1300//2, (255,255,255), 5)
        show_img('current result', temp_current_img)
        cv2.waitKey(1)

    def on_modified(self, event):
        pass

if __name__ == '__main__':
    path = 'D:/data_ROP/test_for_demo_stitch/'
    os.makedirs(path, exist_ok=True)
    event_handler = test_handler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()

    # try:
    #     while True:
    #         time.sleep(1)
    # except KeyboardInterrupt:
    #     observer.stop()
    observer.join()
##==========================================demo for folder watcher======================================================

