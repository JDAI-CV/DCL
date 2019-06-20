import os
import math
import numpy as np
import cv2
import datetime

import torch
from torchvision.utils import save_image, make_grid

import pdb

def dt():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

def set_text(text, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    if isinstance(text, str):
        cont = text
        cv2.putText(img, cont, (20, 50), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    if isinstance(text, float):
        cont = '%.4f'%text
        cv2.putText(img, cont, (20, 50), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    if isinstance(text, list):
        for count in range(len(img)):
            cv2.putText(img[count], text[count], (20, 50), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return img

def save_multi_img(img_list, text_list, grid_size=[5,5], sub_size=200, save_dir='./', save_name=None):
    if len(img_list) > grid_size[0]*grid_size[1]:
        merge_height = math.ceil(len(img_list) / grid_size[0]) * sub_size
    else:
        merge_height = grid_size[1]*sub_size
    merged_img = np.zeros((merge_height, grid_size[0]*sub_size, 3))

    if isinstance(img_list[0], str):
        img_name_list = img_list
        img_list = []
        for img_name in img_name_list:
            img_list.append(cv2.imread(img_name))

    img_counter = 0
    for img, txt in zip(img_list, text_list):
        img = cv2.resize(img, (sub_size, sub_size))
        img = set_text(txt, img)
        pos = [img_counter // grid_size[1], img_counter % grid_size[1]]
        sub_pos = [pos[0]*sub_size, (pos[0]+1)*sub_size,
                   pos[1]*sub_size, (pos[1]+1)*sub_size]
        merged_img[sub_pos[0]:sub_pos[1], sub_pos[2]:sub_pos[3], :] = img
        img_counter += 1

    if save_name is None:
        img_save_path = os.path.join(save_dir, dt()+'.png')
    else:
        img_save_path = os.path.join(save_dir, save_name+'.png')
    cv2.imwrite(img_save_path,  merged_img)
    print('saved img in %s ...'%img_save_path)


def cls_base_acc(result_gather):
    top1_acc = {}
    top3_acc = {}
    cls_count = {}
    for img_item in result_gather.keys():
        acc_case = result_gather[img_item]

        if acc_case['label'] in cls_count:
            cls_count[acc_case['label']] += 1
            if acc_case['top1_cat'] == acc_case['label']:
                top1_acc[acc_case['label']] += 1
            if acc_case['label'] in [acc_case['top1_cat'], acc_case['top2_cat'], acc_case['top3_cat']]:
                top3_acc[acc_case['label']] += 1
        else:
            cls_count[acc_case['label']] = 1
            if acc_case['top1_cat'] == acc_case['label']:
                top1_acc[acc_case['label']] = 1
            else:
                top1_acc[acc_case['label']] = 0

            if acc_case['label'] in [acc_case['top1_cat'], acc_case['top2_cat'], acc_case['top3_cat']]:
                top3_acc[acc_case['label']] = 1
            else:
                top3_acc[acc_case['label']] = 0

    for label_item in cls_count:
        top1_acc[label_item] /= max(1.0*cls_count[label_item], 0.001)
        top3_acc[label_item] /= max(1.0*cls_count[label_item], 0.001)

    print('top1_acc:', top1_acc)
    print('top3_acc:', top3_acc)
    print('cls_count', cls_count)

    return top1_acc, top3_acc, cls_count







