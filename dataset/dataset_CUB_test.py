# coding=utf8
from __future__ import division
import os
import torch
import torch.utils.data as data
import PIL.Image as Image
from PIL import ImageStat
class dataset(data.Dataset):
    def __init__(self, cfg, imgroot, anno_pd, unswap=None, swap=None, totensor=None, train=False):
        self.root_path = imgroot
        self.paths = anno_pd['ImageName'].tolist()
        self.labels = anno_pd['label'].tolist()
        self.unswap = unswap
        self.swap = swap
        self.totensor = totensor
        self.cfg = cfg
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        img_path = os.path.join(self.root_path, self.paths[item])
        img = self.pil_loader(img_path)
        img_unswap = self.unswap(img)
        img_unswap = self.totensor(img_unswap)
        img_swap = img_unswap
        label = self.labels[item]-1
        label_swap = label
        return img_unswap, img_swap, label, label_swap

    def pil_loader(self,imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

def collate_fn1(batch):
    imgs = []
    label = []
    label_swap = []
    swap_law = []
    for sample in batch:
        imgs.append(sample[0])
        imgs.append(sample[1])
        label.append(sample[2])
        label.append(sample[2])
        label_swap.append(sample[2])
        label_swap.append(sample[3])
        # swap_law.append(sample[4])
        # swap_law.append(sample[5])
    return torch.stack(imgs, 0), label, label_swap # , swap_law

def collate_fn2(batch):
    imgs = []
    label = []
    label_swap = []
    swap_law = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[2])
        swap_law.append(sample[4])
    return torch.stack(imgs, 0), label, label_swap, swap_law



