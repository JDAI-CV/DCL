#oding=utf-8
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset.dataset_CUB_test import collate_fn1, collate_fn2, dataset
import torch
import torch.nn as nn
from torchvision import datasets, models
from transforms import transforms
from torch.nn import CrossEntropyLoss
from models.resnet_swap_2loss_add import resnet_swap_2loss_add
from math import ceil
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np

cfg = {}
cfg['dataset'] = 'CUB'
# prepare dataset
if cfg['dataset'] == 'CUB':
    rawdata_root = './datasets/CUB_200_2011/all'
    train_pd = pd.read_csv("./datasets/CUB_200_2011/train.txt",sep=" ",header=None, names=['ImageName', 'label'])
    train_pd, val_pd = train_test_split(train_pd, test_size=0.90, random_state=43,stratify=train_pd['label'])
    test_pd = pd.read_csv("./datasets/CUB_200_2011/test.txt",sep=" ",header=None, names=['ImageName', 'label'])
    cfg['numcls'] = 200
    numimage = 6033
if cfg['dataset'] == 'STCAR':
    rawdata_root = './datasets/st_car/all'
    train_pd = pd.read_csv("./datasets/st_car/train.txt",sep=" ",header=None, names=['ImageName', 'label'])
    test_pd = pd.read_csv("./datasets/st_car/test.txt",sep=" ",header=None, names=['ImageName', 'label'])
    cfg['numcls'] = 196
    numimage = 8144
if cfg['dataset'] == 'AIR':
    rawdata_root = './datasets/aircraft/all'
    train_pd = pd.read_csv("./datasets/aircraft/train.txt",sep=" ",header=None, names=['ImageName', 'label'])
    test_pd = pd.read_csv("./datasets/aircraft/test.txt",sep=" ",header=None, names=['ImageName', 'label'])
    cfg['numcls'] = 100
    numimage = 6667

print('Set transform')
data_transforms = {
        'totensor': transforms.Compose([
            transforms.Resize((448,448)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'None': transforms.Compose([
            transforms.Resize((512,512)),
            transforms.CenterCrop((448,448)),
        ]),

    }
data_set = {}
data_set['val'] = dataset(cfg,imgroot=rawdata_root,anno_pd=test_pd,
                           unswap=data_transforms["None"],swap=data_transforms["None"],totensor=data_transforms["totensor"],train=False
                           )
dataloader = {}
dataloader['val']=torch.utils.data.DataLoader(data_set['val'], batch_size=4,
                                               shuffle=False, num_workers=4,collate_fn=collate_fn1)
model = resnet_swap_2loss_add(num_classes=cfg['numcls'])
model.cuda()
model = nn.DataParallel(model)
resume = './cub_model.pth'
pretrained_dict=torch.load(resume)
model_dict=model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
criterion = CrossEntropyLoss()
model.train(False)
val_corrects1 = 0
val_corrects2 = 0
val_corrects3 = 0
val_size = ceil(len(data_set['val']) / dataloader['val'].batch_size)
for batch_cnt_val, data_val in tqdm(enumerate(dataloader['val'])):
    #print('testing')
    inputs,  labels, labels_swap = data_val
    inputs = Variable(inputs.cuda())
    labels = Variable(torch.from_numpy(np.array(labels)).long().cuda())
    labels_swap = Variable(torch.from_numpy(np.array(labels_swap)).long().cuda())
    # forward
    if len(inputs)==1:
        inputs = torch.cat((inputs,inputs))
        labels = torch.cat((labels,labels))
        labels_swap = torch.cat((labels_swap,labels_swap))
    
    outputs = model(inputs)
    
    outputs1 = outputs[0] + outputs[1][:,0:cfg['numcls']] + outputs[1][:,cfg['numcls']:2*cfg['numcls']]
    outputs2 = outputs[0]
    outputs3 = outputs[1][:,0:cfg['numcls']] + outputs[1][:,cfg['numcls']:2*cfg['numcls']]

    _, preds1 = torch.max(outputs1, 1)
    _, preds2 = torch.max(outputs2, 1)
    _, preds3 = torch.max(outputs3, 1)
    batch_corrects1 = torch.sum((preds1 == labels)).data.item()
    batch_corrects2 = torch.sum((preds2 == labels)).data.item()
    batch_corrects3 = torch.sum((preds3 == labels)).data.item()
    
    val_corrects1 += batch_corrects1
    val_corrects2 += batch_corrects2
    val_corrects3 += batch_corrects3
val_acc1 = 0.5 * val_corrects1 / len(data_set['val'])
val_acc2 = 0.5 * val_corrects2 / len(data_set['val'])
val_acc3 = 0.5 * val_corrects3 / len(data_set['val'])
print("cls&adv acc:", val_acc1, "cls acc:", val_acc2,"adv acc:", val_acc1)









