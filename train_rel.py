#oding=utf-8
import os
import datetime
import pandas as pd
from dataset.dataset_DCL import collate_fn1, collate_fn2, dataset
import torch
import torch.nn as nn
import torch.utils.data as torchdata
from torchvision import datasets, models
from transforms import transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from utils.train_util_DCL import train, trainlog
from  torch.nn import CrossEntropyLoss
import logging
from models.resnet_swap_2loss_add import resnet_swap_2loss_add

cfg = {}
time = datetime.datetime.now()
# set dataset, include{CUB, STCAR, AIR}
cfg['dataset'] = 'CUB'
# prepare dataset
if cfg['dataset'] == 'CUB':
    rawdata_root = './datasets/CUB_200_2011/all'
    train_pd = pd.read_csv("./datasets/CUB_200_2011/train.txt",sep=" ",header=None, names=['ImageName', 'label'])
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

print('Dataset:',cfg['dataset'])
print('train images:', train_pd.shape)
print('test images:', test_pd.shape)
print('num classes:', cfg['numcls'])

print('Set transform')

cfg['swap_num'] = 7

data_transforms = {
       	'swap': transforms.Compose([
            transforms.Resize((512,512)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomCrop((448,448)),
            transforms.RandomHorizontalFlip(),
            transforms.Randomswap((cfg['swap_num'],cfg['swap_num'])),
        ]),
        'unswap': transforms.Compose([
            transforms.Resize((512,512)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomCrop((448,448)),
            transforms.RandomHorizontalFlip(),
        ]),
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
data_set['train'] = dataset(cfg,imgroot=rawdata_root,anno_pd=train_pd,
                           unswap=data_transforms["unswap"],swap=data_transforms["swap"],totensor=data_transforms["totensor"],train=True
                           )
data_set['val'] = dataset(cfg,imgroot=rawdata_root,anno_pd=test_pd,
                           unswap=data_transforms["None"],swap=data_transforms["None"],totensor=data_transforms["totensor"],train=False
                           )
dataloader = {}
dataloader['train']=torch.utils.data.DataLoader(data_set['train'], batch_size=16,
                                               shuffle=True, num_workers=16,collate_fn=collate_fn1)
dataloader['val']=torch.utils.data.DataLoader(data_set['val'], batch_size=16,
                                               shuffle=True, num_workers=16,collate_fn=collate_fn1)

print('Set cache dir')
filename = str(time.month) + str(time.day) + str(time.hour) + '_' + cfg['dataset']
save_dir = './net_model/' + filename
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logfile = save_dir + '/' + filename +'.log'
trainlog(logfile)

print('Choose model and train set')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = resnet_swap_2loss_add(num_classes=cfg['numcls'])
base_lr = 0.0008
resume = None
if resume:
    logging.info('resuming finetune from %s'%resume)
    model.load_state_dict(torch.load(resume))
model.cuda()
model = nn.DataParallel(model)
model.to(device)

# set new layer's lr
ignored_params1 = list(map(id, model.module.classifier.parameters()))
ignored_params2 = list(map(id, model.module.classifier_swap.parameters()))
ignored_params3 = list(map(id, model.module.Convmask.parameters()))

ignored_params = ignored_params1 + ignored_params2 + ignored_params3
print('the num of new layers:', len(ignored_params))
base_params = filter(lambda p: id(p) not in ignored_params, model.module.parameters())
optimizer = optim.SGD([{'params': base_params},
                       {'params': model.module.classifier.parameters(), 'lr': base_lr*10},
                       {'params': model.module.classifier_swap.parameters(), 'lr': base_lr*10},
                       {'params': model.module.Convmask.parameters(), 'lr': base_lr*10},
                      ], lr = base_lr, momentum=0.9)

criterion = CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)
train(cfg,
      model,
      epoch_num=360,
      start_epoch=0,
      optimizer=optimizer,
      criterion=criterion,
      exp_lr_scheduler=exp_lr_scheduler,
      data_set=data_set,
      data_loader=dataloader,
      save_dir=save_dir,
      print_inter=int(numimage/(4*16)),
      val_inter=int(numimage/(16)),)
