#coding=utf-8
import os
import datetime
import argparse
import logging
import pandas as pd

import torch
import torch.nn as nn
from  torch.nn import CrossEntropyLoss
import torch.utils.data as torchdata
from torchvision import datasets, models
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

from transforms import transforms
from utils.train_model import train
from models.LoadModel import MainModel
from config import LoadConfig, load_data_transformers
from utils.dataset_DCL import collate_fn4train, collate_fn4val, collate_fn4test, collate_fn4backbone, dataset

import pdb

os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# parameters setting
def parse_args():
    parser = argparse.ArgumentParser(description='dcl parameters')
    parser.add_argument('--data', dest='dataset',
                        default='CUB', type=str)
    parser.add_argument('--save', dest='resume',
                        default=None,
                        type=str)
    parser.add_argument('--backbone', dest='backbone',
                        default='resnet50', type=str)
    parser.add_argument('--auto_resume', dest='auto_resume',
                        action='store_true')
    parser.add_argument('--epoch', dest='epoch',
                        default=360, type=int)
    parser.add_argument('--tb', dest='train_batch',
                        default=16, type=int)
    parser.add_argument('--vb', dest='val_batch',
                        default=512, type=int)
    parser.add_argument('--sp', dest='save_point',
                        default=5000, type=int)
    parser.add_argument('--cp', dest='check_point',
                        default=5000, type=int)
    parser.add_argument('--lr', dest='base_lr',
                        default=0.0008, type=float)
    parser.add_argument('--lr_step', dest='decay_step',
                        default=60, type=int)
    parser.add_argument('--cls_lr_ratio', dest='cls_lr_ratio',
                        default=10.0, type=float)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        default=0,  type=int)
    parser.add_argument('--tnw', dest='train_num_workers',
                        default=16, type=int)
    parser.add_argument('--vnw', dest='val_num_workers',
                        default=32, type=int)
    parser.add_argument('--detail', dest='discribe',
                        default='', type=str)
    parser.add_argument('--size', dest='resize_resolution',
                        default=512, type=int)
    parser.add_argument('--crop', dest='crop_resolution',
                        default=448, type=int)
    parser.add_argument('--cls_2', dest='cls_2',
                        action='store_true')
    parser.add_argument('--cls_mul', dest='cls_mul',
                        action='store_true')
    parser.add_argument('--swap_num', default=[7, 7],
                    nargs=2, metavar=('swap1', 'swap2'),
                    type=int, help='specify a range')
    args = parser.parse_args()
    return args

def auto_load_resume(load_dir):
    folders = os.listdir(load_dir)
    date_list = [int(x.split('_')[1].replace(' ',0)) for x in folders]
    choosed = folders[date_list.index(max(date_list))]
    weight_list = os.listdir(os.path.join(load_dir, choosed))
    acc_list = [x[:-4].split('_')[-1] if x[:7]=='weights' else 0 for x in weight_list]
    acc_list = [float(x) for x in acc_list]
    choosed_w = weight_list[acc_list.index(max(acc_list))]
    return os.path.join(load_dir, choosed, choosed_w)


if __name__ == '__main__':
    args = parse_args()
    print(args, flush=True)
    Config = LoadConfig(args, 'train')
    Config.cls_2 = args.cls_2
    Config.cls_2xmul = args.cls_mul
    assert Config.cls_2 ^ Config.cls_2xmul

    transformers = load_data_transformers(args.resize_resolution, args.crop_resolution, args.swap_num)

    # inital dataloader
    train_set = dataset(Config = Config,\
                        anno = Config.train_anno,\
                        common_aug = transformers["common_aug"],\
                        swap = transformers["swap"],\
                        totensor = transformers["train_totensor"],\
                        train = True)

    trainval_set = dataset(Config = Config,\
                        anno = Config.train_anno,\
                        common_aug = transformers["None"],\
                        swap = transformers["None"],\
                        totensor = transformers["val_totensor"],\
                        train = False,
                        train_val = True)

    val_set = dataset(Config = Config,\
                      anno = Config.val_anno,\
                      common_aug = transformers["None"],\
                      swap = transformers["None"],\
                      totensor = transformers["test_totensor"],\
                      test=True)

    dataloader = {}
    dataloader['train'] = torch.utils.data.DataLoader(train_set,\
                                                batch_size=args.train_batch,\
                                                shuffle=True,\
                                                num_workers=args.train_num_workers,\
                                                collate_fn=collate_fn4train if not Config.use_backbone else collate_fn4backbone,
                                                drop_last=True if Config.use_backbone else False,
                                                pin_memory=True)

    setattr(dataloader['train'], 'total_item_len', len(train_set))

    dataloader['trainval'] = torch.utils.data.DataLoader(trainval_set,\
                                                batch_size=args.val_batch,\
                                                shuffle=False,\
                                                num_workers=args.val_num_workers,\
                                                collate_fn=collate_fn4val if not Config.use_backbone else collate_fn4backbone,
                                                drop_last=True if Config.use_backbone else False,
                                                pin_memory=True)

    setattr(dataloader['trainval'], 'total_item_len', len(trainval_set))
    setattr(dataloader['trainval'], 'num_cls', Config.numcls)

    dataloader['val'] = torch.utils.data.DataLoader(val_set,\
                                                batch_size=args.val_batch,\
                                                shuffle=False,\
                                                num_workers=args.val_num_workers,\
                                                collate_fn=collate_fn4test if not Config.use_backbone else collate_fn4backbone,
                                                drop_last=True if Config.use_backbone else False,
                                                pin_memory=True)

    setattr(dataloader['val'], 'total_item_len', len(val_set))
    setattr(dataloader['val'], 'num_cls', Config.numcls)


    cudnn.benchmark = True

    print('Choose model and train set', flush=True)
    model = MainModel(Config)

    # load model
    if (args.resume is None) and (not args.auto_resume):
        print('train from imagenet pretrained models ...', flush=True)
    else:
        if not args.resume is None:
            resume = args.resume
            print('load from pretrained checkpoint %s ...'% resume, flush=True)
        elif args.auto_resume:
            resume = auto_load_resume(Config.save_dir)
            print('load from %s ...'%resume, flush=True)
        else:
            raise Exception("no checkpoints to load")

        model_dict = model.state_dict()
        pretrained_dict = torch.load(resume)
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    print('Set cache dir', flush=True)
    time = datetime.datetime.now()
    filename = '%s_%d%d%d_%s'%(args.discribe, time.month, time.day, time.hour, Config.dataset)
    save_dir = os.path.join(Config.save_dir, filename)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.cuda()
    model = nn.DataParallel(model)

    # optimizer prepare
    if Config.use_backbone:
        ignored_params = list(map(id, model.module.classifier.parameters()))
    else:
        ignored_params1 = list(map(id, model.module.classifier.parameters()))
        ignored_params2 = list(map(id, model.module.classifier_swap.parameters()))
        ignored_params3 = list(map(id, model.module.Convmask.parameters()))

        ignored_params = ignored_params1 + ignored_params2 + ignored_params3
    print('the num of new layers:', len(ignored_params), flush=True)
    base_params = filter(lambda p: id(p) not in ignored_params, model.module.parameters())

    lr_ratio = args.cls_lr_ratio
    base_lr = args.base_lr
    if Config.use_backbone:
        optimizer = optim.SGD([{'params': base_params},
                               {'params': model.module.classifier.parameters(), 'lr': base_lr}], lr = base_lr, momentum=0.9)
    else:
        optimizer = optim.SGD([{'params': base_params},
                               {'params': model.module.classifier.parameters(), 'lr': lr_ratio*base_lr},
                               {'params': model.module.classifier_swap.parameters(), 'lr': lr_ratio*base_lr},
                               {'params': model.module.Convmask.parameters(), 'lr': lr_ratio*base_lr},
                              ], lr = base_lr, momentum=0.9)


    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=0.1)

    # train entry
    train(Config,
          model,
          epoch_num=args.epoch,
          start_epoch=args.start_epoch,
          optimizer=optimizer,
          exp_lr_scheduler=exp_lr_scheduler,
          data_loader=dataloader,
          save_dir=save_dir,
          data_size=args.crop_resolution,
          savepoint=args.save_point,
          checkpoint=args.check_point)


