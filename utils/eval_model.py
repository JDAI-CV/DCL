#coding=utf8
from __future__ import print_function, division
import os,time,datetime
import numpy as np
import datetime
from math import ceil

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from utils.utils import LossRecord

import pdb

def dt():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

def eval_turn(model, data_loader, val_version, epoch_num, log_file):

    model.train(False)

    val_corrects1 = 0
    val_corrects2 = 0
    val_corrects3 = 0
    val_size = data_loader.__len__()
    item_count = data_loader.total_item_len
    t0 = time.time()
    get_l1_loss = nn.L1Loss()
    get_ce_loss = nn.CrossEntropyLoss()

    val_batch_size = data_loader.batch_size
    val_epoch_step = data_loader.__len__()
    num_cls = data_loader.num_cls

    val_loss_recorder = LossRecord(val_batch_size)
    val_celoss_recorder = LossRecord(val_batch_size)
    print('evaluating %s ...'%val_version, flush=True)
    with torch.no_grad():
        for batch_cnt_val, data_val in enumerate(data_loader):
            inputs = Variable(data_val[0].cuda())
            labels = Variable(torch.from_numpy(np.array(data_val[1])).long().cuda())
            outputs = model(inputs)
            loss = 0

            ce_loss = get_ce_loss(outputs[0], labels).item()
            loss += ce_loss

            val_loss_recorder.update(loss)
            val_celoss_recorder.update(ce_loss)

            if outputs[1].size(1) != 2:
                outputs_pred = outputs[0] + outputs[1][:,0:num_cls] + outputs[1][:,num_cls:2*num_cls]
            else:
                outputs_pred = outputs[0]
            top3_val, top3_pos = torch.topk(outputs_pred, 3)

            print('{:s} eval_batch: {:-6d} / {:d} loss: {:8.4f}'.format(val_version, batch_cnt_val, val_epoch_step, loss), flush=True)

            batch_corrects1 = torch.sum((top3_pos[:, 0] == labels)).data.item()
            val_corrects1 += batch_corrects1
            batch_corrects2 = torch.sum((top3_pos[:, 1] == labels)).data.item()
            val_corrects2 += (batch_corrects2 + batch_corrects1)
            batch_corrects3 = torch.sum((top3_pos[:, 2] == labels)).data.item()
            val_corrects3 += (batch_corrects3 + batch_corrects2 + batch_corrects1)

        val_acc1 = val_corrects1 / item_count
        val_acc2 = val_corrects2 / item_count
        val_acc3 = val_corrects3 / item_count

        log_file.write(val_version  + '\t' +str(val_loss_recorder.get_val())+'\t' + str(val_celoss_recorder.get_val()) + '\t' + str(val_acc1) + '\t' + str(val_acc3) + '\n')

        t1 = time.time()
        since = t1-t0
        print('--'*30, flush=True)
        print('% 3d %s %s %s-loss: %.4f ||%s-acc@1: %.4f %s-acc@2: %.4f %s-acc@3: %.4f ||time: %d' % (epoch_num, val_version, dt(), val_version, val_loss_recorder.get_val(init=True), val_version, val_acc1,val_version, val_acc2, val_version, val_acc3, since), flush=True)
        print('--' * 30, flush=True)

    return val_acc1, val_acc2, val_acc3

