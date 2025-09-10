from __future__ import print_function

import warnings
from datetime import datetime
import os
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
from info_nce import InfoNCE
from torch.utils.data.dataloader import DataLoader
import numpy as np
from model import SGC
from datesets import get_trainAndtest
from config import class_nums
from config import HyperParams
warnings.filterwarnings("ignore")


def train():
    output_dir = HyperParams['kind'] + '_' + HyperParams['arch'] + '_output'
    try:
        os.stat(output_dir)

    except:
        os.makedirs(output_dir)
    # Data
    trainset, testset = get_trainAndtest()
    trainloader = DataLoader(trainset, batch_size=HyperParams['bs'], shuffle=True, num_workers=6, pin_memory=True)
    testloader = DataLoader(testset, batch_size=HyperParams['bs'], shuffle=False, num_workers=6)

    ####################################################
    print("dataset: ", HyperParams['kind'])
    print("backbone: ", HyperParams['arch'])
    print("trainset: ", len(trainset))
    print("testset: ", len(testset))
    print("classnum: ", class_nums[HyperParams['kind']])
    ####################################################

    net = SGC(class_num=class_nums[HyperParams['kind']], arch=HyperParams['arch'])
    net = net.cuda()

    CELoss = nn.CrossEntropyLoss()
    CRLoss = InfoNCE(temperature=HyperParams['InfoNCE'])

    ########################
    new_params, old_params = net.get_params()
    new_layers_optimizer = optim.SGD(new_params, momentum=0.9, weight_decay=5e-4, lr=0.002)
    old_layers_optimizer = optim.SGD(old_params, momentum=0.9, weight_decay=5e-4, lr=0.0002)
    new_layers_optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(new_layers_optimizer, HyperParams['epoch'], 0)
    old_layers_optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(old_layers_optimizer, HyperParams['epoch'], 0)

    max_val_acc = 0
    if HyperParams['restart']:
        resume_path = f"./{HyperParams['kind']}_resnet50_output/current_model.pth"
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path)
        HyperParams['start_epoch'] = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        new_layers_optimizer.load_state_dict(checkpoint['new_layers_optimizer'])
        old_layers_optimizer.load_state_dict(checkpoint['old_layers_optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume_path, checkpoint['epoch']))

    for epoch in range(HyperParams['start_epoch'], HyperParams['epoch']):
        print('\nEpoch: %d' % epoch)
        start_time = datetime.now()
        print("start time: ", start_time.strftime('%Y-%m-%d-%H:%M:%S'))
        net.train()
        train_loss = 0
        cls_loss = 0
        part_loss = 0
        cr_loss = 0
        kd_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs = inputs.cuda()
            targets = targets.cuda()

            output_1, output_2, output_3, output_concat, output_part, output_f = net(inputs)

            # adjust optimizer lr
            new_layers_optimizer_scheduler.step()
            old_layers_optimizer_scheduler.step()

            # overall update
            loss1 = CELoss(output_1, targets)*2
            loss2 = CELoss(output_2, targets)*2
            loss3 = CELoss(output_3, targets)*2
            concat_loss = CELoss(output_concat, targets)
            loss_part = 0
            loss_cr = 0

            kd_2 = torch.softmax(output_1 / HyperParams['kd_temp'], dim=1)
            kd_3 = torch.softmax(output_2 / HyperParams['kd_temp'], dim=1)
            kd_4 = torch.softmax(output_3 / HyperParams['kd_temp'], dim=1)
            kd1 = CELoss(kd_2, kd_4)
            kd2 = CELoss(kd_3, kd_4)
            for part in output_part:
                loss_part += CELoss(part, targets)
            for i in range(HyperParams['part']):
                loss_cr += CRLoss(output_part[i], output_1, output_f[0])
            for i in range(HyperParams['part'], HyperParams['part'] * 2):
                loss_cr += CRLoss(output_part[i], output_2, output_f[1])
            for i in range(HyperParams['part'] * 2, HyperParams['part'] * 3):
                loss_cr += CRLoss(output_part[i], output_3, output_f[2])

            new_layers_optimizer.zero_grad()
            old_layers_optimizer.zero_grad()

            Lcls = loss1 + loss2 + loss3 + concat_loss
            Lpart = loss_part / (HyperParams['part'] * 2)
            Lcr = loss_cr * 0.1
            Lkd = (kd1 + kd2) * 0.1

            loss = Lcls + Lkd + Lpart + Lcr
            loss.backward()

            new_layers_optimizer.step()
            old_layers_optimizer.step()

            #  training log
            _, predicted = torch.max((output_2+output_3+output_concat+sum(output_part)).data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            train_loss += loss.item()

            cls_loss += Lcls.item()
            part_loss += Lpart.item()
            cr_loss += Lcr.item()
            kd_loss += Lkd.item()

            if batch_idx % 100 == 0:
                print('Step: %d | cls_loss: %.5f | part_loss: %.5f | cr_loss: %.5f | kd_loss: %.5f | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    batch_idx, cls_loss / (batch_idx + 1),
                    part_loss / (batch_idx + 1), cr_loss / (batch_idx + 1), kd_loss / (batch_idx + 1), train_loss / (batch_idx + 1),
                    100. * float(correct) / total, correct, total))

        val_acc = test(net, testloader)
        torch.save({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'new_layers_optimizer': new_layers_optimizer.state_dict(),
            'old_layers_optimizer': old_layers_optimizer.state_dict(),
        }, './' + output_dir + '/current_model.pth')
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'new_layers_optimizer': new_layers_optimizer.state_dict(),
                'old_layers_optimizer': old_layers_optimizer.state_dict(),
            }, './' + output_dir + '/best_model.pth')
        print("best result: ", max_val_acc)
        print("current result: ", val_acc)
        end_time = datetime.now()
        print("end time: ", end_time.strftime('%Y-%m-%d-%H:%M:%S'))


def test(net, testloader):
    net.eval()
    correct_com = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                output_1, output_2, output_3, output_concat, output_part, _ = net(inputs)
                outputs_com = output_1 + output_2 + output_3 + output_concat + sum(output_part)

            _, predicted_com = torch.max(outputs_com.data, 1)
            total += targets.size(0)
            correct_com += predicted_com.eq(targets.data).cpu().sum()
    test_acc_com = 100. * float(correct_com) / total

    return test_acc_com


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    set_seed(666)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train()

