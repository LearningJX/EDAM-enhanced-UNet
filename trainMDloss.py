import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

# from nets.unet import Unet
from nets.unetSimAM import Unet

# from nets.unet_training import CE_Loss
from nets.unet_training import Dice_loss
# from nets.unet_training import Tversky_loss
from nets.unet_training import FocalLoss
from nets.unet_training import active_contour_loss
from utils.dataloader import DeeplabDataset, deeplab_dataset_collate
from utils.metrics import f_score
import csv
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

outCSV = "./LossCSV/Condition_Test2.csv"  # 将损失写入csv


def writeHead2File(str):
    f = open(outCSV, 'w')
    f.write(str + "\n")
    f.close()


def write2File(strlist):
    f = open(outCSV, 'a+')
    writer = csv.writer(f)
    writer.writerow(strlist)
    f.close()


def resultStr(str1, str2, str3, str4, str5, str6, str7, str8, str9):
    liststr = []
    liststr.append(str(str1))
    liststr.append(str(str2))
    liststr.append(str(str3))
    liststr.append(str(str4))
    liststr.append(str(str5))
    liststr.append(str(str6))
    liststr.append(str(str7))
    liststr.append(str(str8))
    liststr.append(str(str9))
    return liststr


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_epoch(net, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda):
    net = net.train()

    total_loss1 = 0
    total_loss2 = 0
    total_loss3 = 0
    total_loss = 0
    total_f_score = 0

    val_total_loss1 = 0
    val_total_loss2 = 0
    val_total_loss3 = 0
    val_total_loss = 0
    val_total_f_score = 0
    start_time = time.time()
    loss_fn = FocalLoss(alpha=0.9, gamma=0.1, reduce=True)
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            imgs, pngs, labels = batch

            with torch.no_grad():
                imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor))
                pngs = Variable(torch.from_numpy(pngs).type(torch.FloatTensor)).long()
                labels = Variable(torch.from_numpy(labels).type(torch.FloatTensor))
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()

            optimizer.zero_grad()
            outputs = net(imgs)
            # loss1    = CE_Loss(outputs, pngs, num_classes = NUM_CLASSES)
            loss1 = loss_fn(outputs, pngs)
            if dice_loss:
                # loss2 = loss1
                loss2 = active_contour_loss(outputs, labels)
                loss3 = Dice_loss(outputs, labels)
                loss = loss1 + loss2 + loss3

            with torch.no_grad():
                # -------------------------------#
                #   计算f_score
                # -------------------------------#
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            total_loss3 += loss3.item()
            total_loss += loss.item()
            total_f_score += _f_score.item()

            waste_time = time.time() - start_time
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'FL_loss': total_loss1 / (iteration + 1),
                                'GC_loss': total_loss2 / (iteration + 1),
                                'Dice_loss': total_loss3 / (iteration + 1),
                                'f_score': total_f_score / (iteration + 1),
                                's/step': waste_time,
                                'lr': get_lr(optimizer)})
            pbar.update(1)

            start_time = time.time()

    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            imgs, pngs, labels = batch
            with torch.no_grad():
                imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor))
                pngs = Variable(torch.from_numpy(pngs).type(torch.FloatTensor)).long()
                labels = Variable(torch.from_numpy(labels).type(torch.FloatTensor))
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()

                outputs = net(imgs)
                # val_loss1 = CE_Loss(outputs, pngs, num_classes = NUM_CLASSES)
                val_loss1 = loss_fn(outputs, pngs)
                if dice_loss:
                    # val_loss2 = val_loss1
                    # val_loss2 = Tversky_loss(outputs, labels, alpha=2, beta=2, sigma=0)
                    val_loss2 = active_contour_loss(outputs, labels)
                    val_loss3 = Dice_loss(outputs, labels)
                    val_loss = val_loss1 + val_loss2 + val_loss3
                # -------------------------------#
                #   计算f_score
                # -------------------------------#
                _f_score = f_score(outputs, labels)

                val_total_loss1 += val_loss1.item()
                val_total_loss2 += val_loss2.item()
                val_total_loss3 += val_loss3.item()
                val_total_loss += val_loss.item()
                val_total_f_score += _f_score.item()

            pbar.set_postfix(**{'val_total_loss': val_total_loss / (iteration + 1),
                                'val_Focal_loss': val_total_loss1 / (iteration + 1),
                                'val_GC_loss': val_total_loss2 / (iteration + 1),
                                'val_Dice_loss': val_total_loss3 / (iteration + 1),
                                'f_score': val_total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print(
        'Train Total Loss: %.4f||Train FL_Loss: %.4f||Train GC_Loss: %.4f||Train Dice_loss: %.4f ||Val Loss: %.4f||Val FL_loss: %.4f||Val GC_loss: %.4f||Val Dice_loss: %.4f' % (
            total_loss / (epoch_size + 1), total_loss1 / (epoch_size + 1), total_loss2 / (epoch_size + 1), total_loss3 / (epoch_size + 1),
            val_total_loss / (epoch_size_val + 1), val_total_loss1 / (epoch_size_val + 1), val_total_loss2 / (epoch_size_val + 1),
            val_total_loss3 / (epoch_size_val + 1)))

    print('Saving state, iter:', str(epoch + 1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
        (epoch + 1), total_loss / (epoch_size + 1), val_total_loss / (epoch_size_val + 1)))

    write2File(
        resultStr(epoch, total_loss / (epoch_size + 1), total_loss1 / (epoch_size + 1), total_loss2 / (epoch_size + 1), total_loss3 / (epoch_size + 1),
                  val_total_loss / (epoch_size_val + 1), val_total_loss1 / (epoch_size_val + 1), val_total_loss2 / (epoch_size_val + 1),
                  val_total_loss3 / (epoch_size_val + 1)))


if __name__ == "__main__":
    time1 = time.time()
    fileHead = "Epoch,Train_total_loss,Train_FL_loss,Train_GC_loss,Train_Dice_loss,Val_total_loss,Val_FL_loss,Val_GC_loss,Val_Dice_loss"
    writeHead2File(fileHead)
    log_dir = "logs/"
    # ------------------------------#
    #   输入图片的大小
    # ------------------------------#
    inputs_size = [512, 512, 3]
    # ---------------------#
    #   分类个数+1
    #   2+1
    # ---------------------#
    NUM_CLASSES = 5
    # --------------------------------------------------------------------#
    #   建议选项：
    #   种类少（几类）时，设置为True
    #   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
    #   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
    # ---------------------------------------------------------------------#
    dice_loss = True
    # -------------------------------#
    #   主干网络预训练权重的使用
    # -------------------------------#
    pretrained = True
    # -------------------------------#
    #   Cuda的使用
    # -------------------------------#
    Cuda = True

    model = Unet(num_classes=NUM_CLASSES, in_channels=inputs_size[-1], pretrained=pretrained).train()

    # -------------------------------------------#
    #   权值文件的下载请看README
    #   权值和主干特征提取网络一定要对应
    # -------------------------------------------#
    model_path = r"model_data/unet_voc.pth"
    # model_path = r"model_data/Condition44Epoch100.pth"
    print('Loading weights into state dict...{0}'.format(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    # 打开数据集的txt
    with open(r"VOCdevkit/VOC2007/ImageSets4/Segmentation4/train.txt", "r") as f:
        train_lines = f.readlines()

    # 打开数据集的txt
    with open(r"VOCdevkit/VOC2007/ImageSets4/Segmentation4/val.txt", "r") as f:
        val_lines = f.readlines()

    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Interval_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
    if True:
        lr = 1e-4
        Init_Epoch = 0
        Interval_Epoch = 0
        Batch_size = 8

        optimizer = optim.Adam(model.parameters(), lr, weight_decay=0)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.92)

        train_dataset = DeeplabDataset(train_lines, inputs_size, NUM_CLASSES, True)
        val_dataset = DeeplabDataset(val_lines, inputs_size, NUM_CLASSES, False)
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=2, pin_memory=True,
                         drop_last=True, collate_fn=deeplab_dataset_collate)
        gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=2, pin_memory=True,
                             drop_last=True, collate_fn=deeplab_dataset_collate)

        epoch_size = max(1, len(train_lines) // Batch_size)
        epoch_size_val = max(1, len(val_lines) // Batch_size)

        for param in model.vgg.parameters():
            param.requires_grad = False

        for epoch in range(Init_Epoch, Interval_Epoch):
            fit_one_epoch(model, epoch, epoch_size, epoch_size_val, gen, gen_val, Interval_Epoch, Cuda)
            lr_scheduler.step()

    if True:
        lr = 1e-4
        Interval_Epoch = 0
        Epoch = 100
        Batch_size = 6

        optimizer = optim.Adam(model.parameters(), lr, weight_decay=0)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.92)

        train_dataset = DeeplabDataset(train_lines, inputs_size, NUM_CLASSES, True)
        val_dataset = DeeplabDataset(val_lines, inputs_size, NUM_CLASSES, False)
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                         drop_last=True, collate_fn=deeplab_dataset_collate)
        gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=deeplab_dataset_collate)

        epoch_size = max(1, len(train_lines) // Batch_size)
        epoch_size_val = max(1, len(val_lines) // Batch_size)

        for param in model.vgg.parameters():
            param.requires_grad = True

        for epoch in range(Interval_Epoch, Epoch):
            fit_one_epoch(model, epoch, epoch_size, epoch_size_val, gen, gen_val, Epoch, Cuda)
            lr_scheduler.step()

    time2 = time.time()
    time3 = time2 - time1
    print('100 epoch costing time{0}h'.format(time3 / 3600))
