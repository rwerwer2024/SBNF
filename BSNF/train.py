#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os

from MyFusionNet import  Bimodal_net
from TaskFusion_dataset import Fusion_dataset
import argparse
import datetime
import logging
from logger import setup_logger
import numpy as np
from loss import FusionLoss
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
import config.config_voc as cfg
import warnings
from torch.autograd import Variable
from PIL import Image
import utils.cosine_lr_scheduler as cosine_lr_scheduler
import time
warnings.filterwarnings('ignore')


def parse_args():
    parse = argparse.ArgumentParser()
    return parse.parse_args()

def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def run_fusion(fusion_model_path,fused_dir):
    fusion_model_path = fusion_model_path
    fused_dir = fused_dir
    os.makedirs(fused_dir, mode=0o777, exist_ok=True)
    fusionmodel = Bimodal_netgsb()
    fusionmodel.eval()
    fusionmodel.cuda()
    fusionmodel.load_state_dict(torch.load(fusion_model_path))
    print('loading model done!')

    test_dataset = Fusion_dataset('test',datatype=cfg.data_type)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    time_all = []

    with torch.no_grad():
        for it, (images_vis, images_ir,_,name) in enumerate(test_loader):
            t_s = time.time()
            #####################格式转换####################
            image_vis = Variable(images_vis).cuda()
            image_vis_ycrcb = RGB2YCrCb(image_vis)
            image_ir = Variable(images_ir).cuda()

            #####################只取彩色图像YCrCb的Y通道进行融合####################
            image_vis_y = image_vis_ycrcb[:, :1]
            logits = fusionmodel(image_vis_y, image_ir)

            #####################Y通道融合后恢复彩色图像YCrCb####################
            fusion_ycrcb = torch.cat(
                (logits, image_vis_ycrcb[:, 1:2, :, :],
                 image_vis_ycrcb[:, 2:, :, :]),
                dim=1,
            )
            fusion_image = YCrCb2RGB(fusion_ycrcb)

            #####################后处理####################
            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)

            fusion_image = (fusion_image - torch.min(fusion_image)) / (
                    torch.max(fusion_image) - torch.min(fusion_image)
            )
            fusion_image = torch.round(255.0 * fusion_image).float()

            #####################融合完成####################
            t_e = time.time()
            time_all.append(t_e - t_s)
            # for k in range(len(image_vis)):
            #     print('test img {} !'.format(name[k]))
            ######################保存本地，需要RGB转成BGR，进行保存#########################
            for k in range(len(image_vis)):
                image = np.uint8(fusion_image[k, :, :, :].cpu().numpy())
                image = image.squeeze()
                image = image.transpose((1, 2, 0))
                image = Image.fromarray(image)
                save_path = os.path.join(fused_dir, name[k])
                image.save(save_path)
                # print('Fusion {0} Sucessfully!'.format(save_path))
            ###############################################


    print('test time {} {} {}!'.format(np.mean(time_all),np.var(time_all),np.std(time_all)))

def train_end2end_fusion( logger=None):

    batch_size = 1
    total_epoch = 10
    writer = SummaryWriter(log_dir='./logs/')

    if logger == None:
        logger = logging.getLogger()#

    fusionmodel = Bimodal_net()

    fusionmodel.cuda()
    fusionmodel.train()
    criteria_fusion = FusionLoss().cuda()

    train_dataset = Fusion_dataset('train',datatype=cfg.data_type)
    print("the training dataset is length:{}".format(train_dataset.length))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = torch.optim.Adam(fusionmodel.parameters(), lr=1e-3, weight_decay=1e-4)

    scheduler = cosine_lr_scheduler.CosineDecayLR(optimizer, T_max=total_epoch * train_dataset.len,
                                                  lr_init=1e-3,
                                                  lr_min=1e-5,
                                                  warmup=0*len(train_dataset))

    train_loader.n_iter = len(train_loader)
    st = glob_st = time.time()
    current_max_norm=1.0


    for epo in range(0, total_epoch):
        print('\n| epo #%s begin...' % epo)

        for it, (image_vis, image_ir,image_label, image_name) in enumerate(train_loader):
            iter = epo * len(train_dataset) + it

            image_vis = Variable(image_vis).cuda()
            bboxes = Variable(image_label).cuda()
            image_vis_ycrcb = RGB2YCrCb(image_vis)
            image_ir = Variable(image_ir).cuda()
            image_vis_y = image_vis_ycrcb[:, :1]
            logits = fusionmodel(image_vis_y, image_ir)

            loss_fusion,loss_Laplacian, loss_grad, loss_ssim = criteria_fusion(
                image_vis_y, image_ir, logits, bboxes
            )

            loss_fusion.backward()

            if ((it + 1) % 32) == 0:
                total_norm = torch.nn.utils.clip_grad_norm_(fusionmodel.parameters(), max_norm=current_max_norm,
                                                            norm_type=2)
                if total_norm > current_max_norm:
                    current_max_norm *= 1.1  # 逐渐放宽限制
                else:
                    current_max_norm *= 0.9  # 逐渐收紧限制

                optimizer.step()  # 更新参数
                optimizer.zero_grad()
                scheduler.step(iter)

                ed = time.time()

                t_intv, glob_t_intv = ed - st, ed - glob_st
                now_it = train_loader.n_iter * epo + it + 1
                eta = int((train_loader.n_iter * total_epoch - now_it)
                          * (glob_t_intv / (now_it)))
                eta = str(datetime.timedelta(seconds=eta))
                st = ed

                msg = ', '.join(
                    [
                        'step: {it}/{max_it}',
                        'fusion_lr: {fusion_lr:.5f}',
                        'loss_Laplacian: {loss_Laplacian:.4f}',
                        'loss_grad: {loss_grad:.4f}',
                        'loss_ssim: {loss_ssim:.4f}',
                        'loss_fusion: {loss_fusion:.4f}',
                        'eta: {eta}',
                        'time: {time:.4f}',
                    ]
                ).format(
                    it=now_it,
                    max_it=train_loader.n_iter * total_epoch,
                    fusion_lr=optimizer.param_groups[0]['lr'],
                    loss_Laplacian=loss_Laplacian.item(),
                    loss_grad=loss_grad.item(),
                    loss_ssim=loss_ssim.item(),
                    loss_fusion=loss_fusion.item(),
                    time=t_intv,
                    eta=eta,
                )

                writer.add_scalar('data/lr', optimizer.param_groups[0]['lr'], iter)
                # writer.add_scalar('data/loss_in', loss_in.data.cpu().numpy(), it)
                writer.add_scalar('data/loss_Laplacian', loss_Laplacian.data.cpu().numpy(), iter)
                writer.add_scalar('data/loss_grad', loss_grad.data.cpu().numpy(), iter)
                writer.add_scalar('data/loss_ssim', loss_ssim.data.cpu().numpy(), iter)
                writer.add_scalar('data/loss_fusion', loss_fusion.data.cpu().numpy(), iter)

                logger.info(msg)

        fusion_model_file = os.path.join('weight/MSRS/', 'fusion_model_backup_epoch%g.pt' % epo)
        torch.save(fusionmodel.state_dict(), fusion_model_file)

    logger.info('\n')



############################查看日志##################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with pytorch')

    parser.add_argument('--batch_size', '-B', type=int, default=4)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    args = parser.parse_args()
    logpath='./logs'
    logger = logging.getLogger()
    setup_logger(logpath)

    train_end2end_fusion(logger)

    # fusion_model_file = os.path.join('./weight/MSRS/fusion_model')
    # fused_dir = './Fusion_results/MSRS'
    # run_fusion(fusion_model_file, fused_dir)


