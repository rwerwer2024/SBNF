# coding:utf-8
import os
import argparse
import time
import numpy as np
import argparse

from tqdm import tqdm
import torch
from torch.autograd import Variable
from PIL import Image
from MyFusionNet import  Bimodal_net
import glob
import cv2


def main():
    fusion_model_path = os.path.join('./weight/MSRS', 'fusion_model.pt')
    saving_path = './Fusion_results/MSRS/'
    os.makedirs(saving_path, exist_ok=True)
    fusionmodel = Bimodal_net()
    fusionmodel.eval()
    fusionmodel.cuda()
    fusionmodel.load_state_dict(torch.load(fusion_model_path))

    print('fusionmodel load done!')
    # ir_path = './test_imgs/ir'
    # vi_path = './test_imgs/vi'

    time_all = []
    with torch.no_grad():
        ir_path = glob.glob(ir_path + '/*')
        vi_path = glob.glob(vi_path + '/*')
        for path1, path2 in zip(tqdm(vi_path), ir_path):
            save_path = saving_path + '/' + path1.strip().split('\\')[-1]

            image_vis = cv2.imread(path1, 1)
            image_ir = cv2.imread(path2, 0)

            image_vis = cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB).transpose(2, 0, 1) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0) / 255.0

            image_vis = image_vis.astype(np.float32)
            image_ir = image_ir.astype(np.float32)

            t_s = time.time()
            #####################格式转换####################
            image_vis = Variable(torch.tensor(image_vis)).cuda()
            image_vis = torch.unsqueeze(image_vis,0)
            image_vis_ycrcb = RGB2YCrCb(image_vis)
            image_ir = Variable(torch.tensor(image_ir)).cuda()
            image_ir = torch.unsqueeze(image_ir, 0)

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

            ######################保存本地，需要RGB转成BGR，进行保存#########################
            for k in range(len(image_vis)):
                image = np.uint8(fusion_image[k, :, :, :].cpu().numpy())
                image = image.squeeze()
                image = image.transpose((1, 2, 0))
                image = Image.fromarray(image)
                image.save(save_path)
                # print('Fusion {0} Sucessfully!'.format(save_path))
            ###############################################

    print('test time {} {} {}!'.format(np.mean(time_all),np.var(time_all),np.std(time_all)))



def YCrCb2RGB(input_im):
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(device)
    temp = (im_flat + bias).mm(mat).to(device)
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

def RGB2YCrCb(input_im):
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).to(device)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test with pytorch')
    parser.add_argument('--batch_size', '-B', type=int, default=4)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    args = parser.parse_args()
    main()
