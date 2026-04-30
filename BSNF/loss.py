
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import config.config_voc as cfg

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# from skimage.measure import compare_ssim, compare_psnr, compare_mse
# from skimage.metrics import compare_ssim, compare_psnr, compare_mse
# from skimage.metrics import structural_similarity as compare_ssim

import cv2
#from skimage import measure
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

from math import exp

# 计算一维的高斯分布向量
def gaussian(window_size, sigma):

    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # import ipdb
    # ipdb.set_trace()
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
    # import ipdb
    # ipdb.set_trace()
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel) # 高斯滤波 求均值
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel) # 求均值

    mu1_sq = mu1.pow(2) # 平方
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq # var(x) = Var(X)=E[X^2]-E[X]^2
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2 # 协方差

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret



def gaussian_blur2d(img, kernel_size, sigma):
    """辅助函数：高斯模糊"""
    return gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)


class FusionLoss(nn.Module):
    def __init__(self,
                 pyramid_levels=3,
                 edge_threshold=0.1,
                 eps=1e-6,
                 weight_laplacian=1000,
                 weight_grad=100,
                 weight_ssim=10):
        super().__init__()

        self.pyramid_levels = pyramid_levels
        self.edge_threshold = edge_threshold
        self.eps = eps

        # 损失权重
        self.weight_laplacian = weight_laplacian
        self.weight_grad = weight_grad
        self.weight_ssim = weight_ssim

        # 统一的Sobel算子
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))

        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))

    def to_gray(self, img):
        if img.shape[1] == 3:
            return 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        return img

    def compute_gradient(self, img):
        """计算梯度幅值 Eq. 22"""
        img = self.to_gray(img)
        grad_x = F.conv2d(img, self.sobel_x, padding=1)
        grad_y = F.conv2d(img, self.sobel_y, padding=1)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + self.eps)

    def laplacian_pyramid(self, x):
        pyr = []
        current = x
        for _ in range(self.pyramid_levels - 1):
            blurred = gaussian_blur2d(current, kernel_size=(5, 5), sigma=(1.5, 1.5))
            down = F.avg_pool2d(blurred, kernel_size=2, stride=2)
            up = F.interpolate(down, size=current.shape[2:], mode='bilinear', align_corners=False)
            pyr.append(current - up)
            current = down
        pyr.append(current)
        return pyr

    # 【修改位置3】: 新增 Psi 算子方法，严格实现 Eq. 23, 24, 25
    def apply_psi_operator(self, H, grad_H):
        """实现边缘掩码驱动的锐化机制"""
        # Eq. 23: 边缘掩码 Omega
        omega = torch.sigmoid(5 * (grad_H - 0.1))

        # Eq. 24: 锐化图像 (论文中系数固定为 10)
        H_blur = gaussian_blur2d(H, kernel_size=[5, 5], sigma=[1.0, 1.0])
        H_sharp = H + 10.0 * (H - H_blur)

        # Eq. 25: 结合掩码得到最终的高频/中频分量
        psi_H = omega * H_sharp + (1 - omega) * H
        return psi_H

    def cal_laplacian_loss(self, vis, ir, gen):
        """计算拉普拉斯金字塔损失 (频域损失)"""
        if vis.shape[2] <= 5 or vis.shape[3] <= 5:
            return F.smooth_l1_loss((vis + ir) / 2, gen)

        pyr_vis = self.laplacian_pyramid(vis)
        pyr_ir = self.laplacian_pyramid(ir)
        pyr_gen = self.laplacian_pyramid(gen)

        loss = 0
        # 高频/中频层处理
        for i in range(self.pyramid_levels - 1):
            grad_vis = self.compute_gradient(pyr_vis[i])
            grad_ir = self.compute_gradient(pyr_ir[i])
            grad_gen = self.compute_gradient(pyr_gen[i])


            psi_vis = self.apply_psi_operator(pyr_vis[i], grad_vis)
            psi_ir = self.apply_psi_operator(pyr_ir[i], grad_ir)
            psi_gen = self.apply_psi_operator(pyr_gen[i], grad_gen)

            weight = torch.sigmoid(10 * (grad_vis - grad_ir))
            fused = weight * psi_vis + (1 - weight) * psi_ir

            loss += F.smooth_l1_loss(fused, psi_gen)

        fused_low = (pyr_vis[-1] + pyr_ir[-1]) / 2
        loss += F.smooth_l1_loss(fused_low, pyr_gen[-1])

        return loss

    def cal_grad_loss(self, vis, ir, gen):
        """计算梯度损失 Eq. 29 - Eq. 34"""
        vis_grad = self.compute_gradient(vis)
        ir_grad = self.compute_gradient(ir)
        gen_grad = self.compute_gradient(gen)

        # Eq. 29: E_fuse
        target_grad = torch.max(vis_grad, ir_grad)
        # Eq. 30: Delta E
        grad_diff = torch.abs(target_grad - gen_grad)

        # Eq. 31: 边缘掩码 M_grad
        edge_mask = (target_grad > self.edge_threshold).float()

        # Eq. 32: 边缘区域损失 (L1)
        edge_loss = torch.mean(edge_mask * grad_diff)

        # Eq. 33: 非边缘区域损失 (Huber)
        non_edge_mask = 1 - edge_mask
        delta = 0.1
        huber = torch.where(
            grad_diff < delta,
            0.5 * grad_diff ** 2 / delta,
            grad_diff - 0.5 * delta
        )
        non_edge_loss = torch.mean(non_edge_mask * huber)

        # Eq. 34: 最终边缘损失 (lambda = 0.7)
        # return 0.3 * edge_loss + 0.7 * non_edge_loss
        return 0.7 * edge_loss + 0.3 * non_edge_loss

    def cal_ssim_loss(self, vis, ir, gen):
        """计算SSIM损失 Eq. 35"""
        # 假设 ssim 函数返回的是结构相似度标量
        return 1 - abs(ssim(vis, gen)) / 2 - abs(ssim(ir, gen)) / 2

    def generate_mask(self,size,bboxes):

        bboxes[:,:, 0] *= size[1]
        bboxes[:,:, 2] *= size[1]
        bboxes[:,:, 1] *= size[0]
        bboxes[:,:, 3] *= size[0]

        lable = (bboxes[..., 4] != -1)
        mask = torch.zeros(bboxes.shape[0], bboxes.shape[1], size[0], size[1])
        bboxes = bboxes.int() # 确保 bboxes 是整数类型
        y1, x1, y2, x2 = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
        rows = torch.arange(mask.shape[2])[:, None].cuda()  # 纵向索引 (H, 1)
        cols = torch.arange(mask.shape[3]).cuda()
        # 横向索引 (W,)
        x_mask = (rows >= x1[..., None, None]) & (rows < x2[..., None, None])  # (N, M, H, 1)
        y_mask = (cols >= y1[..., None, None]) & (cols < y2[..., None, None])  # (N, M, 1, W)
        mask[...] = x_mask & y_mask  # (N, M, H, W)

        return mask,lable


    def cal_loss_mask(self,image_vis,image_ir,generate_img,mask):
        if mask.sum()==0:
            return 0,0,0,0
        image_y = image_vis.mul(mask)
        image_ir = image_ir.mul(mask)
        #####################添加模板########################
        #####################裁剪图像###############
        nonzero_indices =torch.argwhere(mask > 0)
        if nonzero_indices.size == 0:
            # 如果矩阵中没有非零元素，返回 None 或 -1
            loss_ssim = 0
            loss_grad = 0
            loss_laplacian = 0

        else:
            # 提取 y 和 x 维度的索引
            x_indices, y_indices = nonzero_indices[:, 0], nonzero_indices[:, 1]
            # 第一个和最后一个非 0 值的位置
            first_x, last_x = x_indices.min(), x_indices.max()
            first_y, last_y = y_indices.min(), y_indices.max()
            if last_y == first_y or first_x==last_x:
                # 如果矩阵中没有非零元素，返回 None 或 -1
                loss_ssim = 0
                loss_grad = 0
                loss_laplacian = 0
            else:
                image_y_mask = image_y[:,:,first_x:last_x,first_y:last_y]
                image_ir_mask = image_ir[:,:,first_x:last_x,first_y:last_y]
                generate_img_mask = generate_img[:,:,first_x:last_x,first_y:last_y]

                loss_ssim = self.cal_ssim_loss(image_vis, image_ir, generate_img)
                loss_grad = self.cal_grad_loss(image_y_mask, image_ir_mask, generate_img_mask)
                # loss_Laplacian = self.cal_Laplacian_loss_mask(image_y_mask, image_ir_mask, generate_img_mask)
                loss_laplacian = 0  #小图像块不再考虑金字塔损失


        loss_total = self.weight_laplacian * loss_laplacian + self.weight_grad * loss_grad + self.weight_ssim * loss_ssim
        return loss_total, loss_laplacian, loss_grad, loss_ssim


    def cal_loss(self,image_vis,image_ir,generate_img):
        loss_laplacian = self.cal_laplacian_loss(image_vis, image_ir, generate_img)
        loss_grad = self.cal_grad_loss(image_vis, image_ir, generate_img)
        loss_ssim = self.cal_ssim_loss(image_vis, image_ir, generate_img)
        # Eq. 21: 加权求和
        loss_total = (
                self.weight_laplacian * loss_laplacian +
                self.weight_grad * loss_grad +
                self.weight_ssim * loss_ssim
        )

        return loss_total, loss_laplacian, loss_grad, loss_ssim


    def forward(self, image_vis, image_ir, generate_img, bboxes=None):

        loss_total, loss_Laplacian, loss_grad, loss_ssim = self.cal_loss(image_vis, image_ir, generate_img)

        if cfg.data_type=='MSRS' or cfg.data_type=='TNO' : #这两个数据集没有bbox标签
            return loss_total, loss_Laplacian, loss_grad, loss_ssim

        mask,lable = self.generate_mask(image_vis.shape[2:], bboxes)
        mask = mask.cuda()
        # ed = time.time()
        # print("**.makemask_time:  ", ed - st)
        global_lamda = 0.5
        loss_total_global = global_lamda * loss_total
        loss_Laplacian_global = global_lamda * loss_Laplacian
        loss_grad_global = global_lamda * loss_grad
        loss_ssim_global = global_lamda * loss_ssim

        #所有局部区域的面积大小
        mask_sum_all=0
        for i in range(0,lable.shape[0]):
            for j in range(0, lable.shape[1]):
                if lable[i,j]==True:
                    mask_sum_all += mask[i,j].sum()

        for i in range(0,lable.shape[0]):
            for j in range(0, lable.shape[1]):
                if lable[i,j]==True:
                    # st = time.time()
                    alphy = mask[i,j].sum() / mask_sum_all #根据局部区域的面积大小计算权重
                    loss_total_new,loss_Laplacian_new,loss_grad_new,loss_ssim_new = self.cal_loss_mask(image_vis,image_ir,generate_img,mask[i,j])

                    loss_total_global += (1-global_lamda)*loss_total_new*alphy
                    loss_grad_global += (1-global_lamda)*loss_grad_new*alphy
                    loss_Laplacian_global += (1-global_lamda)*loss_Laplacian_new*alphy
                    loss_ssim_global += (1-global_lamda)*loss_ssim_new*alphy


        return loss_total,loss_Laplacian, loss_grad, loss_ssim

