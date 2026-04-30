
import numpy as np
import cv2
import sklearn.metrics as skm
from scipy.signal import convolve2d
import math
from skimage.metrics import structural_similarity as ssim
from piq import multi_scale_ssim,vif_p
import torch
from skimage.filters import sobel
from sklearn.metrics import mutual_info_score
import torch
import torch.nn as nn

from scipy.ndimage import gaussian_filter

import piq

def fsim(reference, distorted):
    reference = torch.tensor(np.array(reference)).unsqueeze(0).unsqueeze(0).float()/255.0
    distorted = torch.tensor(np.array(distorted)).unsqueeze(0).unsqueeze(0).float()/255.0
    fsim_score = piq.fsim(reference, distorted,chromatic=False)
    return fsim_score

def fsim_bk(reference, distorted):
    # 转换为灰度图像
    if len(reference.shape) == 3:
        reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    if len(distorted.shape) == 3:
        distorted = cv2.cvtColor(distorted, cv2.COLOR_BGR2GRAY)

    # 计算相位一致性（Phase Congruency, PC）
    def phase_congruency(img):
        # 使用高斯滤波模拟多尺度分析（简化版本）
        pc = gaussian_filter(img.astype(float), sigma=1)
        return pc

    pc_ref = phase_congruency(reference)
    pc_dist = phase_congruency(distorted)

    # 计算梯度幅度（Gradient Magnitude, GM）
    sobel_x = cv2.Sobel(reference, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(reference, cv2.CV_64F, 0, 1, ksize=3)
    gm_ref = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    sobel_x = cv2.Sobel(distorted, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(distorted, cv2.CV_64F, 0, 1, ksize=3)
    gm_dist = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # 计算相位一致性相似性（S_pc）
    S_pc = (2 * pc_ref * pc_dist + 1e-8) / (pc_ref ** 2 + pc_dist ** 2 + 1e-8)

    # 计算梯度幅度相似性（S_gm）
    S_gm = (2 * gm_ref * gm_dist + 1e-8) / (gm_ref ** 2 + gm_dist ** 2 + 1e-8)

    # 综合相似性（FSIM）
    alpha = 0.5  # 权重参数（根据论文调整）
    fsim_score = np.mean(S_pc * (S_gm ** alpha))

    return fsim_score


def calculate_cv(image):
    """
    计算图像的变异系数（CV）

    参数：
        image: 输入图像（灰度或彩色，numpy数组）

    返回：
        cv_value: 变异系数（百分比）
    """
    # 转换为灰度图（若为彩色）
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算均值和标准差
    mu = np.mean(image)
    sigma = np.std(image)

    # 避免除零错误
    if mu == 0:
        return 0.0

    # 计算CV
    cv_value = (sigma / mu) * 100
    return cv_value

def correlation_coefficient(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mean1 = np.mean(img1)
    mean2 = np.mean(img2)
    numerator = np.sum((img1 - mean1) * (img2 - mean2))
    denominator = np.sqrt(np.sum((img1 - mean1)**2) * np.sum((img2 - mean2)**2))
    return numerator / denominator if denominator != 0 else 0

def contrast_balance(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_normalized = hist / hist.sum()
    entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
    return entropy

def cross_entropy_loss(pred, target):
    criterion = nn.CrossEntropyLoss()
    return criterion(pred, target)

def calculate_rmse(original_img, processed_img):
    """
    计算两幅图像之间的均方根误差（RMSE）

    参数：
        original_img: 原始图像（numpy数组）
        processed_img: 处理后的图像（numpy数组，需与原始图像尺寸相同）

    返回：
        rmse: 均方根误差值
    """
    # 确保输入图像的数据类型为浮点型以避免溢出
    original = original_img.astype(np.float64)
    processed = processed_img.astype(np.float64)

    # 计算均方误差（MSE）
    mse = np.mean((original - processed) ** 2)

    # 计算均方根误差（RMSE）
    rmse = np.sqrt(mse)

    return rmse

def calculate_nabf(original_img, fused_img):
    """
    计算Nabf指标（归一化绝对亮度保真度）

    参数：
        original_img: 原始图像（单通道灰度图, uint8类型）
        fused_img: 融合后的图像（单通道灰度图, uint8类型）

    返回：
        nabf: Nabf值（值越小表示亮度保真度越高）
    """
    # 将图像转换为浮点型 [0,1] 范围
    original = original_img.astype(np.float32) / 255.0
    fused = fused_img.astype(np.float32) / 255.0

    # 计算绝对亮度差异
    abs_diff = np.abs(original - fused)

    # 计算Nabf分子（差异平方和）
    numerator = np.sum(abs_diff ** 2)

    # 计算Nabf分母（原始图像亮度平方和 + 融合图像亮度平方和）
    denominator = np.sum(original ** 2) + np.sum(fused ** 2)

    # 避免除以零
    if denominator == 0:
        return float('inf')

    # 计算Nabf
    nabf = numerator / denominator

    return nabf

def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':  
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img

class Evaluator():
    @classmethod
    def input_check(cls, imgF, imgA=None, imgB=None): 
        if imgA is None:
            assert type(imgF) == np.ndarray, 'type error'
            assert len(imgF.shape) == 2, 'dimension error'
        else:
            assert type(imgF) == type(imgA) == type(imgB) == np.ndarray, 'type error'
            assert imgF.shape == imgA.shape == imgB.shape, 'shape error'
            assert len(imgF.shape) == 2, 'dimension error'

    @classmethod
    def EN(cls, img):  # entropy
        cls.input_check(img)
        a = np.uint8(np.round(img)).flatten()
        h = np.bincount(a) / a.shape[0]
        return -sum(h * np.log2(h + (h == 0)))

    @classmethod
    def SD(cls, img):
        cls.input_check(img)
        return np.std(img)

    @classmethod
    def SF(cls, img):
        cls.input_check(img)
        return np.sqrt(np.mean((img[:, 1:] - img[:, :-1]) ** 2) + np.mean((img[1:, :] - img[:-1, :]) ** 2))

    import numpy as np
    import cv2


    @classmethod
    def AG(cls, img):  # Average gradient
        cls.input_check(img)
        Gx, Gy = np.zeros_like(img), np.zeros_like(img)

        Gx[:, 0] = img[:, 1] - img[:, 0]
        Gx[:, -1] = img[:, -1] - img[:, -2]
        Gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2

        Gy[0, :] = img[1, :] - img[0, :]
        Gy[-1, :] = img[-1, :] - img[-2, :]
        Gy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2
        return np.mean(np.sqrt((Gx ** 2 + Gy ** 2) / 2))

    @classmethod
    def MI(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return skm.mutual_info_score(image_F.flatten(), image_A.flatten()) + skm.mutual_info_score(image_F.flatten(),
                                                                                                   image_B.flatten())

    @classmethod
    def MSE(cls, image_F, image_A, image_B):  # MSE
        cls.input_check(image_F, image_A, image_B)
        return (np.mean((image_A - image_F) ** 2) + np.mean((image_B - image_F) ** 2)) / 2

    @classmethod
    def Nabf(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return calculate_nabf(image_F, image_A)/2 + calculate_nabf(image_F, image_B)/2

    @classmethod
    def CC(cls, image_F, image_A, image_B):
        """
        改进的CC计算，标准化实现
        """
        eps = 1e-10
        F, A, B = map(np.float32, [image_F, image_A, image_B])

        def _pearson(x, y):
            xm, ym = x - np.mean(x), y - np.mean(y)
            return np.sum(xm * ym) / (np.sqrt(np.sum(xm ** 2) * np.sum(ym ** 2)) + eps)

        return (_pearson(A, F) + _pearson(B, F)) / 2

    @classmethod
    def CC_1(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        rAF = np.sum((image_A - np.mean(image_A)) * (image_F - np.mean(image_F))) / np.sqrt(
            (np.sum((image_A - np.mean(image_A)) ** 2)) * (np.sum((image_F - np.mean(image_F)) ** 2)))
        rBF = np.sum((image_B - np.mean(image_B)) * (image_F - np.mean(image_F))) / np.sqrt(
            (np.sum((image_B - np.mean(image_B)) ** 2)) * (np.sum((image_F - np.mean(image_F)) ** 2)))
        return (rAF + rBF) / 2

    @classmethod
    def PSNR(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return 10 * np.log10(np.max(image_F) ** 2 / cls.MSE(image_F, image_A, image_B))


    @classmethod
    def SCD_1(cls, image_F, image_A, image_B):
        """
        改进的SCD计算，严格遵循Aslantas 2015定义
        """
        eps = 1e-10
        F, A, B = map(np.float32, [image_F, image_A, image_B])
        D_FB, D_FA = F - B, F - A

        def _pearson(x, y):
            xm, ym = x - np.mean(x), y - np.mean(y)
            return np.sum(xm * ym) / (np.sqrt(np.sum(xm ** 2) * np.sum(ym ** 2)) + eps)

        return _pearson(A, D_FB) + _pearson(B, D_FA)

    @classmethod
    def SCD(cls, image_F, image_A, image_B): # The sum of the correlations of differences
        cls.input_check(image_F, image_A, image_B)
        imgF_A = image_F - image_A
        imgF_B = image_F - image_B
        corr1 = np.sum((image_A - np.mean(image_A)) * (imgF_B - np.mean(imgF_B))) / np.sqrt(
            (np.sum((image_A - np.mean(image_A)) ** 2)) * (np.sum((imgF_B - np.mean(imgF_B)) ** 2)))
        corr2 = np.sum((image_B - np.mean(image_B)) * (imgF_A - np.mean(imgF_A))) / np.sqrt(
            (np.sum((image_B - np.mean(image_B)) ** 2)) * (np.sum((imgF_A - np.mean(imgF_A)) ** 2)))
        return corr1 + corr2


    @classmethod
    def NCC(cls, image_F, image_A, image_B):

        ncc_AF = cls.cal_NCC(image_F, image_A)  # 融合图与A的非线性相关
        ncc_BF = cls.cal_NCC(image_F, image_B)

        return (ncc_AF + ncc_BF)/2

    @classmethod
    def cal_NCC(cls,image_X, image_Y, sigma=1.0, eps=1e-10):
        """
        非线性相关系数 (NCC)
        参数格式与SCD/CC保持一致:
            image_X: 图像1 (uint8灰度图, 0-255)
            image_Y: 图像2 (uint8灰度图, 0-255)
            sigma: 高斯核带宽(默认1.0)
            eps: 防除零常数
        返回:
            ncc_score: [-1, 1]之间的非线性相关系数
        """
        # 输入验证（与SCD/CC保持一致）
        assert image_X.shape == image_Y.shape, "图像尺寸必须相同"

        # 转换为float32并归一化到[0,1]（与SCD/CC预处理一致）
        X = image_X.astype(np.float32) / 255.0
        Y = image_Y.astype(np.float32) / 255.0

        # 高斯核函数（非线性变换核心）
        def _gaussian_kernel(x, mu):
            return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

        # 计算核变换后的向量
        k_X = _gaussian_kernel(X, np.mean(X))
        k_Y = _gaussian_kernel(Y, np.mean(Y))

        # 计算NCC（公式与Pearson类似，但作用在核空间）
        cov = np.sum((k_X - np.mean(k_X)) * (k_Y - np.mean(k_Y)))
        std_X = np.sqrt(np.sum((k_X - np.mean(k_X)) ** 2))
        std_Y = np.sqrt(np.sum((k_Y - np.mean(k_Y)) ** 2))

        return cov / (std_X * std_Y + eps)

    @classmethod
    def VIFF(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return cls.compare_viff(image_A, image_F)+cls.compare_viff(image_B, image_F)

    @classmethod
    def compare_viff(cls,ref, dist): # viff of a pair of pictures
        sigma_nsq = 2
        eps = 1e-10

        num = 0.0
        den = 0.0
        for scale in range(1, 5):

            N = 2 ** (4 - scale + 1) + 1
            sd = N / 5.0

            # Create a Gaussian kernel as MATLAB's
            m, n = [(ss - 1.) / 2. for ss in (N, N)]
            y, x = np.ogrid[-m:m + 1, -n:n + 1]
            h = np.exp(-(x * x + y * y) / (2. * sd * sd))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            sumh = h.sum()
            if sumh != 0:
                win = h / sumh

            if scale > 1:
                ref = convolve2d(ref, np.rot90(win, 2), mode='valid')
                dist = convolve2d(dist, np.rot90(win, 2), mode='valid')
                ref = ref[::2, ::2]
                dist = dist[::2, ::2]

            mu1 = convolve2d(ref, np.rot90(win, 2), mode='valid')
            mu2 = convolve2d(dist, np.rot90(win, 2), mode='valid')
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = convolve2d(ref * ref, np.rot90(win, 2), mode='valid') - mu1_sq
            sigma2_sq = convolve2d(dist * dist, np.rot90(win, 2), mode='valid') - mu2_sq
            sigma12 = convolve2d(ref * dist, np.rot90(win, 2), mode='valid') - mu1_mu2

            sigma1_sq[sigma1_sq < 0] = 0
            sigma2_sq[sigma2_sq < 0] = 0

            g = sigma12 / (sigma1_sq + eps)
            sv_sq = sigma2_sq - g * sigma12

            g[sigma1_sq < eps] = 0
            sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
            sigma1_sq[sigma1_sq < eps] = 0

            g[sigma2_sq < eps] = 0
            sv_sq[sigma2_sq < eps] = 0

            sv_sq[g < 0] = sigma2_sq[g < 0]
            g[g < 0] = 0
            sv_sq[sv_sq <= eps] = eps

            num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
            den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

        vifp = num / den

        if np.isnan(vifp):
            return 1.0
        else:
            return vifp

    @classmethod
    def Qabf(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        gA, aA = cls.Qabf_getArray(image_A)
        gB, aB = cls.Qabf_getArray(image_B)
        gF, aF = cls.Qabf_getArray(image_F)
        QAF = cls.Qabf_getQabf(aA, gA, aF, gF)
        QBF = cls.Qabf_getQabf(aB, gB, aF, gF)

        # 计算QABF
        deno = np.sum(gA + gB)
        nume = np.sum(np.multiply(QAF, gA) + np.multiply(QBF, gB))
        return nume / deno

    @classmethod
    def Qabf_getArray(cls,img):
        # Sobel Operator Sobel
        h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
        h2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).astype(np.float32)
        h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)

        SAx = convolve2d(img, h3, mode='same')
        SAy = convolve2d(img, h1, mode='same')
        gA = np.sqrt(np.multiply(SAx, SAx) + np.multiply(SAy, SAy))
        aA = np.zeros_like(img)
        aA[SAx == 0] = math.pi / 2
        aA[SAx != 0]= np.arctan(SAy[SAx != 0] / SAx[SAx != 0])
        return gA, aA

    @classmethod
    def Qabf_getQabf(cls,aA, gA, aF, gF):
        L = 1
        Tg = 0.9994
        kg = -15
        Dg = 0.5
        Ta = 0.9879
        ka = -22
        Da = 0.8
        GAF,AAF,QgAF,QaAF,QAF = np.zeros_like(aA),np.zeros_like(aA),np.zeros_like(aA),np.zeros_like(aA),np.zeros_like(aA)
        GAF[gA>gF]=gF[gA>gF]/gA[gA>gF]
        GAF[gA == gF] = gF[gA == gF]
        GAF[gA <gF] = gA[gA<gF]/gF[gA<gF]
        AAF = 1 - np.abs(aA - aF) / (math.pi / 2)
        QgAF = Tg / (1 + np.exp(kg * (GAF - Dg)))
        QaAF = Ta / (1 + np.exp(ka * (AAF - Da)))
        QAF = QgAF* QaAF
        return QAF

    @classmethod
    def SSIM(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return (abs(ssim(image_F,image_A, data_range=255))+abs(ssim(image_F,image_B, data_range=255)))/2

    @classmethod
    def MSSIM(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        image_F = torch.from_numpy(image_F).unsqueeze(0).unsqueeze(0) / 255.0
        image_A = torch.from_numpy(image_A).unsqueeze(0).unsqueeze(0) / 255.0
        image_B = torch.from_numpy(image_B).unsqueeze(0).unsqueeze(0) / 255.0
        return (abs(multi_scale_ssim(image_F,image_A))+abs(multi_scale_ssim(image_F,image_B)))/2

    @classmethod
    def VIF_P(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        image_F = torch.from_numpy(image_F).unsqueeze(0).unsqueeze(0) / 255.0
        image_A = torch.from_numpy(image_A).unsqueeze(0).unsqueeze(0) / 255.0
        image_B = torch.from_numpy(image_B).unsqueeze(0).unsqueeze(0) / 255.0
        return (abs(vif_p(x=image_F, y=image_A))+abs(vif_p(x=image_F, y=image_B))) / 2


    def VIFF(image_F, image_A, image_B):
        refA=image_A
        refB=image_B
        dist=image_F

        sigma_nsq = 2
        eps = 1e-10
        numA = 0.0
        denA = 0.0
        numB = 0.0
        denB = 0.0
        for scale in range(1, 5):
            N = 2 ** (4 - scale + 1) + 1
            sd = N / 5.0
            # Create a Gaussian kernel as MATLAB's
            m, n = [(ss - 1.) / 2. for ss in (N, N)]
            y, x = np.ogrid[-m:m + 1, -n:n + 1]
            h = np.exp(-(x * x + y * y) / (2. * sd * sd))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            sumh = h.sum()
            if sumh != 0:
                win = h / sumh

            if scale > 1:
                refA = convolve2d(refA, np.rot90(win, 2), mode='valid')
                refB = convolve2d(refB, np.rot90(win, 2), mode='valid')
                dist = convolve2d(dist, np.rot90(win, 2), mode='valid')
                refA = refA[::2, ::2]
                refB = refB[::2, ::2]
                dist = dist[::2, ::2]

            mu1A = convolve2d(refA, np.rot90(win, 2), mode='valid')
            mu1B = convolve2d(refB, np.rot90(win, 2), mode='valid')
            mu2 = convolve2d(dist, np.rot90(win, 2), mode='valid')
            mu1_sq_A = mu1A * mu1A
            mu1_sq_B = mu1B * mu1B
            mu2_sq = mu2 * mu2
            mu1A_mu2 = mu1A * mu2
            mu1B_mu2 = mu1B * mu2
            sigma1A_sq = convolve2d(refA * refA, np.rot90(win, 2), mode='valid') - mu1_sq_A
            sigma1B_sq = convolve2d(refB * refB, np.rot90(win, 2), mode='valid') - mu1_sq_B
            sigma2_sq = convolve2d(dist * dist, np.rot90(win, 2), mode='valid') - mu2_sq
            sigma12_A = convolve2d(refA * dist, np.rot90(win, 2), mode='valid') - mu1A_mu2
            sigma12_B = convolve2d(refB * dist, np.rot90(win, 2), mode='valid') - mu1B_mu2

            sigma1A_sq[sigma1A_sq < 0] = 0
            sigma1B_sq[sigma1B_sq < 0] = 0
            sigma2_sq[sigma2_sq < 0] = 0

            gA = sigma12_A / (sigma1A_sq + eps)
            gB = sigma12_B / (sigma1B_sq + eps)
            sv_sq_A = sigma2_sq - gA * sigma12_A
            sv_sq_B = sigma2_sq - gB * sigma12_B

            gA[sigma1A_sq < eps] = 0
            gB[sigma1B_sq < eps] = 0
            sv_sq_A[sigma1A_sq < eps] = sigma2_sq[sigma1A_sq < eps]
            sv_sq_B[sigma1B_sq < eps] = sigma2_sq[sigma1B_sq < eps]
            sigma1A_sq[sigma1A_sq < eps] = 0
            sigma1B_sq[sigma1B_sq < eps] = 0

            gA[sigma2_sq < eps] = 0
            gB[sigma2_sq < eps] = 0
            sv_sq_A[sigma2_sq < eps] = 0
            sv_sq_B[sigma2_sq < eps] = 0

            sv_sq_A[gA < 0] = sigma2_sq[gA < 0]
            sv_sq_B[gB < 0] = sigma2_sq[gB < 0]
            gA[gA < 0] = 0
            gB[gB < 0] = 0
            sv_sq_A[sv_sq_A <= eps] = eps
            sv_sq_B[sv_sq_B <= eps] = eps

            numA += np.sum(np.log10(1 + gA * gA * sigma1A_sq / (sv_sq_A + sigma_nsq)))
            numB += np.sum(np.log10(1 + gB * gB * sigma1B_sq / (sv_sq_B + sigma_nsq)))
            denA += np.sum(np.log10(1 + sigma1A_sq / sigma_nsq))
            denB += np.sum(np.log10(1 + sigma1B_sq / sigma_nsq))

        vifpA = numA / denA
        vifpB =numB / denB

        if np.isnan(vifpA):
            vifpA=1
        if np.isnan(vifpB):
            vifpB = 1
        return vifpA+vifpB

    @classmethod
    def FMI(cls,imgA, imgB, imgF):
        """
        计算 FMI 指标

        参数:
            imgA: 源图像 A (灰度图)
            imgB: 源图像 B (灰度图)
            imgF: 融合图像 (灰度图)
        """
        # 特征提取（以 Sobel 边缘为例）
        edgeA = sobel(imgA)
        edgeB = sobel(imgB)
        edgeF = sobel(imgF)

        # 计算互信息
        mi_FA = mutual_info_score(edgeF.flatten(), edgeA.flatten())
        mi_FB = mutual_info_score(edgeF.flatten(), edgeB.flatten())

        # 计算权重（基于熵）
        histA = np.histogram(edgeA, bins=256, range=(0, 1))[0] + 1e-10
        histB = np.histogram(edgeB, bins=256, range=(0, 1))[0] + 1e-10
        HA = -np.sum(histA * np.log2(histA))
        HB = -np.sum(histB * np.log2(histB))
        wA = HA / (HA + HB)
        wB = HB / (HA + HB)

        # 计算 FMI
        fmi = wA * mi_FA + wB * mi_FB
        return fmi


    @classmethod
    def RMSE(cls,image_A, image_B, image_F):
        cls.input_check(image_F, image_A, image_B)
        return calculate_rmse(image_F, image_A)/2 +  calculate_rmse(image_F, image_B)/2

    @classmethod
    def FSIM(cls,image_A, image_B, image_F):
        cls.input_check(image_F, image_A, image_B)
        return fsim(image_F, image_A)/2 +  fsim(image_F, image_B)/2

    @classmethod
    def FSIM(cls,image_A, image_B, image_F):
        cls.input_check(image_F, image_A, image_B)
        return fsim(image_F, image_A)/2 +  fsim(image_F, image_B)/2
