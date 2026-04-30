import cv2
# from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
import os
import numpy as np
from utils.Metrix_Evaluator import Evaluator
from utils.img_read_save import image_read_cv2
import warnings
import logging
import scipy.io as scio
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)


path_fuse='./Fusion_results/MSRS/'
path_Vis='D:/tan/MSRS/Visible/test/'
path_IR='D:/tan/MSRS/Infrared/test/'


path_test_RoadScene=path_fuse


item_num = len([
        entry for entry in os.listdir(path_test_RoadScene)
        if os.path.isfile(os.path.join(path_test_RoadScene, entry))
    ])


metric_result = np.zeros((item_num,8))
i=0
name_test_save = './metric_results/'+"TNO"+ "_metrix.mat"

for img_name in os.listdir(path_test_RoadScene):
    print(i,img_name)
    ir = image_read_cv2(os.path.join(path_IR, img_name), 'GRAY')
    vi = image_read_cv2(os.path.join(path_Vis, img_name), 'GRAY')
    fi = image_read_cv2(os.path.join(path_fuse, img_name), 'GRAY')
    if fi.shape != vi.shape:
        fi= cv2.resize(fi,(vi.shape[1],vi.shape[0]))

    metric_result[i,0:8] = np.array([
                                Evaluator.PSNR(fi, ir, vi), Evaluator.CC(fi, ir, vi), Evaluator.FSIM(fi, ir, vi), Evaluator.SSIM(fi, ir, vi)
                                ,Evaluator.Nabf(fi, ir, vi),Evaluator.SCD(fi, ir, vi),Evaluator.Qabf(fi, ir, vi), Evaluator.MSSIM(fi, ir, vi)
                                ])
    print(str(np.round(metric_result[i], 2)))
    i=i+1

scio.savemat(name_test_save,{'metric_result': metric_result})
metric_mean = np.mean(metric_result, axis=0)
metric_median = np.median(metric_result, axis=0)
print("="*80)
print("PSNR\t CC\t FSIM\t SSIM\t  Nabf\t SCD\t Qabf\t MSSIM\t")
print(str(np.round(metric_mean[0], 2))+'\t'
        +str(np.round(metric_mean[1], 2))+'\t'
        +str(np.round(metric_mean[2], 2))+'\t'
        +str(np.round(metric_mean[3], 2))+'\t'
        +str(np.round(metric_mean[4], 2))+'\t'
        +str(np.round(metric_mean[5], 2))+'\t'
        +str(np.round(metric_mean[6], 2))+'\t'
        +str(np.round(metric_mean[7], 2))
        )
print("="*80)
print(str(np.round(metric_median[0], 2))+'\t'
        +str(np.round(metric_median[1], 2))+'\t'
        +str(np.round(metric_median[2], 2))+'\t'
        +str(np.round(metric_median[3], 2))+'\t'
        +str(np.round(metric_median[4], 2))+'\t'
        +str(np.round(metric_median[5], 2))+'\t'
        +str(np.round(metric_median[6], 2))+'\t'
        +str(np.round(metric_median[7], 2))
        )

