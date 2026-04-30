# coding:utf-8
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import cv2
import glob
import os
import config.config_voc as cfg


def prepare_data_path(data_dir_label,datatype):
    with open(data_dir_label, 'r') as f:
        lines = f.readlines()
        if datatype=='M3FD':
            filenames = [line.strip().split(' ')[0].split('\\')[-1] for line in lines]
            filenames_vi = [line.strip().split(' ')[0].replace('J:/','D:/').replace('j:/','D:/')  for line in lines]
            filenames_ir = [line.strip().split(' ')[0].replace('J:/','D:/').replace('j:/','D:/').replace('vi','ir') for line in lines]
            label =  [line.strip().split(' ')[1:] for line in lines]
        if datatype=='airport':
            filenames = [line.strip().split(' ')[0].split('\\')[-1] for line in lines]
            # filenames_vi = [line.strip().split(' ')[0].replace('JPEGImages','JPEGImages\\random_level\\airsport_exposure_random')for line in lines]
            filenames_vi = [line.strip().split(' ')[0] for line in lines]
            filenames_ir = [line.strip().split(' ')[0].replace('JPEGImages','IR').replace('.jpg','.png')for line in lines]
            label =  [line.strip().split(' ')[1:] for line in lines]
        if datatype=='FLIR':
            filenames = [line.strip().split(' ')[0].split('\\')[-1] for line in lines]
            filenames_vi = [line.strip().split(' ')[0].replace('J:/','D:/') for line in lines]
            filenames_ir = [line.strip().split(' ')[0].replace('J:/','D:/').replace('vi','ir')for line in lines]
            label =  [line.strip().split(' ')[1:] for line in lines]
        if datatype=='VEDAI':
            filenames = [line.strip().split(' ')[0].split('\\')[-1] for line in lines]
            filenames_vi = [line.strip().split(' ')[0] for line in lines]
            filenames_ir = [line.strip().split(' ')[0].replace('vi','ir')for line in lines]
            label =  [line.strip().split(' ')[1:] for line in lines]
        if datatype=='AVMS':
            filenames = [line.strip().split(' ')[0].split('\\')[-1] for line in lines]
            filenames_vi = [line.strip().split(' ')[0] for line in lines]
            filenames_ir = [line.strip().split(' ')[0].replace('vi','ir')for line in lines]
            label =  [line.strip().split(' ')[1:] for line in lines]
        if datatype=='LLVIP':
            filenames = [line.strip().split(' ')[0].split('\\')[-1] for line in lines]
            filenames_vi = [line.strip().split(' ')[0] for line in lines]
            filenames_ir = [line.strip().split(' ')[0].replace('visible','infrared')for line in lines]
            label =  [line.strip().split(' ')[1:] for line in lines]
        if datatype=='MSRS':
            filenames = [line.strip().split(' ')[0].split('\\')[-1] for line in lines]
            filenames_vi = [line.strip().split(' ')[0].replace('J:/','D:/') for line in lines]
            filenames_ir = [line.strip().split(' ')[0].replace('J:/','D:/').replace('Visible','Infrared')for line in lines]
            label =  [line.strip().split(' ')[1:] for line in lines]
        if datatype=='TNO':
            filenames = [line.strip().split(' ')[0].split('\\')[-1] for line in lines]
            filenames_vi = [line.strip().split(' ')[0] for line in lines]
            filenames_ir = [line.strip().split(' ')[0].replace('VI','IR')for line in lines]
            label =  [line.strip().split(' ')[1:] for line in lines]
        if datatype=='RoadScene':
            filenames = [line.strip().split(' ')[0].split('\\')[-1] for line in lines]
            filenames_vi = [line.strip().split(' ')[0].replace('J:/','D:/') for line in lines]
            filenames_ir = [line.strip().split(' ')[0].replace('J:/','D:/').replace('VI','IR')for line in lines]
            label =  [line.strip().split(' ')[1:] for line in lines]

    lenght = len(filenames_vi)
    # filenames_vi.sort()
    # label.sort()
    return lenght,filenames,filenames_vi,filenames_ir,label
#resize = True
class Fusion_dataset(Dataset):
    def __init__(self, split, datatype, fusion_size=640, fusion_size_random=False, resize = False, ir_path=None, vi_path=None):
        super(Fusion_dataset, self).__init__()
        assert split in ['train', 'val', 'test', 'fusion'], 'split must be "train"|"val"|"test"'

        if datatype == 'M3FD':
            if split == 'train':
                data_dir_label = './data/M3FD/train_annotation.txt'
            elif split == 'test':
                data_dir_label = './data/M3FD/test_annotation.txt'
            elif split == 'fusion':
                data_dir_label = './data/M3FD/train_annotation.txt'
        if datatype == 'airport':
            if split == 'train':
                data_dir_label = './data/airport/train_annotation.txt'
            elif split == 'test':
                data_dir_label = './data/airport/test_annotation.txt'
            elif split == 'night':
                data_dir_label = './data/airport/night_annotation.txt'
        if datatype == 'FLIR':
            if split == 'train':
                data_dir_label = './data/FLIR/train_annotation.txt'
            elif split == 'test':
                data_dir_label = './data/FLIR/test_annotation.txt'
        if datatype == 'AVMS':
            if split == 'train':
                data_dir_label = './data/AVMS/train_annotation.txt'
            elif split == 'test':
                data_dir_label = './data/AVMS/test_annotation.txt'
        if datatype == 'VEDAI':
            if split == 'train':
                data_dir_label = './data/VEDAI/train_annotation.txt'
            elif split == 'test':
                data_dir_label = './data/VEDAI/test_annotation.txt'
        if datatype == 'LLVIP':
            if split == 'train':
                data_dir_label = './data/LLVIP/train_annotation.txt'
            elif split == 'test':
                data_dir_label = './data/LLVIP/test_annotation.txt'
        if datatype == 'MSRS':
            if split == 'train':
                data_dir_label = './data/MSRS/train_annotation.txt'
            elif split == 'test':
                data_dir_label = './data/MSRS/test_annotation.txt'
        if datatype == 'TNO':
            if split == 'train':
                data_dir_label = './data/TNO/train_annotation.txt'
            elif split == 'test':
                data_dir_label = './data/TNO/test_annotation.txt'
        if datatype == 'RoadScene':
            if split == 'train':
                data_dir_label = './data/RoadScene/train_annotation.txt'
            elif split == 'test':
                data_dir_label = './data/RoadScene/test_annotation.txt'

        self.split = split
        self.resize = resize
        self.img_size = cfg.TRAIN["TRAIN_IMG_SIZE"]
        self.len, self.filenames, self.filepath_vis, self.filepath_ir, self.label = prepare_data_path(data_dir_label,
                                                                                                      datatype)
        self.length = min(len(self.filepath_vis), len(self.filepath_ir))

    def __getitem__(self, index):
            # index += 2786
            # print(index)
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            filename = self.filenames[index]
            # cv2.imread读取图片通道为BGR排列顺序
            image_vis = cv2.imread(vis_path, 1)
            image_ir = cv2.imread(ir_path, 0)

            if image_vis is None:
                raise RuntimeError(
                    f"[图像读取错误] idx={index}, vis_path={self.filepath_vis[index]}\n"
                    f"请检查文件是否存在、路径是否正确，且是否为有效图像文件。"
                )

            if image_ir is None:
                raise RuntimeError(
                    f"[图像读取错误] idx={index}, ir_path={self.filepath_ir[index]}\n"
                    f"请检查文件是否存在、路径是否正确，且是否为有效图像文件。"
                )

            height, width = image_ir.shape

            if self.resize==True:
                image_vis = cv2.resize(image_vis,(self.img_size  , self.img_size ))
                image_ir = cv2.resize(image_ir, (self.img_size , self.img_size ))


            image_vis = cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB).transpose(2, 0, 1) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0) / 255.0

            image_vis = image_vis.astype(np.float32)
            image_ir = image_ir.astype(np.float32)

            if len(self.label[index])==0:
                label_sample = np.ones((100, 5)) * (-1)
            else:
                bboxes = np.array([list(map(float, box.split(','))) for box in self.label[index]])[:,:4]
                lables = np.array([list(map(float, box.split(','))) for box in self.label[index]])[:,4]
                # box需要转换成比例
                # height, width, channels = image_ir.shape

                bboxes[:, 0] /= width
                bboxes[:, 2] /= width
                bboxes[:, 1] /= height
                bboxes[:, 3] /= height
                # bboxes_lables = np.concatenate([bboxes, lables[:, np.newaxis], np.full((len(bboxes), 1), 1.0)], axis=-1)
                bboxes_lables = np.concatenate([bboxes, lables[:, np.newaxis]], axis=-1)
                # np.concatenate([bboxes_org, np.full((len(bboxes_org), 1), 1.0)], axis=1)
                label_sample = np.ones((100, 5)) * (-1)
                bbox_count = 0
                for i in range(bboxes_lables.shape[0]):
                    label_sample[int(bbox_count % 100), :] = bboxes_lables[i, :]
                    bbox_count += 1

            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
                torch.tensor(label_sample),
                filename
            )

    def __getitem__bk(self, index):
        #if self.split=='train':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            filename = self.filenames[index]
            # cv2.imread读取图片通道为BGR排列顺序
            image_vis = cv2.imread(vis_path, 1)
            # 转换为rgb
            image_vis = cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB).transpose(2, 0, 1) / 255.0

            image_ir = cv2.imread(ir_path, 0)
            image_ir = np.expand_dims(image_ir, axis=0) / 255.0

            image_vis = image_vis.astype(np.float32)
            image_ir = image_ir.astype(np.float32)

            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
                filename
            )


    def __len__(self):
        return self.length


# if __name__ == '__main__':
    # data_dir = '/data1/yjt/MFFusion/dataset/'
    # train_dataset = MF_dataset(data_dir, 'train', have_label=True)
    # print("the training dataset is length:{}".format(train_dataset.length))
    # train_loader = DataLoader(
    #     dataset=train_dataset,
    #     batch_size=2,
    #     shuffle=True,
    #     num_workers=2,
    #     pin_memory=True,
    #     drop_last=True,
    # )
    # train_loader.n_iter = len(train_loader)
    # for it, (image_vis, image_ir, label) in enumerate(train_loader):
    #     if it == 5:
    #         image_vis.numpy()
    #         print(image_vis.shape)
    #         image_ir.numpy()
    #         print(image_ir.shape)
    #         break
