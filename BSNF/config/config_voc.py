# coding=utf-8
# project

PROJECT_PATH = "D:/VIIF/TIM-fusion-main/"

#data_type = 'voc'
# data_type = 'airport'
#data_type = 'kitti'
# data_type = 'M3FD'
# data_type = 'FLIR'
# data_type = 'AVMS'
# data_type = 'VEDAI'
# data_type = 'TNO'
data_type = 'MSRS'
# data_type = 'RoadScene'
# data_type = 'LLVIP'


if data_type == 'airport':
    DATA_PATH = "D:/new_airport"
    DATA = {"CLASSES":['airplane', 'man', 'car'],
            "NUM":3}
elif data_type == 'AVMS':
    DATA_PATH = "D:/tan/AVMS"
    DATA = {"CLASSES":['airplane', 'man', 'car'],
            "NUM":3}
elif data_type == 'M3FD':
    DATA_PATH = "D:/M3FD/M3FD_Detection"
    DATA = {"CLASSES":['People','Car','Bus', 'Motorcycle', 'Truck','Lamp'],
            "NUM":6}
elif data_type == 'FLIR':
    DATA_PATH = "D:/tan/align_flir"
    DATA = {"CLASSES":['bicycle', 'car', 'person'],
            "NUM":3}
elif data_type == 'LLVIP':
    DATA_PATH = "D:/tan/LLVIP"
    DATA = {"CLASSES":['person'],
            "NUM":1}
elif data_type == 'MSRS':
    DATA_PATH = "D:/tan/MSRS"
    DATA = {"CLASSES":['person'],
            "NUM":1}
elif data_type == 'TNO':
    DATA_PATH = "D:/tan/MSRS"
    DATA = {"CLASSES":['person'],
            "NUM":1}
elif data_type == 'RoadScene':
    DATA_PATH = "D:/tan/MSRS"
    DATA = {"CLASSES":['person'],
            "NUM":1}
# train
TRAIN = {
         "TRAIN_IMG_SIZE":640,
         "AUGMENT":True,
         "BATCH_SIZE":4,
         "MULTI_SCALE_TRAIN":False,
         "IOU_THRESHOLD_LOSS":0.5,
         "NUMBER_WORKERS":0,
         "MOMENTUM":0.9,
         "WEIGHT_DECAY":0.0005,
         }


# test
TEST = {
        "TEST_IMG_SIZE":640,
        "BATCH_SIZE":4,
        "NUMBER_WORKERS":4,
        "CONF_THRESH":0.1,
        "NMS_THRESH":0.5,
        "MULTI_SCALE_TEST":False,
        "FLIP_TEST":False
        }
