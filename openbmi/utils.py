import torch
import numpy as np
from scipy import signal
from scipy.linalg import sqrtm
import logging
import os
import sys
import torch.nn as nn


OpenBMI_CONSTANT = {
    'raw_path': 'datasets/OpenBMI/raw', # raw data path
    'n_subjs': 54,
    'n_trials_2_class': 100,
    'n_trials_3_class': 150, 
    'n_chs': 62,
    'orig_smp_freq': 1000,                  # Original sampling frequency  (Hz)
    'trial_len': 8,                         # 8s (cut-off)
    'MI': {
        'start': 0,                         # start at time = 0 s
        'stop': 4,                          # stop at time = 0 s
        'len': 4,                           # 4s
    },
    'orig_chs': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 
                'C3','Cz','C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 
                'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'FC3', 'FC4', 'C5', 'C1',
                'C2', 'C6', 'CP3','CPz', 'CP4', 'P1', 'P2', 'POz', 'FT9', 'FTT9h', 'TTP7h', 
                'TP7', 'TPP9h', 'FT10','FTT10h','TPP8h', 'TP8', 'TPP10h', 'F9', 'F10', 
                'AF7', 'AF3', 'AF4', 'AF8', 'PO3','PO4'],
    'sel_chs': ['FC5', 'FC3', 'FC1', 'FC2', 'FC4','FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 
                'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6']               # select: 62 -> 20
}


# 1. 对每个样本的所有通道的数据进行独立归一化。保证通道每个样本是独立的
def data_norm(data):
    """
    对数据进行归一化 [-1, 1]
    :param data:   ndarray ,shape[N,channel,samples]
    :return:
    """
    data_copy = np.copy(data)
    for i in range(len(data)):
        data_copy[i] = data_copy[i] / np.max(abs(data[i]))

    return data_copy

def data_norm_one(data):
    """
    对一个数据进行归一化 [-1, 1]
    :param data:  shape[channel, samples]
    :return:
    """
    data_copy = np.copy(data)
    data_copy = data_copy / np.max(abs(data))

    return data_copy

# 2. 对整个数据集进行归一化，计算所有样本的整体平均值和标准差。EEG数据需要被整体分析
def norm_data(data):
    mean = np.mean(data)
    std = np.std(data)
    torch.float32
    normalized_data = (data - mean) / std
    return normalized_data

# 对EEG信号进行欧式对齐（Euclidean Alignment）
def preprocess_ea(data):
    # 一、计算参考矩阵
    R_bar = np.zeros((data.shape[1], data.shape[1]))        # 1. 初始化一个与EEG信号通道数相同维度的零矩阵 R_bar
    for i in range(len(data)):
        R_bar += np.dot(data[i], data[i].T)                 # 2. 对于每个试验（即每个数据片段），计算其协方差矩阵并累加到 R_bar 上
    R_bar_mean = R_bar / len(data)                          # 3. 将累加的协方差矩阵除以试验数，得到参考矩阵

    # 二、数据对齐：对每个试验的数据进行对齐
    R_bar_mean_inv = np.linalg.inv(sqrtm(R_bar_mean))
    for i in range(len(data)):
        data[i] = np.dot(R_bar_mean_inv, data[i])     # 4. 对于每个试验，将其数据与参考矩阵的逆平方根相乘，从而对齐数据
    return data, R_bar_mean_inv


def prepare_data(data):
    data_preprocss = data_norm(data)
    data_ea, _ = preprocess_ea(data_preprocss)
    return data_ea


# 获取 ea 参考矩阵的逆平方根
def get_ea_bar(data):
    # 一、计算参考矩阵
    R_bar = np.zeros((data.shape[1], data.shape[1]))        # 1. 初始化一个与EEG信号通道数相同维度的零矩阵 R_bar
    for i in range(len(data)):
        R_bar += np.dot(data[i], data[i].T)                 # 2. 对于每个试验（即每个数据片段），计算其协方差矩阵并累加到 R_bar 上
    R_bar_mean = R_bar / len(data)                          # 3. 将累加的协方差矩阵除以试验数，得到参考矩阵

    R_bar_mean_inv_sqrt = np.linalg.inv(sqrtm(R_bar_mean))

    return R_bar_mean_inv_sqrt


# 使用 ea 参考矩阵的逆平方根 对数据进行对齐
def align_ea_data(data, R_bar_mean_inv_sqrt):
    aligned_data = np.zeros_like(data)
    for i in range(len(data)):
        aligned_data[i] = np.dot(R_bar_mean_inv_sqrt, data[i])
    return aligned_data


def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
    #                 "%(lineno)d — %(message)s")
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # Check if log file exists
    if os.path.exists(logger_name):
        os.remove(logger_name)

    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


# ------------------------------------------------------------------online----------------------------------------------------------------------------------

# 模型初始化函数
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias) 
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

# Shannon entropy of the predicted class distribution, as in SHOT and DeepTransferEEG
# (MIT License, see THIRD_PARTY_NOTICES.md)
def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

# 重复截取数据
def overlap(data, postdata_len, strip):
    seg_data = []
    lengh = data.shape[1]
    postdata_len = int(postdata_len)
    i = postdata_len
    while 1:
        if i > lengh:
            break
        seg_data.append(data[:, i-postdata_len:i])

        i = i + strip

    return seg_data

# 数据增强
def data_augment(data, args):
    data = np.array(data)
    train_len = args.train_samples
    aug_type = args.augmentation
    stride = args.overlap_stride

    if aug_type=="overlap" :
        for xuhao, i in enumerate(data):
            data_oneaug = overlap(i, train_len, stride) 
            if xuhao == 0:
                data_aug = data_oneaug
            else:
                data_aug = np.vstack((data_aug, data_oneaug))
    elif aug_type=="none" :
        data_aug = data[:, :, 0:train_len]

    return data_aug

# data filter
def preprocessing(data, fs):
    # 设置滤波范围
    lowcut = 0.5
    highcut = 100
    
    filtedData = signal.detrend(data, axis=-1, type='linear', )               # 去趋势化
    b, a = signal.butter(5, [2.0 * lowcut / fs, 2.0 * highcut / fs], 'bandpass')    # 5阶巴特沃斯滤波器
    filtedData = signal.filtfilt(b, a, filtedData, axis=1)

    return filtedData


# 数据增强方法类
class augmentations:

    @staticmethod
    def data_aug_random_mask(input_tensor, during=200):
        # zero a random window of `during` samples (paper: W = 200)
        mask = torch.ones_like(input_tensor).float().to(input_tensor.device)
        start = np.random.randint(0, input_tensor.shape[-1] - during)
        mask[..., start:start + during] = 0
        return input_tensor * mask
