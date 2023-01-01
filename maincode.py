# coding=utf-8

# Note: This is the code of "Binary Change Guided Hyperspectral Multiclass Change Detection"


import numpy as np
import torch
from torch import nn
import random
import matplotlib.pyplot as plt
import time
import math
from PIL import Image
import scipy.io as sio
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from model import UnitedUnmixModule, TemporalCorrelationModule
from train import PreTrainUUModule_train, PreTrainTCModule_train, AlterntvTrain_train
from utils import initNetParams,two_cls_access,mul_cls_access,ChangeDetection_V1_net


#  联合解混模块: UnitedUnmixModule
#  UU-Module预训练
def PreTrainUUModule(data_filename, samp_filename, Y_name):
    print('---func: PreTrainUUModule ---')
    print('data_filename:', data_filename)
    print('samp_filename:', samp_filename)
    data = sio.loadmat(data_filename)
    x_3d = data['X_3d']  # img_channel, img_height, img_width
    y_3d = data[Y_name]
    print('img Y: ' + Y_name)
    endmember = data['endmember']  # img_channel,Num_em
    epoch, batch_sz, patch_size = 100, 64, 7
    learn_rate, w_decay = 0.001, 0.001
    print('learn_rate: ' + str(learn_rate))
    print('w_decay: ' + str(w_decay))
    print('epoch_train: ' + str(epoch))
    sample = sio.loadmat(samp_filename)
    img_channel, img_height, img_width = x_3d.shape
    _, Num_em = endmember.shape
    paramemters = {'x_3d': x_3d, 'y_3d': y_3d, 'patch_size': patch_size,
                   'epoch': epoch, 'batch_sz': batch_sz, 'endmember': endmember,
                   'learn_rate': learn_rate, 'w_decay': w_decay,}

    K_size = math.ceil(math.log(img_channel, 2) / 2 + 0.5) # according to the EfficientChannelAttention paper
    model = UnitedUnmixModule(channel=img_channel, k_size=K_size, num_em=Num_em)
    model.apply(initNetParams)
    prdt_abx, prdt_aby, model_pretrain = PreTrainUUModule_train(paramemters, model, sample)

    # Num_em= 3 for China Dataset
    # endmember extraction: Hyperspectral Signal Identification by Minimum Error and Vertex Component Analysis
    plt.figure('abundance map of pretrain')
    for i in np.arange(Num_em):
        plt.subplot(2, Num_em, i + 1)
        plt.imshow(prdt_abx[i, :, :].squeeze())
    for i in np.arange(Num_em):
        plt.subplot(2, Num_em, i + 1 + Num_em)
        plt.imshow(prdt_aby[i, :, :].squeeze())
    plt.show()
    print('---PreTrainUUModule  Ending---')

    # assessment of estimated abundance map
    prdt_abx_2d = np.reshape(prdt_abx, [Num_em, -1])  # 50, H*W
    prdt_aby_2d = np.reshape(prdt_aby, [Num_em, -1])  # 50, H*W
    eps = data['X_3d'] - np.matmul(data['endmember'], prdt_abx_2d).reshape([img_channel, img_height, img_width])
    eps_2d = np.reshape(eps, [img_channel, -1])
    eps_sum_x = np.sum(eps_2d * eps_2d, axis=0).reshape(img_height, img_width)
    eps_x = np.sum(eps_sum_x) / img_height / img_width / img_channel
    eps = data[Y_name] - np.matmul(data['endmember'], prdt_aby_2d).reshape([img_channel, img_height, img_width])
    eps_2d = np.reshape(eps, [img_channel, -1])
    eps_sum_y = np.sum(eps_2d * eps_2d, axis=0).reshape(img_height, img_width)
    eps_y = np.sum(eps_sum_y) / img_height / img_width / img_channel
    eps_x = round(eps_x, 6)  # round() 方法返回浮点数x的四舍五入值
    eps_y = round(eps_y, 6)  # round() 方法返回浮点数x的四舍五入值
    print('\n')
    print('eps_x:')
    print("%e" % eps_x)
    print('eps_y:')
    print("%e" % eps_y)

    return prdt_abx, prdt_aby, model_pretrain

#  时序相关性模块:TemporalCorrelationModule
#  TC-Module预训练
def PreTrainTCModule(data_filename, samp_filename, unmix_pretrain_file_name, para_set):
    print('---func: PreTrainTCModule ---------------------------------------------------')
    torch.backends.cudnn.deterministic = True
    print('data_filename:', data_filename)
    print('unmix_pretrain_file_name:', unmix_pretrain_file_name)
    print('samp_filename:' + samp_filename)
    data = sio.loadmat(data_filename)
    endmember = data['endmember']  # C,Num_em
    Two_Chge_label = data['Two_CMap']  # H,W
    Mul_Chge_label = data['Mul_CMap']  # H,W
    epoch_train, batch_sz, patch_size = 100, 64, 7
    learn_rate, w_decay = 0.001, 0.001
    print('learn_rate: ', learn_rate)
    print('w_decay: ', w_decay)
    print('epoch_train: ', epoch_train)
    # Urban_sampIdx_8192.mat:{'idx_sample': idx_sample, 'binary_label': binary_label}
    sample = sio.loadmat(samp_filename)  # unchange:16384-400; change:400;  total:16384
    img_height, img_width = Two_Chge_label.shape  # 162,307,307
    _, Num_em = endmember.shape  # 162,4

    unmix_pretrain_result = sio.loadmat(unmix_pretrain_file_name)
    abundance_x = unmix_pretrain_result['abundance_x']
    abundance_y = unmix_pretrain_result['abundance_y']
    model_TC = TemporalCorrelationModule(num_em=Num_em)
    model_TC.apply(initNetParams)
    paramemters = {'X_abundance': abundance_x, 'Y_abundance': abundance_y,
                   'epoch': epoch_train, 'batch_sz': batch_sz,
                   'learn_rate': learn_rate, 'w_decay': w_decay,
                   'alpha': para_set['alpha'], 'gamma': para_set['gamma'],'weight':para_set['weight']}
    model_TC = PreTrainTCModule_train(model_TC, paramemters, sample)
    return model_TC

#  全局训练: 交替更新UU-Module与TC-Module
def AlterntvTrain(data_filename, pretrain_UUModule_filename, pretrain_TCModule_filename, samp_filename, Y_name, para_set):
    print('---func: AlterntvTrain ---')
    print('---交替更新UU-Module与TC-Module---')
    torch.backends.cudnn.deterministic = True
    print('UU-Module预训练模型:', pretrain_UUModule_filename)
    print('TC-Module预训练模型:', pretrain_TCModule_filename)
    print('全局训练样本：', samp_filename)
    print('img Y: ' + Y_name)

    data = sio.loadmat(data_filename)
    Two_Chge_label, Mul_Chge_label = data['Two_CMap'], data['Mul_CMap']  # 307,307
    epoch_pre_train, epoch_train, batch_sz, patch_size = 100, 200, 64, 7  # 400, 64, 7
    learn_rate, w_decay = 0.001, 0.001
    print('learn_rate: ',learn_rate)
    print('w_decay: ' ,w_decay)
    print('epoch_train: ' + str(epoch_train))
    img_channel, img_height, img_width = data['X_3d'].shape  # 162,307,307
    _, Num_em = data['endmember'].shape
    sample_whole_train = sio.loadmat(samp_filename)  # unchange:16384-400; change:400;  total:16384
    paramemters_whole_train = {'x_3d': data['X_3d'], 'y_3d': data[Y_name], 'patch_size': patch_size,
                               'epoch': epoch_train, 'batch_sz': batch_sz, 'endmember': data['endmember'],
                               'learn_rate': learn_rate, 'w_decay': w_decay,
                               'alpha': para_set['alpha'], 'gamma': para_set['gamma'], 'weight': para_set['weight']}
    del epoch_pre_train, epoch_train, batch_sz, learn_rate, w_decay
    del data_filename, samp_filename

    # K_size = math.log(img_channel, 2)/2 +0.5 = 4.17
    print('\n')
    print('---step1:先对UU-Module进行预训练---')
    K_size = math.ceil(math.log(img_channel, 2) / 2 + 0.5)
    UUModule = UnitedUnmixModule(channel=img_channel, k_size=K_size, num_em=Num_em)
    pretrain_UUModule_state_dict = torch.load(pretrain_UUModule_filename)
    UUModule.load_state_dict(pretrain_UUModule_state_dict)
    print('---step2:对TC-Module进行预训练---')
    TCModule = TemporalCorrelationModule(num_em=Num_em)
    pretrain_TCModule_state_dict = torch.load(pretrain_TCModule_filename)
    TCModule.load_state_dict(pretrain_TCModule_state_dict)
    del pretrain_UUModule_state_dict, pretrain_TCModule_state_dict
    print('---step3:交替更新UU-Module与TC-Module---')
    # 接着交替更新UU-Module与TC-Module，每次都是更新一个epoch；
    abundance_x_3d, abundance_y_3d, bi_output, UUModule, TCModule = \
        AlterntvTrain_train(UUModule, TCModule, paramemters_whole_train, sample_whole_train)


    print('---step4:精度评价---')
    unchange = np.zeros([img_height* img_width, 1])
    unchange[np.where(bi_output[:, 0] >= 0.5)] = 1
    unchange = 1 - unchange
    Two_Chge_Map = np.reshape(unchange, [img_height, img_width])
    print('---Two-class assessment---')
    bi_oa_kappa = two_cls_access(Two_Chge_label, Two_Chge_Map)
    print('\n')
    _, Mul_Chge_Map_V1 = ChangeDetection_V1_net(abundance_x_3d, abundance_y_3d, Two_Chge_Map)
    mul_oa_kappa, Mul_Chge_Map_match = mul_cls_access(Mul_Chge_label, Mul_Chge_Map_V1)
    print('\n')
    print('Two-class assessment:')
    print('OA:  ' + str(bi_oa_kappa[1]) + '    ' + 'kappa:  ' + str(bi_oa_kappa[3]))
    print('Multiple-class assessment:')
    print('OA:  ' + str(mul_oa_kappa[4, 0]) + '    ' + 'kappa:  ' + str(mul_oa_kappa[4, 2]))
    print('\n')

    # accessment of estimated abundance map
    prdt_abx_2d = np.reshape(abundance_x_3d, [Num_em, -1])  # 50, H*W
    prdt_aby_2d = np.reshape(abundance_y_3d, [Num_em, -1])  # 50, H*W
    eps = data['X_3d'] - np.matmul(data['endmember'], prdt_abx_2d).reshape([img_channel, img_height, img_width])
    eps_2d = np.reshape(eps, [img_channel, -1])
    eps_sum_x = np.sum(eps_2d * eps_2d, axis=0).reshape(img_height, img_width)
    eps_x = np.sum(eps_sum_x) / img_height / img_width / img_channel
    eps = data[Y_name] - np.matmul(data['endmember'], prdt_aby_2d).reshape([img_channel, img_height, img_width])
    eps_2d = np.reshape(eps, [img_channel, -1])
    eps_sum_y = np.sum(eps_2d * eps_2d, axis=0).reshape(img_height, img_width)
    eps_y = np.sum(eps_sum_y) / img_height / img_width / img_channel
    eps_x = round(eps_x, 6)  # round() 方法返回浮点数x的四舍五入值
    eps_y = round(eps_y, 6)  # round() 方法返回浮点数x的四舍五入值
    print('\n')
    print('eps_x:')
    print("%e" % eps_x)
    print('eps_y:')
    print("%e" % eps_y)

    plt.figure('weight=' + str(para_set['weight']) + ' -bi_output')
    plt.imshow(Two_Chge_Map)
    plt.title('BinaryChangeMap')

    plt.figure('Abundancemap:weight=' + str(para_set['weight']) + ' -ab of whole-train')
    for i in np.arange(Num_em):
        plt.subplot(2, Num_em, i + 1)
        plt.imshow(abundance_x_3d[i, :, :].squeeze())
    for i in np.arange(Num_em):
        plt.subplot(2, Num_em, i + 1 + Num_em)
        plt.imshow(abundance_y_3d[i, :, :].squeeze())
    plt.show()
    print('\n')
    return abundance_x_3d, abundance_y_3d, UUModule, TCModule, Two_Chge_Map, Mul_Chge_Map_match



def maincode_PreTrainUUModule():
    print('\n')
    print('---func: maincode_PreTrainUUModule  begins---')
    print('---Step1: Pre-training of UU-Module----------')
    # Note: warmup samples for UU-Module are opted from the whole images at random, the number of which is all set 16384 for three datasets.

    # data_filename = './data/ChinaData/China_MultChange.mat'  # for China dataset
    # samp_filename = './data/ChinaData/China_sampIdx_16384.mat'  # for China dataset
    # data_name, Y_name = 'China','Y_3d'  # for China dataset


    data_filename = './data/USAData/USA_MultChange.mat'  # for USA dataset
    samp_filename = './data/USAData/USA_sampIdx_16384.mat'  # for USA dataset
    data_name, Y_name = 'USA','Y_3d'  # for USA dataset

    # data_filename = './data/UrbanData/Urban_MultChange.mat'  # for Urban dataset
    # samp_filename = './data/UrbanData/Urban_sampIdx_16384.mat' # for Urban dataset
    # data_name, Y_name = 'Urban_50db', 'Y_3d_50db' #'Y_3d_20db','Y_3d_30db','Y_3d_40db' for Urban dataset


    prdt_abx, prdt_aby, model_pretrain = PreTrainUUModule(data_filename, samp_filename, Y_name)
    path = "./save"
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)
        print('new folder:', path)
    else:
        print('There is this folder: ./save/!')
    model_filename = path +'/'+ data_name +'_PreTrain_UU_Module.pkl'
    file_name = path +'/'+ data_name +'_PreTrain_UU_result.mat'

    torch.save(model_pretrain.state_dict(), model_filename)
    sio.savemat(file_name, {'abundance_x': prdt_abx, 'abundance_y': prdt_aby})
    print('model_filename:', model_filename)
    print('file_name:', file_name)
    return prdt_abx, prdt_aby, model_pretrain

def maincode_PreTrainTCModule():
    print('\n')
    print('---func: maincode_PreTrainTCModule  begins---')
    print('---Step2: Pre-training of TC-Module----------')
    # weight equal to 1 for pretraining of TC-Module, because there is only focal loss for pretraining of TC-Module

    # data_filename = './data/ChinaData/China_MultChange.mat'
    # samp_filename = './data/ChinaData/China_sampIdx_12288_4096.mat'
    # unmix_pretrain_file_name = './save/China_PreTrain_UU_result.mat'
    # para_set = {'alpha': 0.25, 'gamma': 1, 'weight': 1}  # (pretraining of TC-Module) parameter setting of focal loss for China dataset
    # data_name, Y_name = 'China','Y_3d'  # for China dataset

    data_filename = './data/USAData/USA_MultChange.mat'
    samp_filename = './data/USAData/USA_sampIdx_9216_3072.mat'
    unmix_pretrain_file_name = './save/USA_PreTrain_UU_result.mat'
    para_set = {'alpha': 0.5, 'gamma': 2, 'weight': 1}  # (pretraining of TC-Module) parameter setting of focal loss for USA dataset
    data_name, Y_name = 'USA', 'Y_3d'  # for USA dataset


    # data_filename = './data/UrbanData/Urban_MultChange.mat'
    # samp_filename = './data/UrbanData/Urban_sampIdx_50db_2048_400.mat'
    # unmix_pretrain_file_name = './save/Urban_50db_PreTrain_UU_result.mat'
    # para_set = {'alpha': 0.25, 'gamma': 2,'weight': 1}  # (pretraining of TC-Module) parameter setting of focal loss for Urban dataset
    # data_name, Y_name = 'Urban_50db', 'Y_3d_50db' #'Y_3d_20db','Y_3d_30db','Y_3d_40db' for Urban dataset

    model_TC = PreTrainTCModule(data_filename, samp_filename, unmix_pretrain_file_name, para_set)

    path = "./save"
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)
        print('new folder:', path)
    else:
        print('There is this folder: ./save/!')
    model_filename = path + '/' + data_name +'_PreTrain_TC_Module.pkl'
    torch.save(model_TC.state_dict(), model_filename)
    print('model_filename:', model_filename)
    return model_TC

def maincode_AlterntvTrain():
    print('\n')
    print('---func: maincode_AlterntvTrain  begins---')
    print('---Step3: Alternative training of UU-Module and TC-Module--')

    # data_filename = './data/ChinaData/China_MultChange.mat'
    # samp_filename = './data/ChinaData/China_sampIdx_12288_4096.mat'
    # pretrain_UUModule_filename = './save/China_PreTrain_UU_Module.pkl'
    # pretrain_TCModule_filename = './save/China_PreTrain_TC_Module.pkl'
    # data_name, Y_name = 'China','Y_3d'  # for China dataset
    # para_set ={'alpha': 0.25, 'gamma': 1,'weight':20}  # (Alternative training of UU-Module and TC-Module) parameter setting of focal loss for China dataset

    data_filename = './data/USAData/USA_MultChange.mat'
    samp_filename = './data/USAData/USA_sampIdx_9216_3072.mat'
    pretrain_UUModule_filename = './save/USA_PreTrain_UU_Module.pkl'
    pretrain_TCModule_filename = './save/USA_PreTrain_TC_Module.pkl'
    data_name, Y_name = 'USA', 'Y_3d'  # for USA dataset
    para_set ={'alpha': 0.5, 'gamma': 2,'weight': 1}   # (Alternative training of UU-Module and TC-Module) parameter setting of focal loss for USA dataset

    # data_filename = './data/UrbanData/Urban_MultChange.mat'
    # samp_filename = './data/UrbanData/Urban_sampIdx_50db_2048_400.mat'
    # pretrain_UUModule_filename = './save/Urban_50db_PreTrain_UU_Module.pkl'
    # pretrain_TCModule_filename = './save/Urban_50db_PreTrain_TC_Module.pkl'
    # para_set = {'alpha': 0.25, 'gamma': 2, 'weight': 1}  # (Alternative training of UU-Module and TC-Module) parameter setting of focal loss for Urban dataset
    # data_name, Y_name = 'Urban_50db', 'Y_3d_50db' #'Y_3d_20db','Y_3d_30db','Y_3d_40db' for Urban dataset


    abundance_x_3d, abundance_y_3d, UUModule, TCModule, Two_Chge_Map, Mul_Chge_Map_match = \
        AlterntvTrain(data_filename, pretrain_UUModule_filename, pretrain_TCModule_filename, samp_filename,
                  Y_name, para_set)

    path = "./save"
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)
        print('new folder:', path)
    else:
        print('There is this folder: ./save/!')
    UUModule_filename = path + '/' + data_name +'_wholeTrain_UUModel.pkl'
    TCModule_filename = path + '/' + data_name +'_wholeTrain_TCModule.pkl'
    result_filename = path + '/' + data_name +'_wholeTrain_result.mat'

    torch.save(UUModule.state_dict(), UUModule_filename)
    torch.save(TCModule.state_dict(), TCModule_filename)
    sio.savemat(result_filename, {'abundance_X_3d': abundance_x_3d ,'abundance_Y_3d': abundance_y_3d,
                            'Two_Chge_Map': Two_Chge_Map,'Mul_Chge_Map_match': Mul_Chge_Map_match})
    print('UUModule_filename:', UUModule_filename)
    print('TCModule_filename:', TCModule_filename)
    print('result_filename:', result_filename)
    return abundance_x_3d, abundance_y_3d, UUModule, TCModule, Two_Chge_Map, Mul_Chge_Map_match


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def zz():
    print('----Everything will be OK!-------')

if __name__ == "__main__":
    # Step1: Pre-training of UU-Module: endmember matrix and 16384 warmup samples are opted from the whole images at random for UU-Module pre-training
    # Step2: Pre-training of TC-Module: pseudo binary labels are selected from pre-detection result
    # Step3: Alternative training of UU-Module and TC-Module: pseudo binary labels are selected from pre-detection result
    # Note: input data is normalized to range of 0-1 for unmixing
    # Note: endmember extraction: Hyperspectral Signal Identification by Minimum Error and Vertex Component Analysis

    zz()
    start = time.time()
    seed = 23
    setup_seed(seed=seed)  #added in 2022.9.1
    print('seed=', seed)

    """ China dataset; USA dataset; Urban dataset """
    # ---------Step1: Pre-training of UU-Module
    prdt_abx, prdt_aby, UUModule_preTrain = maincode_PreTrainUUModule()
    # ---------Step2: Pre-training of TC-Module
    TCModule_preTrain = maincode_PreTrainTCModule()
    # ---------Step3: Alternative training of UU-Module and TC-Module
    abundance_x_3d, abundance_y_3d, UUModule, TCModule, Two_Chge_Map, Mul_Chge_Map_match = maincode_AlterntvTrain()

    end = time.time()
    print("共用时", (end - start), "秒")









