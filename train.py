# coding=utf-8
# Note: This is the code of "Binary Change Guided Hyperspectral Multiclass Change Detection"

import torch
import numpy as np
from torch.autograd import Variable
import time
from torch.utils.data import DataLoader,TensorDataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
start = time.time()
from torch import nn
from utils import get_sample_patch,test_get_sample_set

class cos(nn.Module):
    #  similar to hinge loss
    def __init__(self, threshold=0.1):
        super(cos, self).__init__()

    def forward(self, input1, input2):
        output = 1 - torch.cosine_similarity(input1, input2, dim=1)       # batch_size, 1
        return torch.sum(output)

# 支持多分类和二分类
class weighted_FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    :param Weight:
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True, Weight=1.0):
        super(weighted_FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average
        self.weight = Weight

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma
        alpha = alpha[idx].squeeze()
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = self.weight * loss.mean()
        else:
            loss = loss.sum()
        return loss


# Pre-training of UU-Module,联合解混模块预训练
def PreTrainUUModule_train(paramemters, model, sample):
    print('---func: PreTrainUUModule_train---')
    endmember = torch.FloatTensor(paramemters['endmember'].T).cuda()
    epoch = paramemters['epoch']
    idx_sample = sample['idx_sample'].squeeze()

    patch_set_x, pixel_set_x, expand_x_3d = get_sample_patch(paramemters['x_3d'], idx_sample, paramemters['patch_size'])
    patch_set_y, pixel_set_y, expand_y_3d = get_sample_patch(paramemters['y_3d'], idx_sample, paramemters['patch_size'])
    train_dataset = TensorDataset(torch.tensor(patch_set_x, dtype=torch.float32).unsqueeze(1),
                                  torch.tensor(pixel_set_x, dtype=torch.float32),
                                  torch.tensor(patch_set_y, dtype=torch.float32).unsqueeze(1),
                                  torch.tensor(pixel_set_y, dtype=torch.float32))
    data_loader = DataLoader(train_dataset, batch_size=paramemters['batch_sz'], shuffle=True)
    iter_num = len(idx_sample) // paramemters['batch_sz']
    optimizer = torch.optim.Adam \
        (model.parameters(), lr=paramemters['learn_rate'], betas=(0.9, 0.99), weight_decay=paramemters['w_decay'])
    cos_mse = cos()
    print('L1 & L2  = cos(sum)')
    model = model.cuda()
    cos_mse = cos_mse.cuda()
    # Training loss
    Tra_ls, L1, L2 = [],[],[]
    del patch_set_x, patch_set_y, pixel_set_x, pixel_set_y, paramemters,sample, train_dataset, idx_sample

    for _epoch in range(0, epoch):
        model.train()
        tra_ave_ls, l1, l2 = 0,0,0
        for i, data in enumerate(data_loader):
            train_patch_x, train_pixel_x, train_patch_y, train_pixel_y = data
            train_patch_x, train_pixel_x = train_patch_x.cuda(), train_pixel_x.cuda()  # transfer the data to GPU
            train_patch_y, train_pixel_y = train_patch_y.cuda(), train_pixel_y.cuda()

            abundance_x, abundance_y = model(train_patch_x, train_patch_y, mode='train')
            loss1 = cos_mse(torch.mm(abundance_x, endmember), train_pixel_x)
            loss2 = cos_mse(torch.mm(abundance_y, endmember), train_pixel_y)
            loss = loss1 + loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tra_ave_ls += loss.item()
            l1 += loss1.item()
            l2 += loss2.item()
        tra_ave_ls /= iter_num
        l1 /= iter_num
        l2 /= iter_num
        Tra_ls.append(tra_ave_ls)
        L1.append(l1)
        L2.append(l2)
        if _epoch % 10 == 0:
            print('epoch [{}/{}],train loss:{:.4f},'.format(_epoch + 1, epoch, tra_ave_ls))

    print('---Pretraining is successfully down---')
    plt.figure('UU-Module-training')
    plt.plot(np.asarray(Tra_ls),  label="train ave loss")
    plt.plot(np.asarray(L1), label="loss1")
    plt.plot(np.asarray(L2), label="loss2")
    plt.legend()
    plt.title('UU-Module-training loss')

    # Prediction
    model.eval()
    expand_x_3d = torch.FloatTensor(expand_x_3d).unsqueeze(0).unsqueeze(0).cuda()  # FloatTensor
    expand_y_3d = torch.FloatTensor(expand_y_3d).unsqueeze(0).unsqueeze(0).cuda()  # FloatTensor
    prdt_abx, prdt_aby = model(expand_x_3d, expand_y_3d, mode='test')
    prdt_abx = prdt_abx.squeeze(0).cpu().detach().numpy()
    prdt_aby = prdt_aby.squeeze(0).cpu().detach().numpy()
    end = time.time()
    print("training 共用时", (end - start), "秒")

    return prdt_abx, prdt_aby, model

# Pre-training of TCModule,时序相关性模块预训练
def PreTrainTCModule_train(model, paramemters, sample):
    """
    Args:
        model:
        paramemters = {'X_abundance': X_abundance, 'Y_abundance': Y_abundance,
                   'epoch': epoch, 'batch_sz': batch_sz,
                   'learn_rate': learn_rate, 'w_decay': w_decay}
        [dict]sample: Urban_sampIdx_8192.mat:{'idx_sample': idx_sample, 'binary_label': binary_label}
    Returns:
    """
    print('---func: PreTrainTCModule_train---')
    Num_em, H, W = paramemters['X_abundance'].shape
    X_abundance_2d = np.reshape(paramemters['X_abundance'], [Num_em, -1])
    Y_abundance_2d = np.reshape(paramemters['Y_abundance'], [Num_em, -1])
    binary_label = sample['binary_label'].squeeze()
    len_bi_label = len(binary_label)
    idx = sample['idx_sample'].squeeze()
    X_abundance_smp = X_abundance_2d[:, idx]  # 4, num_sample
    Y_abundance_smp = Y_abundance_2d[:, idx]  # 4, num_sample
    train_dataset = TensorDataset(torch.tensor(X_abundance_smp.T, dtype=torch.float32),
                                  torch.tensor(Y_abundance_smp.T, dtype=torch.float32),
                                  torch.tensor(binary_label, dtype=torch.long))
    data_loader = DataLoader(train_dataset, batch_size=paramemters['batch_sz'], shuffle=True)
    iter_num = len_bi_label // paramemters['batch_sz']
    optimizer = torch.optim.Adam(model.parameters(), lr=paramemters['learn_rate'],
                                 betas=(0.9, 0.99), weight_decay=paramemters['w_decay'])
    alpha, gamma, weight = paramemters['alpha'], paramemters['gamma'], paramemters['weight']

    mean = True
    binary_lossFc = weighted_FocalLoss(num_class=2, alpha=alpha, gamma=gamma,size_average=mean,  Weight=weight)
    print('FocalLoss:alpha=' + str(alpha)+', gamma=' + str(gamma)+', weight=' + str(weight))
    print('size_average: '+ str(mean))
    model, binary_lossFc = model.cuda(), binary_lossFc.cuda()
    print('---PreTrainTCModule_train begins---')

    # Training loss
    Tra_ls = []
    for _epoch in range(0, paramemters['epoch']):
        model.train()
        tra_ave_ls = 0
        for i, data in enumerate(data_loader):
            abundance_x, abundance_y, bi_label = data
            abundance_x, abundance_y = abundance_x.cuda(), abundance_y.cuda()       # transfer the data to GPU
            bi_label = bi_label.cuda()
            output = model(Variable(abundance_x), Variable(abundance_y), mode='train')
            loss = binary_lossFc(output, bi_label)
            if torch.isnan(loss):
                print('---nan---')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tra_ave_ls += loss.item()
        tra_ave_ls /= iter_num
        Tra_ls.append(tra_ave_ls)
        if (_epoch + 1) % 10 == 0:
            print('epoch [{}/{}],train:{:.4f},'.format(_epoch + 1, paramemters['epoch'], tra_ave_ls))

    print('---PreTrainTCModule_train is successfully down---')
    name = 'FocalLoss:alpha=' + str(alpha) + ', gamma=' + str(gamma) + ', weight=' + str(weight)
    plt.figure(name)
    plt.plot(np.arange(paramemters['epoch']), np.asarray(Tra_ls),'r-o', label="train ave loss")
    plt.legend()
    end = time.time()
    print("training 共用时", (end - start), "秒")
    return model


#  全局训练: 交替更新UU-Module与TC-Module
def AlterntvTrain_train(UUModule, TCModule, paramemters, sample):
    # Args:
    #
    #     paramemters = {'x_3d': x_3d, 'y_3d': y_3d, 'patch_size': patch_size,'abundance_x':abundance_x,
    #                'epoch': epoch_train, 'batch_sz': batch_sz, 'endmember': endmember,
    #                'learn_rate': learn_rate, 'w_decay': w_decay,'abundance_y':abundance_y,
    #                'alpha': para_set['alpha'], 'gamma': para_set['gamma'],'weight':para_set['weight']}
    #     sample[dict]: Urban_sampIdx_8192.mat:{'idx_sample': idx_sample, 'binary_label': binary_label}
    # Returns:
    start = time.time()
    print('---func: AlterntvTrain_train ---')
    endmember = torch.FloatTensor(paramemters['endmember'].T).cuda()
    epoch = paramemters['epoch']
    patch_set_x, pixel_set_x, expand_x_3d, patch_set_y, pixel_set_y, expand_y_3d, binary_label = \
        test_get_sample_set(paramemters['x_3d'], paramemters['y_3d'], paramemters['patch_size'], sample)
    len_bi_label = len(binary_label)
    expand_x_3d = torch.FloatTensor(expand_x_3d).unsqueeze(0).unsqueeze(0).cuda()
    expand_y_3d = torch.FloatTensor(expand_y_3d).unsqueeze(0).unsqueeze(0).cuda()
    # [2048, channel, 7, 7]->[2048, 1, channel, 7, 7] for 3d convolution
    train_dataset = TensorDataset(torch.tensor(patch_set_x, dtype=torch.float32).unsqueeze(1),
                                  torch.tensor(pixel_set_x, dtype=torch.float32),
                                  torch.tensor(patch_set_y, dtype=torch.float32).unsqueeze(1),
                                  torch.tensor(pixel_set_y, dtype=torch.float32),
                                  torch.tensor(binary_label, dtype=torch.long))  # .unsqueeze(1) torch.long
    data_loader = DataLoader(train_dataset, batch_size=paramemters['batch_sz'], shuffle=True,
                             num_workers=3, pin_memory=True)

    iter_num = len_bi_label // paramemters['batch_sz']
    optimizer_UUnet = torch.optim.Adam(UUModule.parameters(), lr=paramemters['learn_rate'],
                                 betas=(0.9, 0.99), weight_decay=paramemters['w_decay'])
    optimizer_BCDnet = torch.optim.Adam(TCModule.parameters(), lr=paramemters['learn_rate'],
                                betas=(0.9, 0.99), weight_decay=paramemters['w_decay'])
    cos_lossFc = cos()  # 1-cosine_similarity, loss fuc of UU-Module
    alpha, gamma, weight = paramemters['alpha'], paramemters['gamma'], paramemters['weight']
    mean = True
    binary_lossFc = weighted_FocalLoss(num_class=2, alpha=alpha, gamma=gamma,size_average=mean,  Weight=weight)
    print('L1 & L2 :cos_sum')
    print('weighted_FocalLoss:alpha=' + str(alpha)+', gamma=' + str(gamma)+', weight=' + str(weight))
    print('size_average: '+ str(mean))
    UUModule, TCModule, mse_lossFc, binary_lossFc = UUModule.cuda(),TCModule.cuda(), cos_lossFc.cuda(), binary_lossFc.cuda()

    print('---whole train begins---')
    # Training loss
    TCModule_ls, UUModule_ls, L1, L2, L3 = [], [], [], [], []
    for _epoch in range(0, epoch):
        TCModule_ave_ls, UUModule_ave_ls, l1, l2, l3 = 0, 0, 0 , 0 ,0
        for i, data in enumerate(data_loader):
            train_patch_x, train_pixel_x, train_patch_y, train_pixel_y, bi_label = data
            train_patch_x, train_pixel_x = train_patch_x.cuda(), train_pixel_x.cuda()       # transfer the data to GPU
            train_patch_y, train_pixel_y = train_patch_y.cuda(), train_pixel_y.cuda()       # transfer the data to GPU
            bi_label = bi_label.cuda()

            # 更新TC-Module权重
            abundance_x, abundance_y = UUModule(train_patch_x, train_patch_y, mode='train')
            output = TCModule(abundance_x.detach(), abundance_y.detach(), mode='train')
            fc_loss = binary_lossFc(output, bi_label)
            optimizer_BCDnet.zero_grad()
            fc_loss.backward()
            optimizer_BCDnet.step()
            TCModule_ave_ls += fc_loss.item()

            # 更新UU-Module权重
            bi_output = TCModule(abundance_x, abundance_y, mode='train')  # TC-Module have been optimized
            loss1 = cos_lossFc(torch.mm(abundance_x, endmember), train_pixel_x)
            loss2 = cos_lossFc(torch.mm(abundance_y, endmember), train_pixel_y)
            loss3 = binary_lossFc(bi_output, bi_label)
            loss_uu_net = loss1 + loss2 + loss3
            optimizer_UUnet.zero_grad()
            loss_uu_net.backward()
            optimizer_UUnet.step()
            l1 += loss1.item()
            l2 += loss2.item()
            l3 += weight * loss3.item()
            UUModule_ave_ls += loss_uu_net.item()

        TCModule_ave_ls /= iter_num
        UUModule_ave_ls /= iter_num
        TCModule_ls.append(TCModule_ave_ls), UUModule_ls.append(UUModule_ave_ls)
        l1/= iter_num
        l2/= iter_num
        l3/= iter_num
        L1.append(l1), L2.append(l2), L3.append(l3)

        if _epoch % 20 == 0:
            print('epoch [{}/{}],train:{:.4f},{:.4f},'.format(_epoch + 1, epoch, UUModule_ave_ls, TCModule_ave_ls))
    print('---Whole Training is successfully down---')

    name = 'weight=' + str(weight) + '- FocalLoss:alpha=' + str(alpha)+', gamma=' + str(gamma)
    plt.figure(name)
    plt.subplot(1, 2, 1)
    plt.plot(np.asarray(TCModule_ls), label="TCModule_ls")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(np.asarray(UUModule_ls),'r-o', label="UUModule_ls")
    plt.plot(np.asarray(L1), label="loss1")
    plt.plot(np.asarray(L2), label="loss2")
    plt.plot(np.asarray(L3), label="loss3")
    plt.legend()

    end = time.time()
    print("training  2 共用时 ", (end - start), "秒")

    abundance_x, abundance_y = UUModule(expand_x_3d, expand_y_3d, mode='test')  # [batch_size, num_em]
    bi_output = TCModule(abundance_x, abundance_y, mode='test')  # TC-Module have been optimized

    abundance_x_3d = abundance_x.squeeze(0).cpu().detach().numpy()
    abundance_y_3d = abundance_y.squeeze(0).cpu().detach().numpy()
    bi_output = bi_output.cpu().detach().numpy()

    end = time.time()
    print("test 共用时", (end - start), "秒")
    return abundance_x_3d, abundance_y_3d, bi_output, UUModule, TCModule




