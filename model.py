# coding=utf-8
# Note: This is the code of "Binary Change Guided Hyperspectral Multiclass Change Detection"

import torch
import torch.nn as nn
import torch.nn.functional as F


# ECA模块
class ECA(nn.Module):
    # Constructs a ECA module.
    # Args:
    #     channel: Number of channels of the input feature map
    #     k_size: Adaptive selection of kernel size

    def __init__(self, K_size):  # def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()  # super类的作用是继承的时候，调用含super的各个基类__init__函数。
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.conv = nn.Conv1d(1, 1, kernel_size=K_size, padding=(K_size - 1) // 2, bias=False)  # 一维卷积
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [batch, channel, h, w]
        # b, c, h, w = x.size()  # b代表b个样本，c为通道数，h为高度，w为宽度
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)  # b, c, 1, 1
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion多尺度信息融合
        y = self.sigmoid(y)
        # 原网络中克罗内克积，也叫张量积，为两个任意大小矩阵间的运算
        return x * y.expand_as(x)

class sumToOne(nn.Module):
    def __init__(self, num_ab):
        super(sumToOne, self).__init__()
        self.num_ab = num_ab

    def forward(self, input, mode):
        if mode == 'train':     # input: [batch_size, num_em]
            sum = torch.sum(input, dim=1,  keepdim=True)
            sum = sum.repeat(1, self.num_ab)
            output = input/(sum + 1e-6)
        elif mode == 'valid':
            sum = torch.sum(input, dim=1, keepdim=True)
            sum = sum.repeat(1, self.num_ab)
            output = input / (sum + 1e-6)
        else:     # input: [num_em, H, W]
            num_em, H, W = input.shape
            input = input.reshape(num_em, -1).t()       # [H*W, num_em]
            sum = torch.sum(input, dim=1, keepdim=True)
            sum = sum.repeat(1, self.num_ab)
            output = input / (sum + 1e-6)
            output = output.t().reshape(num_em, H, W)           # (num_em, H, W)
        return output

class cos(nn.Module):
    #  similar to hinge loss
    def __init__(self):
        super(cos, self).__init__()

    def forward(self, input1, input2):
        output = 1 - torch.cosine_similarity(input1, input2, dim=1)       # batch_size, 1
        return output       # torch.sum(

# United Unmixing Module (UU-Module) 联合解混网络
class UnitedUnmixModule(nn.Module):
    # only sharing weight of feature extraction
    def __init__(self, channel, k_size, num_em):
        super(UnitedUnmixModule,self).__init__()
        self.conv1 = nn.Conv3d(1, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv2 = nn.Conv3d(1, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0))
        self.eca1 = ECA(k_size)
        self.conv3 = nn.Conv3d(1, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0))
        self.conv4 = nn.Conv3d(1, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0))
        self.eca2 = ECA(k_size)
        self.conv5 = nn.Conv3d(1, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv6 = nn.Conv2d(channel, round(channel/2), kernel_size=1)  # 1*1 conv
        self.conv7 = nn.Conv2d(round(channel/2), num_em, kernel_size=1)  # 1*1 conv
        self.conv8 = nn.Conv2d(channel, round(channel/2), kernel_size=1)  # 1*1 conv
        self.conv9 = nn.Conv2d(round(channel/2), num_em, kernel_size=1)  # 1*1 conv
        self.relu = nn.ReLU()
        self.sumToOne = sumToOne(num_em)

    def forward(self, img_x, img_y, mode):
        output11 = self.relu(self.conv1(img_x))  # [N, C=1, D, h-2, w-2]
        output11 = self.conv2(output11)  # [N, C=1, D, h-4, w-4]
        output11 = self.eca1(output11.squeeze(1)).unsqueeze(1)  # [N, C=1, D, h-4, w-4]
        output12 = self.relu(self.conv3(img_x))  # [N, C=1, D, h-2, w-2]
        output12 = self.conv4(output12)  # 接eca模块前去掉激活函数relu [N, C=1, D, h-4, w-4]
        output12 = self.eca2(output12.squeeze(1)).unsqueeze(1)  # [N, C=1, D, h-4, w-4]

        output1 = output11 + output12  # [N, C=1, D, h-4, w-4]
        output1 = self.relu(self.conv5(output1))  # [N, C=1, D, h-6, w-6]
        output1 = output1.squeeze(1)  # [N, D, 1,1]
        output1 = self.relu(self.conv6(output1))  # [N, D/2]
        output1 = self.relu(self.conv7(output1)).squeeze()  # [N, numOfem]
        abundance_x = self.sumToOne(output1, mode)  # [N, numOfem]

        output21 = self.relu(self.conv1(img_y))  # [N, C=1, D, h-2, w-2]
        output21 = self.conv2(output21)  # [N, C=1, D, h-4, w-4]
        output21 = self.eca1(output21.squeeze(1)).unsqueeze(1)  # [N, C=1, D, h-4, w-4]
        output22 = self.relu(self.conv3(img_y))  # [N, C=1, D, h-2, w-2]
        output22 = self.conv4(output22)  # 接eca模块前去掉激活函数relu [N, C=1, D, h-4, w-4]
        output22 = self.eca2(output22.squeeze(1)).unsqueeze(1)  # [N, C=1, D, h-4, w-4]

        output2 = output21 + output22  # [N, C=1, D, h-4, w-4]
        output2 = self.relu(self.conv5(output2))  # [N, C=1, D, h-6, w-6]
        output2 = output2.squeeze(1)  # [N, D, 1,1]
        output2 = self.relu(self.conv8(output2))  # [N, D/2]
        output2 = self.relu(self.conv9(output2)).squeeze()  # [N, numOfem]
        abundance_y = self.sumToOne(output2, mode)  # [N, numOfem]

        return abundance_x, abundance_y

# Temporal Correlation Module (TC-Module)，时序相关性模块
class TemporalCorrelationModule(nn.Module):
    # use relu; add the constrain of sum to one
    def __init__(self, num_em):
        super(TemporalCorrelationModule, self).__init__()
        self.Cos = cos()
        self.liner11 = nn.Linear(num_em, num_em)     # bias = True
        self.liner12 = nn.Linear(num_em, num_em)  # bias = True
        self.liner2 = nn.Linear(2*num_em, num_em)  # bias = True
        self.liner3 = nn.Linear(num_em, 2)  # bias = True
        self.relu = nn.ReLU()

    def forward(self, abundance_x, abundance_y, mode):  # endmember
        if mode == 'train':
            output1 = self.relu(self.liner11(abundance_x))       # batch_size, num_em
            output2 = self.relu(self.liner12(abundance_y))
            output = torch.cat([output1, output2], dim=1)  # [batch_size, 2* num_em]
            output = self.relu(self.liner2(output))  # [batch_size, num_em]
            output = self.liner3(output)  # [batch_size, 2]
        elif mode=='valid':
            output1 = self.relu(self.liner11(abundance_x))  # batch_size, num_em
            output2 = self.relu(self.liner12(abundance_y))
            output = torch.cat([output1, output2], dim=1)  # [batch_size, 2* num_em]
            output = self.relu(self.liner2(output))  # [batch_size, num_em]
            output = self.liner3(output)  # [batch_size, 2]
            output = F.softmax(output, dim=1)

        else:        # abundance_x: (num_em, H, W)
            num_Em, H, W = abundance_x.size()
            prdt_x = torch.reshape(abundance_x, [num_Em, H*W])     # num_Em, H*W
            prdt_y = torch.reshape(abundance_y, [num_Em, H * W])    # num_Em, H*W
            output1 = self.relu(self.liner11(prdt_x.T))    # H*W, num_Em,
            output2 = self.relu(self.liner12(prdt_y.T))
            output = torch.cat([output1, output2], dim=1)  # [H*W, 2* num_em]
            output = self.relu(self.liner2(output))  # [H*W, num_em]
            output = self.liner3(output)   # [H*W, 2]
            output = F.softmax(output, dim=1)  # 按行SoftMax,行和为1; output: [H*W, 2]
        return output


if __name__ == "__main__":
    input1 = torch.rand(1, 1, 162, 307, 307)  # batch, channel, depth, H, W
    input2 = torch.rand(1, 1, 162, 307, 307)  # batch, channel, depth, H, W
    m = UnitedUnmixModule(channel=162, k_size=5, num_em=4)
    abundance_x, abundance_y =m(input1, input2,'test')
    print(abundance_x.size())



