# coding=utf-8
# Note: This is the code of "Binary Change Guided Hyperspectral Multiclass Change Detection"

from torch import nn
import numpy as np


def initNetParams(net):
    '''Init net parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight.data)
            # init.xavier_uniform(m.weight)
            nn.init.constant_(m.bias, 0)
            # if m.bias:
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)      # m.weight.data.normal_(0, 0.001)
            # init.xavier_uniform(m.weight)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight.data)      # init.constant(m.weight, 1)
            # nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            nn.init.constant_(m.bias, 0)        # init.constant(m.bias, 0)
            # m.weight.data.normal_(0, 0.01)  # np.sqrt(2/127))

def get_sample_patch(img_3d, idx_sample, patch_size):
    # img_3d : img_channel, img_height, img_width
    k = patch_size//2
    img_channel, img_height, img_width = img_3d.shape     # 162,307,307
    patch_set = []
    pixel_set = []

    expand_img_3d = img_3d     # 162,307,307

    temp1 = img_3d[:, :, :k]
    temp2 = img_3d[:, :, -k:]
    expand_img_3d = np.concatenate((temp1, expand_img_3d), axis=2)
    expand_img_3d = np.concatenate((expand_img_3d, temp2), axis=2)

    temp3 = img_3d[:, :k, :]
    temp4 = img_3d[:, -k:, :]
    temp3 = np.concatenate((temp1[:, :k, :],temp3), axis=2)
    temp3 = np.concatenate((temp3, temp2[:, :k, :]), axis=2)
    temp4 = np.concatenate((temp1[:, -k:, :],temp4), axis=2)
    temp4 = np.concatenate((temp4, temp2[:, -k:, :]), axis=2)
    expand_img_3d = np.concatenate((temp3, expand_img_3d), axis=1)
    expand_img_3d = np.concatenate((expand_img_3d, temp4), axis=1)

    for idx in idx_sample:
        row = int(np.ceil((idx+1)/img_height)-1)        # 列
        col = int(idx-row*img_height)        # 行
        patch_set.append(expand_img_3d[:, col:(col+7), row:(row+7)])
        pixel_set.append(expand_img_3d[:, (col+3), (row+3)])

    patch_set = np.asarray(patch_set)
    pixel_set = np.asarray(pixel_set)

    return patch_set, pixel_set, expand_img_3d
def test_get_sample_patch(img_3d, idx_sample, patch_size):
    # img_3d : img_channel, img_height, img_width
    k = patch_size//2
    img_channel, img_height, img_width = img_3d.shape     # 162,307,307
    patch_set = []
    pixel_set = []

    expand_img_3d = img_3d     # 162,307,307

    temp1 = img_3d[:, :, :k]
    temp2 = img_3d[:, :, -k:]
    expand_img_3d = np.concatenate((temp1, expand_img_3d), axis=2)
    expand_img_3d = np.concatenate((expand_img_3d, temp2), axis=2)

    temp3 = img_3d[:, :k, :]
    temp4 = img_3d[:, -k:, :]
    temp3 = np.concatenate((temp1[:, :k, :],temp3), axis=2)
    temp3 = np.concatenate((temp3, temp2[:, :k, :]), axis=2)
    temp4 = np.concatenate((temp1[:, -k:, :],temp4), axis=2)
    temp4 = np.concatenate((temp4, temp2[:, -k:, :]), axis=2)
    expand_img_3d = np.concatenate((temp3, expand_img_3d), axis=1)
    expand_img_3d = np.concatenate((expand_img_3d, temp4), axis=1)
    for idx in idx_sample:
        # row = int(np.ceil((idx+1)/img_height)-1)        # 列
        # col = int(idx-row*img_height)        # 行
        col = int(np.ceil((idx+1)/img_width)-1)  # 行
        row = int(idx-col*img_width)  # 列
        patch_set.append(expand_img_3d[:, col:(col+7), row:(row+7)])
        pixel_set.append(expand_img_3d[:, (col+3), (row+3)])

    patch_set = np.asarray(patch_set)
    pixel_set = np.asarray(pixel_set)

    return patch_set, pixel_set, expand_img_3d
def test_get_sample_set(x_3d, y_3d, patch_size, sample):
    # function: get patch for training
    # [dict]sample:
    #     Urban_sampIdx_8192.mat:{'idx_sample': idx_sample, 'binary_label': binary_label}
    #   idx_u_sample: [array] index 1536;   idx_c_sample: [array] index 512 ; bi_label:[array] 94249

    idx_sample = sample['idx_sample'].squeeze()
    binary_label = sample['binary_label'].squeeze()

    patch_set_x, pixel_set_x, expand_x_3d = test_get_sample_patch(x_3d, idx_sample, patch_size)
    patch_set_y, pixel_set_y, expand_y_3d = test_get_sample_patch(y_3d, idx_sample, patch_size)
    return patch_set_x, pixel_set_x, expand_x_3d, patch_set_y, pixel_set_y, expand_y_3d, binary_label


def two_cls_access(reference,result):
    # 对二类变化检测的结果进行精度评价，指标为kappad系数和OA值
    # 输入：
    #      reference：二元变化reference(二值图，H*W)
    #      resultz:算法检测得到的二类变化结果图(二值图，H*W)]
    oa_kappa = []
    m,n = reference.shape
    if reference.shape != result.shape:
        print('the size of reference shoulf be equal to that of result')
        return oa_kappa
    reference = np.reshape(reference, -1)
    result = np.reshape(result, -1)
    label_0 = np.where(reference == 0)
    label_1 = np.where(reference == 1)
    predict_0 = np.where(result == 0)
    predict_1 = np.where(result == 1)
    label_0 = label_0[0]
    label_1 = label_1[0]
    predict_0 = predict_0[0]
    predict_1 = predict_1[0]
    tp = set(label_1).intersection(set(predict_1))  # True Positive
    tn = set(label_0).intersection(set(predict_0))  # False Positive

    oa = (len(tp)+len(tn))/m/n      # Overall precision
    pe = (len(label_1)*len(predict_1)+len(label_0)*len(predict_0))/m/n/m/n
    kappa = (oa-pe)/(1-pe)
    oa = round(oa, 4)
    kappa = round(kappa, 4)
    oa_kappa.append('OA')
    oa_kappa.append(oa)
    oa_kappa.append('kappa')
    oa_kappa.append(kappa)
    print('OA:  ' + str(oa) + '    ' + 'kappa:  ' + str(kappa))
    return oa_kappa

def ChangeDetection_V1_net(abundance_x_3d, abundance_y_3d, Two_Chge_Map):
    # 二类变化：根据神经网络输出二类变化
    # 对变化类：以最大丰度值对应的端元类作为该像元在两个时相的类别，对比得到变化类
    print('---func: ChangeDetection_V1_net---')
    num_em, H, W = abundance_x_3d.shape
    num_pixel = H*W
    abudace_x_2d = np.reshape(abundance_x_3d, [num_em, num_pixel])
    abudace_y_2d = np.reshape(abundance_y_3d, [num_em, num_pixel])
    # 区分变化与未变化类: 主要成分不一致，视为变化
    max_idx1 = np.argmax(abudace_x_2d, axis=0)  # 取每个像元丰度向量的最大值的位置 : num_pixel,1
    max_idx2 = np.argmax(abudace_y_2d, axis=0)  # 取每个像元丰度向量的最大值的位置
    # temp = max_idx1 == max_idx2      # logical=1 表示主要成分一致;未变化
    # Two_Chge_Map = 1 - temp      # 未变化用0表示；变化用1表示
    # Two_Chge_Map = np.reshape(Two_Chge_Map, [H, W])

    Two_Chge_Map_2d = np.reshape(Two_Chge_Map, [H * W])
    temp = np.where(Two_Chge_Map_2d == 0)

    max_idx1 = max_idx1 + 1  #
    max_idx2 = max_idx2 + 1
    # 对变化类: 计算各丰度向量最大的丰度值
    max_idx1[temp] = 0      #只保留变化类像元的最大端元类
    max_idx2[temp] = 0
    # 某变化像元在T1中主要端元成为为第2个；在T2中为第3个，则组成23变化类
    Change_Transf = max_idx1 * 10 + max_idx2
    Clusters = set(Change_Transf)
    Clusters = list(Clusters)
    print('the change classes of the multple map are')
    print(np.array(Clusters))
    Mul_Chge_Map = np.zeros([H * W, 1])
    if len(Clusters)>1:
        # T1->T2, 找出不同类变化的个数
        num = np.zeros([len(Clusters) - 1, 2])
        for i in np.arange(len(Clusters) - 1):
            num[i, 0] = Clusters[i + 1]
            num[i, 1] = len(np.where(Change_Transf == num[i, 0])[0])
        print('the num of multi-class change is ' + str(len(Clusters) - 1))

        for i in np.arange(len(num)):
            Mul_Chge_Map[np.where(Change_Transf == num[i, 0])] = i+1
    Mul_Chge_Map = np.reshape(Mul_Chge_Map, [H, W])
    # plt.figure('ChangeDetection_V1_net')
    # plt.subplot(1, 2, 1)
    # plt.imshow(Two_Chge_Map)
    # plt.title('Two-Classes')
    # plt.subplot(1, 2, 2)
    # plt.imshow(Mul_Chge_Map)
    # plt.show()
    # plt.title('Multi-Class')
    return Two_Chge_Map, Mul_Chge_Map


def two_cls_access_for_mulipleAccess(reference,result):
    # 对二类变化检测的结果进行精度评价，指标为kappad系数和OA值
    # 输入：
    #      reference：二元变化reference(二值图，H*W)
    #      resultz:算法检测得到的二类变化结果图(二值图，H*W)]
    oa_kappa = []
    m,n = reference.shape
    if reference.shape != result.shape:
        print('the size of reference shoulf be equal to that of result')
        return oa_kappa
    reference = np.reshape(reference, -1)
    result = np.reshape(result, -1)
    label_0 = np.where(reference == 0)
    label_1 = np.where(reference == 1)
    predict_0 = np.where(result == 0)
    predict_1 = np.where(result == 1)
    label_0 = label_0[0]
    label_1 = label_1[0]
    predict_0 = predict_0[0]
    predict_1 = predict_1[0]
    tp = set(label_1).intersection(set(predict_1))  # True Positive
    tn = set(label_0).intersection(set(predict_0))  # False Positive
    oa = (len(tp)+len(tn))/m/n      # Overall precision
    pe = (len(label_1)*len(predict_1)+len(label_0)*len(predict_0))/m/n/m/n
    kappa = (oa-pe)/(1-pe)
    oa = round(oa, 4)
    kappa = round(kappa, 4)
    oa_kappa.append('OA')
    oa_kappa.append(oa)
    oa_kappa.append('kappa')
    oa_kappa.append(kappa)

    print('OA:  ' + str(oa) + '    ' + 'kappa:  ' + str(kappa))
    return oa_kappa
def mul_cls_access(label, predict):
    """
    对多类变化检测的结果进行精度评价，指标为kappad系数和OA值
    输入：
        label：多元变化reference(0，1，2，...，H * W)
        predict: 算法检测得到的多类变化结果图(0，1，2，H * W)]
    输出：
        OA_kappa：4 * 7的cell，第1行为label的类序号；
            第2行为predict的类序号，第3行为OA；第4行为kappa
        match_predict: 对各个类的序号按与ref匹配的序号赋值，
            match_predict取值为 - 1，0，1，2，3等，-1为未匹配的类；0为变化类
    当predict为3个变化类，ref为5个时，3个类与对应匹配为1，3，5；未变化为0
    当predict为7个变化类，ref为5个时，5个类命名未1，2，3，4，5，未变化为0，剩余两个命名为6，7
    """
    print('---Multiple change Accessment---')
    m, n = label.shape
    OA_kappa, match_predict = [], []
    if label.shape != predict.shape:
        print('the size of label should be equal to that of predict')
        return OA_kappa, match_predict
    label = np.reshape(label, -1)
    predict = np.reshape(predict, -1)
    num_label = int(np.max(label))  # ref中变化类别的总数
    num_predict = int(np.max(predict))  # 预测结果中变化类别的总数
    print('-----num_predict :  ' + str(num_predict))
    # 对变化类进行匹配
    label_2d = np.zeros([m * n, num_label])
    for i in np.arange(num_label):
        label_2d[np.where(label == i+1), i] = 1

    predict_2d = np.zeros([m * n, num_predict])
    for i in np.arange(num_predict):
        predict_2d[np.where(predict == i+1), i] = 1

    label_2d_sum = np.sum(label_2d, axis=0)  # 按列求和
    label_sum_2d = np.tile(label_2d_sum, (num_predict, 1))      # [num_predict,num_label]
    predict_2d_sum = np.sum(predict_2d, axis=0)  # 按列求和
    predict_sum_2d = np.tile(predict_2d_sum, (num_label,1)).T

    if np.any(predict_sum_2d==0):
        print('there are 0 in array [label_sum_2d]')
        predict_sum_2d[predict_sum_2d == 0] = -1
    tabel = np.matmul(np.transpose(predict_2d), label_2d)/label_sum_2d + np.matmul(np.transpose(predict_2d) , label_2d)/predict_sum_2d
    if np.isnan(tabel).any():
        print('there are nans in array [tabel]')
        tabel[np.isnan(tabel)]=0
    tabel0 = tabel      # [num_predict, num_label]
    match_predict = np.zeros([m*n,1])   # 对各个类的序号按与ref匹配的序号赋值
    # SAD第1行是label的1，2，3，等
    # SAD第2行是predict匹配上label对应的类别序号，如3，2，5，等
    # SAD第3行是OA
    # SAD第4行是kappa
    # 第5行表示整体精度的oa与kappa
    idx = np.arange(1, num_label+1)     # 变化类的序号：1,2,...,num_label

    if num_label <= num_predict:  # 实际有5种变化，但检测出7种
        # 当实际检测的类更多时，除了与真实的类匹配外，剩下的类被赋值为-1
        SAD = np.zeros([5, num_label + 1])  # 加的1指的是未变化类
        SAD[0, :num_label] = idx
        j = 1
        while j <= num_label:  # 变化类的OA与kappa
            label_i = np.zeros([m*n,1])
            predict_i = np.zeros([m*n,1])
            [row, col] = np.where(tabel0 == np.max(tabel0))    # find the col and row idx of max value
            SAD[1, col[0]] = row[0]+1
            tabel0[row[0], :] = -1
            tabel0[:, col[0]] = -1
            label_i[np.where(label == col[0]+1)] = 1
            predict_i[np.where(predict == row[0]+1)] = 1
            oa_kappa_i = two_cls_access_for_mulipleAccess(label_i, predict_i)
            SAD[2, col[0]] = oa_kappa_i[1]      # oa
            SAD[3, col[0]] = oa_kappa_i[3]      # kappa
            match_predict[np.where(predict == row[0]+1)] = col[0]+1
            j = j+1
    # predict有7类，匹配了ref中的5类, 找出剩下的未匹配的
        t=SAD[1, :num_label]
        idx_predict = np.arange(1, num_predict + 1)  # 变化类的序号：1,2,...,num_label
        remain = list(set(idx_predict).difference(set(SAD[1, :num_label])))
        for i in remain:  # remain = [1，7]，表示predict中这两类没有匹配项
            match_predict[np.where(predict == i)] = -1  # 未匹配的类用 - 1表示

    # 未变化类的OA与kappa
        label_i = np.zeros([m*n,1])
        predict_i = np.zeros([m*n,1])
        label_i[np.where(label == 0)] = 1
        predict_i[np.where(predict == 0)] = 1
        oa_kappa_i = two_cls_access_for_mulipleAccess(label_i, predict_i)
        SAD[2, num_label] = oa_kappa_i[1]      # oa
        SAD[3, num_label] = oa_kappa_i[3]      # kappa
    else:  # num_label > num_predict 实际有5种变化，但只检测出3种
        # 当实际检测的类更少时，只需将这些类与真实的类匹配
        SAD = np.zeros([5, num_label + 1])
        SAD[0, :num_label] = idx
        j = 1
        while j <= num_predict:  # 变化类的OA与kappa        #变化类的OA与kappa
            label_i = np.zeros([m*n,1])
            predict_i = np.zeros([m*n,1])
            [row, col] = np.where(tabel0 == np.max(tabel0))  # find the col and row idx of max value
            SAD[1, col[0]] = row[0]+1
            tabel0[row[0], :] = -1
            tabel0[:, col[0]] = -1
            label_i[np.where(label == col[0]+1)] = 1
            predict_i[np.where(predict == row[0]+1)] = 1
            oa_kappa_i = two_cls_access_for_mulipleAccess(label_i, predict_i)
            SAD[2, col[0]] = oa_kappa_i[1]  # oa
            SAD[3, col[0]] = oa_kappa_i[3]  # kappa
            match_predict[np.where(predict == row[0]+1)] = col[0]+1
            j =j +1
    # 未变化类的OA与kappa
        label_i = np.zeros([m*n,1])
        predict_i = np.zeros([m*n,1])
        label_i[np.where(label == 0)] = 1
        predict_i[np.where(predict == 0)] = 1
        oa_kappa_i = two_cls_access_for_mulipleAccess(label_i, predict_i)
        SAD[2, num_label] = oa_kappa_i[1]      # oa
        SAD[3, num_label] = oa_kappa_i[3]      # kappa
    SAD[0, num_label] = 0

    # 计算总体分类精度：每一类正数确分类的样本量之和除以总样本数
    idx = 0  # 0, 1, 2, 其中0表示未变化类
    IDX = 0
    for i in np.arange(num_label+1):
        idx1 = np.where(match_predict == i)
        idx2 = np.where(label == i)
        idx = idx +len(set(idx1[0]).intersection(set(idx2[0])))
        IDX = IDX + len(idx1[0]) * len(idx2[0])
    OA = idx / m / n
    pe = IDX / m / n / m / n
    kappa = (OA - pe) / (1 - pe)
    SAD[4, 0] = round(OA, 4)
    SAD[4,2] = round(kappa, 4)
    OA_kappa = SAD
    OA = round(OA, 4)       # round() 方法返回浮点数x的四舍五入值
    kappa = round(kappa, 4)
    print('whole OA is   ' + str(OA))
    print('whole kappa is   ' + str(kappa))
    match_predict = np.reshape(match_predict, [m, n])
    return OA_kappa, match_predict
