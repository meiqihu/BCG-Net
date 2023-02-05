# BCG-Net
This is the Pytorch code of TIP paper "Binary Change Guided Hyperspectral Multiclass Change Detection".

The paper is available on https://arxiv.org/abs/2112.04493v2

My personal google web: https://scholar.google.com.hk/citations?hl=zh-CN&user=jxyAHdkAAAAJ

-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- 

The data file, including input data and training samples, is available in 

Baidu Netdisk, Link：https://pan.baidu.com/s/1AVG7YhU1e9NYSgcruL7PcQ 
code：h3ul

or google drive,Link: https://drive.google.com/drive/folders/1qxtbLm4zu6pNvN25ypfarssGSSJV7U-B.

The structure of data file is show as follows:

-data

  --ChinaData
  
    ---China_MultChange.mat (input data)
    
    ---China_sampIdx_12288_4096.mat (used for pre-training of United Unmixing Module.)
    
    ---China_sampIdx_16384.mat (for pre-training of Temporal Correlation Module and alternative optimization of the two modules)
    
  --UrbanData
  
  --USAData
  
  -=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

  [Usage]: maincode.py
  
  
  
 # Binary Change Guided Hyperspectral Multiclass Change Detection
Pytorch implementation of TIP paper "Binary Change Guided Hyperspectral Multiclass Change Detection"
![image](https://github.com/meiqihu/HyperNet/blob/main/Figure-HyperNet.jpg)
# Paper
[Binary Change Guided Hyperspectral Multiclass Change Detection](https://ieeexplore.ieee.org/document/10011164)

Please cite our paper if you find it useful for your research.

>@ARTICLE{9934933,
  author={Hu, Meiqi and Wu, Chen and Zhang, Liangpei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={HyperNet: Self-Supervised Hyperspectral Spatial–Spectral Feature Understanding Network for Hyperspectral Change Detection}, 
  year={2022},
  volume={60},
  number={},
  pages={1-17},
  doi={10.1109/TGRS.2022.3218795}}
# Installation
Install Pytorch 1.10.2 with Python 3.6
# Dataset
Download the [dataset of Viareggio 2013 and Simulated Hymap for HACD task, and USA, Bay, and Barbara dataset for HBCD task](https://pan.baidu.com/s/1c5Bi8bkqUolWdGKbNmfU1Q),passcode提取密码：af57
> Viareggio_data.mat
> 
> num_idx_ex1.mat, num_idx_ex2.mat
> 
> Bay.mat
> 
> num_idx_Bay.mat

# More
[My personal google web](https://scholar.google.com.hk/citations?hl=zh-CN&user=jxyAHdkAAAAJ)




  
  
