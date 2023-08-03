 # Binary Change Guided Hyperspectral Multiclass Change Detection
Pytorch implementation of TIP paper "Binary Change Guided Hyperspectral Multiclass Change Detection"
![image](https://github.com/meiqihu/BCG-Net/blob/main/Figure_BCG-Net.png)
# Paper
[Binary Change Guided Hyperspectral Multiclass Change Detection](https://ieeexplore.ieee.org/document/10011164)

Please cite our paper if you find it useful for your research.

>@ARTICLE{10011164,
  author={Hu, Meiqi and Wu, Chen and Du, Bo and Zhang, Liangpei},
  journal={IEEE Transactions on Image Processing}, 
  title={Binary Change Guided Hyperspectral Multiclass Change Detection}, 
  year={2023},
  volume={32},
  number={},
  pages={791-806},
  doi={10.1109/TIP.2022.3233187}}
  
# Installation
Install Pytorch 1.10.2 with Python 3.6
# Dataset
Download the [data file, including input data and training samples](https://pan.baidu.com/s/1AVG7YhU1e9NYSgcruL7PcQ ),passcode提取密码：h3ul
Baidu Netdisk, Link：https://pan.baidu.com/s/1AVG7YhU1e9NYSgcruL7PcQ 
code：h3ul

or google drive,Link: https://drive.google.com/drive/folders/1qxtbLm4zu6pNvN25ypfarssGSSJV7U-B.

> ChinaData
>> China_MultChange.mat (input data)
>>> X_3d：channel, height, width                
>>> Y_3d：channel, height, width                
>>> endmember：channel, num_em               
>>> Two_CMap：height, width （0 means unchanged, 1 means changed）
>>> Mul_CMap：height, width   （0 means unchanged, 1 means changed）

>> China_sampIdx_12288_4096.mat (used for pre-training of United Unmixing Module.)
>>> binary_label, 1,12288（the first 8192 samples are unchanged（labeled value as 0）; the remained 4096 samples are changed, labeled value as 1）         
>>> idx_sample: 1, 12288（index of the samples, used for python index）          

>> China_sampIdx_16384.mat (for pre-training of Temporal Correlation Module and alternative optimization of the two modules)
>>> idx_sample
# Usage
maincode.py

# More
🌷[Homepage](https://meiqihu.github.io/)🌷  </br>
🔴[Google web](https://scholar.google.com.hk/citations?hl=zh-CN&user=jxyAHdkAAAAJ) 🔴 </br>
🌏[ResearchGate](https://www.researchgate.net/profile/Humeiqi-humeiqi) 🌍




  
  
