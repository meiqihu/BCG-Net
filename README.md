# BCG-Net
This is the Pytorch code of "Binary Change Guided Hyperspectral Multiclass Change Detection".
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
  
  
