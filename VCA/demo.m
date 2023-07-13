

clc,clear;
close all
load('');
[C,H,W]=size(img_3d);
R=reshape(img_3d,[C,H*W]);R=double(R);
% img_2d:[C,H*W]
[n_em,~]=HySime(y);
[predict_em, ~, ~ ] = hyperVca(img_2d,n_em);


figure;
color = colormap(jet(n_em));  % M 是你要用的颜色数量
for i=1:n_em
    plot(predict_em(:,i),'color',color(i,:),'LineWidth',2);hold on
end

addpath F:\3月算法\对比算法代码
S1=fcls(predict_em,R);%丰度矩阵, [4, 94294]
% 丰度
predict_ab=reshape(S1,[n_em,H,W]);
figure;
for i=1:n_em
    subplot(1,n_em,i);
    imshow(squeeze(predict_ab(i,:,:)));colorbar;colormap jet
    hold on
end
legend('1','2','3','4');

path='H:\hmq\220电脑\E-光谱解混\unmixing\unmixing-data\Urban（辐射值）\groundTruth_Urban_end4\';
file='end4_groundTruth.mat';
load([path,file]);
endmember=M;
abundance=A;
% 光谱角距离SAD
% 预测端元和真实端元之间
predict_em=sum(predict_em.*endmember);
z1=sqrt(sum(predict_em.*predict_em));
z2=sqrt(sum(endmember.*endmember));
SAD=predict_em./(z1.*z2);SAD=acos(SAD);

%  RMSE(Root Mean Square Error)
%预测丰度和真实丰度之间
z1 = S1 - abundance;
z2=sum(z1.*z1,2)./H./W;
RMSE=sqrt(z2);
%% 
load('F:\3月算法\解混数据集\Urban_188_em4.mat');
% step1:先提取端元
[C,H,W]=size(img_3d);
img_2d=reshape(img_3d,[C,H*W]);img_2d=double(img_2d);
n_em=size(endmember,2);
addpath F:\3月算法\对比算法代码
[predict_em, indicies, snrEstimate ] = hyperVca(img_2d,n_em);

figure;
color = colormap(jet(n_em));  % M 是你要用的颜色数量
for i=1:n_em
    plot(predict_em(:,i),'color',color(i,:),'LineWidth',2);hold on
end
legend('#1','#2','#3','#4');

% step2:计算丰度
addpath F:\3月算法\对比算法代码
S1=fcls(predict_em,img_2d);%丰度矩阵, [4, 94294]
% 丰度
predict_ab=reshape(S1,[n_em,H,W]);
figure;
for i=1:n_em
    subplot(1,n_em,i);
    imshow(squeeze(predict_ab(i,:,:)));colorbar;colormap jet
    title(['#',num2str(i)]);
    hold on
end




