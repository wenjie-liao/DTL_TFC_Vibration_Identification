# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import numpy as np
#小波变换库pywt
import pywt
import matplotlib.colors as col

# 图像输出
def figure_out(outpath,times,freqs,cwts):
    # 生成小波云图
    plt.xlim(0,times[-1]+1)
    plt.ylim(-0.05,20)
#    lim=np.arange(0.5,10.01,0.4)
#    lim=np.arange(0.5,10,0.5)
    lim=np.arange(0.005,1,0.04)
    color1 = ['white','gray','violet','purple','royalblue','blue','navy','cyan','springgreen','lime',
              'green','yellow','gold','orange','darkorange','red']
    cmap2 = col.LinearSegmentedColormap.from_list('own2',color1)
    plt.contourf(times,freqs,cwts,lim,cmap=cmap2)
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
#    plt.axis('off')
    plt.savefig(outpath,dpi=200,bbox_inches='tight')
    plt.close('all')
    
    return None
    
if __name__ == '__main__':
    rootpath = ".\\5_dynwt\\MyShake"
    root_pathout = ".\\5_dynwt\\MyShake"
#    root_pathout = ".\\datasets_20200909\\earthquake_new"
    if not (os.path.exists(root_pathout)):
        os.makedirs(root_pathout) #创建输出文件夹
    filenames= os.listdir(rootpath) #得到文件夹下的所有文件名称
        
    for i,filename in enumerate(filenames):
        if i >= 0 and i<=500:
            if filename.split(".")[-1] == "txt": # 判断是否为txt文件
                data_path = rootpath + "\\" + filename #构造文件路径
                times = list(np.arange(0,40,0.01)) #时间
                frequencies = np.loadtxt(".\\5_dynwt\\datasets_20200909\\WT_freq_128scale.txt") #频率
                Abs_cwtmatr = np.loadtxt(data_path) #小波系数矩阵
                # 输出数据
                fig_path = root_pathout + "\\" + filename.split(".")[0] + ".png"
                figure_out(fig_path,times,frequencies,Abs_cwtmatr)
