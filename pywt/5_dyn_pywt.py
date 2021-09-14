# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 14:08:43 2020

This part is used to conduct wavelet transform analysis, to get the signal characteristics in the time-frequency domain.
The core function is signal_pywt.
More information about pywt package can be found in https://pywavelets.readthedocs.io/en/latest/

@author: lwjnn
"""

import os
import matplotlib.pyplot as plt
import numpy as np
#小波变换库pywt
import pywt
import matplotlib.colors as col

# 图像输出
def figure_out(outpath,times,freqs,cwts):
    # 生成小波云图    
    plt.figure()
    plt.xlim(0,times[-1]+1)
    plt.ylim(-0.05,20)
#    lim=np.arange(0.5,10.01,0.4)
#    lim=np.arange(0.5,10,0.5)
    lim=np.arange(0.01,0.5,0.03)
    color1 = ['#FFFFFF','#00FF00','#0000FF','#FF0000','#000000']
    color2 = ['#FFFFFF', '#F3F3F3', '#D6D6D6', '#A3A3A3', '#808080', '#696969', '#000000']
    color3 = ['#FFFFFF','r', 'orange', 'yellow', 'green', 'b','purple']
    cmap2 = col.LinearSegmentedColormap.from_list('own2',color3)
    plt.contourf(times,freqs,cwts,lim,cmap=cmap2)
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
#    plt.axis('off')
    plt.savefig(outpath,dpi=200)
    
    return None

# 读取信号文件
def readdata(data_path):
    file = open(data_path,"r")
    datas = file.readlines()
#    sample_rate = len(datas) #采样率
    times, signals = [],[]
    for i,line in enumerate(datas):
        time = float(line.split(",")[0])
        signal = float(line.split(",")[-1])
        times.append(time)
        signals.append(signal)
    
    return times,signals

def signal_pywt(signals):
    #幅值数据
    data_amp = signals
    #小波名称
    wavename = 'cgau8'
    #采样频率
    sampling_rate = 100
#    小波计算scale，决定变换后频率下限
    totalscal = 128
    fc = pywt.central_frequency(wavename)
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(totalscal, 1, -1)
#    小波变换，得到小波系数与频率
    [cwtmatr, frequencies] = pywt.cwt(data_amp, scales, wavename, 1.0 / sampling_rate)
    Abs_cwtmatr=abs(cwtmatr)
    
    return frequencies,Abs_cwtmatr
    
if __name__ == '__main__':
    rootpath = ".\\4_dynanoise\\CESMD_data"
    root_pathout = ".\\5_dynwt\\CESMD_data"
#    root_pathout = ".\\datasets_20200909\\earthquake_new"
    if not (os.path.exists(root_pathout)):
        os.makedirs(root_pathout) #创建输出文件夹
    filenames= os.listdir(rootpath) #得到文件夹下的所有文件名称
#    for i,filename in enumerate(filenames):
#        if filename.split(".")[-1] == "txt": # 判断是否为txt文件
#            data_path = rootpath + "\\" + filename #构造文件路径
#            times,signals = readdata(data_path) # 读取信号文件
#            # 输出数据
#            txt_pathout = root_pathout + "\\" + filename.split(".")[0] + ".txt"
#            np.savetxt(txt_pathout,signals,fmt = '%.8e')
        
    for i,filename in enumerate(filenames):
        if i <= 2:
            if filename.split(".")[-1] == "txt": # 判断是否为txt文件
                data_path = rootpath + "\\" + filename #构造文件路径
                times,signals = readdata(data_path) # 读取信号文件
                frequencies,Abs_cwtmatr = signal_pywt(signals) #小波变换
                # 输出数据
                fig_path = root_pathout + "\\" + filename.split(".")[0] + "_wt.png"
                figure_out(fig_path,times,frequencies,Abs_cwtmatr)
                txt_pathout = root_pathout + "\\" + filename.split(".")[0] + "_wt.txt"
                np.savetxt(txt_pathout,Abs_cwtmatr,fmt = '%.8e')
                txt_pathout_freq = root_pathout + "\\" + filename.split(".")[0] + "_freq.txt"
                np.savetxt(txt_pathout_freq,frequencies,fmt = '%.8e')
