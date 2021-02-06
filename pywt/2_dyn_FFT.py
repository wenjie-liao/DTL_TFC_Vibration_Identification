# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 23:48:57 2020

@author: lwjnn
"""

import os
import xlwt
import numpy as np
import numpy.fft as nf
import matplotlib.pyplot as plt

# 图像输出
def figure_out(path,x,y):
    plt.figure()
    plt.plot(x, y, color='b', linestyle='--')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.savefig(path)
    
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
    maxwave = max(max(signals),abs(min(signals)))
    
    return times,signals,maxwave

# 信号傅里叶变换
def signalfft(filename,rootpath,times,signals):
    # 傅里叶变换后，绘制频域图像
    freqs = nf.fftfreq(len(times), times[1] - times[0])
    complex_array = nf.fft(signals)
    pows = np.abs(complex_array)
    
    freqs = freqs[10:]
    pows = pows[10:]
    
    # 输出傅里叶变换图像
#    fig_path = rootpath + "\\" + filename.split(".")[0] + "FFT.png"
#    figure_out(fig_path,freqs,pows)
    
    # 寻找能量最大的频率值
    fund_freq = freqs[pows.argmax()]
    
    return fund_freq

if __name__ == '__main__':
    rootpath = ".\\2_dynaout\\MyShake"
#    rootpath = ".\\datasets_20200909\\earthquake"
    filenames= os.listdir(rootpath) #得到文件夹下的所有文件名称
    fund_freqs = []
    excelfile = xlwt.Workbook()
    sheet1 = excelfile.add_sheet(u'sheet1',cell_overwrite_ok=True) #创建sheet
    excelpath = (rootpath + "\\fund_freqs.xls")
    
    max_poss = []
    excelfile_max = xlwt.Workbook()
    sheet1_max = excelfile_max.add_sheet(u'sheet1',cell_overwrite_ok=True) #创建sheet
    excelpath_max = (rootpath + "\\max_waves.xls")
    
    row = 1
    for i,filename in enumerate(filenames):
        if filename.split(".")[-1] == "txt": # 判断是否为txt文件
            data_path = rootpath + "\\" + filename #构造文件路径
            times,signals,maxwave = readdata(data_path) # 读取信号文件
            fund_freq = signalfft(filename,rootpath,times,signals)
            fund_freqs.append(abs(fund_freq))
            sheet1.write(row,0,filename)
            sheet1.write(row,1,abs(fund_freq))
            sheet1_max.write(row,0,filename)
            sheet1_max.write(row,1,abs(maxwave))
            row += 1
    excelfile.save(excelpath)
    excelfile_max.save(excelpath_max)
    
    