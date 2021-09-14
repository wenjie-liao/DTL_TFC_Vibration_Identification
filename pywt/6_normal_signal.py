# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 22:09:51 2020

This part is used to deal with normal signals, not useful, [emoji]

@author: lwjnn
"""

import os
import xlwt
import math
import numpy as np
import numpy.fft as nf
import matplotlib.pyplot as plt
import pandas as pd

# 图像输出
def figure_out(path,x,y):
    plt.figure()
    plt.plot(x, y, color='black', linestyle='-')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.savefig(path)
    
    return None

# 文本输出
def txt_out(path,xs,ys):
    file = open(path, 'w')
    new_ys = []
    for i,x in enumerate(xs):
        try:
            new_y = float(ys[i])
        except ValueError:
            continue
        file.write("%.3f , %.10f \n"%(x,new_y))
        new_ys.append(new_y)
    file.close()
    max_y = max(new_ys)
        
    return new_ys,max_y

# CSV文件读取
def readcsv(path):
    datas = pd.read_csv(path)
    acc_xs = list(np.array(datas.iloc[:,2]))
    acc_ys = list(np.array(datas.iloc[:,3]))
    
    return acc_xs,acc_ys

def noise_data(rootpath,noises):
    filenames= os.listdir(rootpath) #得到文件夹下的所有文件名称
    for i,filename in enumerate(filenames):
        file_path = os.path.join(rootpath, filename) #构造文件路径        
        if os.path.isdir(file_path):
            noises = noise_data(file_path,noises)
#        elif filename.split(".")[-1] == "csv": # 判断是否为txt文件
        elif filename == "AccelerometerLinear.csv": # 判断是否为txt文件
            # 读取噪音文件
            # 读取txt
#            file = open(noisepath,"r")
#            noise = file.readlines()
#            noises.extend(noise)
#            # 读取csv
            acc_xs,acc_ys = readcsv(file_path)
            noises.extend(acc_xs)
            noises.extend(acc_ys)
    return noises

# 信号傅里叶变换
def signalfft(filename,rootpath,times,signals):
    # 傅里叶变换后，绘制频域图像
    freqs = nf.fftfreq(len(times), times[1] - times[0])
    complex_array = nf.fft(signals)
    pows = np.abs(complex_array)
    
    freqs = freqs[10:]
    pows = pows[10:]
    
    # 输出傅里叶变换图像
#    fig_path = rootpath + "\\" + filename.split(".")[0] + ".png"
#    figure_out(fig_path,freqs,pows)
    
    # 寻找能量最大的频率值
    fund_freq = freqs[pows.argmax()]
    
    return fund_freq

if __name__ == '__main__':
    rootpath = ".\\SensorRecord"
    root_pathout = ".\\6_noises\\dynamic"
    if not (os.path.exists(root_pathout)):
        os.makedirs(root_pathout) #创建输出文件夹
    
    noises = []
    noises = noise_data(rootpath,noises)
    
    totallen = len(noises)
    time_len = 40
    times = list(np.arange(0,time_len,0.01))
    Snoiselen = len(times)
    numnoise = math.floor(totallen/Snoiselen)
    # 准备输出噪音的频率特征
    fund_freqs = []
    excel_freq_file = xlwt.Workbook()
    excel_max_file = xlwt.Workbook()
    sheet1 = excel_freq_file.add_sheet(u'sheet1',cell_overwrite_ok=True) #创建sheet
    sheet_max = excel_max_file.add_sheet(u'sheet1',cell_overwrite_ok=True) #创建sheet
    excel_freq_path = (root_pathout + "\\dynamic_fundfreqs.xls")
    excel_max_path = (root_pathout + "\\dynamic_max.xls")
    for i in range(numnoise):
#    for i in range(10):
        if i < 2760:
            endline = Snoiselen*(i+1)
            startline = endline - Snoiselen
            Snoise = noises[startline:endline]
            # 输出数据
            txt_pathout = root_pathout + "\\dynamic_%d.txt"%i
            new_Snoise,max_Snoise = txt_out(txt_pathout,times,Snoise)
#            fig_path = root_pathout + "\\noise_liaowj_%d.png"%i
#            figure_out(fig_path,times,new_Snoise)
            # FFT分析
            filename = "dynamic_FFT_%d.txt"%i
            fund_freq = signalfft(filename,root_pathout,times,new_Snoise)
            fund_freqs.append(abs(fund_freq))
            sheet1.write(i,0,filename)
            sheet1.write(i,1,abs(fund_freq))
            sheet_max.write(i,0,filename)
            sheet_max.write(i,1,abs(max_Snoise))

    excel_freq_file.save(excel_freq_path)
    excel_max_file.save(excel_max_path)
        
