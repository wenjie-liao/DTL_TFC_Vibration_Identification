# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 14:08:43 2020

This part is used to equivalently simulate collected signals considering the noise of the phone sensor.
the noise file can be obtained by the smartphone.

@author: lwjnn
"""

import os
import xlwt
import random
import numpy as np
import matplotlib.pyplot as plt

# 图像输出
def figure_out(path,x,y):
    plt.figure()
    plt.plot(x, y, color='r', linestyle='-')
    plt.xlabel("Time (s)")
    plt.ylabel("Energy")
    plt.savefig(path)
    
    return None

# 文本输出
def txt_out(path,xs,ys):
    file = open(path, 'w')
    for i,x in enumerate(xs):
#        file.write("%.2f , %.10f \n"%(xs[i],ys[i]))
        file.write("%.10f \n"%(ys[i]))
    file.close()
        
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

# 读取噪音文件并叠加 read and plus
def RPnoise(noises,signals):
    startline = random.randint(0,(len(noises)-len(signals)-1))
    signals_noise = []
    for i,signal in enumerate(signals):
        signal_noise = signal + float(noises[(startline + i)])
        signals_noise.append(signal_noise)
    max_signals_noise = max(signals_noise)

    return signals_noise,max_signals_noise

if __name__ == '__main__':
    rootpath = ".\\3_dynacut\\MyShake"
    root_pathout = ".\\4_dynanoise\\MyShake"
    if not (os.path.exists(root_pathout)):
        os.makedirs(root_pathout) #创建输出文件夹
    filenames= os.listdir(rootpath) #得到文件夹下的所有文件名称
    # 读取噪音文件
    noisepath = ".\\6_noises\\AccelerometerLinear.txt"
    file = open(noisepath,"r")
    noises = file.readlines()
    # 准备信号最大值输出
    excel_max_file = xlwt.Workbook()
    sheet_max = excel_max_file.add_sheet(u'sheet1',cell_overwrite_ok=True) #创建sheet
    excel_max_path = (root_pathout + "\\max.xls")
    row = 0
    for filename in filenames:
        if filename.split(".")[-1] == "txt": # 判断是否为txt文件
            data_path = rootpath + "\\" + filename #构造文件路径
            times,signals = readdata(data_path) # 读取信号文件
            signals_noise,max_signal = RPnoise(noises,signals)
            # 输出数据
            fig_path = root_pathout + "\\" + filename.split(".")[0] + "_noise.png"
#            figure_out(fig_path,times,signals_noise)
            txt_pathout = root_pathout + "\\" + filename.split(".")[0] + "_noise.txt"
            txt_out(txt_pathout,times,signals_noise)
            sheet_max.write(row,0,filename)
            sheet_max.write(row,1,abs(max_signal))
            row += 1

    excel_max_file.save(excel_max_path)
