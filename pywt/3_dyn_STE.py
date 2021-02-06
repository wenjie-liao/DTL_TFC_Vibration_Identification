# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 01:24:05 2020

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
    plt.xlabel("Time (s)")
    plt.ylabel("Energy")
    plt.savefig(path)
    
    return None

# 文本输出
def txt_out(path,xs,ys):
    file = open(path, 'w')
    for i,x in enumerate(xs):
        file.write("%.2f , %.10f \n"%(xs[i],ys[i]))
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

# 信号短时能量计算，N个采样点为一帧
def calEnergy(wave_data) :
    energys,positions = [],[]
    framelen = 100 #每100个采样点为1帧
    energy,position = 0,0
    positions.append(position)
    for i in range(len(wave_data)) :
        energy = energy + (float(wave_data[i]) * float(wave_data[i]))
        if (i + 1) % framelen == 0 :
            energys.append(energy)
            energy = 0
            position += 1
            positions.append(position)
        elif i == len(wave_data) - 2 :
#        elif i == len(wave_data):
            energys.append(energy)
    # 寻找能量最大的时刻
    energys = np.array(energys)
    max_pos = (positions[energys.argmax()])*framelen
    return positions,energys,max_pos

def signalcut(wave_data,max_pos,framelen,time_len):
    start_line = int(max_pos - time_len*framelen/2)
    if start_line <= 0:
        start_line = 0
    end_line = (start_line + time_len*framelen)
    cut_datas = wave_data[start_line:end_line]
    new_times = list(np.arange(0,time_len,0.01))
    max_wave = abs(wave_data[max_pos])
    
    return new_times,cut_datas,max_wave

if __name__ == '__main__':
    rootpath = ".\\2_dynaout\\MyShake"
    root_pathout = ".\\3_dynacut\\MyShake"
    if not (os.path.exists(root_pathout)):
        os.makedirs(root_pathout) #创建输出文件夹
    filenames= os.listdir(rootpath) #得到文件夹下的所有文件名称
    max_poss = []
#    excelfile = xlwt.Workbook()
#    sheet1 = excelfile.add_sheet(u'sheet1',cell_overwrite_ok=True) #创建sheet
#    excelpath = (root_pathout + "\\max_waves.xls")
    row = 1
    for i,filename in enumerate(filenames):
        if filename.split(".")[-1] == "txt": # 判断是否为txt文件
            data_path = rootpath + "\\" + filename #构造文件路径
            times,signals = readdata(data_path) # 读取信号文件
            positions,energys,max_pos = calEnergy(signals)
            # 输出短时能量图像
            fig_path = rootpath + "\\" + filename.split(".")[0] + "STE.png"
            minlen = min(len(positions),len(energys))
            figure_out(fig_path,positions[:minlen],energys[:minlen])
            # 输出短时能量的位置信息
            max_poss.append(max_pos)
            # 输出短时能量数据
#            txt_path = rootpath + "\\" + filename.split(".")[0] + "STE.txt"
#            txt_out(txt_path,list(positions),list(energys))
            # 裁剪后数据输出
            time_len = 40 #初定裁剪后时间长度40s
            framelen = 100 #一帧内数据数量
            new_times,cut_datas,max_wave = signalcut(signals,max_pos,framelen,time_len)
            txt_pathout = root_pathout + "\\" + filename.split(".")[0] + "_cut.txt"
            txt_out(txt_pathout,new_times,cut_datas)
#            sheet1.write(row,0,filename)
#            sheet1.write(row,1,max_wave)
            row += 1
#    excelfile.save(excelpath)