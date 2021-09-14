# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 10:03:42 2020

This part is used to deal with the raw ground motions downloaded from CESMD, maybe useful, [emoji]

@author: lwjlalala
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def txt_out(path,accs,times):
    file = open(path, 'w')
    for i,acc in enumerate(accs):
        file.write("%.2f , %.10f \n"%(times[i],accs[i]))
    file.close()
        
    return None

def figure_out(path,accs,times):
    plt.figure()
    plt.plot(times, accs, color='gray', linestyle='-')
    plt.xlabel("time (s)")
    plt.ylabel("acceleration (m/s/s)")
    plt.savefig(path)
    
    return None

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def dyna_read(filename,out_path,name):
    dyna_file = pd.read_table(filename, encoding="gb2312",header=None,skiprows=46) #读取文件
    dyna_matrix = dyna_file.values.tolist()
    dyna_list = []
    for i,line in enumerate(dyna_matrix):
        line_str = line[0]
        for i in range(8):
            data = line_str[(0+(i*10)):(10+(i*10))].strip()
            if is_number(data):
                data = float(data)/100  #convert unit from cm/s/s to m/s/s
                dyna_list.append(data)
            else:
                break
    
    totallen = len(dyna_list)
    time_len = totallen/100
    times = list(np.arange(0,time_len,0.01))
    txt_out((out_path + name + ".txt"),dyna_list,times)
    figure_out((out_path + name + ".png"),dyna_list,times)
    
    return dyna_list
    
if __name__ == '__main__':
    path = ".\\CESMD_data\\Redlands - 7-story Commercial"
    out_path = ".\\2_dynaout\\CESMD_data\\"
    if not (os.path.exists(out_path)):
        os.makedirs(out_path) #创建输出文件夹
    
    files = os.listdir(path)
    for i,filename in enumerate(files):
        if filename.split(".")[-1] == "V2":
            name = path.split("\\")[-1] + filename.split(".")[0]
            readpath = os.path.join(path,filename)
            dyna_list = dyna_read(readpath,out_path,name) #读取Excel数据
        
