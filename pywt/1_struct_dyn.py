# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 08:55:17 2020

This part is used to deal with the structural dynamic response.
read the response from Excel and rewrite the acceleration data to text in another format, and plot the corresponding figure

@author: 12437
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

def dyna_read(path,out_path):
    dyna_file = pd.read_table(path, encoding="gb2312") #读取文件
    dyna_datas = dyna_file.values.tolist()
    
    dyna_times = []
    dyna_accs = []
    for i,line in enumerate(dyna_datas):
        strline = line[0]
        jointname = strline.split(",")[0]
        thaname = strline.split(",")[1]
        thaname = thaname.rstrip('"')
        thaname = thaname.lstrip('"')
        dyna_time = float(strline.split(",")[2])
        dyna_acc = float(strline.split(",")[3])
        dyna_times.append(dyna_time)
        dyna_accs.append(dyna_acc)
        
        if len(dyna_accs) == 10001:
            txt_path = out_path + "\\conframe_" + str(jointname) + str(thaname) + ".txt"
            txt_out(txt_path,list(dyna_accs),list(dyna_times))
            fig_path = out_path + "\\conframe_" + str(jointname) + str(thaname) + ".png"
            figure_out(fig_path,list(dyna_accs),list(dyna_times))
            
            dyna_times = []
            dyna_accs = []
    
    return None

def myshake_read(path,out_path):
    myshake_names = os.listdir(path)
    for i, myshake_name in enumerate(myshake_names):
        myshake_dir = os.path.join(path,myshake_name)
        myshake_data = np.loadtxt(myshake_dir)
        myshake_time,myshake_acc = myshake_data[:,0],myshake_data[:,1]*10 # time unit = s; acceleration unit should be converted from "g" to "m/s/s"
        ### txt and png output
        txt_path = os.path.join(out_path,myshake_name)
        txt_out(txt_path,list(myshake_acc),list(myshake_time))
        myshake_name_png = myshake_name.split(".txt")[0]+".png"
        fig_path = os.path.join(out_path,myshake_name_png) 
        figure_out(fig_path,list(myshake_acc),list(myshake_time))
    
    return None
    
if __name__ == '__main__':
    path = ".\\1_dynamics\\MyShake"
    out_path = ".\\2_dynaout\\MyShake"
    if not (os.path.exists(out_path)):
        os.makedirs(out_path) #创建输出文件夹
        
#    dyna_read(path,out_path) #读取Excel数据
    myshake_read(path,out_path)
        
