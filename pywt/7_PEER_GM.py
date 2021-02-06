# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 17:54:19 2020

@author: 12437
"""
import shutil,os
import xlwt
import math
import numpy as np
import numpy.fft as nf
import matplotlib.pyplot as plt
import pandas as pd

def readexcel(path):
    NGA_file = pd.read_excel(path)
    record_nums = list(np.array(NGA_file.iloc[:,0]))
    Mgs = list(np.array(NGA_file.iloc[:,9]))
    
    return record_nums,Mgs

def movefile(filenames,record_nums,Mgs):
    for filename in filenames:
        for i,record_num in enumerate(record_nums):
            try:
                filenum = int(filename.split("NGA")[-1])
            except ValueError:
                print("Error: invalid literal for int()")
                break
            else:
                if filenum == record_num:
                    if (float(Mgs[i]) >= 4.0 and float(Mgs[i]) < 5.0):
                        shutil.move((".\\NGA_data\\"+filename),".\\NGA_data\\Mg4_5")
                    elif (float(Mgs[i]) >= 5.0 and float(Mgs[i]) < 6.0):
                        shutil.move((".\\NGA_data\\"+filename),".\\NGA_data\\Mg5_6")
                    elif (float(Mgs[i]) >= 6.0):
                        shutil.move((".\\NGA_data\\"+filename),".\\NGA_data\\Mg6_8")
                    else:
                        print("Error: no match move path!")
                    break
                else:
                    print ("Error: no match filename")
    return None

def check_file(file_path):
    os.chdir(file_path)
    print(os.path.abspath(os.curdir))
    all_file = os.listdir()
    files = []
    for f in all_file:
        if os.path.isdir(f):
            files.extend(check_file(file_path+'\\'+f))
            os.chdir(file_path)
        else:
            files.append(os.path.abspath(os.curdir)+'\\'+f)
    return files

# 读取信号文件
def readdata(data_path):
    file = open(data_path,"r")
    datas = file.readlines()
#    sample_rate = len(datas) #采样率
    signals = []
    for i,line in enumerate(datas):
        if i == 3:
            tempdt = line.split(".")[-1]
            dt = list(filter(str.isdigit, tempdt))
            dt = "0." + ''.join(dt)
            dt = float(dt)
        elif i>3 and i<(len(datas)-1):
            signals.append(line.split()[0])
            signals.append(line.split()[1])
            signals.append(line.split()[2])
            signals.append(line.split()[3])
            signals.append(line.split()[4])
    times = list(np.arange(0, len(signals)*dt, dt))
    if len(times) != len(signals):
        minlen = min(len(times),len(signals))
        times = times[:minlen]
        signals = signals[:minlen]
    return times,signals

# 文本输出
def txt_out(path,xs,ys):
    file = open(path, 'w')
    new_ys = []
    for i,x in enumerate(xs):
        new_y = float(ys[i])
        file.write("%.4f    %.4e \n"%(x,new_y))
        new_ys.append(new_y)
    file.close()
    max_y = max(new_ys)
        
    return new_ys,max_y

def createGM(rootpath,outrootpath):
    files = check_file(rootpath)
    num = 0
    for file in files:
        filename_1 = file.split(".")[-1]
        filename_2 = file.split(".")[-2]
        if filename_1 == "AT2" and (filename_2[-2:] != "UP" and filename_2[-3:] != "DWN" ):
            times,signals = readdata(file)
            outpath = outrootpath + "\\" + outrootpath.split("\\")[-1] + "_%d.txt"%num
            new_ys,max_y = txt_out(outpath,times,signals)
            num += 1
    
    return None

if __name__ == '__main__':
    rootpath = "E:\\liaowj\\2_codes\\34_vibration_identify\\NGA_data\\Mg6_8"
    outrootpath = "E:\\liaowj\\2_codes\\34_vibration_identify\\8_peer_GM\\Mg6_8"
    if not os.path.exists(outrootpath):
        os.makedirs(outrootpath)
#    NGA_Excelpath = ".\\NGA_data\\NGA_Flatfile.xls"
#    record_nums,Mgs = readexcel(NGA_Excelpath)
#    filenames= os.listdir(rootpath) #得到文件夹下的所有文件名称
#    movefile(filenames,record_nums,Mgs)
    createGM(rootpath,outrootpath)