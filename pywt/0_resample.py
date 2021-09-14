# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 13:04:36 2021

This part is used to resample the signal, if needed.
down-sample or up-sample

@author: Administrator
"""

import os
import scipy.signal as ss
import numpy as np

def MyShake_re(x,num):
    y = ss.resample(x, num, t=None, axis=0, window=None, domain='time')
    
    return y

def main_re():
    data_name = "MyShake_data_4-4.txt"
    raw_data_dir = os.path.join(".\\MyShake_data",data_name)
    num_data = 2000
    raw_data = np.loadtxt(raw_data_dir)
    re_data = MyShake_re (raw_data,num_data)
    re_data_dir = os.path.join(".\\MyShake_data",("re_"+data_name))
    np.savetxt(re_data_dir,re_data)

main_re()    
