# -*- coding: utf-8 -*-
import os
import shutil
import random
import numpy as np
from keras.preprocessing.sequence import pad_sequences
#import cv2
#import sys
#from keras.models import load_model
    

# 从指定路径读取训练数据
def read_path(path_name,matrixs_path,labels):
    
    for dir_item in os.listdir(path_name):
        full_path = os.path.abspath(os.path.join(path_name, dir_item))# 从初始路径开始叠加，合并成可识别的操作路径
        #print(full_path)
        if os.path.isdir(full_path):  # 如果是文件夹，继续递归调用
            read_path(full_path,matrixs_path,labels)
        else:  # 文件
            if dir_item.endswith('.txt'):
                matrix_path = full_path
                matrixs_path.append(matrix_path) 
                labels.append(str(path_name).split("\\")[-1])
    return matrixs_path, labels
	
# 从指定路径读取训练数据
def load_dataset(path_name):
    matrixs_path = []
    labels = []
    matrixs_path, labels = read_path(path_name,matrixs_path,labels)
#    将数据转化为数组
    matrixs_path = np.array(matrixs_path)
    labels = np.array(labels)

    return matrixs_path, labels

def read_shape(path_name,shapes):
    
    for dir_item in os.listdir(path_name):
        full_path = os.path.abspath(os.path.join(path_name, dir_item))# 从初始路径开始叠加，合并成可识别的操作路径
        #print(full_path)
        if os.path.isdir(full_path):  # 如果是文件夹，继续递归调用
            read_shape(full_path,shapes)
        else:  # 文件
            if dir_item.endswith('.txt'):
                matrix = np.loadtxt(full_path)
                shape = matrix.shape
                shapes.append(shape)
                break
    return shapes

def rand_split(path,train_size=0.727,test_size=0.182,val_size=0.091):
    datatype = 1
    old_path = path + "\\"+ str(datatype)
    train_path = path +"\\train\\" + str(datatype)
    test_path = path +"\\test\\" + str(datatype)
    val_path = path +"\\val\\" + str(datatype)
    os.makedirs(train_path)
    os.makedirs(test_path)
    os.makedirs(val_path)
    filelist = os.listdir(old_path) #列出该目录下的所有文件,listdir返回的文件列表是不包含路径的。
    num_file = len(filelist)
    for i in range(num_file):
        if i < num_file*train_size:
            filelist = os.listdir(old_path)
            pos = random.randint(0, (len(filelist)-1))
            file = filelist[pos]
            src = os.path.join(old_path, file)
            dst = os.path.join(train_path, file)
            shutil.move(src, dst)
        elif i >= num_file*train_size and i < num_file*(train_size+test_size):
            filelist = os.listdir(old_path)
            pos = random.randint(0, (len(filelist)-1))
            file = filelist[pos]
            src = os.path.join(old_path, file)
            dst = os.path.join(test_path, file)
            shutil.move(src, dst)
        else:
            filelist = os.listdir(old_path)
            file = filelist[0]
            src = os.path.join(old_path, file)
            dst = os.path.join(val_path, file)
            shutil.move(src, dst)
    
    return None
    
def temp_generator(train_Mpaths,train_labels,batch_size):
    x_matrixs = []
    row = np.random.randint(0,len(train_Mpaths),size=batch_size)
    x = np.zeros((batch_size,train_Mpaths.shape[-1]))
    y = np.zeros((batch_size,))
    x_paths = train_Mpaths[row]
    for x_path in x_paths:
        x_matrix = np.loadtxt(x_path)
        x_matrix = np.array(x_matrix)
        x_matrixs.append(x_matrix)
    x = pad_sequences(x_matrixs,dtype='float64',padding='post')
    y = train_labels[row]
    return x,y

if __name__ == '__main__':
    path_name =".\\datasets_M"
    rand_split(path_name,train_size=0.727,test_size=0.182,val_size=0.091)
#    shapes = []
#    shapes = read_shape(path_name,shapes)
#    train_Mpath, train_labels = load_dataset(path_name)
#    batch_size = 8
#    x,y = temp_generator(train_Mpath,train_labels,batch_size)

#加载预测数据集
#path_name = './wtcoefs'
#data, labels = load_dataset(path_name)

#预测
#predicts = model.predict(data)
#predictlabel = []
#for predict in predicts:
#    predictlabel.append(np.argmax(predict))