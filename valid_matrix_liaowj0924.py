#-*- coding: utf-8 -*-
import random
import numpy as np
from keras.utils import np_utils
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from load_matrix_liaowj0831 import load_dataset
import tensorflow as tf
import tensorflow.keras
import time
import os
import cv2
from sklearn.metrics import confusion_matrix    # 生成混淆矩阵函数

class Dataset:
    def __init__(self, valid_path):
        # 验证集
        self.valid_Mpaths = None
        self.valid_matrixs = None
        self.valid_labels = None

        # 数据集加载路径
        self.valid_path = valid_path

        # 当前库采用的维度顺序
        self.input_shape = None
        
        # 分类的数量
        self.nb_classes = 2

    # 加载数据集并按照交叉验证的原则划分数据集并进行相关预处理工作
    def load(self,valid_size):#加载数据，并进行分类
        # 加载数据的路径及labels到内存
        valid_Mpaths, valid_labels = load_dataset(self.valid_path)
        
        # 输出训练集、验证集、测试集的数量
        print(valid_Mpaths.shape[0], 'valid samples')

        self.valid_Mpaths = valid_Mpaths
        self.valid_labels = valid_labels

# CNN网络模型类
class Model_transfer:
    def __init__(self):
        self.model = None
        self.valid_acc = None
        self.valid_loss = None

    def load_model(self, model_path):
        self.model = load_model(model_path)
        # 输出模型概况
        self.model.summary()

    def evaluate(self, dataset, batch_size):
        def generator(out):
            truth,true_names = [],[]
            while 1:              
                row = np.random.randint(0,len(dataset.valid_Mpaths),size=batch_size)
                x_paths = dataset.valid_Mpaths[row]
                x_matrixs,x_temp_labels,x_names = [],[],[]
                for x_path in x_paths:
                    try:
                        x_matrix = np.loadtxt(x_path)
                        matrix_shape = x_matrix.shape
                        x_matrix = np.array(x_matrix).reshape((matrix_shape[0]*5,int(matrix_shape[1]/5)))
                        x_matrix = x_matrix[:,:,np.newaxis]
                        x_matrix = np.concatenate((x_matrix,x_matrix,x_matrix),axis=2)
                        x_matrixs.append(x_matrix)
                        x_temp_labels.append(str(x_path).split("\\")[-2])
                        x_names.append(str(x_path).split("\\")[-1])
                    except:
                        continue                            
                x = np.array(x_matrixs)
                if out:
                    truth.extend(x_temp_labels)
                    true_names.extend(x_names)
                    with open("./predicts/ground_truth.txt","w") as f:
                        truth_out = "\n".join(truth)
                        f.write(truth_out)
                    with open("./predicts/ground_truth_name.txt","w") as f:
                        truth_name_out = "\n".join(true_names)
                        f.write(truth_name_out)
                x_temp_labels = np.array(x_temp_labels)
                y = np_utils.to_categorical(x_temp_labels, nb_classes)

                yield x,y
        
        score = self.model.evaluate_generator(generator(out=False),steps=int((dataset.valid_Mpaths.shape[0])/batch_size),
                                              callbacks=None,max_queue_size=10,
                                              workers=1,use_multiprocessing=False,verbose=0)
        predicts = self.model.predict_generator(generator(out=True),steps=int((dataset.valid_Mpaths.shape[0])/batch_size),
                                              callbacks=None,max_queue_size=10,
                                              workers=1,use_multiprocessing=False,verbose=0)
        predicts_out = []
        for predict in predicts:
            predict = np.argmax(predict)
            predicts_out.append(predict)
        
        self.valid_loss, self.valid_acc = score[0], score[1]
        print("%s: %.2f" % (self.model.metrics_names[0], score[0]))
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))
        np.savetxt("./predicts/predicts.txt",predicts_out,fmt="%d")
        np.savetxt("./predicts/accuracy_loss.txt",score,fmt="%s")

def confuse_matrix(pred_path):
    predicts_path = pred_path + "\\predicts.txt"
    groud_truth_path = pred_path + "\\ground_truth.txt"
    predicts = np.array(np.loadtxt(predicts_path))
    groud_truth =  np.array(np.loadtxt(groud_truth_path))
    groud_truth = groud_truth[0:len(predicts)]
    
    # confusion matrix
    conf_matrix=confusion_matrix(predicts, groud_truth)
    # confusion matrix out
    pred_path_out = pred_path + "\\confusion_matrix.txt"
    np.savetxt(pred_path_out,conf_matrix)
    
    return conf_matrix

if __name__ == '__main__':
    pred_path = "./predicts"
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)
    if not os.path.exists("./model"):
        print("请补充预训练模型")
    if not os.path.exists("./datasets_M"):
        print("请补充数据集")

    valid_path = ".\\datasets_M\\CESMD_data_M"
    dataset = Dataset(valid_path)

    model_name="model"
    nb_classes = 2
    valid_size = 32
    batch_size = 16
    dataset.nb_classes = nb_classes #统一所有的分类数
    train_type="GPU"

    # 创建session时，对session进行参数配置
    if train_type == "GPU":  
        sess =tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(device_count={'gpu': 0}))
    elif train_type == "CPUS":
        sess = tf.Session()
        K.set_session(sess)

    model_path = './model/VGG19/%s.h5' %model_name
    
    dataset.load(valid_size)
    model = Model_transfer()
    model.load_model(model_path)
    
    start=time.time()
    model.evaluate(dataset, batch_size)
    confuse_matrix(pred_path)
    end=time.time()