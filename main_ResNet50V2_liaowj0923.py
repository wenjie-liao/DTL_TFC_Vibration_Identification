#-*- coding: utf-8 -*-
import random
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.sequence import pad_sequences
from keras.backend.tensorflow_backend import set_session
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD,Adam,RMSprop,Adadelta
from keras.utils import np_utils
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.applications import VGG16, VGG19,InceptionV3,resnet_v2
from tensorflow.keras.models import Model
from load_matrix_liaowj0831 import load_dataset
from load_matrix_liaowj0831 import read_shape
import tensorflow as tf
import tensorflow.keras
import keras
import matplotlib.pyplot as plt
import time
import os
import random as rn
import datetime
testaccuracy=[]


class LossHistory(keras.callbacks.Callback):#定义一个类，用于记录训练过程中的数据，继承自callback
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, epoch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type,model_name):
        iters = range(len(self.losses[loss_type]))
        #创建一个图
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'b', label='train acc')#plt.plot(x,y)，这个将数据画成曲线
        # loss
        plt.plot(iters, self.losses[loss_type], 'k', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'g', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'r', label='val loss')
        plt.title(model_name)
        plt.grid(True)#设置网格形式
        plt.xlabel(loss_type)
        plt.ylabel('acc/loss')#给x，y轴加注释
        plt.legend(loc="upper right")#设置图例显示位置

        plt.savefig("./result_data/%s.png"%(model_name))
        np.savetxt("./result_data/%s_train_acc_epoch.txt" %(model_name),self.accuracy['epoch'],fmt="%s")
        np.savetxt("./result_data/%s_train_acc_batch.txt" %(model_name),self.accuracy['batch'],fmt="%s")
        np.savetxt("./result_data/%s_train_loss_epoch.txt" %(model_name),self.losses['epoch'],fmt="%s")
        np.savetxt("./result_data/%s_train_loss_batch.txt" %(model_name),self.losses['batch'],fmt="%s")
        np.savetxt("./result_data/%s_test_acc_epoch.txt" %(model_name),self.val_acc['epoch'],fmt="%s")
        np.savetxt("./result_data/%s_test_loss_epoch.txt" %(model_name),self.val_loss['epoch'],fmt="%s")

        plt.show()
        

class Dataset:
    def __init__(self, train_path, test_path, valid_path):
        # 训练集
        self.train_Mpaths = None
        self.train_labels = None

        # 验证集
        self.valid_Mpaths = None
        self.valid_labels = None

        # 测试集
        self.test_Mpaths = None
        self.test_labels = None

        # 数据集加载路径
        self.train_path = train_path
        self.test_path = test_path
        self.valid_path = valid_path

        # 当前库采用的维度顺序
        self.input_shape = None
        
        # 分类的数量
        self.nb_classes = 2

    # 加载数据集并按照交叉验证的原则划分数据集并进行相关预处理工作
    def load(self):#加载数据，并进行分类
        # 加载数据的路径及labels到内存
        train_Mpaths, train_labels = load_dataset(self.train_path)
        test_Mpaths, test_labels = load_dataset(self.test_path)
        valid_Mpaths, valid_labels = load_dataset(self.valid_path)

        # train_test_split为sklearn的函数，作用是随机划分测试集
#        train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size=0.2,
#                                                                                  random_state=random.randint(0, 100))

            # 输出训练集、验证集、测试集的数量
        print(train_Mpaths.shape[0], 'train samples')
        print(test_Mpaths.shape[0], 'test samples')
        print(valid_Mpaths.shape[0], 'valid samples')

        # 我们的模型使用categorical_crossentropy作为损失函数，因此需要根据类别数量nb_classes将
        # 类别标签进行one-hot编码使其向量化，在这里我们的类别只有两种，经过转化后标签数据变为二维
        train_labels = np_utils.to_categorical(train_labels, self.nb_classes)
        test_labels = np_utils.to_categorical(test_labels, self.nb_classes)

        # 将其归一化,值归一化到0~1区间
#        threshold 的值为8，所以归一化设置为8
#        train_images /= 1
#        valid_images /= 1

        self.train_Mpaths = train_Mpaths
        self.test_Mpaths = test_Mpaths
        self.train_labels = train_labels
        self.test_labels = test_labels

# CNN网络模型类
class Model_transfer:
    def __init__(self):
        self.model = None

    def build_model(self, dataset,drop_out,nb_classes,shapes):
#        创建一个贯序模型
        self.model = Sequential()

        # 调用VGG模型，进行迁移学习，冻结部分层
        self.conv_base = resnet_v2.ResNet50V2(weights="imagenet",include_top=False,input_shape=(shapes[0][0]*5,int((shapes[0][1])/5),3))
        
        # 冻结直到某一层的所有层
        #仅微调卷积基的最后的两三层
        self.conv_base.trainable = True
        
        set_trainable = False
        for i,layer in enumerate(self.conv_base.layers[:-1]):
            if layer.name == 'block5_conv3':
                set_trainable = True
            if set_trainable:
                print(layer)
                self.conv_base.layers[i].trainable = True
            else:
                self.conv_base.layers[i].trainable = False
        
        # 输出模型概况
        self.conv_base.summary()
        
        # 增加展平层输出
        self.model.add(self.conv_base)
        self.model.add(Flatten())  #  Flatten层
        self.model.add(Dense(128))  #  Dense层,又被称作全连接层
        self.model.add(Activation('relu'))  #  激活函数层
        self.model.add(Dropout(drop_out))  #  Dropout层
        self.model.add(Dense(nb_classes))  # Dense层
        self.model.add(Activation('softmax'))  #  分类层，输出最终结果

        # 输出模型概况
        self.model.summary()

        # 训练模型
    def train(self, dataset, batch_size, nb_epoch, test_size, history, model_name, steps_per_epoch, data_augmentation=False):
        # 控制显存使用
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        set_session(tf.Session(config=config))

#        sgd = SGD(lr=0.001, decay=10e-6, momentum=0.98 , nesterov=True)
#        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
        adadelta = Adadelta(lr=0.5, rho=0.8, epsilon=None, decay=0.010) #keras优化器
#        rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
#        adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-06, amsgrad=True)
        self.model.compile(loss='binary_crossentropy',
                           optimizer=adadelta,
                           metrics=['accuracy'])  # 完成实际的模型配置#工作

        #   
        # 训练数据，有意识的提升训练数据规模，增加模型训练量
        model_path="./model/%s_" %model_name
        filename=model_path+"{epoch:02d}_{val_loss:.4f}.h5"
        #checkpoint为保存模型
        log_dir = os.path.join("callbacks" )
        checkpoint=ModelCheckpoint(filepath=filename,monitor="val_loss",mode="min",
                                   save_weights_only=False,save_best_only=False,verbose=1,period=32)
        callback_lists=[history,checkpoint]

        #采用fit_generator方式，可以节约内存
        row_test =  np.linspace(0, (len(dataset.test_Mpaths)-1), test_size).astype(int)
        test_matrix_paths = dataset.test_Mpaths[row_test]
        test_matrixs,temp_labels = [],[]
        for test_path in test_matrix_paths:
            try:
                test_matrix = np.loadtxt(test_path)
                matrix_shape = test_matrix.shape
                test_matrix = np.array(test_matrix).reshape((matrix_shape[0]*5,int(matrix_shape[1]/5)))
    #            test_matrix = np.resize((matrix_shape[0],matrix_shape[0]/25))
                test_matrix = test_matrix[:,:,np.newaxis]
                test_matrix = np.concatenate((test_matrix,test_matrix,test_matrix),axis=2)
                test_matrixs.append(test_matrix)
                temp_labels.append(str(test_path).split("\\")[-2])
            except:
                continue
        test_matrixs = np.array(test_matrixs)
        print("\n test_matrixs shape=")
        print(str(test_matrixs.shape))
        temp_labels = np.array(temp_labels)
        test_labels = np_utils.to_categorical(temp_labels, nb_classes)
        print("\n test_labels shape=")
        print(str(test_labels.shape))
            
        def generator():
            while 1:
                row = np.random.randint(0,len(dataset.train_Mpaths),size=batch_size)
#                x = np.zeros((batch_size,dataset.train_Mpaths.shape[-1]))
                y = np.zeros((batch_size,))
                x_paths = dataset.train_Mpaths[row]
                x_matrixs,x_temp_labels = [],[]
                for i,x_path in enumerate(x_paths):
                    try:
                        x_matrix = np.loadtxt(x_path)
                        matrix_shape = x_matrix.shape
                        x_matrix = np.array(x_matrix).reshape((matrix_shape[0]*5,int(matrix_shape[1]/5)))
                        x_matrix = x_matrix[:,:,np.newaxis]
                        x_matrix = np.concatenate((x_matrix,x_matrix,x_matrix),axis=2)
                        x_matrixs.append(x_matrix)
                        x_temp_labels.append(str(x_path).split("\\")[-2])
                    except:
                        continue
                x = np.array(x_matrixs)
                x_temp_labels = np.array(x_temp_labels)
                y = np_utils.to_categorical(x_temp_labels, nb_classes)
                
                yield x,y
                
        self.model.fit_generator(generator(),
                                 epochs=nb_epoch,
                                 steps_per_epoch=steps_per_epoch,
#                                 steps_per_epoch=len(dataset.train_Mpaths)//(batch_size),
                                 validation_data=(test_matrixs,test_labels),
                                 shuffle=False,
                                 callbacks = callback_lists)


    def save_model(self, file_path):
        self.model.save(file_path)

    def load_model(self, file_path):
        self.model = load_model(file_path)

    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose=1)
        testaccuracy.append(score[1])
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

if __name__ == '__main__':

    if not os.path.exists("./result_data"):#如果不存在这个文件夹或目录，就执行下一步
        os.mkdir("./result_data")#创建新目录
    if not os.path.exists("./model"):
        os.mkdir("./model")
    if not os.path.exists("./datasets_M"):
        print("请补充数据集")

    history = LossHistory()
    train_path, test_path, valid_path = ".\\datasets_M\\train",".\\datasets_M\\test",".\\datasets_M\\val"
    dataset = Dataset(train_path, test_path, valid_path)

    shapes = []
    shapes = read_shape(train_path,shapes)
    line = shapes[0][0]
    row = (shapes[0][1]-1)
    dropout, batch_size, nb_epoch, steps_per_epoch =  0.5, 16, 128, 16
    test_size = 1024
    model_name="model"
    nb_classes = 2
    dataset.nb_classes = nb_classes #统一所有的分类数
    train_type="GPU"

    # 创建session时，对session进行参数配置
    if train_type == "GPU":
        sess =tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(device_count={'gpu': 0}))
    elif train_type == "CPUS":
        sess = tf.Session()
        K.set_session(sess)

    model_path = './model/%s.h5' %model_name
    
    dataset.load()
    model = Model_transfer()
    model.build_model(dataset,drop_out=dropout,nb_classes=nb_classes,shapes=shapes)
    start=time.time()
    model.train(dataset, batch_size, nb_epoch, test_size, history, model_name=model_name, steps_per_epoch=steps_per_epoch)

    end=time.time()
    model.save_model(file_path=model_path)
    print(end-start)
    history.loss_plot('epoch',model_name=model_name)#画图并且记录训练过程
 

