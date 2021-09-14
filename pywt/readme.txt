本文件夹中的代码主要用于信号特征的初步处理，包括采用快速傅里叶变换（FFT），短时能量分析（STE），小波变换（WT），基于特征变换的结果再进一步用于深度神经网络的特征提取与分类。
关于特征处理方法更加深入的研究，我们在工程力学发表了相应的论文：A VIBRATION RECOGNITION METHOD BASED ON DEEP LEARNING AND SIGNAL PROCESSING, https://kns.cnki.net/kcms/detail/detail.aspx?filename=GCLX202104023&dbcode=CJFQ&dbname=CJFD2021&v=moJN17kfB5mHgnzlX3ZfYaHcOAi-sL3c8L3FdlzACARlzGTveKbUE9yXhTTyNSMq

The code in this folder is mainly used for the preliminary processing of signal features, including Fast Fourier Transform (FFT), Short-Time Energy Analysis (STE), Wavelet Transform (WT). based on the feature transform analysis results, depth feature extraction and classification can be conducted by deep neural networks.
For more in-depth research on feature processing methods, we have published corresponding papers in Engineering Mechanics:
https://oversea.cnki.net/KCMS/detail/detail.aspx?dbcode=CJFD&dbname=CJFDLAST2021&filename=GCLX202104023&v=moJN17kfB5kbgaHgfg60sBQAPuVwaRpQ4gEqtRIKTU0kp%25mmd2FVpCHEkOi8r4wVYuu2H

0_resample.py
如果需要，这部分用于重新采样信号。下采样或上采样。

1_struct_dyn.py
该部分用于处理结构动力响应。从Excel中读取响应并将加速度数据改写为另一种格式的文本，并绘制相应的图形

2_dyn_FFT.py
这部分用于进行快速傅里叶变换（FFT），分析频域中的信号特性。核心函数是signalfft

3_dyn_STE.py
该部分用于进行短时能量分析，获取能量域中的信号特征。核心功能是calEnergy

4_dyn_Pnoise.py
这部分用于等效模拟考虑到手机传感器噪声的采集信号。可以通过智能手机获取噪声文件。

5_dyn_pywt.py
该部分用于进行小波变换分析，得到时频域中的信号特征。核心函数是signal_pywt。输出是保存在.txt文件中的时频矩阵。有关 pywt 包的更多信息可以在 https://pywavelets.readthedocs.io/en/latest/ 中找到

5_dyn_pywt_txt2png.py
这部分用于将时频特征矩阵绘制成对应的小波图。

6_normal_signal.py
这部分是用来处理正常信号的，没用，[笑哭, emoji]

7_PEER_GM.py
这部分用于处理从PEER下载的原始地面运动，也许有用，[笑哭, emoji]

8_CESMD_data.py
这部分用于处理从CESMD下载的原始地面运动，也许有用，[笑哭, emoji]
