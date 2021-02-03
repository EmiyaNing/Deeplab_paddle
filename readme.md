# Deeplab 

1. 该程序在paddle平台实现了deeplab v3模型
2. 其中dataload.py文件主要完成数据读入工作
3. 其中deeplab.py文件主要包含deeplab v3 的模型搭建工作
4. Lossfunc.py文件实现了交叉熵损失函数
5. resnet_multi_grid.py文件实现了dilate_ResNet模型
6. train.py 实现了模型的训练功能
7. utils.py 实现了一个用来记录训练过程中模型损失值动态改变的工具类

### 代码的使用方式
> python train.py 

### 需要预先安装的库
1. paddlepaddle2.0
2. numpy >1.8
3. opencv > 4.1

### 实现过程中的一些注意事项
1. 使用交叉熵损失函数来训练模型时，模型的最后一层必须使用softmax激活函数。
2. 采用opencv接口读出来的灰度图像每个像素元素是按照uint8来进行存储的，在喂给模型之前需要转换为int64格式（paddle平台适用）。
3. 在linux环境下采用多线程读取数据时，注意线程数不能超过机器本身的性能极限。
4. 采用resnet作为骨干网络的缺点是网络结构巨大，训练代价极大。
5. 百度提供的dummy_data数据集中的一个图片是错误的。。。

### TODO_List
1. 实现deeplab v1， v2， v3+等模型
2. 骨干网络换用mobilenet
3. 在百度提供的数据集上训练模型
