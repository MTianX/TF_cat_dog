# TF cat_dog

**基于tensorflow的猫狗识别**

网络结构：
- 卷积层x3(一层卷积层有卷积、池化和局部响应归一化，当ues_pool参数为True使用后面两个）
- 全连接层x2(加了个dropout层，避免过拟合)

一些相关说明
- 使用tf.data建立pipeline
- 图片预处理：剪切(24x24)，标准化(0-1)
- Dataset.py进行数据处理，ops.py网络结构，trian.py网络建立及训练

keras文件夹
- data.py数据的预处理
- perd.py根据模型预测输出
- train.py网络的搭建及训练
