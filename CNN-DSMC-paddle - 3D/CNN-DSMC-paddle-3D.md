# CNN-DSMC-paddle-3D

## 1. 问题描述

数据集形状，输入为$(1,2,250,250,250)$，输出为$(1,3,250,250,250)$，共20组，精度为"float32"。

网络基于Unet，参数设置为

```python
lr = 0.001
kernel_size = 5
filters = [8, 16, 32, 32, 64, 64, 128]
bn = True
wn = False
model = CNN_DSMC(2, 3, filters=filters, kernel_size=kernel_size, batch_norm=bn, weight_norm=wn)
wd = 0.005
optimizer = paddle.optimizer.AdamW(learning_rate=lr, parameters=model.parameters(), weight_decay=wd)
epochs = 10000
batch_size = 1
```

基于此参数进行训练，每个Epoch训练时间为

![image-20220113151639897](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220113151639897.png)

显存及核心利用率情况为（batch_size设置为2则显存不够）：

![image-20220113151455772](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220113151455772.png)

网络参数大小为171 MB。

## 2. 目前进展

使用AMP训练模型，其它不变。

![image-20220113154648767](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220113154648767.png)

使用AMP训练模型，同时将数据集形状减小，输入为$(1,2,250,100,100)$，输出为$(1,3,250,100,100)$。batch_size=2。

![image-20220113153317712](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220113153317712.png)

![truth](D:\Users\zby\Research\paddle\truth.png)

![prediction](D:\Users\zby\Research\paddle\prediction.png)