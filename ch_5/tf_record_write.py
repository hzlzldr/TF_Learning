#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = lanzili
__mtime__ = '2018/6/1'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
             ┏┓   ┏┓
            ┏┛┻━━━┛┻┓
            ┃   ☃    ┃
            ┃ ┳┛  ┗┳ ┃
            ┃   ┻    ┃
            ┗━┓    ┏━┛
              ┃    ┗━━━┓
              ┃  神兽保佑 ┣┓
              ┃  永无BUG！ ┏┛
              ┗┓┓┏━┳┓┏┛
               ┃┫┫ ┃┫┫
               ┗┻┛ ┗┻┛

when I wrote this,only God and I understood what I was doing.
Now,God only knows.

"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
"""
    tfrecord其实是一种数据存储形式。使用tfrecord时，实际上是先读取原生数据，然后转换成tfrecord格式，
    再存储在硬盘上。而使用时，再把数据从相应的tfrecord文件中解码读取出来。
"""

#生成整数型特征
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))#要以列表的形式赋值

#生成浮点型特征
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

#生成字串型特征
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

mnist=input_data.read_data_sets("./mnist/data",dtype=tf.uint8,one_hot=True)
images=mnist.train.images
labels=mnist.train.labels
#训练图像的分辨率
pixels=images.shape[1]#784  shape[0]代表数量
num_examples=mnist.train.num_examples

#输出文件目录
output_file="./output.tfrecords"
writer=tf.python_io.TFRecordWriter(output_file)

for index in range(num_examples):
    #将图像矩阵化为一个字符串
    image_raw=images[index].tostring()

    #将样例转化为example protoc，并写入数据结构
    example=tf.train.Example(features=tf.train.Features(feature={
        'pixel':_int64_feature(pixels),
        'label':_int64_feature(np.argmax(labels[index])),
        'image_raw':_bytes_feature(image_raw)
    }))

    #将一个example写入tfrecord
    writer.write(example.SerializeToString())

writer.close()