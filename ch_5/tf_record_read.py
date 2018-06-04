#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = lanzili
__mtime__ = '2018/6/2'
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
"""
    注意和写入文件的键（key）保持一致

"""

import tensorflow as tf


#创建一个reader来读取TFRecorder
reader=tf.TFRecordReader()

#创建一个队列来维护输入文件
#tf.train.string_input_producer()
input_queue=tf.train.string_input_producer(["./output.tfrecords"])

#从文件中读取一个样例，也可以一次性读取多个样例read_up_to
_,serialized_example=reader.read(input_queue)
#解析读入的样例，如果多个样例，使用parse_example()
features=tf.parse_single_example(
    serialized_example,
    features={
        #tf提供两种方法的特征解析，一种是tf.tf.FixedLenFeature(),解析成一个tensor
        #tf.VarLenFeature()解析得到sparsetensor，用于处理稀疏矩阵
        #数据解析的格式要和写入的格式一致
        'image_raw':tf.FixedLenFeature([],tf.string),
        'pixel':tf.FixedLenFeature([],tf.int64),
        'label':tf.FixedLenFeature([],tf.int64)
    }
)

#tf.decode_raw()可以将字符串解析成图像对应的像素数组
image_raws=tf.decode_raw(features['image_raw'],tf.uint8)
pixels=tf.cast(features['pixel'],tf.int32)
labels=tf.cast(features['label'],tf.int32)

with tf.Session() as sess:
    #启用多线程处理数据
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
    for i in range(10):
        image_raw,pixel,label=sess.run([image_raws,pixels,labels])
        print(image_raw,label)

