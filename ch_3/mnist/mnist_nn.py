#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = lanzili
__mtime__ = '2018/5/25'
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
    定义nn的结构
"""

import tensorflow as tf


INPUT_NODE=784
OUTPUT_NODE=10
LAYER_NODE=500

def get_weight_variable(shape,regularizer):
    weights=tf.get_variable("weights",shape=shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
    #tf.get_variable()的好处是，在创建模型时，创建变量，在训练模型时，则根据名称可以直接获取

    if regularizer!=None:#提供了正则化函数，测试时用不到正则的
        tf.add_to_collection('losses',regularizer(weights))

    return weights


def nn(input_tensor,regularizer):

    with tf.variable_scope("layer1"):
        weights=get_weight_variable(shape=[INPUT_NODE,LAYER_NODE],regularizer=regularizer)
        biases=tf.get_variable("biase1",[LAYER_NODE],initializer=tf.constant_initializer(0.0))
        layer1=tf.nn.relu(tf.matmul(input_tensor,weights)+biases)

    with tf.variable_scope("layer2"):
        weights=get_weight_variable(shape=[LAYER_NODE,OUTPUT_NODE],regularizer=regularizer)
        biases=tf.get_variable("biase2",[OUTPUT_NODE],initializer=tf.constant_initializer(0.0))
        layer2=tf.matmul(layer1,weights)+biases

    return layer2

