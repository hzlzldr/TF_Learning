#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = lanzili
__mtime__ = '2018/5/29'
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


#define params
INPUT_NODE=784
OUTPUT_NODE=10

IMAGE_SIZE=28#尺寸
NUM_CHANNEL=1#图片的深度
NUM_LABEL=10#标签

CONV1_SIZE=5
CONV1_DEEP=32
CONV2_SIZE=5
CONV2_DEEP=64
FC_SIZE=512#fullt connected
BATCH_SIZE=100

def nn(input_tensor,train,regularizer):

    with tf.name_scope("reshape"):
        input_tensor=tf.reshape(input_tensor,[BATCH_SIZEIMAGE_SIZE,IMAGE_SIZE,NUM_CHANNEL])

    #第一层参数定义和前向传播
    with tf.variable_scope("layer1_conv1"):
        conv1_weights=tf.get_variable(
            "weights",[CONV1_SIZE,CONV1_SIZE,NUM_CHANNEL,CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv1_bias=tf.get_variable(
            "bias",[CONV1_DEEP],initializer=tf.constant_initializer(0.0)
        )

        #filter:5X5,stride=1，paddings=same
        #全0填充的话，输出的结果矩阵式28*28*CONV1_DEEP
        conv1=tf.nn.conv2d(input_tensor,conv1_weights,[1,1,1,1],padding="SAME")

        relu1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_bias))

    #第一层池化层
    #filter 2*2
    with tf.variable_scope("layer2_pool1"):
        pool1=tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,1,1,1],padding="SAME")
        #输出14*14*CONV1_DEEP

    with tf.variable_scope("layer3_conv2"):
        conv2_weights=tf.get_variable("weights",[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))

        conv2_bias=tf.get_variable("bias",[CONV2_DEEP],initializer=tf.constant_initializer(0.0))

        #全0填充，输出14*14*CONV2_DEEP
        conv2=tf.nn.conv2d(pool1,conv2_weights,[1,1,1,1],padding="SAME")
        relu2=tf.nn.relu(tf.nn.bias_add(conv2,conv2_bias))

    #第二层池化层
    #filter 2*2
    with tf.variable_scope("layer4_pool2"):
        pool2=tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,1,1,1],padding="SAME")
        #out:7*7*CONV2_DEEP

    #把第四层的输出拉直成一个向量，作为下一层全连接的输入层

    #pool2.get_shape()可以得到pool2的维度
    #由于每一层神经网络的输入输出都包含batch这一维度，所以，get_shape()所得到的维度列表的第一数字代表batch的中数据的个数
    pool2_shape=pool2.get_shape().as_list()
    node=pool2_shape[1]*pool2_shape[2]*pool2_shape[3]#长*宽*深度

    #通过tf.reshape()将第四层所有的输出变成一个包含batch里面所有节点的矩阵
    reshaped=tf.reshape(pool2,[pool2_shape[0],node])

    #input:batch_size*7*7*64
    #out:3136*512(FC_SIZE)
    with tf.variable_scope("layer5_fcl"):
        weights=tf.get_variable("weights",[node,FC_SIZE],initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias=tf.get_variable("bias",[FC_SIZE],initializer=tf.constant_initializer(0.0))

        if regularizer!=None:
            tf.add_to_collection("losses",regularizer(weights))

        fcl_1=tf.nn.relu(tf.matmul(reshaped,weights)+bias)
        if train:
            fcl_1=tf.nn.dropout(fcl_1,0.5)

    #input :layer5
    #out:10
    with tf.variable_scope("layer6"):
        weights=tf.get_variable("weights",[FC_SIZE,OUTPUT_NODE],initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias=tf.get_variable("bias",[OUTPUT_NODE],initializer=tf.constant_initializer(0.0))

        if regularizer!=None:
            tf.add_to_collection("losses", regularizer(weights))
        logit=tf.matmul(fcl_1,weights)+bias#这一层不用relu回归，因为后面会用到稀疏softmax来处理


    return logit
