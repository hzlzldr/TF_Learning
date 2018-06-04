#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = lanzili
__mtime__ = '2018/5/22'
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
from ch1_1 import time_count
from tensorflow.examples.tutorials.mnist import input_data



def data_info():
    #相关数据集大小
    print("training set size is {0}".format(mnist_data.train.num_examples))
    print("validating set size is {0}".format(mnist_data.validation.num_examples))
    print("testing set size is {0}".format(mnist_data.test.num_examples))

    print("the data is {0}".format(mnist_data.train.images[0]))
    print("the label is {0}".format(mnist_data.train.labels[0]))


#constant setting
INPUT_NODE=784
OUTPUT_NODE=10
LAYER_NODE=300
BATCH_SIZE=100
LEARNING_RATE_BASE=0.8
LEARNING_RATE_DECAY=0.99
REGULARIZATION_RATE=0.0001
TRAINING_STEP=5000
MOVING_AVERAGE_DECAY=0.99

def inference(input_tensor,avg_class,weight1,bias1,weight2,bias2):
    # Relu激活函数（The Rectified Linear Unit）表达式为：f(x)=max(0,x)

    if avg_class==None:#无滑动平均类
        layer1=tf.nn.relu(tf.matmul(input_tensor,weight1)+bias1)
        return tf.matmul(layer1,weight2)+bias2

    else:#有滑动平均类
        layer1=tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weight1))+avg_class.average(bias1))
        return tf.matmul(layer1,avg_class.average(weight2))+avg_class.average(bias2)

@time_count
def train_step(mnist):
    x=tf.placeholder(shape=[None,INPUT_NODE],dtype=tf.float32,name='x_input')
    y_=tf.placeholder(shape=[None,OUTPUT_NODE],dtype=tf.float32,name='y_input')

    weight1=tf.Variable(tf.truncated_normal(shape=[INPUT_NODE,LAYER_NODE],dtype=tf.float32,stddev=0.1))
    bias1=tf.Variable(tf.constant(0.1,shape=[LAYER_NODE]))

    weight2=tf.Variable(tf.truncated_normal(shape=[LAYER_NODE,OUTPUT_NODE],dtype=tf.float32,stddev=0.1))
    bias2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))


    #不使用滑动平均
    y=inference(input_tensor=x,avg_class=None,weight1=weight1,bias1=bias1,weight2=weight2,bias2=bias2)

    global_step=tf.Variable(0,trainable=False)

    avg_class=tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DECAY,num_updates=global_step)
    #在所有代表模型参数的变量上使用滑动模型，tf.trainable_variables()返回所有trainable不为False的值，如global_step
    varibale_op=avg_class.apply(tf.trainable_variables())

    avg_y=inference(input_tensor=x,avg_class=avg_class,weight1=weight1,bias1=bias1,weight2=weight2,bias2=bias2)

    """
        用交叉熵来作为刻画预测的准确性
        这里使用tf.nn.sparse_softmax_cross_entropy_with_logits(),但面对的分类问题只有一个正确答案时，该函数可以加速
        计算交叉熵损失；
        该函数的第一个参数是神经网络不包含softmax层的前向传播结果，第二个参数是训练集的正确结果。
        因为标准答案的是一个长度为10的数据，这里要求提供一个正确答案的数字，所以需要使用tf.argmax函数来得到正确答案对应的编号
        tf.argmax(input, dimension, name=None) ，dimension=0.按行，dimension=1 按列
    """
    cross_entropy_mean=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1)))

    #cal L2
    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization=regularizer(weight1)+regularizer(weight2)#一般不正则化bias

    total_loss=cross_entropy_mean+regularization

    #设置指数衰减learning_rate
    """
    learning_rate=initial_rate*decay_rate^(global_step/decay_step)
    if staircase is  True, decay the learning rate at discrete intervals
    """
    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,#基础xuexilv
                                             global_step,#当前迭代次数
                                             mnist_data.train.num_examples/BATCH_SIZE,#所有数据需要的迭代次数
                                             LEARNING_RATE_DECAY,#学习率衰减速度
                                             staircase=True
                                             )

    train=tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss,global_step=global_step)

    #在训练神经网络时，每过一遍训练数据，既需要通过方向传播来更新nn中的参数，又需要更新每一个参数的滑动平均值
    #tf中提供了一次完成多个操作的函数tf.control_dependencies()和tf.group()

    #train_op=tf.group([train,varibale_op]) 和下方代码等价
    with tf.control_dependencies([train,varibale_op]):#一个用来处理依赖性的上下文管理器
        train_op=tf.no_op(name='train')#什么也不做

    #判断滑动模型预测是否准确
    avg_correct_predict=tf.equal(tf.argmax(avg_y,1),tf.argmax(y_,1))
    avg_accuracy=tf.reduce_mean(tf.cast(avg_correct_predict,tf.float32))

    #判断不使用滑动模型预测结果
    correct_predict=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #准备验证数据，一般nn的训练上会通过验证数据集来判断停止的条件和结果
        #在小量数据的情况下，可以使用cress-validation。但在大量数据上，由于计算的原因，一般选取验证数据集的方法
        validate_set={x:mnist_data.validation.images,y_:mnist_data.validation.labels}

        #在实际中，测试数据一般是不可见的
        test_set={x:mnist_data.test.images,y_:mnist_data.test.labels}

        for i in range(TRAINING_STEP):
            xs,ys=mnist_data.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})

            if i%1000==0:
                validate_acc=sess.run(accuracy,feed_dict=validate_set)
                avg_validate_acc=sess.run(avg_accuracy,feed_dict=validate_set)

                print("after {0} iter the validate_acc is {1} and avg_validate_acc is {2}".
                      format(i,validate_acc,avg_validate_acc))


        test_acc=sess.run(accuracy,feed_dict=test_set)
        avg_test_acc=sess.run(avg_accuracy,feed_dict=test_set)
        print("after  iter the test_acc is {0} and avg_test_acc is {1}".
              format( test_acc, avg_test_acc))

if __name__ == '__main__':
    mnist_data = input_data.read_data_sets("./MNIST_data", one_hot=True)
    train_step(mnist_data)
    print("hello world")