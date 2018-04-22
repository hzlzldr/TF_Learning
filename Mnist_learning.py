#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = lanzili
__mtime__ = '2018/4/22'
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
import time

#使用官方提供的数据
#60000行的训练数据集（mnist.train）和10000行的测试数据集（mnist.test）
import tensorflow.examples.tutorials.mnist.input_data as mnist_data
mnist=mnist_data.read_data_sets("MNIST_data/", one_hot=True)

#创建占位符变量
x=tf.placeholder(tf.float32,[None,784])#None表示第一维数据可以任意长度
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))


#创建训练模型
y=tf.nn.softmax(tf.matmul(x,W)+b)

#创建损失函数（交叉熵），模型训练和收敛
y_=tf.placeholder("float",[None,10])
cross_entropy=-tf.reduce_sum(y_*tf.log(y))
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#初始化变量，开始训练
init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)

#训练模型，重复1000
start=time.time()
for i in range(1000):
    print("the "+str(i)+"th is running...")
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

end=time.time()
all=end-start
print(all)

#模型评估
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
correct_precent=tf.reduce_mean(tf.cast(correct_prediction,"float"))
print(sess.run(correct_precent,feed_dict={x:mnist.test.images, y_:mnist.test.labels}))
