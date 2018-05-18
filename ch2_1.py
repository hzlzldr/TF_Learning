#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = lanzili
__mtime__ = '2018/5/17'
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
from numpy.random import RandomState


def func1():
    a=tf.constant([[1,2,3,4]])
    b=tf.constant([[3,4,5,6]])

    print(tf.select(tf.greater(a,b),a,b).eval())

    rdm=RandomState(seed=1)
    y=rdm.rand()#不设置参数，则会产生0-1之间的随机数
    y=rdm.rand(1024,2)#生成1024*2的矩阵

def learn_rate():

    """
    learning_rate=initial_rate*decay_rate^(global_step/decay_step)
    if staircase is  True, decay the learning rate at discrete intervals

    """
    initial_rate=tf.Variable(0.1,dtype=tf.float32)
    global_step=tf.Variable(0)
    step=tf.Variable(24,dtype=tf.int32)
    decay_rate=tf.Variable(0.9,dtype=tf.float32)

    lr=tf.train.exponential_decay(learning_rate=initial_rate,global_step=global_step,decay_steps=step,
                                  decay_rate=decay_rate,staircase=True)


    opt=tf.train.GradientDescentOptimizer(lr)
    add_global = global_step.assign_add(1)  # increase the count of global

    with tf.Session() as sess:

        init=tf.initialize_all_variables()
        sess.run(init)

        for i in range(1000):
            _,rate=sess.run([add_global,lr])
            if i%step.eval()==0:#using eval() can fetch the value from tensor
                print(rate)

if __name__ == '__main__':


    #func1()
    learn_rate()
    print("hello world!")