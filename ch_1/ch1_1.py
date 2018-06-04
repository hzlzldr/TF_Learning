#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = lanzili
__mtime__ = '2018/5/15'
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
import numpy as np
from numpy.random import RandomState
import time

def time_count(func):
    def wrapper(*args,**kwargs):
        start=time.time()
        func(*args,**kwargs)
        print(time.time()-start)
        return
    return wrapper


@time_count
def func1():
    a=tf.constant([1.0,2.0])
    b=tf.constant([4.5,6.7])
    result=a+b
    sess=tf.Session()
    print(sess.run(result))
    sess.close()


def graph():
    g1=tf.Graph() #define a graph
    with g1.as_default():
        v=tf.get_variable('v1',initializer=tf.zeros_initializer(),shape=[1])
        #initialize variables
    g2=tf.Graph()


    with g2.as_default():
        v=tf.get_variable(name='v2',initializer=tf.ones_initializer(),shape=[1])

    #call g1
    with tf.Session(graph=g1) as sess:
        tf.initialize_all_variables().run()
        with tf.variable_scope("",reuse=True):
            print(sess.run(tf.get_variable("v1")))

    with tf.Session(graph=g2) as sess2:
        tf.initialize_all_variables().run()
        with tf.variable_scope("",reuse=True):
            print(sess2.run(tf.get_variable("v2")))
    with g1.device('/gpu:0'):
        print("hello world")

def configuration():
    config=tf.ConfigProto(allow_soft_placement=True,#when GPU doesn't work，then the CPU will place on
                          )#log_device_placement=True)#record the running params

    sess=tf.Session(config=config)

    weights=tf.Variable(tf.random_normal([2,3],stddev=2))#标准差=2
    w1=tf.Variable(weights.initial_value*2)
    w2=tf.Variable(tf.random_normal([3,1],stddev=2,seed=1))
    x=tf.constant([[1.0,2.2]])

    a=tf.matmul(x, w1)
    y=tf.matmul(a,w2)
    # sess.run(w1.initializer)#要先初始化
    # sess.run(w2.initializer)
    init=tf.initialize_all_variables()
    sess.run(init)
    print(sess.run(y))

    #assing  can't change the type,but can change the data-dimension
    t1=tf.Variable(([1,2]))
    t2=tf.Variable((2.2,3.4))
    t3=tf.Variable(([12,3,4],[4,5,6]))

    #tf.assign(t1,t2)#error
    tf.assign(t1,t3,validate_shape=False)#it is not to encourage to use

    sess.close()

def placeholder_learn():
    x=tf.placeholder(shape=(1,2),name="input",dtype=tf.float32)

    w1=tf.Variable(tf.random_normal([2,3],stddev=2,seed=1))
    w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=2))

    a=tf.matmul(x,w1)
    y=tf.matmul(a,w2)

    init=tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)

        print(sess.run(y,feed_dict={x:[[1.2,3.4]]}))#assign for x  [[ , ]] attention!!! [,] is wrong

if __name__ == '__main__':
    #func1()
    #graph()
    #configuration()
    #placeholder_learn()

    print("hello world!")
