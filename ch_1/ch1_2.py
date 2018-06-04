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
from numpy.random import RandomState
from ch1_1 import time_count

@time_count
def nn():

    #set batch size
    batch_size=8

    #set weights
    w1=tf.Variable(tf.random_normal([2,3],stddev=2,seed=1))
    w2=tf.Variable(tf.random_normal([3,1],stddev=2,seed=2))

    #input x,y
    x=tf.placeholder(tf.float32,shape=(None,2),name='x_input')
    y_=tf.placeholder(tf.float32,shape=(None,1),name='y_input')#lable

    #forward propogation
    a=tf.matmul(x,w1)
    y=tf.matmul(a,w2)

    #set the optimizer
    cross_entropy=-tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))#y y_ 注意区分
    #tf.clip_by_value 会将值y限制在一个区间内（le-10，1），大于1则使其等于1，小之亦然
    train_step=tf.train.AdamOptimizer(0.01).minimize(cross_entropy)#0.01=learning rate

    #set data
    data_size=1024
    rmd=RandomState(seed=1)
    X=rmd.rand(data_size,2)#生成一个1024*2的matrix
    Y=[[int(x1+x2<1)] for x1,x2 in X]#将y的取值设定为判断x1+x2是否大于1的int（bool）

    #training
    with tf.Session() as sess:
        init=tf.initialize_all_variables()
        sess.run(init)
        print(sess.run(w1))
        print(sess.run(w2))

        #set iter-num
        step=5000
        for i in range(step):
            start=(batch_size*i)%data_size
            end=min(start+batch_size,data_size)

            sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})

            if i%500==0:
                total_cross_entropy=sess.run(cross_entropy,feed_dict={x:X,y_:Y})
                print("After the {0}th training step(s),the cross entropy is {1}..."
                      "\n".format(i,total_cross_entropy))


if __name__ == '__main__':
    nn()
    print("hello world!")