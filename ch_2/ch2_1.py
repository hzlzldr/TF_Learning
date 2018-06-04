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
import matplotlib.pyplot as plt
import numpy as np

def data_production():
    sample_size = 200
    x1=np.random.normal(loc=0,scale=0.5,size=sample_size)#loc均值，scale标准差
    x2=np.random.normal(loc=1,scale=0.2,size=sample_size)
    data=np.column_stack((x1,x2))
    lable=[]
    for i in range(sample_size):
        if (pow(data[i,0],2)+pow(data[i,1],2)) <1:
            lable.append(0)
        else:
            lable.append(1)

    lable=np.hstack(lable).reshape(-1,1)#-1 的含义代表total/other
    # plt.scatter(x=data[:, 0], y=data[:, 1],  cmap="RdBu", edgecolor="white",
    #             c=lable.reshape(200))#在py3中，需要把label reshape一下
    # plt.show()

    return data,lable

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

        init=tf.global_variables_initializer()
        sess.run(init)

        for i in range(1000):
            _,rate=sess.run([add_global,lr])
            if i%step.eval()==0:#using eval() can fetch the value from tensor
                print(rate)



def get_weight(shape,var_lambda):
    """
    shape:dimension,[n,m]
    lambda:regulation rate
    """
    var=tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    #add_to_collection 将新生成的变量的l2正则的损失加入到集合中
    #该函数的第一个参数是集合的名字，第二个参数是要加入的内容
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(var_lambda)(var))

    return var

def nn_construction(data,label):

    """
    构建一个5层的神经网络
    """
    batch_size=24
    #各层的节点数
    layer_dimension=[2,10,10,10,1]
    #输入数据
    x = tf.placeholder(tf.float32, shape=(None, 2), name='x_input')
    y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y_input')  # lable
    #总共的层数和当前层以及节点个数
    n_layer=len(layer_dimension)
    cur_layer=x #该变量维护神经网络传播时的最深层
    in_dimension=layer_dimension[0]
    #通过循环创建nn
    for i in range(1,n_layer):
        out_dimension=layer_dimension [i]#下一层
        bias=tf.Variable(tf.constant(0.1,shape=[out_dimension]))
        weight=get_weight([in_dimension,out_dimension],0.001)
        cur_layer=tf.nn.relu(tf.matmul(cur_layer,weight)+bias)#using relu,refresh cur_layer
        #更新当层的节点个数
        in_dimension=out_dimension


    #define loss-function
    mes_loss=tf.reduce_mean(tf.square(cur_layer-y_))
    tf.add_to_collection('losses',mes_loss)#add to losses
    total_loss=tf.add_n(tf.get_collection('losses'))
    #tf.add_n  将列表item相加返回
    #tf.get_collection()  获取列表

    #setting the lr
    initial_rate=tf.Variable(0.1,dtype=tf.float32)
    global_step=tf.Variable(0,trainable=False)
    step=tf.Variable(240,dtype=tf.int32)
    decay_rate=tf.Variable(0.9,dtype=tf.float32)

    lr=tf.train.exponential_decay(learning_rate=initial_rate,global_step=global_step,decay_steps=step,
                                  decay_rate=decay_rate,staircase=True)

    add_global=global_step.assign_add(1)

    #train_step setting
    train_step=tf.train.AdamOptimizer(lr).minimize(total_loss,global_step=global_step)
    
    sample_size=len(label)
    
    iter_num=5000

    with tf.Session() as sess:
        init = tf.global_variables_initializer().run()

        for i in range(iter_num):
            start=i*batch_size%sample_size#这里要用取余
            end=min((i+1)*batch_size,sample_size)
            sess.run(add_global)
            sess.run(train_step, feed_dict={x: data[start:end], y_: label[start:end]})
            if i%500==0:
                rate = sess.run(lr)
                print("learn_rate is {0}".format(rate))
                t_loss=sess.run(total_loss,feed_dict={x:data,y_:label})
                mes=sess.run(mes_loss,feed_dict={x:data,y_:label})
                print("the {0}th iter the total_loss is {1},mes is {2}".format(i,t_loss,mes))

        saver=tf.train.Saver()
        saver.save(sess,"./ch2_1.model.ckpt")

if __name__ == '__main__':


    data,label=data_production()
    #lr=learn_rate()
    nn_construction(data,label)
    print("hello world!")