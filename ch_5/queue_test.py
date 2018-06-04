#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = lanzili
__mtime__ = '2018/6/3'
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
    tensorflow 队列的学习
"""

import tensorflow as tf
import numpy as np
import time
import threading


def queue():
    #a queue include two tf.int8 items
    #FIFO:first in first out
    q=tf.FIFOQueue(capacity=2,dtypes="int32")
    #initialize
    init=q.enqueue_many(([12,30],))#注意这里的赋值

    #fetch a value and add1
    x=q.dequeue()
    y=x+1

    #enqueue a value
    q_inc=q.enqueue([y])

    with tf.Session() as sess:
        init.run()
        for i in range(5):
            v,_=sess.run([x,q_inc])
            print(v)

"""
    tf提供先进先出和随机调度两种队列模型
    tf.RandomShuffleQueue()
"""



def myloop(coord,worker_id):
    while not coord.should_stop():
        if np.random.rand()<0.1:
            print("stopping from id :{}".format(worker_id))
            coord.request_stop()

        else:
            print("working :{}".format(worker_id))

        time.sleep(1)

def loop():
    coord=tf.train.Coordinator()
    threads=[
        threading.Thread(target=myloop,args=(coord,i,)) for i in range(5)
    ]
    for t in threads:
        t.start()
    #等待所有线程退出
    coord.join(threads)

if __name__ == '__main__':
    try:
        #queue()
        loop()
    except Exception as e:
        print(e)
    finally:
        print("sometime!")