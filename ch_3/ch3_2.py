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

"""
    
"""

def func():
    a=tf.constant(1)
    b=tf.Variable(2)
    result=a+b

    with tf.control_dependencies([result]):
        op = b.assign_add(1)#op是一个operation
        print(result)
    with tf.Session() as sess:
        init=tf.global_variables_initializer()
        sess.run(init)
        for i in range(5):
            print(sess.run(op))#out:3,4,5,6,7

def func2():
    a=tf.constant(1)
    b=tf.Variable(2)
    result=a+b

    with tf.control_dependencies([result]):
        b+=1
        print(result)

    with tf.Session() as sess:
        init=tf.global_variables_initializer()
        sess.run(init)
        for i in range(5):
            print(sess.run(b))


def func3():
    a=tf.constant(1)
    b=tf.Variable(2)
    result=a+b

    with tf.control_dependencies([result]):
        b=tf.identity(result,name='result')
        print(b)

    with tf.Session() as sess:
        init=tf.global_variables_initializer()
        sess.run(init)
        for i in range(5):
            print(sess.run(b))

if __name__ == '__main__':
    func()
    func2()
    func3()




