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
from inspect import v



"""
    tf变量管理
"""

import tensorflow as tf

def func1():
    a=tf.get_variable(name="v",shape=[1],dtype=tf.float32,initializer=tf.constant_initializer(1.0),
                      trainable=False)
    #name为必须参数

    b=tf.Variable(tf.constant(1.0,shape=[1]),name="v")
    #name为可选参数

    #总的来说，a,b 两种变量的定义是大体上是一致的
    print(a,b)


    """
    tf的7个初始化函数
    tf.constant_initializer()
    tf.random_normal_initializer()
    tf.random_uniform_initializer()
    tf.truncated_normal_initializer()#不超过两个stddev
    tf.ones_initializer()
    tf.zeros_initializer()
    tf.uniform_unit_scaling_initializer()#满足随机分布但不影响输出数量级的随机值
    """

def func2():
    """
    tf.variable_scope()控制tf.get_variable()获取已创建的变量
    """
    with tf.variable_scope("var1"):
        v1=tf.get_variable('v1',[1],initializer=tf.constant_initializer(1.0))
        """
        在var1这个命名空间中定义了一个变量v1,当下次出现在var1中重复定义时，将会报错
        """
        print(v1.name)#var1/v1:0
    with tf.variable_scope("var1",reuse=True):
        #将reuse设为true后，get variable将直接获取之前定义过的变量,未定义将会报错
        #reuse为False时，get_variable 在遇到之前未定义过变量时，将会新定义该变量；出现重复则会报错
        v = tf.get_variable('v1')
        print(v==v1)#True
        print(v1.name)#var1/v1:0

    #嵌套 nest
    with tf.variable_scope("var1"):
        print(tf.get_variable_scope().reuse)#False

        with tf.variable_scope("var2"):
            v2=tf.get_variable("v2",[1],tf.float32,tf.constant_initializer(1.2))
            print(v2.name)#var1/var2/v2:0



if __name__ == '__main__':
    #func1()
    func2()
    print("hello tf")