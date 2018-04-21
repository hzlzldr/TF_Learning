#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = lanzili
__mtime__ = '2018/4/20'
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

Pycharm多行注释解注释：按快捷键Ctrl + /

"""

import tensorflow as tf
import pandas as pd
import time

hello=tf.constant('hello tensorflow')
sess=tf.Session()
print(sess.run(hello))#b'hello tensorflow'

a=tf.constant(21)
b=tf.constant(22)
print(sess.run(a+b))

matrix_a=tf.constant([[1,2]])
matrix_b=tf.constant([[2],[4]])

matrix_result=sess.run(tf.matmul(matrix_a,matrix_b))
print(matrix_result)

#Variables
state=tf.Variable(0,name="counter")
step=tf.constant(10)
new_value=tf.add(state,step)
init=tf.initialize_all_variables()#初始化op的操作
update=tf.assign(state,new_value)

#with 代码下的自动关闭session
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(4):
        sess.run(update)
        print(sess.run(state))

#代码中 assign() 操作是图所描绘的表达式的一部分, 正如 add() 操作一样. 所以在调用 run() 执行表达式
#之前, 它并不会真正执行赋值操作
sess.close()#记得关闭会话，显式调用close