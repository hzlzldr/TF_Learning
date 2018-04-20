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
