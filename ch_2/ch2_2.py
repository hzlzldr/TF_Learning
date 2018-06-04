#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = lanzili
__mtime__ = '2018/5/21'
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
    滑动平均模型
"""

import tensorflow as tf


v1=tf.Variable(0,dtype=tf.float32)

#迭代次数
step=tf.Variable(0,trainable=False)

#定义一个滑动模型的类，decay_rate=0.99,迭代步数为step
ema=tf.train.ExponentialMovingAverage(decay=0.99,num_updates=step)
ema_1=tf.train.ExponentialMovingAverage(decay=0.99)

#定义一个更新滑动平均的op，需要一个列表，每次执行该op时，这个列表中的变量会被更新
maintain_ema_op=ema.apply([v1])
maintain_ema_op_1=ema_1.apply([v1])

with tf.Session() as sess:
    init=sess.run(tf.global_variables_initializer())

    #通过ema.average(v1)获取滑动平均之后变量的取值。在初始化和滑动平均后，两个值均为0
    print(sess.run([v1,ema.average(v1)]))#[0.0,0.0]

    sess.run(tf.assign(v1,5))#给v1赋值

    sess.run(maintain_ema_op)
    sess.run(maintain_ema_op_1)#没有选用num_updates参数，则decay_rate则直接为所提供的值
    #当设定了迭代次数参数（num_updates)，那么decay_rate=min{decay_rate，（1+num_updates)/(10+num_updates)}
    #print(sess.run([v1,ema.average(v1)]))#[5.0,4.5]
    print(sess.run([v1,ema_1.average(v1)]))

    add_step=step.assign_add(10)
    for i in range(20,200,10):
        sess.run(add_step)
        sess.run(maintain_ema_op)
        print(sess.run([v1,ema.average(v1)]))

