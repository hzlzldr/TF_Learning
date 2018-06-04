#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = lanzili
__mtime__ = '2018/5/25'
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

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from ch_3.mnist import mnist_nn

BATCH_SIZE=100
LEARNING_RATE_BASE=0.8
LEARNING_RATE_DECAY=0.99
REGULARIZATION_RATE=0.0001
TRAINING_STEP=30000
MOVING_AVERAGE_DECAY=0.99
MODEL_SAVE_DIR="./mnist/model/"
MODEL_FILENAME="mnist.ckpt"


def train(mnist):

    x=tf.placeholder(tf.float32,shape=[None,mnist_nn.INPUT_NODE],name='x_input')
    y_=tf.placeholder(tf.float32,shape=[None,mnist_nn.OUTPUT_NODE],name='y_input')


    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y=mnist_nn.nn(x,regularizer)
    global_step=tf.Variable(0,trainable=False)

    variable_average=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_average_op=variable_average.apply(tf.trainable_variables())
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    """
    第一个参数logits：就是神经网络最后一层的输出，如果有batch的话，它的大小就是[batchsize，num_classes]，num_classes就是分类的数量。单样本的话，大小就是num_classes
    第二个参数labels：实际的标签，它的shape同上
    """
    cross_entropy_mean=tf.reduce_mean(cross_entropy)

    total_loss=cross_entropy_mean+tf.add_n(tf.get_collection('losses'))

    learning_rate=tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples /BATCH_SIZE,
         LEARNING_RATE_DECAY,
        staircase=True
    )

    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss,
                global_step=global_step)

    with tf.control_dependencies([train_step,variable_average_op]):
        train_op=tf.no_op(name='train')


    saver=tf.train.Saver()

    with tf.Session() as sess:

        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEP):
            xs, ys =mnist.train.next_batch(BATCH_SIZE)
            _,loss_value,step=sess.run([train_op,total_loss,global_step],feed_dict={x:xs,y_:ys})

            if i%1000==0:
                print("after {0}th iter ,the total_loss is {1}".format(step,loss_value))
                saver.save(sess,os.path.join(MODEL_SAVE_DIR,MODEL_FILENAME),global_step=global_step)

def main():
    mnist=input_data.read_data_sets("./mnist/MNIST_data",one_hot=True)
    train(mnist)

if __name__ == '__main__':
    main()
    print("hello mnist")


