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
r              ┗┓┓┏━┳┓┏┛
               ┃┫┫ ┃┫┫
               ┗┻┛ ┗┻┛

when I wrote this,only God and I understood what I was doing.
Now,God only knows.

"""
import sys
sys.path.append("C:/Users/acer/Documents/GitHub/TF_Learning")#将目录加入python的搜索list

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from  ch_3.mnist import mnist_nn
from ch_3.mnist import train
import time

#每10s加载一次最新模型进行评估，一般工程中不会这么密集，因为模型训练需要时间
EVAL_INTERVAL_SECS=10


def evaluate(mnist_data):
    mygraph=tf.Graph()
    with mygraph.as_default() as g:
        x=tf.placeholder(tf.float32,[None,mnist_nn.INPUT_NODE],name='x_input')
        y_=tf.placeholder(tf.float32,[None,mnist_nn.OUTPUT_NODE],name='y_input')

        validate_feed={x:mnist_data.validation.images,y_:mnist_data.validation.labels}

        y=mnist_nn.nn(x,None)

        correct_prec=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prec,tf.float32))

        variable_average=tf.train.ExponentialMovingAverage(train.MOVING_AVERAGE_DECAY)
        variabless_to_restore=variable_average.variables_to_restore()

        saver=tf.train.Saver(variabless_to_restore)

    temp_ckpt=None
    while True:

        with tf.Session(graph=mygraph) as sess:
            #tf.train.get_checkpoint_state()会通过checkpoints文件自动找到目录中最新模型的名字
            ckpt=tf.train.get_checkpoint_state(train.MODEL_SAVE_DIR)
            #print(ckpt.model_checkpoint_path)

            if ckpt and ckpt.model_checkpoint_path !=temp_ckpt:
                """
                    增加temp_ckpt来去重复，如果出现重复的模型，则认为训练已经结束
                """
                temp_ckpt =ckpt.model_checkpoint_path
                saver.restore(sess,ckpt.model_checkpoint_path)
                #get the para step

                global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_score=sess.run(accuracy,feed_dict=validate_feed)
                print("after {0}th the accuracy is {1}".format(global_step,accuracy_score))
                time.sleep(EVAL_INTERVAL_SECS)
                print("Waiting...")

            else:
                print("No checkpoint file found!")
                return
def main():

    mnist_data = input_data.read_data_sets("./MNIST_val_data", one_hot=True)
    evaluate(mnist_data)

if __name__ == '__main__':
    main()
    print("hello tf.mnist")