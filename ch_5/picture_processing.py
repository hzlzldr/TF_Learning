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

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

"""
    图像片段的截取、大小的调整、图像翻转和色彩调整等预处理
"""

#给定一张图片，不同的处理流程，最后得到的图片结果会不一样，因此可以通过随机性来减少这种误差

def distort_color(image,color_ordering=0):

    if color_ordering==0:
        image=tf.image.random_brightness(image,max_delta=32./255.)#亮度
        image=tf.image.random_saturation(image,lower=0.5,upper=1.5)#饱和度
        image=tf.image.random_hue(image,max_delta=0.2)#色相
        image=tf.image.random_contrast(image,lower=0.5,upper=1.5)#对比度
    elif color_ordering==1:
        image = tf.image.random_hue(image, max_delta=0.2)  # 色相
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)  # 对比度
        image = tf.image.random_brightness(image, max_delta=32. / 255.)  # 亮度
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)  # 饱和度
    elif color_ordering==2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)  # 对比度
        image = tf.image.random_brightness(image, max_delta=32. / 255.)  # 亮度
        image = tf.image.random_hue(image, max_delta=0.2)  # 色相
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)  # 饱和度
    else:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)  # 亮度
        image = tf.image.random_hue(image, max_delta=0.2)  # 色相
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)  # 饱和度
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)  # 对比度

    return tf.clip_by_value(image,clip_value_min=0.0,clip_value_max=1.0)

def preprocess_for_train(image,height,width,bbox):
    """
    对训练的数据集进行预处理，输入是原始图片，返回是处理过的作为神经网络输入的节点数据
    一般对测试集数据略过这一步
    """

    #如果没有预设标注框，则默认为全图
    if bbox is None:
        bbox=tf.constant([0.0,0.0,1.0,1.0],dtype=tf.float32,shape=[1,1,4])
        #注意这里的维度，114

    #转换图像张量类型
    if image.dtype != tf.float32:
        image=tf.image.convert_image_dtype(image,dtype=tf.float32)

    #随机裁剪
    begin,size,_=tf.image.sample_distorted_bounding_box(tf.shape(image),bbox)

    distorted_image=tf.slice(image,begin,size)

    #将图像调整为神经网络输入层大小
    resized_image=tf.image.resize_images(distorted_image,[height,width],
                                         method=np.random.randint(4))
    #method代表几种调整图像大小的方法，如双线性插值等

    #随机左右翻转
    fliped_image=tf.image.random_flip_left_right(resized_image)

    image=distort_color(fliped_image,np.random.randint(4))

    return image

if __name__ == '__main__':

    raw_image=tf.gfile.FastGFile("./scarlett.jpgg.jpg",'rb').read()
    img=tf.image.decode_jpeg(raw_image)

    with tf.Session() as sess:
        boxes=tf.constant([[[0.05,0.05,0.9,0.7],[0.34,0.32,0.54,0.56]]])
        processed=preprocess_for_train(img,78,100,boxes)
        plt.imshow(processed.eval())
        plt.show()