#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = lanzili
__mtime__ = '2018/6/2'
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
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def picture_encode_decode():
    raw_image=tf.gfile.FastGFile("./sc2.jpeg",'rb').read()

    with tf.Session() as sess:
        img_data=tf.image.decode_jpeg(raw_image)
        #print(img_data.eval())

        #plt.imshow(img_data.eval())
        #plt.show()

        #将数据类型转换成实数，方便下文的处理
        #img_data=tf.image.convert_image_dtype(img_data,dtype=tf.float32)

        #得到一张新的和之前的一样的图片
        encode_image=tf.image.encode_jpeg(img_data)

        with tf.gfile.FastGFile("./sc2_copy.jpeg",'wb') as f:
            f.write(encode_image.eval())


def picture_size_adjust():
    """
    因为从网络上获取的图片大小可能不一致，而神经网络输入节点确实一定的
    """
    raw_image = tf.gfile.FastGFile("./sc2.jpeg", 'rb').read()

    with tf.Session() as sess:
        img_data=tf.image.decode_jpeg(raw_image)
        print(img_data.get_shape())

        resized=tf.image.resize_images(img_data,[300,300],method=0)
        """
            双线性插值、双三次线性插值、最近邻法、面积插值法
            （均属图像处理过程中，缩放的方法）
        """

        print(resized.get_shape())
        #(300, 300, ?) ?代表深度
        #dtype=tf.float32,需要转化成uint8

        image = np.asarray(resized.eval(), dtype='uint8')
        #image=tf.image.convert_image_dtype(resized, tf.uint8)
        plt.imshow(image)

        plt.show()

def picture_crop_or_pad():
    raw_image = tf.gfile.FastGFile("./sc2.jpeg", 'rb').read()
    """
        进行图片的裁剪和填充
    """
    image=tf.image.decode_jpeg(raw_image)
    with tf.Session() as sess:#这句代码必备

        croped=tf.image.resize_image_with_crop_or_pad(image,300,300)
        plt.subplot((231)),plt.title("croped")
        plt.imshow(croped.eval())


        padded=tf.image.resize_image_with_crop_or_pad(image,800,800)
        plt.subplot((232)),plt.title("padded")
        plt.imshow(padded.eval())

        #截取中心部分
        central_cropped=tf.image.central_crop(image,0.5)
        plt.subplot((233)),plt.title("central_cropped")
        plt.imshow(central_cropped.eval())

        fliped=tf.image.flip_up_down(image)#可以左右，上下和对角线翻转
        plt.subplot((234)),plt.title("flip")
        plt.imshow(fliped.eval())

        """
            在图像识别问题中，大部分情况下，图像的翻转不会影响识别，所以可以通过随机翻转的方式，使得训练出来的
            模型能够识别不同角度的目标
        """
        fliped_left=tf.image.random_flip_left_right(image)#以一定的概率左右翻转
        plt.subplot((235)),plt.title("left")
        plt.imshow(fliped_left.eval())

        """
            还可以通过众多的api来调整亮度（brightness），对比度（contrast），色相（hue）等
            可以视要求而调用
        """
        plt.show()

def draw_bounding_box():
    raw_image=tf.gfile.FastGFile("./scarlett.jpgg.jpg",'rb').read()
    img_data=tf.image.decode_jpeg(raw_image)
    """
        给图片加入标注框
    """
    with tf.Session() as sess:
        #将图像缩小一些，标注会更清晰
        img=tf.image.resize_images(img_data,[245,178],method=1)
        """
            tf.image.draw_bounding_boxes()要求图像矩阵的数字为实数，所以先将图像矩阵转化为实数类型
            该函数的输入是一个batch的数据（即4维），所以需要将解码后的图像加一维
        """
        batched=tf.expand_dims(
            tf.image.convert_image_dtype(img,tf.float32),0
        )#0代表插入到首维

        boxes=tf.constant([[[0.05,0.05,0.9,0.7],[0.34,0.32,0.54,0.56]]])
        #[ymin,xmin,ymax,xmax]

        begin,size,_=tf.image.sample_distorted_bounding_box(tf.shape(img),boxes)
        print(begin.eval(),size.eval())
        """
        Returns:
    A tuple of `Tensor` objects (begin, size, bboxes).

    begin: A `Tensor`. Has the same type as `image_size`. 1-D, containing
    `[offset_height, offset_width, 0]`. Provide as input to
      `tf.slice`.
    size: A `Tensor`. Has the same type as `image_size`. 1-D, containing
    `[target_height, target_width, -1]`. Provide as input to
      `tf.slice`.
    bboxes: A `Tensor` of type `float32`. 3-D with shape `[1, 1, 4]` containing
    the distorted bounding box.
      Provide as input to `tf.image.draw_bounding_boxes`.
  
        """
        result=tf.image.draw_bounding_boxes(batched,boxes)#batched 是4维，boxes是3维
        distorted_img=tf.slice(img,begin,size)
        plt.imshow(distorted_img.eval())
        plt.show()


if __name__ == '__main__':

    try:
        # picture_encode_decode()
        # picture_size_adjust()
        #picture_crop_or_pad()
        draw_bounding_box()

    except Exception as e:
        print(e)

    finally:
        print("tired! ")