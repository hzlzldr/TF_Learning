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

"""
    实现一组图片的存储为tfrecord文件和从tfrecord文件中读取数据
"""

import tensorflow as tf
import numpy as np
import os

def encode_to_tfrecord(filename_path,data_num):

    if os.path.exists(filename_path):
        os.remove(filename_path)

    #创建写入器
    writer=tf.python_io.TFRecordWriter("./"+filename_path)

    for i in range(data_num):
        raw_image=np.random.randint(0,255,size=[22,22])
        raw_image=raw_image.tostring()

        example=tf.train.Example(features=tf.train.Features(
            feature={
                'raw_image':tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw_image])),
                'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[i]))
            }
        ))

        writer.write(example.SerializeToString())

    writer.close()
    return 0

def decode_to_tfrecord(filename_queue,is_batch):

    if not os.path.exists(filename_queue):
        return

    reader=tf.TFRecordReader()
    input_queue=tf.train.string_input_producer([filename_queue])#这里要列表的形式
    _,serialized_example=reader.read(input_queue)

    features=tf.parse_single_example(
        serialized_example,
        features={
        'raw_image':tf.FixedLenFeature([],tf.string),
        'label':tf.FixedLenFeature([],tf.int64)
        }
    )

    image=tf.decode_raw(features['raw_image'],tf.uint8)
    #image=tf.reshape(image,[22,22])#
    label=tf.cast(features['label'],tf.int64)

    if is_batch:
        batch_size=5
        min_after_dequeue=10
        num_threads=3
        capacity = min_after_dequeue + num_threads * batch_size

        image,label=tf.train.shuffle_batch(
            batch_size=batch_size,
            num_threads=num_threads,
            min_after_dequeue=min_after_dequeue,
            capacity=capacity
        )

    return image,label


if __name__ == '__main__':
    filename_path="./test.rfrecords"
    data_num=42
    filename_queue="./test.rfrecords"
    is_batch=None

    encode_to_tfrecord(filename_path,data_num)
    images,labels=decode_to_tfrecord(filename_queue,is_batch)

    with tf.Session() as sess:
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(sess,coord)

        try:
            for i in range(10):
                iamge,label=sess.run([images,labels])
                print(iamge,label)
        except tf.errors.OutOfRangeError:
            print("Over!")
        finally:
            coord.request_stop()

    coord.request_stop()
    coord.join(threads)

    print("Far cry _v3")
    print("This is a trick!")