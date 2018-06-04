#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = lanzili
__mtime__ = '2018/5/30'
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
Inception-v3模型： https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip

数据集： http://download.tensorflow.org/example_images/flower_photos.tgz

数据集文件解压后，包含5个子文件夹，子文件夹的名称为花的名称，代表了不同的类别。平均每一种花有734张图片，图片是RGB色彩模式，大小也不相同。
"""

import tensorflow as tf
import glob
import numpy as np
import os.path
import random
from tensorflow.python.platform import gfile

#Inception-v3 瓶颈节点个数
BOTTLENECK_TENSOR_NODE=2048

#在谷歌提供的inception-v3的模型中，代表瓶颈层结果的张量名称是“pool_3/_reshape:0"
#在训练的时候，可以通过tensor.name 来获取
BOTTLENECK_TENSOR_NAME='pool_3/_reshape:0'

#图片输入张量对应的名称
JPEG_DATA_TENSOR_NAME='DecodeJpeg/contents:0'

#下载的谷歌的inception_v3模型
MODEL_DIR="./inception_dec_2015"

#下载的谷歌的inception_v3模型的名字
MODEL_NAME="tensorflow_inception_graph.pb"

#训练后得到的特征向量的保存文件
CACHE_DIR="./temp_bottleneck"

#input
INPUT_DATA="./flower_photos/flower_photos/"

#percentage of validation/test set
VALIDATION_PERCENTAGE=10
TEST_PERCENTAGE=10

#神经网络相关参数
learning_rate=0.01
STEPS=4000
BATCH_SIZE=100


#该函数读取输入文件并按训练、测试、验证三个数据集分开
def create_images_list(testing_percentage=TEST_PERCENTAGE,
                        validation_percentage=VALIDATION_PERCENTAGE):
    #将结果存储在字典中，key为类别，值为一个字典集合
    result={}

    #读取所有子目录
    sub_dirs=[x[0] for x in os.walk(INPUT_DATA)]
    #print(sub_dirs)

    #第一个目录是当前目录，不考虑
    is_root_dir=True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir=False
            continue

        extensions=['jpg','jpeg','JPG','JPEG']
        file_list=[]
        dir_name=os.path.basename(sub_dir)
        #print("dir_name is :{}".format(dir_name))
        for extension in extensions:
            file_glob=os.path.join(INPUT_DATA,dir_name,'*.'+extension)
            #print("file_glob is :{}".format(file_glob))
            file_list.extend(glob.glob(file_glob))#将每一张图片的路径名都加入列表
            #print(file_list)
        if not file_list:
            continue

        #类别名称
        label_name=dir_name.lower()
        #print(label_name)
        training_images=[]
        testing_images=[]
        validation_images=[]

        for file_name in file_list:
            basename=os.path.basename(file_name)#获取文件名

            chance=np.random.randint(100)#生成100以内的自然数
            if chance<VALIDATION_PERCENTAGE:
                validation_images.append(basename)
            elif chance<VALIDATION_PERCENTAGE+TEST_PERCENTAGE:
                testing_images.append(basename)
            else:
                training_images.append(basename)

        result[label_name]={'dir':dir_name,
                            'training':training_images,
                            'testing':testing_images,
                            'validation':validation_images}

    #print(result.keys())
    return result

def get_image_path(image_lists,image_dir,label_name,index,category):
    """
    通过类别名称，所属数据集和图片编号获取一张图片的地址
    :param image_lists:所有图片的信息
    :param image_dir:根目录，注意存放图片数据的根目录和存放特征向量的根目录不一致
    :param label_name:类别名称,什么花
    :param index:需要获取的图片的编号
    :param category:获取的图片所在的类别，测试，验证和训练集
    :return:full_path，返回全路径
    """

    #获取给定类别中所有图片的信息
    label_lists=image_lists[label_name]
    #根据数据集的名称获取集合中所有图片的信息
    category_lists = label_lists[category]
    mod_index=index%len(category_lists)

    #获取图片的文件名
    base_name=category_lists[mod_index]
    sub_dir=label_lists['dir']

    full_path=os.path.join(image_dir,sub_dir,base_name)

    return full_path

def get_bottleneck_path(image_lists,label_name,index,category):
    """
    获取inception-v3模型处理后特征向量的文件地址
    :param image_lists:
    :param label_name:
    :param index:
    :param category:
    :return:
    """
    return get_image_path(image_lists,CACHE_DIR,label_name,index,category)+'.txt'

def run_bottleneck_on_image(sess,image_data,image_data_tensor,bottleneck_tensor):
    """
    使用训练好的inception-v3模型处理每一张图片，得到对应的特征向量
    """
    bottleneck_value=sess.run(bottleneck_tensor,{image_data_tensor:image_data})

    #输出一个4维向量
    #将其压缩成一个一维的向量
    bottleneck_value=np.squeeze(bottleneck_value)

    return bottleneck_value

def get_or_create_bottleneck(sess,image_lists,label_name,index,category,jpeg_data_tensor,bottleneck_tensor):
    """
    获取一张图片的经过inception-v3模型处理后的特征向量，如果没有找到对应图片的特征向量，则计算该特征向量，并保存
    """

    #获取一张图片对应的特征向量的路径
    label_list=image_lists[label_name]
    sub_dir=label_list['dir']
    sub_dir_path=os.path.join(CACHE_DIR,sub_dir)
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
    bottleneck_path=get_bottleneck_path(image_lists,label_name,index,category)

    #如果特征向量不存在，则用inception-v3计算
    if not os.path.exists(bottleneck_path):
        #获取图片原始路径
        image_path=get_image_path(image_lists,INPUT_DATA,label_name,index,category)
        image_data=gfile.FastGFile(image_path,'rb').read()
        bottleneck_value=run_bottleneck_on_image(sess,image_data,jpeg_data_tensor,bottleneck_tensor)

        bottleneck_string=','.join(str(x) for x in bottleneck_value)
        with open(bottleneck_path,'w') as f:
            bottleneck_string=f.write(bottleneck_string)

    else:#存在则直接读取
        with open(bottleneck_path,'r') as f:
            bottleneck_string=f.read()
        bottleneck_value=[float(x) for x in bottleneck_string.split(',')]

    #返回特征向量
    return bottleneck_value


def get_random_cached_bottlenecks(sess,n_class,image_lists,how_many,category,jpeg_data_tensor,bottleneck_tensor):

    """
    随机取一个batch的图片作为训练
    """
    bottlenecks=[]
    ground_truths=[]

    for _ in range(how_many):
        label_index=random.randrange(n_class)
        label_name=list(image_lists.keys())[label_index]
        image_index=random.randrange(2**16)
        bottleneck=get_or_create_bottleneck(sess,image_lists,label_name,image_index,
                                             category,jpeg_data_tensor,bottleneck_tensor)

        ground_truth=np.zeros(n_class,dtype=np.float32)
        #ground_truth=tf.get_variable(name='ground_truth_1',shape=[n_class,1],initializer=tf.zeros_initializer())
        ground_truth[label_index]=1.0
        ground_truth=tf.constant(ground_truth)
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks,ground_truths


def get_test_bottlenecks(sess,image_lists,n_class,jpeg_data_tensor,bottleneck_tensor):
    """
    获取全部的测试集
    """

    bottlenecks=[]
    groundtruths=[]
    label_name_list=list(image_lists.keys())

    for label_index,label_name in enumerate(label_name_list):
        category='testing'
        for index,unused_base_name in enumerate(image_lists[label_name][category]):
            #通过inception-v3计算图片对应的特征向量，并将其加入到最终数据列表中
            bottleneck=get_or_create_bottleneck(sess,image_lists,label_name,index,category,
                                                jpeg_data_tensor,bottleneck_tensor)
            #groundtruth = tf.get_variable(name='ground_truth_2', shape=[n_class, 1],
             #                              initializer=tf.zeros_initializer())
            groundtruth = np.zeros(n_class, dtype=np.float32)
            groundtruth[label_index]=1.0
            groundtruth=tf.constant(groundtruth)
            bottlenecks.append(bottleneck)
            groundtruths.append(groundtruth)


    return bottlenecks,groundtruths

def main():
    image_lists=create_images_list()
    n_class=len(image_lists.keys())

    with gfile.FastGFile(os.path.join(MODEL_DIR,MODEL_NAME),'rb') as f:
        graph_def=tf.GraphDef()
        graph_def.ParseFromString(f.read())

    bottleneck_tensor,jpeg_data_tensor=tf.import_graph_def(
        graph_def,
        return_elements=[BOTTLENECK_TENSOR_NAME,JPEG_DATA_TENSOR_NAME]
    )


    #定义新的神经网络，输入是瓶颈层的输出结果
    bottleneck_input=tf.placeholder(dtype=tf.float32,shape=[None,BOTTLENECK_TENSOR_NODE],name='x_input')
    ground_truth_input=tf.placeholder(dtype=tf.float32,shape=[None,n_class],name='y_input')

    with tf.variable_scope("fcl"):
        weights=tf.get_variable("weights",[BOTTLENECK_TENSOR_NODE,n_class],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias=tf.get_variable("bias",[n_class],initializer=tf.constant_initializer(0))

        logit=tf.matmul(bottleneck_input,weights)+bias
        final_result=tf.nn.softmax(logit)

    cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=logit,labels=ground_truth_input)
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_mean)

    with tf.name_scope("evaluation"):
        correct_prediction=tf.equal(tf.argmax(final_result,1),tf.argmax(ground_truth_input,1))
        evaluation_step=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(STEPS):
            train_bottleneck,train_groundtruth=get_random_cached_bottlenecks(sess,n_class,image_lists,BATCH_SIZE,'training',
                                                                             jpeg_data_tensor,bottleneck_tensor)
            sess.run(train_step,feed_dict={bottleneck_input:train_bottleneck,ground_truth_input:train_groundtruth})

            if i%100==0 or i+1==STEPS:
                validation_bottleneck,validation_groundtruth=get_random_cached_bottlenecks(sess,n_class,image_lists,BATCH_SIZE,
                                    'validation',jpeg_data_tensor,bottleneck_input)

                validation_accuracy=sess.run(evaluation_step,feed_dict={bottleneck_input:validation_bottleneck,
                                                                        ground_truth_input:validation_groundtruth})
                print("After {0}th iter ,the accuracy in validation set is {1}".format(i,validation_accuracy))

            test_bottleneck,test_groundtruth=get_test_bottlenecks(sess,image_lists,n_class,
                                                                  jpeg_data_tensor,bottleneck_tensor)
            test_accuracy=sess.run(evaluation_step,feed_dict={bottleneck_input:test_bottleneck,
                                                             ground_truth_input:test_groundtruth})
            print("the accuracy of the model in test-set is {}".format(test_accuracy))

if __name__ == '__main__':

    main()
    print("hello inception_v3!")
