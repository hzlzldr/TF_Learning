#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = lanzili
__mtime__ = '2018/5/23'
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
    代码持久化，模型保存与复用
"""
import tensorflow as tf

def func1():
    a=tf.Variable(1.2,name='a')
    b=tf.Variable(2.3,name='b')
    result=a+b

    saver=tf.train.Saver()#创建保存类
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(result))
        for var in tf.global_variables():
            print(var)
        saver.save(sess,'./ch3_4_model.ckpt')
        #save() required positional arguments: 'sess' and 'save_path'

        #生产3个文件
        #filename.ckpt.meta保存了tf图结构
        #filename.ckpt 保存了每一个变量的取值
        #checkpoint 保存了目录下所有模型的列表

def func2():
    a=tf.Variable(0.0,name='other_a')#注意和模型变量类型一致
    b=tf.Variable(0.0,name='other_b')
    ret=a+b
    #定义变量和图的运算

    saver=tf.train.Saver({"a":a,"b":b})

    with tf.Session() as sess:
        #没有变量初始化过程
        saver.restore(sess,"./ch3_4_model.ckpt")
        print(sess.run(ret,feed_dict={a:3.4,b:6.5}))#out：9.9

def func3():
    """
    直接加载已经持久化图，不重新定义变量和运算
    """
    saver=tf.train.import_meta_graph('./ch3_4_model.ckpt.meta')#加载之前的结构
    with tf.Session() as sess:
        saver.restore(sess,"./ch3_4_model.ckpt")
        print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))
        for var in tf.global_variables():
            print(var)

def func4():
    v=tf.Variable(0,dtype=tf.float32,name='v')
    for var in tf.global_variables():
        print(var.name)

    ema=tf.train.ExponentialMovingAverage(0.99)
    #声明滑动模型后，tf会自动生成一个影子变量  v/ExponentialMovingAverage:0
    ema_op=ema.apply(tf.global_variables())
    for var in tf.global_variables():
        print(var.name)
    """
 
apply(var_list=None)
Maintains moving averages of variables.

var_list must be a list of Variable or Tensor objects. This method creates shadow variables for all elements of var_list. Shadow variables for Variable objects are initialized to the variable's initial value. They will be added to the GraphKeys.MOVING_AVERAGE_VARIABLES collection. For Tensor objects, the shadow variables are initialized to 0 and zero debiased (see docstring in assign_moving_average for more details).

shadow variables are created with trainable=False and added to the GraphKeys.ALL_VARIABLES collection. They will be returned by calls to tf.global_variables().

Returns an op that updates all shadow variables as described above.

Note that apply() can be called multiple times with different lists of variables.

Args:
var_list: A list of Variable or Tensor objects. The variables and Tensors must be of types float16, float32, or float64.
Returns:
An Operation that updates the moving averages.
    """
    saver=tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.assign_add(v,10))
        sess.run(ema_op)
        saver.save(sess,"./ema.ckpt")
        print(sess.run([v,ema.average(v)]))#ema.average()获取变量的的滑动均值


    v1=tf.Variable(0,dtype=tf.float32,name='v1')

    saver1=tf.train.Saver({"v/ExponentialMovingAverage":v1})
    with tf.Session() as sess:
        saver1.restore(sess,"./ema.ckpt")
        print(sess.run(v1))


def func5():
    from tensorflow.python.framework import graph_util
    """
    将图中的变量和取值通过常量来存储
    """
    v1=tf.Variable(tf.constant(1.0,shape=[1]),name='v1')
    v2=tf.Variable(tf.constant(2.2,shape=[1]),name='v2')

    ret=v1+v2

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #导出当前图的计算过程，只需要这一部分就可以完成输入层到输出层的计算过程
        graph_def=tf.get_default_graph().as_graph_def()

        #将图中的变量存储为常量，同时去掉无关节点，{add]是只保存了这个节点操作，从而不必写成[add:0]
        #[add] represent a node,[add:0] represent the first output_result of the node
        output_graph_def=graph_util.convert_variables_to_constants(sess,graph_def,['add'])

        with tf.gfile.GFile("./ch3_4.model.pb","wb") as f:
            f.write(output_graph_def.SerializeToString())
            #out:two variables convert to constant
            #把add那个节点以及v1，v2这两个变量保存了下来

def func6():

    from tensorflow.python.platform import gfile
    """
    读取上个函数生成的pb文件
    """
    with tf.Session() as sess:
        filename="./ch3_4.model.pb"
        #读取保存的模型，将其转化成对应的GraphDef Protocal Buffer
        with gfile.GFile(filename,'rb') as f:
            grahp_def=tf.GraphDef()
            grahp_def.ParseFromString(f.read())

        ret=tf.import_graph_def(grahp_def,return_elements=['add:0'])
        print(sess.run(ret))


if __name__ == '__main__':

    #func1()
    #func2()
    #func3()
    #func4()
    #func5()
    func6()
    print("hello zergs")