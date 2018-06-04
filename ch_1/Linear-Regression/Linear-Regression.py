#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = lanzili
__mtime__ = '2018/5/2'
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
    to practice the related knowledge about the Linear-Regression
    Data source：https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/
    predict the rings of abalone
    Name		Data Type	Meas.	Description
	----		---------	-----	-----------
	Sex		nominal			M, F, and I (infant)
	Length		continuous	mm	Longest shell measurement
	Diameter	continuous	mm	perpendicular to length
	Height		continuous	mm	with meat in shell
	Whole weight	continuous	grams	whole abalone
	Shucked weight	continuous	grams	weight of meat
	Viscera weight	continuous	grams	gut weight (after bleeding)
	Shell weight	continuous	grams	after being dried
	Rings		integer			+1.5 gives the age in years
	
	M=(4177,8)

"""
import tensorflow as tf
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.stats import pearsonr
import random


def pick_feature(datafile):

    data=pd.read_csv(datafile)

    #数据摸底，看一些各列的波动情况，方差，相关性等，好减少特征
    #print(data.describe())#看一些各列的数据整体情况，注意方差


    #feature scale  z-score
    for colname in data.columns.values.tolist():
        if colname=="Sex" or colname=="Rings":
            continue
        else:
            # plt.scatter(data[colname], data["Rings"])
            # plt.show()#发现第三个特征最好
            mean=data[colname].mean()
            std=data[colname].std()
            for i in range(len(data)):#测试集也要缩放
                data.ix[i,colname]=(data.ix[i,colname]-mean)/std
    #
    #         # maxrow=data.idxmax(colname)
    #         # minrow=data.idxmin(colname)
    #         # maxrange=data.ix[maxrow,colname]-data.ix[minrow,colname]


    #pearson correlation
    # for i in range(1,7):
    #     for j in range(i+1,7):
    #         print("this is pearson correlation of {0}th and {1}th".format(i,j))
    #         print(data.ix[:, i].corr(data.ix[:,j]))
    # for i in range(1,8):
    #     print(data.ix[:, i].corr(data.ix[:,8]))

    return data

def linear_regression(data):

    #params
    w_1=[random.uniform(-1,1) for i in range(7)]
    w_2= [random.uniform(-1, 1) for i in range(7)]
    w_3= [random.uniform(-3, 3) for i in range(7)]
    #w=[70.0]
    b=[random.uniform(-3,3) for i in range(3)]
    β=0.5#Regulation
    learn_rate=0.05
    iteration_num=100

    #model
    #fx=tf.matmul(tf.transpose(w),x)+b
    #err_sum=pow(fx-y,2)+beta*Σβ*pow(wi,2)
    #

    for iter in range(iteration_num):
        print("the {0}th iteration...".format(iter))
        err_sum=[0.0,0.0,0.0]
        residual_1=[]
        residual_2 = []
        residual_3 = []


        count=[0,0,0]

        for i in range(train_num):
            x=[data.ix[i,1],data.ix[i,2],data.ix[i,3],data.ix[i,4],data.ix[i,5],
               data.ix[i, 6],data.ix[i,7]]

            if data.ix[i,0]=="M":
                fx_1 = w_1[0] * x[0] + w_1[1] * x[1] + w_1[2] * x[2] + w_1[3] * x[3] + \
                       w_1[4] * x[4] + w_1[5] * x[5] + w_1[6] * x[6] + b[0]
                count[0]+=1
                err_sum[0]+=pow((data.ix[i,8]-fx_1),2)
                residual_1.append(data.ix[i,8]-fx_1)
            elif data.ix[i,0]=="F":
                fx_2 = w_2[0] * x[0] + w_2[1] * x[1] + w_2[2] * x[2] + w_2[3] * x[3] + \
                       w_2[4] * x[4] + w_2[5] * x[5] + w_2[6] * x[6] + b[0]
                count[1]=1
                err_sum[1]+= pow((data.ix[i, 8] - fx_2), 2)
                residual_2.append(data.ix[i, 8] - fx_2)
            else:
                fx_3 = w_3[0] * x[0] + w_3[1] * x[1] + w_3[2] * x[2] + w_3[3] * x[3] + \
                       w_3[4] * x[4] + w_3[5] * x[5] + w_3[6] * x[6] + b[0]
                count[2]+=1
                err_sum[2] += pow((data.ix[i, 8] - fx_3), 2)
                residual_3.append(data.ix[i, 8] - fx_3)


        for i in range(3):
            print("{0}th mean_err_sum: {1}".format(i,err_sum[i]/count[i]))

        w_grad_1=[0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        w_grad_2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        w_grad_3 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        b_grad_1=0.0
        b_grad_2 = 0.0
        b_grad_3 = 0.0

        for j in range(count[0]):
            w_grad_1[0] +=   - 2 * (w_1[0]) * residual_1[i]+2*β*w_1[0]
            w_grad_1[1] +=   - 2 * (w_1[1]) * residual_1[i]+2*β*w_1[1]
            w_grad_1[2] +=   - 2 * (w_1[2]) * residual_1[i]+2*β*w_1[2]
            w_grad_1[3] +=   - 2 * (w_1[3]) * residual_1[i]+2*β*w_1[3]
            w_grad_1[4] +=   - 2 * (w_1[4]) * residual_1[i]+2*β*w_1[4]
            w_grad_1[5] +=   - 2 * (w_1[5]) * residual_1[i]+2*β*w_1[5]
            w_grad_1[6] +=   - 2 * (w_1[6]) * residual_1[i]+2*β*w_1[6]
            b_grad_1    +=   - 2 * residual_1[i]

        for j in range(count[1]):
            w_grad_2[0] +=   - 2 * (w_2[0]) * residual_2[i]+2*β*w_2[0]
            w_grad_2[1] +=   - 2 * (w_2[1]) * residual_2[i]+2*β*w_2[1]
            w_grad_2[2] +=   - 2 * (w_2[2]) * residual_2[i]+2*β*w_2[2]
            w_grad_2[3] +=   - 2 * (w_2[3]) * residual_2[i]+2*β*w_2[3]
            w_grad_2[4] +=   - 2 * (w_2[4]) * residual_2[i]+2*β*w_2[4]
            w_grad_2[5] +=   - 2 * (w_2[5]) * residual_2[i]+2*β*w_2[5]
            w_grad_2[6] +=   - 2 * (w_2[6]) * residual_2[i]+2*β*w_2[6]
            b_grad_2    +=   - 2 * residual_2[i]

        for j in range(count[2]):
            w_grad_3[0] +=   - 2 * (w_3[0]) * residual_3[i]+2*β*w_3[0]
            w_grad_3[1] +=   - 2 * (w_3[1]) * residual_3[i]+2*β*w_3[1]
            w_grad_3[2] +=   - 2 * (w_3[2]) * residual_3[i]+2*β*w_3[2]
            w_grad_3[3] +=   - 2 * (w_3[3]) * residual_3[i]+2*β*w_3[3]
            w_grad_3[4] +=   - 2 * (w_3[4]) * residual_3[i]+2*β*w_3[4]
            w_grad_3[5] +=   - 2 * (w_3[5]) * residual_3[i]+2*β*w_3[5]
            w_grad_3[6] +=   - 2 * (w_3[6]) * residual_3[i]+2*β*w_3[6]
            b_grad_3    +=   - 2 * residual_3[i]

        print((w_grad_1,b_grad_1),(w_grad_2,b_grad_2),(w_grad_3,b_grad_3))

        gt_1 = np.zeros([iteration_num, len(w_1) + 1])  # 这里要加[]
        gt_2 = np.zeros([iteration_num, len(w_2) + 1])
        gt_3 = np.zeros([iteration_num, len(w_3) + 1])
        for i in range(len(w_1)):
            gt_1[iter,i]=pow(w_grad_1[i],2)#储存每一次计算得到的各个参数的偏微分的平方值
            w_1[i]=w_1[i]-(learn_rate*w_grad_1[i]/pow(sum(gt_1[:,i]),0.5))
            gt_2[iter, i] = pow(w_grad_2[i], 2)  # 储存每一次计算得到的各个参数的偏微分的平方值
            w_2[i] = w_2[i] - (learn_rate * w_grad_2[i] / pow(sum(gt_2[:, i]), 0.5))
            gt_3[iter, i] = pow(w_grad_3[i], 2)  # 储存每一次计算得到的各个参数的偏微分的平方值
            w_3[i] = w_3[i] - (learn_rate * w_grad_3[i] / pow(sum(gt_3[:, i]), 0.5))

        #print(gt)


        gt_1[iter,-1]=pow(b_grad_1,2)
        b[0]=b[0]-(learn_rate*b_grad_1/pow(sum(gt_1[:,-1]),0.5))
        gt_2[iter, -1] = pow(b_grad_2, 2)
        b[1] = b[1] - (learn_rate * b_grad_2 / pow(sum(gt_2[:, -1]), 0.5))
        gt_3[iter, -1] = pow(b_grad_3, 2)
        b[2] = b[2] - (learn_rate * b_grad_3 / pow(sum(gt_3[:, -1]), 0.5))

        print("w_1:{0},b_1:{1}".format(w_1,b[0]))
        print("w_2:{0},b_2:{1}".format(w_2, b[1]))
        print("w_3:{0},b_3:{1}".format(w_3, b[2]))

    w=[w_1,w_2,w_3]
    return data,w,b

def test(data,w,b):

    w_1=w[0]
    w_2=w[1]
    w_3=w[2]
    err_sum =[0.0,0.0,0.0]
    prediction_value_1=[]
    prediction_value_2 = []
    prediction_value_3 = []
    realvalue_1=[]
    realvalue_2 = []
    realvalue_3 = []

    count=[0,0,0]
    for i in range(train_num,len(data)):
        x=[data.ix[i,1],data.ix[i,2],data.ix[i,3],data.ix[i,4],data.ix[i,5],
           data.ix[i, 6],data.ix[i,7]]

        if data.ix[i, 0] == "M":
            fx_1 = w_1[0] * x[0] + w_1[1] * x[1] + w_1[2] * x[2] + w_1[3] * x[3] + \
                   w_1[4] * x[4] + w_1[5] * x[5] + w_1[6] * x[6] + b[0]
            count[0] += 1
            err_sum[0] += pow((data.ix[i, 8] - fx_1), 2)
            prediction_value_1.append(fx_1)
            realvalue_1.append(data.ix[i, 8])
        elif data.ix[i, 0] == "F":
            fx_2 = w_2[0] * x[0] + w_2[1] * x[1] + w_2[2] * x[2] + w_2[3] * x[3] + \
                   w_2[4] * x[4] + w_2[5] * x[5] + w_2[6] * x[6] + b[0]
            count[1] = 1
            err_sum[1] += pow((data.ix[i, 8] - fx_2), 2)
            prediction_value_2.append(fx_2)
            realvalue_2.append(data.ix[i, 8])
        else:

            fx_3 = w_3[0] * x[0] + w_3[1] * x[1] + w_3[2] * x[2] + w_3[3] * x[3] + \
                   w_3[4] * x[4] + w_3[5] * x[5] + w_3[6] * x[6] + b[0]
            count[2] += 1
            err_sum[2] += pow((data.ix[i, 8] - fx_3), 2)
            prediction_value_3.append(fx_3)
            realvalue_3.append(data.ix[i, 8])



    for i in range(3):
        print(err_sum[i]/count[0])




    # plt.scatter(realvalue,prediction_value,edgecolors='blue')
    # plt.show()

    print(pearsonr(realvalue_1,prediction_value_1))
    print(pearsonr(realvalue_2, prediction_value_2))
    print(pearsonr(realvalue_3, prediction_value_3))


if __name__ == '__main__':
    start=time.time()
    train_num = 2800
    datafile="abalone.csv"
    data=pick_feature(datafile)
    (data,w,b)=linear_regression(data)
    test(data,w,b)
    end=time.time()
    print(end-start)

#iter-num=30   (0.5532245517026284, 3.1552518035750408e-111)