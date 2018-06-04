#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = lanzili
__mtime__ = '2018/5/13'
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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression as LR
from sklearn.decomposition import PCA


def load_visualize__data(filename):
    data=pd.read_csv(filename)
    #print(data.describe())
    #print(data.head())
    #plot=sn.pairplot(data,x_vars=["Length","Diameter","Height","Whole-weight","Shucked-weight",
    #                              "Viscera-weight","Shell-weight"],y_vars="Rings",aspect=0.8,kind='reg')
    #plt.show(plot)

    plot=sn.pairplot(data,x_vars=["Length"],y_vars=["Rings"],aspect=0.8,kind='reg')

    plt.show(plot)

    feature_col=["Length","Diameter","Height","Whole-weight","Shucked-weight",
                                  "Viscera-weight","Shell-weight"]

    x_data=data[feature_col]
    y_data=data["Rings"]

    pca=PCA(4)
    pca.fit(x_data)
    print(pca.explained_variance_,pca.explained_variance_ratio_)

    return x_data,y_data

def train_model_and_predict(x_data,y_data):
    """
    cross_validation
    """
    x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,random_state=1)

    #print(x_test.shape,y_train.shape)#default splited by 75% and 25%
    linereg=LR()

    #training
    model=linereg.fit(x_train,y_train)
    #print(model)
    #print(linereg.intercept_)
    #print(linereg.coef_)

    #predicting
    y_pred=linereg.predict(x_test)
    #print(y_pred)
    #cal MSE
    MSE_sum=0
    for i in range(len(y_pred)):
        MSE_sum+=pow(y_pred[i]-y_test.values[i],2)
    print(MSE_sum/len(y_pred))


    #result_visulation
    plt.plot(range(len(y_pred)),y_pred,'b',label="predict_value")
    plt.plot(range(len(y_test)),y_test,'g',label="true_value")
    plt.legend(loc="upper right")
    plt.xlabel("number of test_data")
    plt.ylabel("the value of Rings")
    plt.show()


if __name__ == '__main__':
    filename="abalone.csv"
    x_data,y_data=load_visualize__data(filename)
    #train_model_and_predict(x_data,y_data)