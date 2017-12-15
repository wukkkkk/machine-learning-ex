# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 17:13:25 2017

@author: wk
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import train_test_split 

def Computecost(X,y,w,m):
    e = y - np.dot(X,w)
    J = 1/(2*m) * np.dot(e.T,e)
    return J

def gradientDescent(X,y,w,alpha,num_iters):
    m,n= np.shape(X)
    L_train = []
    L_validation = []
    for i in range(num_iters):
        J = Computecost(X,y,w,m)
        L_train.append(J)
        J = Computecost(X_test,y_test,w,m)
        L_validation.append(J)
        w = w + (alpha/m * np.dot((y - np.dot(X,w)).transpose(),X))
    return w,L_train,L_validation

##载入数据
x,y = load_svmlight_file("E:/housing_scale.txt")
x=x.toarray()
m, n = np.shape(x)
a=np.ones(m)
X=np.column_stack((x,a))
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)


w = np.zeros(n+1).transpose()
Iteration = range(1000)

w,L_train,L_validation =gradientDescent(X_train,y_train,w,0.01,1000) 

Computecost

plt.plot(Iteration,L_train,'r')
plt.plot(Iteration,L_validation,'b')   
 