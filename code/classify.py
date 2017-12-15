# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 19:29:39 2017

@author: wk
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import train_test_split 

def Computecost(X,y,w,C,m):
    e = 1 - y * (np.dot(X,w.T) + b)
    for i in range(m):
        if e[i] <0 :
            e[i] = 0
    J = 2*np.dot(w,w.T) + C/m * np.sum(e)
    return J,e

def accuracy(X,y,w,b,m):
    s = 0
    predict = np.dot(X,w)+b
    for j in range(m):
        if predict[j] > 0:
                predict[j] = 1
        else:
                predict[j] = -1
  
        if y[j] == predict[j]:
                s = s+1
    loss = s / m
    return loss
  

def gradientDescent(X,y,w,b,alpha,C,num_iters):
    m,n= np.shape(X)
    L = []
    A = []
    for i in range(num_iters):
        J,e = Computecost(X,y,w,C,m)
        for k in range(m):
            if e[k] >0 :
                e[k] = 1
        L.append(J)
        w = w - (alpha/m * C*(w - 1/m*np.dot(e.T,(np.tile(y,n).reshape(n,m).T * X)))).T
        b = b - (alpha/m * C*np.sum(1/m*(-y)*e))
        
        loss = accuracy(X,y,w,b,m)
        A.append(loss)
    return w,b,L,A

      

x,y = load_svmlight_file("E:/australian_scale.txt")
X=x.toarray()
#m, n = np.shape(X)
#a=np.ones((m))
#X=np.column_stack((x,a))
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)
m_test,n_test = np.shape(X_test)

w = np.zeros(n_test).transpose()
b = 0
Iteration = range(1000)
loss = accuracy(X_train,y_train,w,b,m_test)
print(loss)

w,b,L,A =gradientDescent(X_train,y_train,w,b,0.1,5,1000)

plt.plot(Iteration,L,'r')
plt.show
print(L[-1])


loss = accuracy(X_test,y_test,w,b,m_test)
print(loss)

