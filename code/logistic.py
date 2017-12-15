# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 19:50:57 2017

@author: wk
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file

def accuracy(X,y,w,m):
    s = 0
    predict = np.dot(X,w)
    for j in range(m):
        if predict[j] > 0:
                predict[j] = 1
        else:
                predict[j] = -1
  
        if y[j] == predict[j]:
                s = s+1
    loss = s / m
    return loss


def sigmoid(z):
    return 1/(1+np.exp(-z))

def Computecost(X,y,w,m):
    J = -1/m * np.sum((1+y_train).T*np.log(sigmoid(np.dot(X_train,w))) + (1-y_train).T*np.log(1-sigmoid(np.dot(X_train,w))))
               
    return J

def gradientDescent(X,y,w,alpha,num_iters):
    m,n= np.shape(X)
    L = []
    A = []
    for i in range(num_iters):
        J = Computecost(X,y,w,m)
        L.append(J)
        w = w + (alpha/m * np.dot((y - np.dot(X,w)).transpose(),X).T)
        acc = accuracy(X,y,w,m)
        A.append(acc)
    return w,L,A

def Adadelta(X,y,w,p,e,num_iters):
    m,n = np.shape(X)
    L = []
    A = []
    Eg = np.zeros(n)
    exs = np.zeros(n)
    for i in range(num_iters):
        J = Computecost(X,y,w,m)
        L.append(J)
        g = 1/m * np.dot((y - np.dot(X,w)).transpose(),X).T
        Eg = p*(Eg)+(1-p)*(g**2)
        delta = (np.sqrt(exs**2 + e)/np.sqrt(Eg**2 + e))*(-g)
        exs = p*exs + (1-p)*delta**2
        w = w - delta
        acc = accuracy(X,y,w,m)
        A.append(acc)
    return L

def RMSprop(X,y,w,alpha,e,num_iters):
    m,n = np.shape(X)
    L = []
    A = []
    Eg = np.zeros(n).T
    for i in range(num_iters):
        J = Computecost(X_test,y_test,w,m)
        L.append(J)
        g = 1/m * np.dot((y - np.dot(X,w)).transpose(),X).T
        Eg = 0.5 * (Eg+(g**2))
        RMS = np.sqrt(Eg + e)
        w = w + alpha/RMS * g
        acc = accuracy(X,y,w,m)
        A.append(acc)
    return L
        

X_train,y_train = load_svmlight_file("E:/a9a_train.txt")
X_train = X_train.toarray()
X_test,y_test = load_svmlight_file("E:/a9a.txt")
X_test = X_test.toarray()
m_train, n_train = np.shape(X_train)
m_test,n_test = np.shape(X_test)
X_train = np.column_stack((X_train,np.ones(m_train).T))
X_test = np.column_stack((X_test,np.zeros(m_test).T))
X_test = np.column_stack((X_test,np.ones(m_test).T))
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)
m_train, n_train = np.shape(X_train)
m_test,n_test = np.shape(X_test)
'''
for i in range(m_train):
    if y_train[i] == -1:
        y_train[i] = 0
for i in range(m_test):
    if y_test[i] == -1:
        y_test[i] = 0
'''
w = np.zeros(n_train).transpose()
Iteration = range(100)

L = gradientDescent(X_train,y_train,w,0.1,100)
L = Adadelta(X_train,y_train,w,0.95,1e-6,100)#
L = RMSprop(X_train,y_train,w,0.01,1e-6,100)

print(accuracy(X_test,y_test,w,m_test))

plt.plot(Iteration,L,'r')
plt.plot(Iteration,A,'b')
print(L[-1])




