#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 11:08:13 2019

@author: yanyifan
"""
import tensorflow as tf
import scipy.io as sio
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def build_model():
  model = keras.Sequential([
    layers.Dense(30, activation='relu',input_shape = [2]),
    layers.Dense(30, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.Adam()

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mse'])
  return model

def build_model_bn():
  model = keras.Sequential([
    layers.BatchNormalization(trainable = True),
    layers.Dense(30, activation='relu',input_shape = [2]),
    layers.BatchNormalization(trainable = True),
    layers.Dense(30, activation='relu'),
    layers.BatchNormalization(trainable = True),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.Adam()

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mse'])
  return model

  
def change_batch_size():
    data = sio.loadmat('a2data.mat')
    x_train = data['X']
    y_train = data['Y']
    
    test = sio.loadmat('a2test.mat')
    x_test = test['Xc']
    y_test = test['Yc']
    batch_size = [20,40,80,200,500,800,1000,1500,2000]
    losses = []
    for bs in batch_size:
        model = build_model()
        model.fit(x_train, y_train, batch_size=bs, epochs=200, verbose=1, validation_split=0.0)
        loss = np.mean(np.square(model.predict(x_test) - y_test))   
        losses.append(loss)
    
    plt.figure(figsize = (12,6))
    plt.xlabel('batch_size')
    plt.ylabel('Mean Square Error on test set')
    plt.plot(batch_size, losses)
    plt.legend()
    plt.show()
    
def normalization():
    data = sio.loadmat('a2data.mat')
    x_train = data['X']
    y_train = data['Y']
    test = sio.loadmat('a2test.mat')
    x_test = test['Xc']
    y_test = test['Yc']
    mu = np.mean(x_train,axis = 0)
    sigma = np.std(x_train,axis = 0)
    
    x_train_transform = (x_train - mu)/sigma
    model1 = build_model()
    model1.fit(x_train_transform,y_train,batch_size = 1000,epochs=200,verbose = 1,validation_split = 0)
    
    x_test_transform = (x_test - mu) / sigma
    mse1 = np.mean(np.square(model1.predict(x_test_transform) - y_test)) 
    
    model2 = build_model()
    model2.fit(x_train,y_train,batch_size = 1000,epochs=200,verbose = 1,validation_split = 0)
    mse2 = np.mean(np.square(model2.predict(x_test) - y_test)) 
    
    return mse1,mse2
    
def batch_normalization():
    data = sio.loadmat('a2data.mat')
    x_train = data['X']
    y_train = data['Y']
    
    test = sio.loadmat('a2test.mat')
    x_test = test['Xc']
    y_test = test['Yc']
    model = build_model_bn()
    model.fit(x_train, y_train, batch_size=1000, epochs=200, verbose=1, validation_split=0.0)
    return np.mean(np.square(model.predict(x_test) - y_test))
    
def cross_validation():
    data = sio.loadmat('a2data.mat')
    x_train = data['X']
    y_train = data['Y']
    
    test = sio.loadmat('a2test.mat')
    x_test = test['Xc']
    y_test = test['Yc']
    
    cross_validation_rate = [0.0,0.1,0.2,0.3,0.4,0.5]
    
    res = []
    for rate in cross_validation_rate:
        model = build_model()
        model.fit(x_train,y_train,batch_size = 1000,epochs = 200,verbose = 1,validation_split = rate)
        res.append(np.mean(np.square(model.predict(x_test) - y_test)))
    
    return res

def train_more_epoch():
    data = sio.loadmat('a2data.mat')
    x_train = data['X']
    y_train = data['Y']
    
    test = sio.loadmat('a2test.mat')
    x_test = test['Xc']
    y_test = test['Yc']
    
    model = build_model()
    model.fit(x_train,y_train,batch_size = 1000,epochs = 1000,verbose = 2,validation_split = 0)
    print(np.mean(np.square(model.predict(x_test) - y_test)))

if __name__=='__main__':
    #测试是否标准化的影响
#    mse1s,mse2s=[],[]
#    for i in range(10):
#        mse1,mse2 = normalization()
#        mse1s.append(mse1)
#        mse2s.append(mse2)
#        print("With normalization:{} \t Without normalization:{}".format(mse1,mse2))
#    
#    print(mse1s,mse2s)
#    
#    #测试不同cross_validation rate的影响
#    mse = []
#    for i in range(5):
#        mse.append(cross_validation())
#       
#    #测试batch_normalizaton的效果
#    mse=[]
#    for i in range(5):
#        mse.append(batch_normalization())
#        
#    print(mse)
#        
#    
#    #测试batch_size对loss的影响
#    change_batch_size()
    
    #训练1000轮
    train_more_epoch()
    