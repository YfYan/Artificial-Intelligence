#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 22:54:40 2019

@author: yanyifan
"""

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import scipy.io as sio

from keras.initializers import Initializer, Constant


from keras import backend as K
from keras.engine.topology import Layer

from sklearn.cluster import KMeans

#随机选取x的的部分点作为基
class InitS(Initializer):
    def __init__(self, x):
        self.x=x

    def __call__(self, shape, dtype=None):
        idx = np.random.randint(self.x.shape[0], size=shape[0])
        return self.x[idx, :]
    
#用KMeans方法做初始化
class Init_S_with_KMeans(Initializer):
    def __init__(self, x, max_iter=100):
        self.x = x
        self.max_iter = max_iter

    def __call__(self, shape, dtype=None):
        model = KMeans(n_clusters=shape[0], max_iter=self.max_iter, verbose=0)
        model.fit(self.x)
        return model.cluster_centers_

#自定义RBFlayer
class RBFLayer(Layer):

    def __init__(self, output_dim, initializer, betas=1.0, **kwargs):
        self.output_dim = output_dim
        self.init_betas = betas
        self.initializer = initializer
        super(RBFLayer, self).__init__(**kwargs)

    #初始化
    def build(self, input_shape):
        self.s = self.add_weight(name='s',
                                 shape=(self.output_dim, input_shape[1]),
                                 initializer=self.initializer,
                                 trainable=True)
        self.e = self.add_weight(name='epsilon',
                                 shape=(self.output_dim,),
                                 initializer=Constant(
                                     value=self.init_betas),
                                 # initializer='ones',
                                 trainable=True)

        super(RBFLayer, self).build(input_shape)
    
    #前向传播
    def call(self, x):
        C = K.expand_dims(self.s)
        H = K.transpose(C-K.transpose(x))
        return K.exp(-self.e * K.sum(H ** 2, axis=1))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


#用Keras框架训练的rbf
class Rbf_keras(object):
    
    def __init__(self,x,y,rbf_dim,kmeans = False):
        self.x = x
        self.y = y
        self.dim = rbf_dim
        self.model = Sequential()
        if kmeans:
            self.model.add(RBFLayer(self.dim,
                              initializer=Init_S_with_KMeans(self.x),
                              input_shape=(self.x.shape[1],)))
        else:
            self.model.add(RBFLayer(self.dim,
                              initializer=InitS(self.x),
                              input_shape=(self.x.shape[1],)))
            
        self.model.add(Dense(1))
        self.model.compile(loss='mse',
                      optimizer='adam',
                      metrics=['mse'])
        
    def fit(self,epoch,batch_size):
        self.model.fit(self.x,self.y,batch_size=batch_size,epochs=epoch,verbose=1,validation_split=0.0,use_multiprocessing=True)
        
        
    def calc_mse(self,x_,y_):
        return np.mean(np.square(self.model.predict(x_) - y_))
   
    
if __name__ == '__main__':
    data = sio.loadmat('a2data.mat')
    x_train = data['X']
    y_train = data['Y']
        
    test = sio.loadmat('a2test.mat')
    x_test = test['Xc']
    y_test = test['Yc']
    
    rbf = Rbf_keras(x_train,y_train,100)
    
    epoch = 1000
    batch_size = 1000
    rbf.fit(epoch,batch_size)
    print("The MSE on test set is ",rbf.calc_mse(x_test,y_test))