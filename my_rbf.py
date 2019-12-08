#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 10:32:59 2019

@author: yanyifan
"""

import numpy as np
from tqdm import tqdm
import scipy.io as sio

class Rbf(object):
    
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.dim = 100
        self.a = np.random.normal(loc = 0,scale = 1,size = (self.dim,1))
        self.e = np.zeros(shape = (self.dim,1))
        self.e = np.random.uniform(low = -1,high = 0,size = (self.dim,1))
        self.s = np.random.uniform(low = -6,high = 6,size = (self.dim,2))
        self.b = 0
    
        #Adam所需要的历史信息
        self.av = None
        self.au = None
        
        self.ev = None
        self.eu = None
        
        self.sv = None
        self.su = None
        
        self.bv = None
        self.bu = None
        
        self.mu = 0.9
        self.p = 1
        
        
        self.cur_loss = 0
    
        
#    def initilize_e(self):
#        for i in range(self.dim):
#            sigma = np.std(np.linalg.norm(self.s[i,:] - self.x ,axis = 1)**2)
#            self.e[i][0] = -1/sigma
    
    def init_loss(self):
        for i in range(self.x.shape[0]):
            self.loss[i] = np.square((self.predict(self.x[i,:])-self.y[i,:])[0])
            
        self.cur_loss = np.mean(self.loss)
            
    def predict(self,x_data):
        s_ = (np.linalg.norm(self.s-x_data.reshape((1,2)),axis = 1)**2).reshape((self.dim,1))
        r = self.a * ( (np.exp(self.e * s_)).reshape((self.dim,1)) )
        return np.sum(r)+self.b
    
    #计算单个梯度
    def gradient(self,index):
        delta = (self.predict(self.x[index,:]) - self.y[index,:])[0]
        
        self.loss[index] = delta**2
        
        grad_b = delta
        
        grad_a = delta * np.exp(self.e * (np.linalg.norm(self.s - self.x[index,:],axis = 1)**2).reshape((self.dim,1)))
        #print(grad_a)
        grad_e = delta * self.a * (np.linalg.norm(self.x[index,:]-self.s,axis = 1)**2).reshape((self.dim,1)) * grad_a
        #print(grad_e)
        grad_s = delta * self.a * self.e * 2 * (self.s - self.x[index,:]) * grad_a
        #print(grad_s)
        return grad_a,grad_e,grad_s,grad_b
    
    #计算batch梯度
    def batch_gradient(self,choice):
        ag,eg,sg=np.zeros(shape = self.a.shape),np.zeros(shape = self.e.shape),np.zeros(shape = self.s.shape)
        bg = 0
        for i in range(choice.shape[0]):
            index = choice[i]
            cur_a,cur_e,cur_s,cur_b = self.gradient(index)
            ag+=cur_a
            eg+=cur_e
            sg+=cur_s
            bg+=cur_b
        ag/=choice.shape[0]
        eg/=choice.shape[0]
        sg/=choice.shape[0]
        bg/=choice.shape[0]
        
        return ag,eg,sg,bg
    
    def update_gradient(self,gradients,learning_rate):
        ag,eg,sg,bg = gradients
        
        if type(self.av).__name__ == 'NoneType':
            self.av = ag
            self.ev = eg
            self.sv = sg
            self.bv = bg
            
            self.au = ag**2
            self.eu = eg**2
            self.su = sg**2
            self.bu = bg**2
            
        else:
            self.av = self.mu*self.av + (1-self.mu)*ag
            self.ev = self.mu*self.ev + (1-self.mu)*eg
            self.sv = self.mu*self.sv + (1-self.mu)*sg
            self.bv = self.mu*self.bv + (1-self.mu)*bg
            
            self.au = self.mu*self.au + (1-self.mu)*ag**2
            self.eu = self.mu*self.eu + (1-self.mu)*eg**2
            self.su = self.mu*self.su + (1-self.mu)*sg**2
            self.bu = self.mu*self.bu + (1-self.mu)*bg**2
          
        self.p*=self.mu
            
        self.a -= learning_rate * (self.av/(1-self.p))/np.sqrt(self.au/(1-self.p)+1e-8)
        self.e -= learning_rate * (self.ev/(1-self.p))/np.sqrt(self.eu/(1-self.p)+1e-8) * 0.01
        self.s -= learning_rate * (self.sv/(1-self.p))/np.sqrt(self.su/(1-self.p)+1e-8)
        self.b -= learning_rate * (self.bv/(1-self.p))/np.sqrt(self.bu/(1-self.p)+1e-8)
#        
#        for i in range(self.dim):
#            self.e[i][0] = min(self.e[i][0],10)
#            self.e[i][0] = max(self.e[i][0],-10)
                
    
    def batch_iteration(self,batch_size = 200):
        total = self.x.shape[0]
        permutation = np.arange(total)
        np.random.shuffle(permutation)
        
        start=0
        pbar = tqdm(total=total)
        while start < total:
            if start+batch_size<=total:
                choice = permutation[start:start+batch_size]
            else:
                choice = permutation[start:]
            
            batch_gradient = self.batch_gradient(choice)
            self.update_gradient(batch_gradient,self.alpha)
            
            pre_loss = self.cur_loss 
            self.cur_loss = np.mean(self.loss)
            
            if self.cur_loss > pre_loss:
                self.alpha*=0.99
            
            
            start+=batch_size
            pbar.update(batch_size)
        pbar.close()
        
    def fit(self,epoch = 100,batch_size = 200):
        history_mse = []
        for cnt in range(1,epoch+1):
            self.batch_iteration(batch_size)
            history_mse.append(np.mean(self.loss))
            #print(self.e)
            print("The MSE at epoch {} : {}".format(cnt,history_mse[-1]))
        return history_mse
    
    def calc_total_mse(self,x_,y_):
        s=0
        for i in range(x_.shape[0]):
            s += np.square(self.predict(x_[i,:])-y_[i,:])
        
        return s/x_.shape[0]
    
if __name__ == '__main__':
    data = sio.loadmat('a2data.mat')
    x_train = data['X']
    y_train = data['Y']
    
#    x_train = x_train[:5000,:]
#    y_train = y_train[:5000,:]
    test = sio.loadmat('a2test.mat')
    x_test = test['Xc']
    y_test = test['Yc']
    
    rbf = Rbf(x_train,y_train)
    epoch = 100
    batch_size = 200
    
    rbf.fit(epoch,batch_size)
    
    print("The MSE on the test set is: ",rbf.calc_total_mse(x_test,y_test))
   
    
    
        