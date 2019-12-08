import numpy as np
import scipy.io as sio
from scipy.spatial import distance
from tqdm import tqdm

import matplotlib.pyplot as plt

class myNetwork(object):

    def __init__(self,input_dim,output_dim):
        self.weights = [np.random.normal(loc = 0,scale = 1,size = (30,input_dim)),
                        np.random.normal(loc = 0,scale = 1,size = (30,30)),
                        np.random.normal(loc = 0,scale = 1,size = (output_dim,30))]
        #偏置项
        self.biases = [np.random.normal(loc = 0,scale = 1,size = (30,1)),
                       np.random.normal(loc = 0,scale = 1,size = (30,1)),
                       np.random.normal(loc = 0,scale = 1,size = (output_dim,1))]
        #保留隐藏层未激活前的数据 用于反向传播
        self.z = []
        #保留隐藏层激活后的数据 用于反向传播
        self.a = []

    @staticmethod
    def relu(x):
        for i in range(x.shape[0]):
            x[i][0] = max(x[i][0],0)


    @staticmethod
    def relu_gradient(x):
        #对ReLU求梯度
        g = np.zeros(x.shape)
        for i in range(x.shape[0]):
            if x[i][0]>0:
                g[i][0]=1
            elif x[i][0] == 0:
                #该点不可导 采用subgradient
                g[i][0]=0.5
            else:
                g[i][0]= 0
        return g

    def forward(self,x_):
        # 隐藏层初始化
        self.a=[]
        self.z=[]
        # 先作copy防止篡改原数据
        x = x_.copy()
        self.a.append(x.copy())

        for i in range(len(self.weights)):
            weight = self.weights[i]
            b = self.biases[i]
            x = np.dot(weight,x) + b
            self.z.append(x.copy())
            if(i != len(self.weights)-1):
                self.relu(x)
            self.a.append(x.copy())
        return x
    
    #对一个输入列向量作预测
    def predict(self,x_):
        x = x_.copy()
        for i in range(len(self.weights)):
            weight = self.weights[i]
            b = self.biases[i]
            x = np.dot(weight,x) + b
            if(i != len(self.weights)-1):
                self.relu(x)
        return x
    
    # 反向传播求梯度的过程
    def backward(self,x,y,verbose = False):
        y_hat = self.forward(x)
        weight_gradients = []
        bias_gradients = []
        delta = (y_hat - y)

        if verbose:
            print(delta)
        for i in range(len(self.weights)-1,-1,-1):
            weight = self.weights[i]
            weight_gradients.insert(0,np.dot(delta,self.a[i].T))
            bias_gradients.insert(0,delta.copy())
            if i!= 0 :
                delta = self.relu_gradient(self.z[i-1])*(np.dot(weight.T,delta))
                if verbose:
                    print(delta)

        return weight_gradients,bias_gradients,(y_hat-y)**2
    
    # 根据给定的调整量对梯度作调整
    def weights_adjust(self,gradients,learning_rate):
        #print(len(weight_gradients))
        weight_gradients,bias_gradients= gradients
        for i in range(len(weight_gradients)):
            w = self.weights[i]
            g = weight_gradients[i]
            m,n = w.shape
            for j in range(m):
                for k in range(n):
                    w[j][k] -= learning_rate*g[j][k]

        for i in range(len(bias_gradients)):
            b = self.biases[i]
            g = bias_gradients[i]
            m = b.shape[0]
            for j in range(m):
                b[j] -= learning_rate*g[j]

# 用来对神经网络参数作优化的类
class nn_fit(object):
    def __init__(self,x,y,adam = False):
        self.x = x.copy()
        self.y = y.copy()
        self.myNetwork = myNetwork(x.shape[0], y.shape[0])
        self.alpha = 0.0001
        self.adam = adam
        self.loss = np.zeros((x.shape[1],))
        self.history_loss = []
        
        #对Adam做初始化
        if adam:
            self.alpha = 0.001
            self.wv=[]
            self.wu=[]
            self.mu1 = 0.9
            self.mu2 = 0.99
            self.p1 = self.mu1
            self.p2 = self.mu2
#            for i in range(len(self.myNetwork.weights)):
#                self.wv.append(np.zeros(self.myNetwork.weights[i].shape))
#                self.wu.append(np.zeros(self.myNetwork.weights[i].shape))
            self.bv=[]
            self.bu=[]
#            for i in range(len(self.myNetwork.biases)):
#                self.bv.append(np.zeros(self.myNetwork.biases[i].shape))
#                self.bu.append(np.zeros(self.myNetwork.biases[i].shape))
    
    #对一个batch算平均梯度
    def cal_gradient(self,choice):
        wg=[]
        for i in range(len(self.myNetwork.weights)):
            wg.append(np.zeros(self.myNetwork.weights[i].shape))
        
        bg=[]

        for i in range(len(self.myNetwork.biases)):
            bg.append(np.zeros(self.myNetwork.biases[i].shape))

        for i in range(choice.shape[0]):
            col = choice[i]
            x_data = self.x[:,col].reshape((-1,1)).copy()
            y_data = self.y[:,col].reshape((-1,1)).copy()
            cur_wg,cur_bg,cur_loss = self.myNetwork.backward(x_data, y_data)
            self.loss[col] = cur_loss
            for i in range(len(cur_wg)):
                wg[i]+=cur_wg[i]

            for i in range(len(cur_bg)):
                bg[i]+=cur_bg[i]

        for i in range(len(wg)):
            wg[i]/=choice.shape[0]

        for i in range(len(bg)):
            bg[i]/=choice.shape[0]
            
        return wg,bg
    
    #一个epoch的迭代
    def batch_iteration(self,batch_size = 80):
        total = self.x.shape[1]
        permutation = np.arange(total)
        np.random.shuffle(permutation)
        
        start=0
        pbar = tqdm(total=total)
        while start < total:
            if start+batch_size<=total:
                choice = permutation[start:start+batch_size]
            else:
                choice = permutation[start:]
            wg,bg = self.cal_gradient(choice)
            
            gs=()
            if self.adam:
                if self.wv==[]:
                    for i in range(len(wg)):
                        self.wv.append(wg[i])
                        self.wu.append(wg[i]**2)
    
                    for i in range(len(bg)):
                        self.bv.append(bg[i])
                        self.bu.append(bg[i]**2)
                else:
                    for i in range(len(self.wv)):
                        self.wv[i] = self.mu1*self.wv[i] + (1-self.mu1)*wg[i]
                        self.wu[i] = self.mu2*self.wu[i] + (1-self.mu2)*(wg[i]**2)
                        self.bv[i] = self.mu1 * self.bv[i] + (1 - self.mu1) * bg[i]
                        self.bu[i] = self.mu2 * self.bu[i] + (1 - self.mu2) * (bg[i] ** 2)
    
                wv_hat,wu_hat,bv_hat,bu_hat=[],[],[],[]
                for i in range(len(self.wu)):
                    wv_hat.append(self.wv[i].copy() / (1 - self.p1))
                    wu_hat.append(self.wu[i].copy() / (1 - self.p2))
                    bv_hat.append(self.bv[i].copy() / (1 - self.p1))
                    bu_hat.append(self.bu[i].copy() / (1 - self.p2))
    
                nwg,nbg=[],[]
                for i in range(len(wv_hat)):
                    nwg.append(wv_hat[i]/np.sqrt(wu_hat[i]+1e-8))
                    nbg.append(bv_hat[i]/np.sqrt(bu_hat[i]+1e-8))
    
                gs = (nwg,nbg)
                self.p1*=self.mu1
                self.p2*=self.mu2
            else:
                gs = (wg,bg)
    
            self.myNetwork.weights_adjust(gs, self.alpha)
            
            start+=batch_size
            pbar.update(batch_size)
#            print(c)
#            print(start,total)
        pbar.close()
        print("")

    
    def fit(self,epoch = 1000, batch_size = 80):
        cnt = 0
        while cnt < epoch:
            self.batch_iteration(batch_size)
            self.history_loss.append(np.mean(self.loss))
            print("The MSE at train epoch {} is {}".format(cnt+1,np.mean(self.loss)))
            cnt += 1

        return self.history_loss

#            if cnt % 100 == 0 :
#                print("{} iterations has complete".format(cnt))


    def cal_total_loss(self,x_data,y_data):
        s = 0
        for i in range(x_data.shape[1]):
            s += distance.euclidean(self.myNetwork.predict(x_data[:,i].reshape((-1,1)).copy()),
                          y_data[:,i].reshape((-1,1)).copy())**2
        s/=x_data.shape[1]
        return s

if __name__ == '__main__':
    # x=np.array([2,1]).reshape((2,1))
    # y=np.array([1]).reshape((1,1))
    #
    # nf = nn_fit(x,y,adam=False)
    # print(nf.net.forward(x))
    #
    # nf.fit(1000,1)
    #
    # print(nf.net.forward(x))
    data = sio.loadmat('a2data.mat')
    x=data['X'].T
    y=data['Y'].T

    test = sio.loadmat('a2test.mat')

    x_test = test['Xc'].T
    y_test = test['Yc'].T

    network_fitting = nn_fit(x,y,adam = True)
    epoch_time = 200
    batch_size = 1000

    history_loss = network_fitting.fit(epoch = epoch_time,batch_size=batch_size)
    
#    network_fitting2 = nn_fit(x,y,adam = False)
#
#    history_loss2 = network_fitting2.fit(epoch = epoch_time,batch_size=batch_size)
#    
    plt.figure(figsize = (12,6))
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.plot(range(1,epoch_time+1), history_loss,label = 'Adam')
#    plt.plot(range(1,epoch_time+1), history_loss2,label = 'SGD')
    plt.ylim([0, 500])
    plt.legend()
    plt.show()


#    print("The final mse for the training set is {}".format(network_fitting.cal_total_loss(x,y)))
#
#    print("The final mse for the test set is {}".format(network_fitting.cal_total_loss(x_test,y_test)))
#
#    plt.figure(figsize = (12,6))
#    plt.xlabel('Epoch')
#    plt.ylabel('Mean Square Error')
#    plt.plot(range(1,epoch_time+1),history_loss,label='Train Error')
#    plt.ylim([0, 300])
#    plt.legend()
#    plt.show()



    