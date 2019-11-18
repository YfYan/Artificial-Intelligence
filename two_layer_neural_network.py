import numpy as np


class myNetwork(object):

    def __init__(self,input_dim,output_dim):
        self.weights = [np.random.random((30,input_dim)),
                        np.random.random((30,30)),
                        np.random.random((output_dim,30))]
        #偏置项
        self.biases = [np.random.random((30,1)),
                      np.random.random((30,1)),
                      np.random.random((output_dim,1))]
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
            self.relu(x)
            self.a.append(x.copy())
        return x

    def backward(self,x,y,verbose = False):
        y_hat = self.forward(x)
        weight_gradients = []
        bias_gradients = []
        delta = (y_hat - y) * self.relu_gradient(self.z[-1])

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

        return weight_gradients,bias_gradients

    def weights_adjust(self,gradients,learning_rate = 0.00001):
        #print(len(weight_gradients))
        weight_gradients,bias_gradients = gradients
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


if __name__ == '__main__':
    x=np.array([2,1]).reshape((2,1))
    y=np.array([1]).reshape((1,1))
    net = myNetwork(2,1)
    print(net.forward(x))
    for cnt in range(100):
        gs = net.backward(x,y)
        net.weights_adjust(gs)

    print(net.forward(x))
