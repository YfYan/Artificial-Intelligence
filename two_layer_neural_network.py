import numpy as np


class myNetwork(object):

    def __init__(self,input_dim,output_dim):
        self.weights = [np.ones((30,input_dim)),
                        np.ones((30,30)),
                        np.ones((output_dim,30))]

        #保留隐藏层未激活前的数据
        self.a = []
        self.z = []

    #前向传播过程
    @staticmethod
    def relu(x):
        for i in range(x.shape[0]):
            x[i][0] = max(x[i][0],0)


    @staticmethod
    def relu_gradient(x):
        g = np.zeros(x.shape)
        for i in range(x.shape[0]):
            if x[i][0]>0:
                g[i][0]=1
            elif x[i][0] == 0:
                g[i][0]=0.5
            else:
                g[i][0]= 0
        return g

    def forward(self,x_):
        #先作copy防止篡改原数据
        self.a=[]
        self.z=[]
        x = x_.copy()
        self.a.append(x.copy())

        for weight in self.weights:
            x = np.dot(weight,x)
            self.z.append(x.copy())
            self.relu(x)
            self.a.append(x.copy())
        return x


    def backward(self,x,y):
        y_hat = self.forward(x)
        weight_gradients = []
        delta = (y_hat - y) * self.relu_gradient(self.z[-1])

        for i in range(len(self.weights)-1,0,-1):
            weight = self.weights[i]
            weight_gradients.insert(0,np.dot(delta,self.a[i].T))
            if i!= 0 :
                delta = self.relu_gradient(self.z[i-1])*(np.dot(weight.T,delta))

        return weight_gradients

    def weights_adjust(self,weight_gradients):
        

if __name__ == '__main__':
    x=np.array([2,1]).reshape((2,1))
    y=np.array([1]).reshape((1,1))
    net = myNetwork(2,1)
    print(net.backward(x,y))