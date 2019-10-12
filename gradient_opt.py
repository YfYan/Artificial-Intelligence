import scipy.io as sio
import numpy as np
from scipy.spatial import distance

def my_linear_model(x,theta):
    return theta[0] + theta[1] * x

def loss(x,y,theta):
    return my_linear_model(x,theta) - y

def single_grad(x,y,theta):
    return np.array([2* loss(x,y,theta),2*loss(x,y,theta)*x],dtype = 'float64')

def batch_grad(x,y,theta,batch_size):
    total = x.shape[0]
    choice = np.random.choice(total,batch_size,replace = False)
    grad = np.array([0.0,0.0],dtype='float64')
    for i in range(batch_size):
        index = choice[i]
        grad += single_grad(x[index],y[index],theta)
    return grad / batch_size

def gradient_optmization(x,y,args,method,batch_size = 80):
    theta = np.array([1.0,1.0],dtype='float64')
    max_itr = 100000
    if method == 'sgd':
        alpha = args[0]
        itr_cnt = 0
        convergence = False
        while itr_cnt < max_itr and not convergence:
            grad = batch_grad(x,y,theta,batch_size)
            new_theta = theta - alpha*grad
            if distance.euclidean(theta,new_theta) < 1e-4:
                convergence = True
                print("sgd converges")
            theta = new_theta
            itr_cnt+=1
        if itr_cnt >= max_itr:
            print("sgd does not converge")
        return theta,convergence

if __name__ =='__main__':
    data = sio.loadmat('a1data.mat')
    x = data['x'].T[0]
    y = data['y'].T[0]
    theta,con = gradient_optmization(x,y,[0.001],'sgd')
    print(theta)



