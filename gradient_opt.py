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
    itr_cnt = 0
    convergence = False
    if method == 'sgd':
        alpha = args[0]
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

    if method =='momentum':
        alpha = args[0]
        mu = args[1]
        v0 = theta
        v1 = np.array([0, 0], dtype='float64')
        while itr_cnt < max_itr and not convergence:
            grad = batch_grad(x,y,theta,batch_size)
            v1 = mu*v0 + alpha*grad
            v0=v1
            new_theta = theta - v1
            if distance.euclidean(theta, new_theta) < 1e-4:
                convergence = True
                print("momentum converges")
            theta = new_theta
            itr_cnt += 1
        if itr_cnt >= max_itr:
            print("momentum does not converge")
        return theta, convergence

    if method == 'nestrov':
        alpha = args[0]
        mu = args[1]
        v0 = theta
        v1 = np.array([0, 0], dtype='float64')
        while itr_cnt < max_itr and not convergence:
            theta_ahead = theta - mu*v0
            grad = batch_grad(x, y, theta_ahead, batch_size)
            v1 = mu * v0 + alpha * grad
            v0 = v1
            new_theta = theta - v1
            if distance.euclidean(theta, new_theta) < 1e-4:
                convergence = True
                print("nestrov converges")
            theta = new_theta
            itr_cnt += 1
        if itr_cnt >= max_itr:
            print("nestrov does not converge")
        return theta, convergence

    if method =='adagrad':
        alpha = args[0]
        G = theta**2
        eps = 1e-8
        while itr_cnt < max_itr and not convergence:
            grad = batch_grad(x,y,theta,batch_size)
            new_theta = theta - alpha/np.sqrt(G+eps)*grad
            G += grad**2
            if distance.euclidean(theta, new_theta) < 1e-4:
                convergence = True
                print("adagrad converges")
            theta = new_theta
            itr_cnt += 1
        if itr_cnt >= max_itr:
            print("adagrad does not converge")
        return theta, convergence

    if method == 'rmsprop':
        alpha = args[0]
        mu=args[1]
        eps = 1e-8
        expect = theta**2
        while itr_cnt < max_itr and not convergence:
            grad = batch_grad(x,y,theta,batch_size)
            expect = mu*expect + (1-mu)*grad**2
            new_theta = theta - alpha/np.sqrt(expect+eps)*grad
            if distance.euclidean(theta, new_theta) < 1e-4:
                convergence = True
                print("rmsprop converges")
            theta = new_theta
            itr_cnt += 1
        if itr_cnt >= max_itr:
            print("rmsprop does not converge")
        return theta, convergence

if __name__ =='__main__':
    data = sio.loadmat('a1data.mat')
    x = data['x'].T[0]
    y = data['y'].T[0]
    theta,con = gradient_optmization(x,y,[0.001,0.9],'rmsprop')
    print(theta)



