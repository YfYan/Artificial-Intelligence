from gradient_opt import gradient_optmization
import scipy.io as sio
import time

def time_stat(x,y,method):
    args=[0.001, 0.9, 0.9]
    time1 = time.time()
    theta, conver = gradient_optmization(x,y,args,method)
    print(theta,conver)
    time2 = time.time()
    print(time2-time1)


if __name__ == '__main__':
    data = sio.loadmat('a1data.mat')
    x = data['x'].T[0]
    y = data['y'].T[0]
    methods = ['sgd','momentum','nestrov','adagrad','rmsprop','adam']
    time_stat(x,y,'adam')