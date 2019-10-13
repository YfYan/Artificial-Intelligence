from gradient_opt import gradient_optmization
import scipy.io as sio
import time

def time_stat(x,y):
    args=[0.001, 0.9, 0.9]
    

if __name__ == '__main__':
    data = sio.loadmat('a1data.mat')
    x = data['x'].T[0]
    y = data['y'].T[0]