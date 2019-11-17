from gradient_opt import gradient_optmization
import scipy.io as sio
import time
import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model as lm

def time_stat(x,y,method,plot):
    args=[0.001, 0.9, 0.9]
    time1 = time.time()
    theta, conver,itr_cnt,all_theta,total_loss = gradient_optmization(x,y,args,method,batch_size=100)
    print(theta,conver)
    time2 = time.time()
    print(time2 - time1)
    if plot:
        plt.plot(total_loss)
        plt.show()
        plt.plot(all_theta)
        plt.show()
    return time2 - time1,itr_cnt


if __name__ == '__main__':
    data = sio.loadmat('a1data.mat')
    x = data['x'].T[0]
    y = data['y'].T[0]
    methods = ['sgd','momentum','nestrov','adagrad','rmsprop','adadelta','adam']
    #time_stat(x,y,'adam',True)
    times = []
    cnt = []
    if True:
        for i in range(30):
            t,c = time_stat(x,y,'adam',False)
            times.append(t)
            cnt.append(c)
        print(np.mean(times),np.std(times))
        print(np.mean(cnt),np.std(cnt))
    # plt.scatter(x,y)
    # plt.show()
    # model = lm.LinearRegression()
    # model.fit(x,y)
    # print(model.intercept_,model.coef_)
    # print(model.score(x,y))
