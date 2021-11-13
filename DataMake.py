import sys
import h5py
import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def FindSum(a):
    Sum=0
    (m,n)=a.shape
    for i in range (0,m):
        for j in range (0,n):
            Sum=Sum+a[i,j]
    return Sum


def SumNormalization(Img):
    B=np.zeros([1000,240,240])
    print(B.shape)
    for k in range(0,1000):
        for i in range (0,12):
            for j in range(0,12):
                num=Img[k,20*i:20*(i+1),20*j:20*(j+1)]
                #print(num.shape)
                Sum=FindSum(num)
                if Sum!=0:
                    Img[k,20*i:20*(i+1),20*j:20*(j+1)]=Img[k,20*i:20*(i+1),20*j:20*(j+1)]/Sum
                B[k,20*i:20*(i+1),20*j:20*(j+1)]=Img[k,20*i:20*(i+1),20*j:20*(j+1)]
    return B


def save_h5(times=0,str1='',str2='',batchlen=1000,imgsize=240,nmode=299):
    if times == 0:
        h5f = h5py.File('Sum_NewData_299_10.h5', 'w')
        X_train = h5f.create_dataset(name="Xtrain",
                                     shape=(batchlen, imgsize, imgsize),
                                     maxshape=(None, imgsize, imgsize),
                                     # chunks=(1, 1000, 1000),
                                     dtype='float64')
        Y_train = h5f.create_dataset(name="Ytrain",
                                    shape=(batchlen, nmode),
                                     maxshape=(None, nmode),
                                     # chunks=(1, 1000, 1000),
                                     dtype='float64')
    else:
        h5f = h5py.File('Sum_NewData_299_10.h5', 'a')
        X_train = h5f['Xtrain']
        Y_train = h5f['Ytrain']
        
    str1 = str1 + str((times + 1) * batchlen) + '.mat'
    # print(str1)
    str2 = str2 + str((times + 1) * batchlen) + '.mat'
    # print(str2)
    data_az = scipy.io.loadmat(str1) 
    data_Img = scipy.io.loadmat(str2)
    Zer = data_az['Zer'].astype('float64')
    Zer=np.transpose(Zer)
    Img = data_Img['ISHWFS_3D'].astype('float64')
    Img=np.transpose(Img)
    Img.resize([batchlen,imgsize,imgsize])
    Img=SumNormalization(Img)

    X_train.resize([times*batchlen+batchlen,imgsize,imgsize])
    X_train[times*batchlen:times*batchlen+batchlen] = Img
    Y_train.resize([times*batchlen+batchlen,nmode])
    Y_train[times*batchlen:times*batchlen+batchlen] = Zer

    h5f.close()




if __name__ == '__main__':
    # save_h5(0)
    str1 = 'TurbPhaseGeneration/DataSet/Zer_'
    str2 = 'TurbPhaseGeneration/DataSet/ISHWFS_'

    for i in range(100):
        print(i)
        save_h5(i,str1,str2,1000,240,299)










