import numpy as np
import random
from scipy.io import loadmat

def readMFile(datapath):
	m = loadmat(datapath)
	T11 = m['T11']
	T12 = m['T12']
	T13 = m['T13']
	T22 = m['T22']
	T23 = m['T23']
	T33 = m['T33']
	return T11,T12,T13,T22,T23,T33

def padding(path,savepath,wsize=128):
	T11,T12,T13,T22,T23,T33 = readMFile(path)
	T1 = np.real(T11)
	T2 = np.real(T12)
	T3 = np.real(T13)
	T4 = np.real(T22)
	T5 = np.real(T23)
	T6 = np.real(T33)
	
	T7 = np.imag(T12)
	T8 = np.imag(T13)
	T9 = np.imag(T23)
	data = np.array([T1,T2,T3,T4,T5,T6,T7,T8,T9])
	data = np.transpose(data,[1,2,0])
	#padding 
	m,n = np.shape(T1)
	pdata = np.zeros([m+wsize-1,n+wsize-1,9])
	
	l_n = wsize//2-1
	r_n = wsize//2
	pdata[0:l_n,0:l_n,:] = np.flip(np.flip(data[:l_n,:l_n,:],0),1)
	pdata[0:l_n,l_n:n+l_n,:] = np.flip(data[:l_n,:,:],1)
	pdata[0:l_n,n+l_n:,:] = np.flip(np.flip(data[:l_n,n-r_n:,:],0),1)
	
	pdata[l_n:m+l_n,0:l_n,:] = np.flip(data[:,:l_n,:],0)
	pdata[l_n:m+l_n,l_n:n+l_n,:] = data
	pdata[l_n:m+l_n,n+l_n:,:] = np.flip(data[:,n-r_n:,:],0)
	
	pdata[m+l_n:,0:l_n,:] = np.flip(np.flip(data[m-r_n:,:l_n,:],0),1)
	pdata[m+l_n:,l_n:n+l_n,:] = np.flip(data[m-r_n:,:,:],1)
	pdata[m+l_n:,n+l_n:,:] = np.flip(np.flip(data[m-r_n:,n-r_n:,:],0),1)
	np.save(savepath+str(wsize)+'.npy',pdata)
	
if __name__ == "__main__":
	path = './DATA/T_SanFrancisco.mat'
	savepath = './DATA/San_padding'
	padding(path,savepath,wsize=36)