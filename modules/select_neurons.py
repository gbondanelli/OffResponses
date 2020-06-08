from numpy import *

def compute_SNR(X):
    N=X.shape[0]; C=X.shape[2]
    SNR=empty((N,C))
    for i_stim in range(C):
        for i_cell in range(N):
            SNR[i_cell,i_stim]=amax(X[i_cell,:,i_stim])-amin(X[i_cell,:,i_stim])
    return SNR

def select_most_responsive_neurons(X,stimuli,percentage):
    N=X.shape[0]; C=X.shape[2]
    if stimuli=='all':
        SNR=compute_SNR(X)
    else:
        SNR=compute_SNR(X[:,:,stimuli])
    SNRmean=mean(SNR,axis=1)
    a=argsort(SNRmean)[::-1]
    n=int(percentage*N)
    selection=a[:n]
    return selection


