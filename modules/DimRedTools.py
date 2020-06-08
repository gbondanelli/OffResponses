from numpy import *
from numpy import random
from random import sample
from numpy.linalg import cond
from numpy.linalg import eigh,norm,inv, multi_dot, svd
from scipy.linalg import eig,svdvals,block_diag
from scipy.sparse import csr_matrix
from sklearn.metrics import r2_score
from time import*
from sklearn.utils.extmath import randomized_svd

from matplotlib.pyplot import *

def PCA(X):
    # X is a matrix #cells x #timepoints
    # returns the eigvals and eigvects of the corrMat in descending order
    if len(X.shape)==3:
        S1=X.shape[0]; S2=X.shape[1]; S3=X.shape[2]
        X=reshape(X,(S1,S2*S3),order='F')
    X=X-mean(X,axis=1,keepdims=True)
    C=dot(X,X.T)
    d,V=eigh(C)
    V=V[:,argsort(d)[::-1]]
    d=d[argsort(d)[::-1]]
    return d,V

def PCA_rand(X,n_components,n_iter):
    X=X-mean(X,axis=1,keepdims=True)
    V,d,_=randomized_svd(X,n_components=n_components,n_iter=n_iter)
    V=V[:,argsort(d)[::-1]]
    d=d[argsort(d)[::-1]]
    return d,V


def cvPCA_2trials(X):
    # X is the data matrix: N x T x #stim x #trials
    ## Method from Stringer et al. 2018
    N = X.shape[0]
    T = X.shape[1]
    C = X.shape[2]
    Ntrials = X.shape[3]
    n = int(Ntrials*(Ntrials-1)/2)

    cvLambdas = empty((n,N))
    m=0
    for i in range(Ntrials):
        for j in range(i+1,Ntrials):

            F1 = reshape(X[:,:,:,i],(N,T*C), order='F')
            F1 = F1-mean(F1,axis=1,keepdims=True)

            F2 = reshape(X[:,:,:,j],(N,T*C), order='F')
            F2 = F2-mean(F2,axis=1,keepdims=True)

            _,V = PCA(F1)
            cvLambdas[m,:] = diag(multi_dot([V.T,F2,F1.T,V]))
            m+=1
    return cvLambdas
