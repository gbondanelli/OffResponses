from numpy          import *
from numpy          import random
from numpy.linalg   import cond,inv,multi_dot,eigvals,matrix_rank
from scipy.linalg   import svdvals,svd
from scipy.linalg   import  polar as polar_decomp
from modules.DimRedTools    import*
from modules.StatTools      import *

def gaussian(t,m,s):
    return exp(-(t-m)**2/2/s**2)

def basis_gaussian(nsteps,width,number):
    tnormal = linspace(0,1,nsteps)
    latency = linspace(0,1,number)
    f=empty((number,nsteps))
    for i in range(number):
        f[i,:] = gaussian(tnormal,latency[i],width)
    return f

def basis_alpha(nsteps,tau,number):
    tnormal = linspace(0,1,nsteps)
    latency = linspace(0,1,number)
    f=empty((number,nsteps))
    for i in range(number):
        f[i,:] = e*(tnormal-latency[i])*exp(-(tnormal-latency[i])/tau)*(tnormal>latency[i])/tau
    return f

def evaluate_fit(Y,Yrec,N,number_par,dt):
    R2_single = empty(N)
    Y_dot=diff(Y,axis=1)/dt
    Yrec_dot=diff(Yrec,axis=1)/dt
    MSE = mean((Y-Yrec)**2)
    n = float(len(Y.ravel()))
    R2 = 1-MSE/var(Y)
    R2adj=1-(1-R2)*(n-1)/(n-number_par-1)

    MSE_dot = mean((Y_dot-Yrec_dot)**2)
    AIC=n*log(MSE_dot)+2.*number_par+n*log(2*pi)+n
    BIC=n*log(MSE_dot)+number_par*log(n)+n*log(2*pi)+n

    for i in range(N):
        R2_single[i] = 1-var(Y[i,:]-Yrec[i,:])/var(Y[i,:])
    return MSE,R2,R2_single,R2adj,AIC,BIC

def normalize_by_range(Y):
    N = Y.shape[0]; T=Y.shape[1]; C=Y.shape[2]
    Y2=empty((N,T,C)); R=empty((N,C))
    for i_c in range(C):
        for i_n in range(N):
            y=Y[i_n,:,i_c]
            Range=amax(y)-amin(y)
            Range=Range*sign(y[argmax(abs(y))])
            R[i_n,i_c] = Range
            Y2[i_n,:,i_c] = Y[i_n,:,i_c]/Range
    return Y2,R

def normalize_by_ic(Y):
    N = Y.shape[0]; T=Y.shape[1]; C=Y.shape[2]
    Y2=empty((N,T,C))
    R=empty((N,C))
    for i_c in range(C):
        for i_n in range(N):
            y = Y[i_n,:,i_c]
            ic = y[0]; R[i_n,i_c] = ic
            Y2[i_n,:,i_c] = Y[i_n,:,i_c]/ic
    return Y2,R

def normalize_back(Y,R):
    N = Y.shape[0]; T=Y.shape[1]; C=Y.shape[2]
    Y2=empty((N,T,C))
    for i_c in range(C):
        for i_n in range(N):
            Y2[i_n,:,i_c] = R[i_n,i_c]*Y[i_n,:,i_c]
    return Y2

def LinReg_basis_functions(Y,X,l,dt):
    # Y is N x T x C, Y1 is N x TC, M is N x Nbasis, X is Nbasis x T x C
    # fit of Y1.T=X.T*MT
    N = Y.shape[0]; T=Y.shape[1]; C=Y.shape[2]; Nbasis = X.shape[0]
    Y1 = reshape(Y,(N,T*C),order='F')
    X = reshape(X, (Nbasis,T*C),order='F')
    MT,_  = approx_LS_solution(X,l,Y1.T,'none')
    Yrec = dot(MT.T,X)
    number_par = df_ridge(X,l)*N
    MSE,R2,R2_single,R2adj, AIC, BIC = evaluate_fit(Y1,Yrec,N,number_par,dt)
    return [R2, R2_single, R2adj, AIC, BIC], MT, [Y1,Yrec]

def LinReg_basis_functions_CV(Y,X,l):
    N = Y.shape[0]; T=Y.shape[1]; C=Y.shape[2]
    R2_single = empty((N,C))
    R2 = empty(C)
    R2adj = empty(C)
    for i_c in range(C):
        train = [True]*C; train[i_c] = False
        Ytrain=Y[:,:,train]; Ytest=Y[:,:,i_c]
        _,_,_,MT,_=LinReg_basis_functions(Ytrain,X,l)
        Yrec = dot(MT.T,X)
        number_par = float(len(MT.ravel()))
        _,R2[i_c],R2_single[:,i_c], R2adj[i_c] = evaluate_fit(Ytest,Yrec,N,number_par)
    return R2, R2_single, R2adj

def LinReg_basis_functions_Kfold_time(Y,X,l,K,dt):
    N = Y.shape[0]; T=Y.shape[1]; C=Y.shape[2]; Nbasis = X.shape[0]
    size_test_set = int(T/K)
    R2 = empty(K)
    for c in range(K):
        if c == K-1:
            msk_train = ones(T,dtype=bool); msk_train[c*size_test_set:]=False
        else:
            msk_train = ones(T,dtype=bool); msk_train[c*size_test_set:(c+1)*size_test_set]=False
        msk_test = ~msk_train

        Ytrain=Y[:,msk_train,:]
        Xtrain=X[:,msk_train,:]
        Ytest=Y[:,msk_test,:]
        Xtest=X[:,msk_test,:]

        _,MT,_ = LinReg_basis_functions(Ytrain,Xtrain,l,dt)

        Xtest2 = reshape(Xtest, (Nbasis,C*sum(msk_test)), order='F')
        Ytest2 = reshape(Ytest, (N,C*sum(msk_test)), order='F')

        Yrec=dot(MT.T,Xtest2)

        MSE = mean( (Yrec-Ytest2)**2 )
        R2[c] = 1 - MSE/var(Ytest2)

    return R2
