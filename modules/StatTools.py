from numpy import *
from numpy import random
from numpy.linalg import cond,inv,multi_dot,eigvals,matrix_rank, multi_dot
from scipy.linalg import svdvals,svd
from scipy.linalg import  polar as polar_decomp
from modules.DimRedTools import*
from sklearn.linear_model import LinearRegression

def pseudo_inverse(X,l):
    #X is N x TC
    #l is the ridge param
    W=inv(dot(X,X.T)+l*identity(X.shape[0]))
    return W.dot(X), cond(W)

def df_ridge(X,l):
    b=dot(X,X.T)
    d=eigvals(b)
    d=d.real
    df=sum(d/(d+l));
    return df

def approx_LS_solution(X,l,Y,constraint):
    A = pseudo_inverse(X,l)[0].dot(Y)
    cond=pseudo_inverse(X,l)[1]

    if constraint == 'none':
        return A, cond

    if constraint == 'nearest_orth':
        U,P=polar_decomp(A+identity(A.shape[0]))
        return U-identity(U.shape[0]), cond

    if constraint == 'procustes':
        B=dot(X,X.T+Y)
        U,P=polar_decomp(B)
        return U-identity(U.shape[0]), cond

    if constraint == 'SP':
        return symmetric_procustes(X.T,Y), cond

    if constraint == 'nearest_symm':
        return .5*(A+A.T), cond

def Rank_Reduced_Regression(X,BOLS,rank,l):
    X=vstack((X,sqrt(l)*identity(X.shape[1])))
    Yhat=dot(X,BOLS)
    _,s,v=svd(Yhat)
    v1=v[:rank,:]
    BRRR=multi_dot([BOLS,v1.T,v1])

    #evaluate degrees of freedom
    n=max(Yhat.shape); q=min(Yhat.shape); rx=matrix_rank(X); r_=min(rx,q)
    dfNaive=rank*(n+q-rank);
    sl=s[rank+1:]; sk=s[:rank]; Sl,Sk=meshgrid(sl,sk); F=(Sk**2+Sl**2)/(Sk**2-Sl**2)
    df=float(rx)*q*(rank==r_ )+( max(rx,q)*rank+sum(F) )*(rank<r_)

    return BRRR, [dfNaive, df]

def compute_Jsum(X1, l, X2, constraint, C, T, rank, nDim):
    Jsum=0
    for stim in range(C):
        x1=X1[:,stim*(T-1):(stim+1)*(T-1)]
        x2=X2[stim*(T-1):(stim+1)*(T-1),:]
        mt = approx_LS_solution(x1,l,x2,constraint)[0];
        if  rank != 'none':
            mt = Rank_Reduced_Regression(x1.T,mt+identity(nDim),rank,l)[0] - identity(nDim)
        Jsum+=mt
    return Jsum


def fit_dynamical_system(X,nDim,cv,par):
### X is #CELLS x #TIMEPOINTS x #CONDITIONS

    N = X.shape[0]
    T = X.shape[1]
    C = X.shape[2]

    method = par[0];
    nchunks = par[1];
    nrandsampl = par[2];
    l = par[3]  # ridge parameter
    number_par = par[4]## for AIC and BIC formulae
    constraint = par[5] ## None, nearest_orth, procustes, SP, nearest_symm
    rank = par[6] ## rank for rank-reduced regression
    dt = par[7]

    size_test_set = int((T-1)/nchunks)
    size_training_set = T-1-size_test_set

    X = reshape(X,(N,T*C),order = 'F')
    d,V = PCA(X)
    x_pc = dot(V.T[:nDim,:],X)#-mean(X,axis=1,keepdims=True))

    if len(par) == 8: # for fitting all stimuli
        X_PC = x_pc

    if len(par) == 9:  #for fitting single stimuli
        stim = par[8]
        X_PC = dot(V.T[:nDim,:],X[:,stim*T:(stim+1)*T]);

    C = int(X_PC.shape[1]/T)

    mask1 = ones(T,dtype=bool)
    mask1[-1] = False
    mask1 = tile(mask1,(C,))
    mask2 = ones(T,dtype=bool)
    mask2[0] = False
    mask2 = tile(mask2,(C,))
    X_PC2 = X_PC[:,mask2]
    X_PC1 = X_PC[:,mask1]
    Xdot = X_PC2-X_PC1; #X_PC=X_PC1
    Xdot = Xdot/dt

    # Xdot.T=X_PC.T*MT
    #No cross-validation
    if cv == False:
        MT, cond = approx_LS_solution(X_PC1, l, Xdot.T, constraint)

        if rank != 'none':
            A, DF = Rank_Reduced_Regression(X_PC1.T, MT+identity(nDim), rank, l)
            MT = A-identity(nDim)
            number_par = DF[1]
        else:
            number_par = df_ridge(X_PC1,l)*X_PC1.shape[0]

        Xdot_predictedT = dot(X_PC1.T,MT)
        MSE = mean((Xdot.T-Xdot_predictedT)**2)
        sr = var(Xdot)
        SE = sum(Xdot.T-Xdot_predictedT)**2

        n = float(len(X_PC1.ravel())); #print n, number_par
        R2 = 1-MSE/sr
        R2adj = 1-(1-R2)*(n-1)/(n-number_par-1)
        AIC = n*log(MSE)+2.*number_par+n*log(2*pi)+n
        BIC = n*log(MSE)+number_par*log(n)+n*log(2*pi)+n
        GCV = n*SE/(n-number_par)**2

        return [R2,R2adj,AIC,BIC,GCV, cond], MT ,[V, d, x_pc, Xdot, Xdot_predictedT.T, X_PC1, X_PC2]

    #with cross-validation
    elif cv == True:

        nshuffles = 10
        R2 = empty(nchunks)
        R2_Jsum = empty(nchunks)
        R2_Jsum_sh = empty((nshuffles,nchunks))
        MSE = empty(nchunks)

        #K-fold method
        if method == 'kfold':
            for c in range(nchunks):
                if c==nchunks-1:
                    msk_cv_train = ones(T-1,dtype=bool)
                    msk_cv_train[c*size_test_set:] = False; \
                    msk_cv_train = tile(msk_cv_train,(C,))
                else:
                    msk_cv_train = ones(T-1,dtype=bool)
                    msk_cv_train[c*size_test_set:(c+1)*size_test_set] = False; \
                    msk_cv_train = tile(msk_cv_train,(C,))
                msk_cv_test = ~(msk_cv_train)
                X_PC1train = X_PC1[:,msk_cv_train]
                X_PC1test = X_PC1[:,msk_cv_test]
                X_PC2train = X_PC2[:,msk_cv_train]
                X_PC2test = X_PC2[:,msk_cv_test]
                Xdot_train = Xdot[:,msk_cv_train]
                Xdot_test = Xdot[:,msk_cv_test]
                MT, cond = approx_LS_solution(X_PC1train,l,Xdot_train.T,constraint)

                if rank != 'none':
                    MT = Rank_Reduced_Regression(X_PC1train.T,MT+identity(nDim),rank,l)[0] - identity(nDim)

                Xdot_predictedT = dot(X_PC1test.T,MT)
                MSE[c] = mean((Xdot_test.T-Xdot_predictedT)**2)
                sr = var(Xdot_test)
                R2[c] = 1-MSE[c]/sr

                # performance with Jsum
                Jsum = compute_Jsum(X_PC1train,5,Xdot_train.T,constraint, C, T, 6, nDim)

                Xdot_predictedT_Jsum = dot(X_PC1test.T,Jsum)
                mse = mean((Xdot_test.T-Xdot_predictedT_Jsum)**2)
                sr = var(Xdot_test)
                R2_Jsum[c] = 1-mse/sr

                #quality of fit with Jsum_shuffle
                A = Jsum.ravel()
                for i_shuffles in range(nshuffles):
                    A_sh = A[random.permutation(range(nDim*nDim))]
                    Jsum_sh = reshape(A_sh,(nDim,nDim))
                    Xdot_predictedT_Jsum_sh = dot(X_PC1test.T,Jsum_sh)
                    mse = mean((Xdot_test.T-Xdot_predictedT_Jsum_sh)**2)
                    sr = var(Xdot_test)
                    R2_Jsum_sh[i_shuffles,c] = 1-mse/sr

            return R2, cond, MSE, R2_Jsum, mean(R2_Jsum_sh,axis=0)


def OneD_OLD_LOOCV(x,y):
    n = x.shape[0]
    R2 = empty(n)
    for i in range(n):
        X = delete(x,i)
        Y = delete(y,i)
        x1 = X.reshape((-1,1))
        model = LinearRegression()
        model.fit(x1,Y)
        b = model.intercept_
        slope = model.coef_
        R2[i] = 1-(y[i]-b-slope*x[i])**2/y[i]**2
    return R2

def fit_dyn_best_ridge(X,nDim,par,l,K):
    NL = len(l)
    par[0] = 'kfold'
    par[1] = K
    R2 = empty((K,NL))
    for i in range(NL):
        par[3] = l[i]
        R2[:,i],_,_ = fit_dynamical_system(X,nDim,True,par)
    a = argmax(mean(R2,axis = 0))
    return R2[:,a], l[a]
