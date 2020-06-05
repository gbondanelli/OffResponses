from numpy import *
from numpy import random
from random import sample
from numpy.linalg import cond
from numpy.linalg import eigh,norm,inv
from scipy.linalg import eig,svdvals

from matplotlib.pyplot import *

def smooth_cov(N,t,tau):
    cov=empty((N,N))
    for i in range(N):
        for j in range(N):
            cov[i,j]=exp(-(t[i]-t[j])**2/2/tau**2)
    return cov

def euler_initial_condition(J,r0,t,nsteps,N,sigma,tau,tau_sigma=0):
    dt=t[1]-t[0]
    r=empty((N,nsteps))
    r[:,0]=r0
    if tau_sigma == 0:
        Sigma=random.normal(0,1,(N,nsteps-1))
    if tau_sigma != 0:
        cov=smooth_cov(nsteps-1,t,tau_sigma)
        Sigma=random.multivariate_normal(zeros(nsteps-1), cov, N)
    for i in range(nsteps-1):
        dr=(-r[:,i]+ dot(J,(r[:,i])) +sigma/sqrt(dt)*Sigma[:,i])*dt/tau
        r[:,i+1]=r[:,i]+dr
    return r

def euler_input(J,INP,t,nsteps,N,sigma,tau,tau_sigma=0):
    dt=t[1]-t[0]
    r=empty((N,nsteps))
    r[:,0]=zeros(N)
    if tau_sigma == 0:
        Sigma=random.normal(0,1,(N,nsteps-1))
    if tau_sigma != 0:
        cov=smooth_cov(nsteps-1,t,tau_sigma)
        Sigma=random.multivariate_normal(zeros(nsteps-1), cov, N)
    for i in range(nsteps-1):
        dr=(-r[:,i]+ dot(J,(r[:,i])) +sigma/sqrt(dt)*Sigma[:,i]+INP[:,i])*dt/tau
        r[:,i+1]=r[:,i]+dr
    return r

def rank1_random(N,cosphi):
    x1=random.normal(0,1./sqrt(N),N); x2=random.normal(0,1./sqrt(N),N)
    v=x1; u=cosphi*v+sqrt(1-cosphi**2)*x2
    return u,v

def simulated_data(par):
<<<<<<< HEAD

=======
    
>>>>>>> 58af17a6174d73be89d658c4e0e44077675fdf27
    T = par['T']
    N = par['N']
    nsteps = par['nsteps']
    n_trajectories = par['n_trajectories']
    connectivity = par['connectivity']
    noise = par['noise']
    g = par['g']
    P = par['rank']
    D = par['Delta']
    initial_conditions = par['initial_conditions']
    tau_sigma = par['tau_sigma']
<<<<<<< HEAD

    t = linspace(0,T,nsteps)
    Data = empty((N,nsteps,n_trajectories))

    if connectivity == 'gaussian':
        J = random.normal(0,g/sqrt(N),(N,N))
        for i in range(n_trajectories):
            r0 = random.normal(0,1/sqrt(N),N)
            Data[:,:,i] = euler_initial_condition(J,r0,t,nsteps,N,noise,1, tau_sigma)

    if connectivity == 'gaussian_symm':
        J = random.normal(0,g/sqrt(N),(N,N))
        J = .5*(J+J.T)
        for i in range(n_trajectories):
            r0 = random.normal(0,1/sqrt(N),N)
            Data[:,:,i] = euler_initial_condition(J,r0,t,nsteps,N,noise,1,tau_sigma)

    elif connectivity == 'low_rank':
        J = zeros((N,N))
        V = empty((N,P))
        U = empty((N,P))
        for p in range(P):
            u,v = rank1_random(N,0)
            V[:,p] = v
            U[:,p] = u
            J = J+D*outer(u,v)
        J = J+g*random.normal(0,1/sqrt(N),(N,N))
        if initial_conditions == 'amplified':
            for i in range(n_trajectories):
                r0 = V[:,i]
                Data[:,:,i] = euler_initial_condition(J,r0,t,nsteps,N,noise,1,tau_sigma)
        elif initial_conditions == 'random':
            for i in range(n_trajectories):
                r0 = random.normal(0,1/sqrt(N),N)
                Data[:,:,i] = euler_initial_condition(J,r0,t,nsteps,N,noise,1,tau_sigma)

    return Data, J, [U,V]
=======
    
    t=linspace(0,T,nsteps)
    Data=empty((N,nsteps,n_trajectories))
    
    if connectivity == 'gaussian':
        J=random.normal(0,g/sqrt(N),(N,N))
        for i in range(n_trajectories):
            r0=random.normal(0,1/sqrt(N),N)
            Data[:,:,i]=euler_initial_condition(J,r0,t,nsteps,N,noise,1, tau_sigma)
            
    if connectivity == 'gaussian_symm':
        J=random.normal(0,g/sqrt(N),(N,N))
        J=.5*(J+J.T)
        for i in range(n_trajectories):
            r0=random.normal(0,1/sqrt(N),N)
            Data[:,:,i]=euler_initial_condition(J,r0,t,nsteps,N,noise,1,tau_sigma)
            
    elif connectivity == 'low_rank':
        J=zeros((N,N))
        V=empty((N,P))
        U=empty((N,P))
        for p in range(P):
            u,v=rank1_random(N,0)
            V[:,p]=v; U[:,p]=u
            J=J+D*outer(u,v)
        J=J+g*random.normal(0,1/sqrt(N),(N,N))
        if initial_conditions == 'amplified':
            for i in range(n_trajectories):
                r0=V[:,i]
                Data[:,:,i]=euler_initial_condition(J,r0,t,nsteps,N,noise,1,tau_sigma)
        elif initial_conditions == 'random':
            for i in range(n_trajectories):
                r0=random.normal(0,1/sqrt(N),N)
                Data[:,:,i]=euler_initial_condition(J,r0,t,nsteps,N,noise,1,tau_sigma)
               
    return Data, J, [U,V]

def OFF_filter(t,alpha,tau):
    return (alpha*t+1)*exp(-alpha*t/tau)

def OFF_gaussian(t,peak,width):
    return exp(-(t-peak)**2/2/width**2)

def generate_single_cell_responses(par):
    T = par['T']
    N = par['N']
    nsteps = par['nsteps']
    nstim = par['nstim']
    noise = par['noise']
    tau = par['tau']
    a = par['alpha']
    t=linspace(0,T,nsteps)
    dt=t[1]-t[0]
    r=empty((N,nsteps,nstim))
    for j in range(nstim):
        x0=random.normal(0,1/sqrt(N),N); 
        x0=x0/norm(x0)
        for i in range(N):
            r[i,:,j]=x0[i]*(OFF_filter(t,a[i],tau)+noise/sqrt(dt)*random.normal(0,1.,nsteps))
    return r

def generate_single_cell_responses_gaussian(par):
    T = par['T']
    N = par['N']
    nsteps = par['nsteps']
    nstim = par['nstim']
    noise = par['noise']
    width = par['width']
    a = par['alpha']
    t=linspace(0,T,nsteps)
    dt=t[1]-t[0]
    r=empty((N,nsteps,nstim))
    for j in range(nstim):
        x0=random.normal(0,1/sqrt(N),N); 
        x0=x0/norm(x0)
        for i in range(N):
            r[i,:,j]=x0[i]*(OFF_gaussian(t,a[i],width)+noise/sqrt(dt)*random.normal(0,1.,nsteps))
    return r


        
        
>>>>>>> 58af17a6174d73be89d658c4e0e44077675fdf27
