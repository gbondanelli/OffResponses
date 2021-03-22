from myimports import*

path    = './data'
Data=load(path+'/off_responses_trialavg.npy')

DT = 0.031731
total_time = 108*DT
time_stim_presentation = 33*DT
nsteps = 1000
dt = total_time/nsteps


## Figure 3C

ntrials = 100
ncells = Data.shape[0]
nsteps = Data.shape[1]
percentage = .4
stim = [1,5]
SingleTrialTraj_SingleCell_2= empty((int(percentage*ncells), nsteps, ntrials, len(stim)))

s = select_most_responsive_neurons(Data,stim,percentage)
Data_sel = Data[:,:,stim]; Data_sel=Data_sel[s,:,:]

nbasis = 10; l = 1

D = Data_sel
Y = copy(D)
N = Y.shape[0]
T = Y.shape[1]
C = Y.shape[2]

Y,R = normalize_by_range(Y)

f = basis_gaussian(D.shape[1],.1,nbasis)[:,:,None]
X = repeat(expand_dims(f,axis=2),len(stim),axis=2)

_,_, [Y1,Yrec] = LinReg_basis_functions(Y,X,l,dt)
Y1 = reshape(Y1,(N,T,C),order='F')
Yrec = reshape(Yrec,(N,T,C),order='F')

Yrec_renorm = normalize_back(Yrec, R )
Y1_renorm = normalize_back(Y1, R)

for j in range(len(stim)):
    for i in range(ntrials):
        Rnoisy = random.multivariate_normal(R[:,j], 0.0001*identity(len(R[:,j])),1)[0,:]
        SingleTrialTraj_SingleCell_2[:, :, i, j] = dot(diag(Rnoisy), Yrec[:,:,j])


col=['#F14377','#569FD1']

m=0
i=stim[m]
D = Data_sel[:,:,m]
Drec = SingleTrialTraj_SingleCell_2[:,:,:,m]

figure(m+1, figsize=(2,2))
ax=subplot(111)
gca().set_aspect('equal', adjustable='box')
ax.spines['left'].set_bounds(-.5, .5)
ax.spines['bottom'].set_bounds(-.5,.5)
ax.spines['bottom'].set_position(('axes', -0.1))
ax.spines['left'].set_position(('axes', -0.))

d1,V1=PCA(D)
V1[:,1]=-V1[:,1]

for i_trial in range(ntrials):
    plot(dot(V1[:,0],Drec[:,:,i_trial]),dot(V1[:,1],Drec[:,:,i_trial]),lw=2,color='#D1D9DC')
Drecmean = mean(Drec,axis=2)
plot(dot(V1[:,0],Drecmean),dot(V1[:,1],Drecmean),'--', lw=1,color='#5C7C99')


plot(dot(V1[:,0],D),dot(V1[:,1],D),lw=1.5,color=col[m])
plot(dot(V1[:,0],D[:,0]),dot(V1[:,1],D[:,0]),'.',markersize=7,color=col[m])

xticks([-.5,0,.5])
yticks([-.5,0,.5])

if m == 0:
    xlabel('PC1 - OFF (8kHz)')
    ylabel('PC2 - OFF (8kHz)')
if m == 1:
    xlabel('PC1 - OFF (WN)')
    ylabel('PC2 - OFF (WN)')

tight_layout()

#------------------
stim = [1]
SingleTrialTraj_SingleCell_stim1 = empty((int(percentage*ncells), nsteps, ntrials, len(stim)))

Data_sel = Data[:,:,stim]; Data_sel=Data_sel[s,:,:]

nbasis = 10; l = 1

D = Data_sel
Y = copy(D)
N = Y.shape[0]
T = Y.shape[1]
C = Y.shape[2]

Y,R = normalize_by_range(Y)

f = basis_gaussian(D.shape[1],.1,nbasis)[:,:,None]
X = repeat(expand_dims(f,axis=2),len(stim),axis=2)

_,_, [Y1,Yrec] = LinReg_basis_functions(Y,X,l,dt)
Y1 = reshape(Y1,(N,T,C),order='F')
Yrec = reshape(Yrec,(N,T,C),order='F')

Yrec_renorm = normalize_back(Yrec, R )
Y1_renorm = normalize_back(Y1, R)

for j in range(len(stim)):
    for i in range(ntrials):
        Rnoisy = random.multivariate_normal(R[:,j], 0.0001*identity(len(R[:,j])),1)[0,:]
        SingleTrialTraj_SingleCell_stim1[:, :, i, j] = dot(diag(Rnoisy), Yrec[:,:,j])

i=stim[0]
D = Data_sel[:,:,0]
Drec = SingleTrialTraj_SingleCell_stim1[:,:,:,0]

figure(1, figsize=(2,2))

d1,V1 = PCA(D)
V1[:,1] = -V1[:,1]

Drecmean = mean(Drec,axis=2)
plot(dot(V1[:,0],Drecmean),dot(V1[:,1],Drecmean),'--', lw=1,color='#5C7C99')
xticks([-.5,0,.5])
yticks([-.5,0,.5])


## Figure 3B
stim = [1]
Data_sel = Data[:,:,stim]; Data_sel=Data_sel[s,:,:]

figure(figsize = (2.1,1.7))
t = dt*arange(T)
plot(t,norm(Data_sel[:,:,0],axis=0),lw=1.5,c=col[0])
for i in range(ntrials):
    plot(t,norm(SingleTrialTraj_SingleCell_2[:,:,i,0],axis=0),c='#D1D9DC')
plot(t,norm(mean(SingleTrialTraj_SingleCell_2,2)[:,:,0],axis=0),'--',lw =0.7,c='#5C7C99')
plot(t,norm(mean(SingleTrialTraj_SingleCell_stim1,2)[:,:,0],axis=0),'--',lw=0.7,c='#5C7C99')
plot([0.05,0.05],[0.4,1.4],'--',lw=.6, color='grey')
xlabel('Time (ms)')
ylabel('Activity norm')
ylim([0.4,1.4])
yticks([0.6,1,1.4])
xticks([.05,0.15,0.25,0.35],['$0$','$100$','$200$','$300$'])
tight_layout()

##

