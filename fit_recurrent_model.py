from myimports import*

path = './data'
Data = load(path+'/off_responses_trialavg.npy')

DT = 0.031731
total_time = 108*DT
time_stim_presentation = 33*DT
nsteps = 1000
dt = total_time/nsteps

#####  select stimulus  #####
stim=7
########################

T=Data.shape[1]
dims=100; nchunks=20; nrandsampl=20
cv=False; n_free_params=dims*dims; constraint1='none'
l=2
rank=70  ##### rank='none' for ordinary linear regression 
n_free_params1=int(dims*dims)
par1=['random',nchunks,nrandsampl,l,n_free_params1,constraint1,rank,dt]
ntrials=100
col1 = '#F14377'
col2 = '#569FD1'

rM,MT,E=fit_dynamical_system(Data,dims,cv,par1)

X_PC1=E[2]
X_dot=E[3]
X_dotP=E[4]
X_PC2=E[5]

#generate fit through exponential mapping
X_PC_P=empty((dims, ntrials, T))
X_PC_P[:,:,0]=random.multivariate_normal(X_PC2[:,stim*(T-1)],.0005*identity(dims),ntrials).T
for j in range(T-1):
    X_PC_P[:,:,j+1]=dot(expm(dt*(j+1)*MT.T),X_PC_P[:,:,0])

#PCA on data
X=X_PC2[:,stim*(T-1):(stim+1)*(T-1)]
d1,V1 = PCA(X)
V1[:,0] = -V1[:,0]
V1[:,1] = -V1[:,1]

figure(figsize=(2,2))
ax=subplot(111)
gca().set_aspect('equal', adjustable='box')
ax.spines['left'].set_bounds(-.5, .5)
ax.spines['bottom'].set_bounds(-.5,.5)
ax.spines['bottom'].set_position(('axes', -0.1))
ax.spines['left'].set_position(('axes', -0.))

for i_trial in range(ntrials):
    ln,=ax.plot(dot(V1[:,0],X_PC_P[:,i_trial,:]),dot(V1[:,1],X_PC_P[:,i_trial,:]),lw=1.2,color='#D1D9DC')
    ln.set_solid_capstyle('round')

X_PC_Pmean=mean(X_PC_P,axis=1)
plot(dot(V1[:,0],X_PC_Pmean[:,:]),dot(V1[:,1],X_PC_Pmean[:,:]),'--', lw=1,color='#5C7C99')

plot(dot(V1[:,0],X),dot(V1[:,1],X),lw=1.5,color=col2)
plot(dot(V1[:,0],X[:,0]),dot(V1[:,1],X[:,0]),'.',markersize=7,color=col2)

xticks([-.5,0,.5])
yticks([-.5,0,.5])

if stim == 1:
    xlabel('PC1 - OFF (8kHz)')
    ylabel('PC2 - OFF (8kHz)')
if stim == 5:
    xlabel('PC1 - OFF (WN)')
    ylabel('PC2 - OFF (WN)')

tight_layout()
