from myimports import*

colors = ['#FFFFFF', '#FFBABA', '#F75656', '#7C0D0D']
cm = LinearSegmentedColormap.from_list(
        'new_cm', colors, N=100)

path    = './data'
Data   = load(path+'/off_responses_trialavg.npy')

ncells  = Data.shape[0]
nstim   = Data.shape[2]
dt = 108*0.031731/1000

nchunks = 10
nrandsampl = 10
l = 2
cv = True
cv_type = 'kfold'
dims = 100
rank = 70
n_free_params = dims*dims

par = [cv_type,nchunks,nrandsampl,l,n_free_params,'none',rank,dt]
R2_J,_,_,R2_JTOT, R2_sh = fit_dynamical_system(Data,dims,cv,par)

figure(figsize=(1.45,2.05))
ax = subplot(111)

eps = -.25
col = '#1E1212'
plot([1-eps,1+eps],[mean(R2_J),mean(R2_J)],c=col,lw=1.3)
plot([2-eps,2+eps],[mean(R2_JTOT),mean(R2_JTOT)],c=col,lw=1.3)
plot([3-eps,3+eps],[mean(R2_sh),mean(R2_sh)],c=col,lw=1.3)

plot(1*ones(nchunks),R2_J,'o',markersize=3.5, color='#FF6672', \
     markeredgewidth=.3,markeredgecolor='#4C4C4C')
plot(2*ones(nchunks),R2_JTOT,'o',markersize=3.5, color='#FF6672',\
     markeredgewidth=.3,markeredgecolor='#4C4C4C')
plot(3*ones(nchunks),R2_sh,'o',markersize=3.5,color='#DEF4E7', \
     markeredgewidth=.4,markeredgecolor='#4C4C4C')

ax.plot([.5,3.5],[0,0],'--', lw=0.7,c='grey')
ax.set_xlim(left=.5,right=3.5)
ax.set_ylim(top=1,bottom=-1.6)
ax.spines['left'].set_bounds(-1.,1)
ax.spines['bottom'].set_bounds(1,3)
xticks([1,2,3],[r'Full', r'Sum', 'Shuffle'], rotation=60, fontsize=9)
yticks([-1,0,1])
ylabel(r'Goodness of fit $R^2$')
tight_layout()
