from myimports import*

path = './data'
Data = load(path+'/off_responses_trialavg.npy')
ncells      = Data.shape[0]
nsteps = Data.shape[1]
nstim = Data.shape[2]
dt = 108*0.031731/1000.

stimuli=arange(nstim)
percentage=.5

l=5
constraint='none'
nrandsampl = 1
nchunks = 1
nsubsamplings = 10
dims=50
rank=6; #rank='none'
overlaps = empty((rank,rank,nsubsamplings, nstim))
initial_condition = empty((nsubsamplings, nstim))
initial_condition_along_v = empty((nsubsamplings, nstim))

initial_condition_rand = empty((nsubsamplings, nstim))
initial_condition_along_v_rand = empty((nsubsamplings, nstim))

for i_n in range(nstim):
    print(i_n)
    for i_sub in range(nsubsamplings):
        f=int(percentage*ncells)
        cells=random.choice(range(ncells), f,replace=False)
        D=Data[cells,:,i_n]
        D = D[:,:,None]
        par=['kfold',nchunks,nrandsampl,l,dims*dims,constraint,rank,dt]
        statistics, MT, R=fit_dynamical_system(D,dims,False,par)
        # print(statistics)
        X_PC = R[2]
        r0 = X_PC[:,0]
        J = MT.T+identity(dims)
        JS = (J+J.T)/2.
        # print(max(eigvals(J).real)-1, max(eigvals(JS)))
        U,S,VT = svd(J)

        U = U[:,:rank]
        VT = VT[:rank,:]
        S = S[:rank]

        B = dot(VT,U)

        r0=r0/norm(r0)
        alphas = dot(VT,r0)
        betas = dot(U.T,r0)
        alpha_beta_S = hstack( ( alphas[:,None],betas[:,None],S[:, None]) )

        overlaps[:,:,i_sub,i_n] = B
        initial_condition[i_sub,i_n] = dot(alphas**2, S**2 )+2*sum(prod( alpha_beta_S, axis = 1 ))
        initial_condition_along_v[i_sub,i_n] = sqrt(sum(dot(VT,r0)**2))

        r0=random.normal(0,1,dims)
        r0=r0/norm(r0)
        alphas = dot(VT,r0)
        betas = dot(U.T,r0)
        alpha_beta_S = hstack( ( alphas[:,None],betas[:,None],S[:, None]) )

        initial_condition_rand[i_sub,i_n] = dot(alphas**2, S**2 )+2*sum(prod( alpha_beta_S, axis = 1 ))
        initial_condition_along_v_rand[i_sub,i_n] = sqrt(sum(dot(VT,r0)**2))


########### plot elements of V.T dot U - overlaps between u's and v's  #############

colors = ['#ed876b', 'w', '#85aded']
cm = LinearSegmentedColormap.from_list('new_cm', colors, N=100)

fig,ax=subplots(1,figsize=(2.4,2.4))
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
m = overlaps[:,:,0,0]
c=ax.imshow(m,cmap=cm,clim=(-1,1))
fig.colorbar(c,ticks=[-1,0,1],fraction=0.045).set_label('$\mathbf{u}^{(i)}-\mathbf{v}^{(j)}$ overlap')
ylabel('Right connectivity vector $\mathbf{v}^{(i)}$')
xlabel('Left connectivity vector $\mathbf{u}^{(j)}$')
rect1 = patches.Rectangle((-.5,-.5),2,2,linewidth=.7,edgecolor='k',facecolor='none')
rect2 = patches.Rectangle((1.5,1.5),2,2,linewidth=.7,edgecolor='k',facecolor='none')
rect3 = patches.Rectangle((3.5,3.5),2,2,linewidth=.7,edgecolor='k',facecolor='none')
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
xticks([])
yticks([])
tight_layout()

########### plot histogram elements of V.T dot U ##############
mask_high_overlap = zeros((rank,rank))
for i in range(rank):
    if i%2==0:
        mask_high_overlap[i,i+1]=1
    else:
        mask_high_overlap[i,i-1]=1
mask_high_overlap=mask_high_overlap.astype(bool)

fig,ax=subplots(figsize=(1.5,2.))
hist(abs(overlaps).ravel(),40,normed=1, color='#92b8f0')
xlim([0,1])
ylim([0,8])
ytop=ax.get_ylim()[1]
m_low = mean(abs(overlaps)[~mask_high_overlap,:,:].ravel())
m_high = mean(abs(overlaps)[mask_high_overlap,:,:].ravel())
plot([m_low],[5],'v',markersize=4,color='#476480')
plot([m_high],[5],'v', markersize=4,color='#476480')
xlabel('Overlap magnitude')
ylabel('Number of pairs (norm.)')
tight_layout()

###############  weighted component of r0 along v's #############
m = mean(sqrt(initial_condition),0)
m_rand= mean(sqrt(initial_condition_rand),0)

fig=figure(figsize=(1.4,1.7))
ax = fig.add_subplot(1,1,1)
ax.spines['bottom'].set_visible(False)

col='#f79040'
flierprops = dict(marker='o',markerfacecolor=col, markersize=4,markeredgewidth=.3,
                  linestyle='none')
whiskerprops=dict(linewidth=1.,color=col)
boxprops=dict(linewidth=1.,color=col)
capprops=dict(linewidth=1.,color=col)
bp=ax.boxplot([m], flierprops=flierprops,\
              widths=.3, patch_artist=True, whiskerprops=whiskerprops,\
              boxprops=boxprops,capprops=capprops)# width=0.5)
for box in bp['boxes']:
    box.set(facecolor='w')
for median in bp['medians']:
    median.set(color=col, linewidth=.8)
plot([.5,1.5], sqrt((e**2-1))*ones(2), '--', lw=1.2,color='#aab4bd')
ylabel('Weighted component of $\mathbf{r}_0$ \n along $\mathbf{v}$\'s')
xticks([])
yticks([0,5,10,15])
tight_layout()


############# component of r0 along v's ###############
m = mean(sqrt(initial_condition_along_v),0)
m_rand = mean(sqrt(initial_condition_along_v_rand),0)


fig=figure(figsize=(1.4,1.7))
ax = fig.add_subplot(1,1,1)
ax.spines['bottom'].set_visible(False)

col='#0bb1ba'
flierprops = dict(marker='o',markerfacecolor=col, markersize=4,markeredgewidth=.3,
                  linestyle='none')
whiskerprops=dict(linewidth=1.,color=col)
boxprops=dict(linewidth=1.,color=col)
capprops=dict(linewidth=1.,color=col)
bp=ax.boxplot([m], flierprops=flierprops,\
              widths=.35, patch_artist=True, whiskerprops=whiskerprops,\
              boxprops=boxprops,capprops=capprops)# width=0.5)

for box in bp['boxes']:
    box.set(facecolor='w', )
for median in bp['medians']:
    median.set(color=col, linewidth=.8)

col='#bdbdbd'
flierprops = dict(marker='o',markerfacecolor=col, markersize=4,markeredgewidth=.3,
                  linestyle='none')
whiskerprops=dict(linewidth=1.,color=col)
boxprops=dict(linewidth=1.,color=col)
capprops=dict(linewidth=1.,color=col)
bp=ax.boxplot([m_rand], flierprops=flierprops,\
              widths=.35, patch_artist=True, whiskerprops=whiskerprops,\
              boxprops=boxprops,capprops=capprops)# width=0.5)

for box in bp['boxes']:
    box.set(facecolor='w', )
for median in bp['medians']:
    median.set(color=col, linewidth=.8)


ylabel('Component of $\mathbf{r}_0$ along $\mathbf{v}$\'s')
xticks([])
# ylim([0,1])
yticks([0,.25,.5,.75,1])
tight_layout()
