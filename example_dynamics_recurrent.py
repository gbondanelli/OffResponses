from myimports import*

colors = ['#C64343','#FFA0A0','w', '#9AB5D3','#6397D3']
cmm = LinearSegmentedColormap.from_list(
'new_cm', colors[::-1], N=100)

parameters = {
    'T' : 10,
    'nsteps' : 400,
    'N' : 2000,
    'n_trajectories' : 10,
    'connectivity' : 'low_rank',
    'noise' : 0.00,
    'g' : 0.,
    'rank' :30,
    'Delta' : 5,
    'initial_conditions' : 'amplified',
    'tau_sigma' : 0.
    }

Data0, J, [U,V] = simulated_data(parameters)
dt = parameters['T']/float(parameters['nsteps'])

A1 = Data0[:,:,0]
A2 = Data0[:,:,1]

########## plot population responses to the two stimuli on principal components #########

d1,V1 = PCA(A1[:,:150])
d2,V2 = PCA(A2[:,:150])
col1 = '#F24378'
col2 = '#1A88AD'

figure(figsize = (2.2,2.2))
ax = subplot(111)
gca().set_aspect('equal', adjustable = 'box')
ax.spines['left'].set_bounds(-.5, .5)
ax.spines['bottom'].set_bounds(-.5,.5)
ax.spines['bottom'].set_position(('axes', -0.1))
ax.spines['left'].set_position(('axes', -0.))

plot(dot(V1[:,0].T,A1),dot(V1[:,1].T,A1),lw = 1.3,c = col1)
plot(dot(V1[:,0].T,A2),dot(V1[:,1].T,A2),lw = 1.5,c = col2)

vx = dot(V[:,0],V1[:,0]); vy = dot(V[:,0],V1[:,1])
ux = dot(U[:,0],V1[:,0]); uy = dot(U[:,0],V1[:,1])
plot(1*array([0,vx]),1*array([0,vy]),'--',lw = 1,c = '#F46831')
plot(1*array([0,ux]),1*array([0,uy]),lw = 1,c = '#F46831')

xticks([-.5,0,.5])
yticks([-.5,0,.5])
xlabel(r'PC1 (stim 1)')
ylabel(r'PC2 (stim 1)')
tight_layout()
show()

figure(figsize = (2.2,2.2))
ax = subplot(111)
gca().set_aspect('equal', adjustable = 'box')
ax.spines['left'].set_bounds(-.5, .5)
ax.spines['bottom'].set_bounds(-.5,.5)
ax.spines['bottom'].set_position(('axes', -0.1))
ax.spines['left'].set_position(('axes', -0.))

plot(dot(V2[:,0].T,A1[:,:400]),dot(V2[:,1].T,A1[:,:400]),lw=1.5,c=col1)
plot(dot(V2[:,0].T,A2[:,:400]),dot(V2[:,1].T,A2[:,:400]),lw=1.3,c=col2)

vx = dot(V[:,1],V2[:,0]); vy = dot(V[:,1],V2[:,1])
ux = dot(U[:,1],V2[:,0]); uy = dot(U[:,1],V2[:,1])
plot(1*array([0,vx]),1*array([0,vy]),'--',lw=1,c='#F46831')
plot(1*array([0,ux]),1*array([0,uy]),lw=1,c='#F46831')

#legend(['OFF - 8Hz','OFF - WN'], frameon=0)
xticks([-.5,0,.5])
yticks([-.5,0,.5])
xlabel(r'PC1 (stim 2)')
ylabel(r'PC2 (stim 2)')
tight_layout()
show()

############ plot responses of 30 example neurons to the two stimuli ############
n = 30
ncommon = 30
i0 = where(amax(A1,axis = 1)>0)[0]
i1 = amax(A1[i0,:],axis = 1)-A1[i0,0]
i2 = amax(A2[i0,:],axis = 1)-A2[i0,0]
i3 = multiply(i1,i2)
index_common = argsort(i3)[::-1][:ncommon]
other = delete(arange(parameters['N']),index_common)
index = random.permutation(hstack((index_common,other[:n-ncommon])))


n2 = 70
C1 = empty((n,parameters['nsteps']+n2))
C2 = empty((n,parameters['nsteps']+n2))

for i in range(n):
    v1 = hstack((A1[i,0]*ones(n2),A1[i,:]))
    v2 = hstack((A2[i,0]*ones(n2),A2[i,:]))
    C1[i,:] = v1
    C2[i,:] = v2

figure(figsize = (2.,1.55))
imshow(C1,cmap = cmm,aspect = 'auto',clim = (-.1,.1))
colorbar(ticks = [-.1,0,.1]).set_label(r'Firing rate (a.u.)')
plot([n2,n2],[-.5,n-1+0.5],c = 'k',lw = 0.7, ls = '--')
xlabel('Time (a.u.)')
ylabel('Neurons')
xticks([70,270,470],['$0$','$200$','$400$'])
tight_layout()
show()

figure(figsize=(2.,1.55))
imshow(C2,cmap=cmm,aspect='auto',clim=(-.1,.1))
colorbar(ticks=[-.1,0,.1]).set_label(r'Firing rate (a.u.)')
plot([n2,n2],[-.5,n-1+0.5],c='k',lw=0.7, ls='--')
xlabel('Time (a.u.)')
ylabel('Neurons')
xticks([70,270,470],['$0$','$200$','$400$'])
tight_layout()
show()

#########  plot responses of the same neuron to two example stimuli (5 example neurons) ##########
for neuron in range(5):
    figure(figsize=(2.3,1.75))
    Min = min(amin(C1[neuron,:]), amin(C2[neuron,:]))
    Max = max(amax(C1[neuron,:]), amax(C2[neuron,:]))
    plot(C1[neuron,:], lw=1.5, color=col1, label='Stim 1')
    plot(C2[neuron,:], lw=1.5, color=col2, label='Stim 2')
    xlabel('Time (a.u.)')
    ylabel('Firing rate (a.u.)')
    tight_layout()

############## plot population activity norm for the two stimuli #############
nsteps = parameters['nsteps']
N = parameters['N']
C1=empty((N,nsteps+n2)); C2=empty((N,nsteps+n2))
for i in range(N):
    v1=hstack((A1[i,0]*ones(n2),A1[i,:]))
    v2=hstack((A2[i,0]*ones(n2),A2[i,:]))
    C1[i,:]=v1; C2[i,:]=v2

figure(figsize=(2,1.5))
plot(norm(C1,axis=0),color=col1,lw=1.5)
a=max(norm(C1,axis=0))
xlim(0,400)
xlabel('Time (a.u.)')
ylabel('Distance from \n baseline')
xticks([70,270,470],['$0$','$200$','$400$'])
plot([70,70],[0,a],c='grey',lw=1, ls='--')
tight_layout()

figure(figsize=(2,1.5))
plot(norm(C2,axis=0),color=col2,lw=1.5)
a=max(norm(C2,axis=0))
xlim(0,400)
xlabel('Time (a.u.)')
ylabel('Distance from \n baseline')
xticks([70,270,470],['$0$','$200$','$400$'])
plot([70,70],[0,a],c='grey',lw=1, ls='--')
tight_layout()
