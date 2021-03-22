from myimports import*

path = './data'
Data = load(path+'/off_responses_trialavg.npy')

DT = 0.031731
total_time = 108*DT
time_stim_presentation = 33*DT
nsteps = 1000
dt = total_time/nsteps
DT = 0.031731
total_time = 108 * DT
time_stim_presentation = 33 * DT
nsteps = 1000
dt = total_time / nsteps

## %%  fit
stim = [1, 5]

dims = 100
nchunks = 20
nrandsampl = 20
cv = False
n_free_params = dims * dims
constraint1 = 'none'
l = 1
rank = 'none'
n_free_params1 = int(dims * dims)
par1 = ['random', nchunks, nrandsampl, l, n_free_params1, constraint1, rank, dt]

ntrials = 100
col = ['#F14377', '#569FD1']

for m in range(2):
    i = stim[m]
    figure(figsize=(2, 2))

    X = expand_dims(Data[:, :, i], axis=2)
    # d,_=PCA(X)
    # dims = where(cumsum(d/sum(d))>0.99)[0][0]
    # print(dims)
    rM, MT, E = fit_dynamical_system(X, dims, cv, par1)

    X_PC1 = E[2]
    X_dot = E[3]
    X_dotP = E[4]
    X_PC2 = E[5]

    X_PC_P = empty((dims, ntrials, X_PC1.shape[1]))
    X_PC_P[:, :, 0] = random.multivariate_normal(X_PC2[:, 0], .0005 * identity(dims), ntrials).T
    for j in range(X_PC1.shape[1] - 1):
        X_PC_P[:, :, j + 1] = dot(expm(dt * (j + 1) * MT.T), X_PC_P[:, :, 0])

    #    ax=subplot(4,4,i+1)
    ax = subplot(111)
    gca().set_aspect('equal', adjustable='box')
    ax.spines['left'].set_bounds(-.5, .5)
    ax.spines['bottom'].set_bounds(-.5, .5)
    ax.spines['bottom'].set_position(('axes', -0.1))
    ax.spines['left'].set_position(('axes', -0.))

    d1, V1 = PCA(X_PC2)
    V1[:, 1] = -V1[:, 1]

    for i_trial in range(ntrials):
        ln, = ax.plot(dot(V1[:, 0], X_PC_P[:, i_trial, :]), dot(V1[:, 1], X_PC_P[:, i_trial, :]), lw=2, color='#D1D9DC')
        ln.set_solid_capstyle('round')
    X_PC_Pmean = mean(X_PC_P, axis=1)
    plot(dot(V1[:, 0], X_PC_Pmean[:, :]), dot(V1[:, 1], X_PC_Pmean[:, :]), '--', lw=1, color='#5C7C99')

    #    plot(dot(V1[:,0],X_PC2),dot(V1[:,1],X_PC2),lw=1.5,color='#EA1744'*(i<8)+'#6BAB90'*(i>=8))
    #    plot(dot(V1[:,0],X_PC2[:,0]),dot(V1[:,1],X_PC2[:,0]),'.',markersize=7,color='#EA1744'*(i<8)+'#6BAB90'*(i>=8))
    plot(dot(V1[:, 0], X_PC2), dot(V1[:, 1], X_PC2), lw=1.5, color=col[m])
    plot(dot(V1[:, 0], X_PC2[:, 0]), dot(V1[:, 1], X_PC2[:, 0]), '.', markersize=7, color=col[m])

    xticks([-.5, 0, .5])
    yticks([-.5, 0, .5])
    #    title('Stimulus %s'%(i+1))
    if m == 0:
        xlabel('PC1 - OFF (8kHz)')
        ylabel('PC2 - OFF (8kHz)')
    if m == 1:
        xlabel('PC1 - OFF (WN)')
        ylabel('PC2 - OFF (WN)')

    tight_layout()
##

