from myimports import*

colors = ['#FFFFFF', '#FFBABA', '#F75656', '#7C0D0D']
cm = LinearSegmentedColormap.from_list(
        'new_cm', colors, N=100)

path    = './data'
Data   = load(path+'/off_responses_trialavg.npy')

ncells  = Data.shape[0]
nstim   = Data.shape[2]
dt = 108*0.031731/1000

nchunks = 10;
nrandsampl = 10
l = 5;
cv = False;
cv_type = 'kfold'

dims = 150;
rank = 6
nPC = rank;
n_free_params = dims * dims

LEFT = empty((dims, nPC * nstim))
RIGHT = empty((dims, nPC * nstim))
SINGVALS = empty(nPC * nstim)

JTOT = 0

for stim in range(nstim):
    print(stim)
    par = [cv_type, nchunks, nrandsampl, l, n_free_params, 'none', rank, dt, stim]
    r, M, E = fit_dynamical_system(Data, dims, cv, par)
    X_PC = E[2]
    J = M + identity(dims)

    L, S, RT = svd(J)
    LEFT[:, stim * nPC:(stim + 1) * nPC] = L[:, :nPC]
    RIGHT[:, stim * nPC:(stim + 1) * nPC] = RT.T[:, :nPC]
    SINGVALS[stim * nPC:(stim + 1) * nPC] = S[:nPC]

theta = empty((nstim, nstim))
n = nPC
for i in range(nstim):
    for j in range(nstim):
        Q1 = LEFT[:, i * n:(i + 1) * n]
        Q2 = LEFT[:, j * n:(j + 1) * n]
        s = svdvals(dot(Q1.T, Q2))
        theta[i, j] = s[0]

##
fig, ax = subplots(1, figsize=(1.9, 1.9))
c = ax.imshow(theta, cmap=cm, clim=(0, 1))
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
xticks([])
yticks([])
fig.colorbar(c, ticks=[0, .5, 1], fraction=0.046, pad=0.04).set_label('Connectivity overlap')
tight_layout()




##

