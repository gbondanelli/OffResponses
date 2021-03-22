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
overlaps = load('./data/subspace_overlaps.npy')

theta2 = theta[~np.eye(theta.shape[0], dtype=bool)].reshape(theta.shape[0], -1).ravel()
overlaps2 = overlaps[~np.eye(overlaps.shape[0], dtype=bool)].reshape(overlaps.shape[0], -1).ravel()

y = theta2
x = overlaps2

N = 5000
P = pearsonr(x, y)[0]
P_shuffled = empty(N)
for i in range(N):
    y_shuffled = y[random.permutation(y.shape[0])]
    P_shuffled[i] = abs(pearsonr(x, y_shuffled)[0])
P_pear = sum(P_shuffled > P) / float(N)
print(P)

##
fig, ax = subplots(figsize=(1.85, 1.85))
xlabel('Subspace overlap')
ylabel('Connectivity overlap')

X = linspace(0, .6, 100)
# plot(X,X)

x1 = x.reshape((-1, 1))
model = LinearRegression()
model.fit(x1, y)
r2 = model.score(x1, y)
b = model.intercept_
slope = model.coef_

plot(X, slope * X + b, '-', lw=.9, color='#abbef7')
scatter(x, y, s=20, color='#ff8c9f', lw=.5, edgecolor='#cce4eb')
xlim(-.04, .6)
xticks([0, .2, .4, .6])
yticks([0, .1, .2, .3, .4])
ylim([0.0, .4])
ax.spines['bottom'].set_bounds(0, 0.6)
ax.spines['left'].set_bounds(0, .4)
tight_layout()

print(pearsonr(theta2, overlaps2)[0])

##

