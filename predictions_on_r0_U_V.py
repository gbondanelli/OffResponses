from myimports import*

path = './data'
Data = load(path+'/off_responses_trialavg.npy')
ncells      = Data.shape[0]
nsteps = Data.shape[1]
nstim = Data.shape[2]
dt = 108*0.031731/1000.

stimuli=arange(nstim)

percentage = .8

l = 5
constraint = 'none'
nrandsampl = 1
nchunks = 1
nsubsamplings = 20
dims = 100
rank = 6;  # rank='none'
overlaps = empty((rank, rank, nsubsamplings, nstim))
initial_condition_along_v2 = empty((nsubsamplings, nstim))
initial_condition_along_v2_rand = empty((nsubsamplings, nstim))
diffD = empty((nsubsamplings, int(rank / 2), nstim))
indeces_v2 = empty(int(rank / 2))

for i_n in range(nstim):
    print(i_n)
    for i_sub in range(nsubsamplings):
        f = int(percentage * ncells)
        cells = random.choice(range(ncells), f, replace=False)
        D = Data[cells, :, i_n]
        D = D[:, :, None]
        par = ['kfold', nchunks, nrandsampl, l, dims * dims, constraint, rank, dt]
        statistics, MT, R = fit_dynamical_system(D, dims, False, par)

        X_PC = R[2]
        tstar = dt * argmax(norm(X_PC, axis=0))

        r0 = X_PC[:, 0]
        J  = MT.T + identity(dims)
        JS = (J + J.T) / 2.
        U, S, VT = svd(J)

        U   = U[:, :rank]
        VT  = VT[:rank, :]
        S   = S[:rank]

        for i_R in range(int(rank / 2)):
            indeces_v2[i_R] = int(2 * i_R) + argmax([S[i_R], S[i_R + 1]])
        V2T = VT[indeces_v2.astype(int), :]

        diffD[i_sub, :, i_n] = diff(S)[[0, 2, 4]]
        B = dot(VT, U)
        r0      = r0 / norm(r0)
        overlaps[:, :, i_sub, i_n] = B
        initial_condition_along_v2[i_sub, i_n] = sqrt(sum(dot(V2T, r0) ** 2))

        r0 = random.normal(0, 1, dims)
        r0 = r0 / norm(r0)
        initial_condition_along_v2_rand[i_sub, i_n] = sqrt(sum(dot(V2T, r0) ** 2))

## Figure 5B left panel

colors = ['#ed4276', 'w', '#85aded']
cm = LinearSegmentedColormap.from_list('new_cm', colors, N=100)

fig, ax = subplots(1, figsize=(2.4, 2.4))
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
m = overlaps[:, :, 0, 0]
c = ax.imshow(m, cmap=cm, clim=(-1, 1))
fig.colorbar(c, ticks=[-1, 0, 1], fraction=0.045).set_label('$\mathbf{u}^{(i)}-\mathbf{v}^{(j)}$ overlap')
ylabel('Right connectivity vector $\mathbf{v}^{(i)}$')
xlabel('Left connectivity vector $\mathbf{u}^{(j)}$')
rect1 = patches.Rectangle((-.5, -.5), 2, 2, linewidth=.7, edgecolor='k', facecolor='none')
rect2 = patches.Rectangle((1.5, 1.5), 2, 2, linewidth=.7, edgecolor='k', facecolor='none')
rect3 = patches.Rectangle((3.5, 3.5), 2, 2, linewidth=.7, edgecolor='k', facecolor='none')
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
xticks([])
yticks([])
tight_layout()

## Figure 5B right panel

mask_high_overlap = zeros((rank, rank))
for i in range(rank):
    if i % 2 == 0:
        mask_high_overlap[i, i + 1] = 1
    else:
        mask_high_overlap[i, i - 1] = 1
mask_high_overlap = mask_high_overlap.astype(bool)

fig, ax = subplots(figsize=(1.5, 2.))
hist(abs(overlaps).ravel(), 40, density=1, color='#439ee8')
xlim([0, 1])
ylim([0, 8])
ytop = ax.get_ylim()[1]
m_low = mean(abs(overlaps)[~mask_high_overlap, :, :].ravel())
m_high = mean(abs(overlaps)[mask_high_overlap, :, :].ravel())
plot([m_low], [5], 'v', markersize=4, color='#476480')
plot([m_high], [5], 'v', markersize=4, color='#476480')
xlabel('Overlap magnitude')
ylabel('Number of pairs (norm.)')
tight_layout()

## Figure 5C right panel

m = mean(initial_condition_along_v2, 0)
m_rand = mean(initial_condition_along_v2_rand, 0)

data = [m, m_rand]
labels = ['Data', 'Random']
col1 = '#0bb1ba';
col2 = '#bdbdbd'
facecolor = [col1, col2]
colorwhisk = 2 * [col1] + 2 * [col2]
colorcaps = colorwhisk
colorfliers = 'w'
ax = my_boxplot((1.6, 2.05), data, labels, 40, facecolor, colorwhisk, colorcaps, colorfliers, .8)
ylabel('Component of $\mathbf{r}_0$ along $\mathbf{v}$\'s')
# ax.spines['left'].set_bounds(-2.,1)
# ax.spines['bottom'].set_bounds(1,3)
yticks([0, .25, .5, .75, 1])
tight_layout()

## Figure 5C left panel

D = amax(abs(diffD) / 2, 1)
m = D

data = [m.ravel()]
labels = ['']
col1 = '#e0466a'
facecolor = [col1]
colorwhisk = 2 * [col1]
colorcaps = colorwhisk
colorfliers = 'w'
ax = my_boxplot((1.3, 2.05), data, labels, 40, facecolor, colorwhisk, colorcaps, colorfliers, .4)
plot([0.5, 1.5], [1, 1], '-', lw=1.5, color='#8c979c')
ylabel(r'$|\,\Delta_2-\Delta_1\,|/2$')
ax.spines['left'].set_bounds(0., 3)
ax.spines['bottom'].set_visible(False)
yticks([0, 1, 2, 3])
xticks([])
tight_layout()

