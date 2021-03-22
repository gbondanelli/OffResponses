from myimports import *

colors = ['#C64343', '#FFA0A0', 'w', '#9AB5D3', '#6397D3']
cmm = LinearSegmentedColormap.from_list(
    'new_cm', colors[::-1], N=100)


parameters = {'T': 4, 'nsteps': 2000, 'N': 1000, 'n_trajectories': 2, 'connectivity': 'rotational',
              'noise': 0.0, 'g': 0, 'rank': 20, 'Delta': 5, 'initial_conditions': 'amplified', 'tau_sigma': 0.,
              'Delta_rot': [1, 7], 'tau': 1
              }

Data0, w1, w2 = simulated_data(parameters)
dt = parameters['T'] / float(parameters['nsteps'])

## Figure 6C
A1 = Data0[:, :, 0]
A2 = Data0[:, :, 1]
# %% Plot trajectories on PC1 PC2
d1, V1 = PCA(A1[:, :300])
d2, V2 = PCA(A2[:, :300])
col1 = '#F24378'
col2 = '#1A88AD'
figure(figsize=(1.8, 1.8))
ax = subplot(111)
gca().set_aspect('equal', adjustable='box')
ax.spines['left'].set_bounds(-.5, .5)
ax.spines['bottom'].set_bounds(-.5, .5)
ax.spines['bottom'].set_position(('axes', -0.1))
ax.spines['left'].set_position(('axes', -0.))

plot(dot(V1[:, 0].T, A1), dot(V1[:, 1].T, A1), lw=1.3, c=col1)
plot(dot(V1[:, 0].T, A2), dot(V1[:, 1].T, A2), lw=1.5, c=col2)

vx = dot(-w1[:, 0], V1[:, 0]);
vy = dot(-w1[:, 0], V1[:, 1])
ux = dot(w2[:, 0], V1[:, 0]);
uy = dot(w2[:, 0], V1[:, 1])
plot(1 * array([0, vx]), 1 * array([0, vy]), '--', lw=1, c='#F46831')
plot(1 * array([0, ux]), 1 * array([0, uy]), lw=1, c='#F46831')

xticks([-.5, 0, .5])
yticks([-.5, 0, .5])
xlabel(r'PC1 (stim 1)')
ylabel(r'PC2 (stim 1)')
tight_layout()
show()

figure(figsize=(1.8, 1.8))
ax = subplot(111)
gca().set_aspect('equal', adjustable='box')
ax.spines['left'].set_bounds(-.5, .5)
ax.spines['bottom'].set_bounds(-.5, .5)
ax.spines['bottom'].set_position(('axes', -0.1))
ax.spines['left'].set_position(('axes', -0.))

plot(dot(V2[:, 0].T, A1[:, :]), dot(V2[:, 1].T, A1[:, :]), lw=1.5, c=col1)
plot(dot(V2[:, 0].T, A2[:, :]), dot(V2[:, 1].T, A2[:, :]), lw=1.3, c=col2)

vx = dot(-w1[:, 1], V2[:, 0]);
vy = dot(-w1[:, 1], V2[:, 1])
ux = dot(w2[:, 1], V2[:, 0]);
uy = dot(w2[:, 1], V2[:, 1])
plot(1 * array([0, vx]), 1 * array([0, vy]), '--', lw=1, c='#F46831')
plot(1 * array([0, ux]), 1 * array([0, uy]), lw=1, c='#F46831')

xticks([-.5, 0, .5])
yticks([-.5, 0, .5])
xlabel(r'PC1 (stim 2)')
ylabel(r'PC2 (stim 2)')
tight_layout()
show()

## Figure 6A

n = 30
ncommon = 30
i0 = where(amax(A1, axis=1) > 0)[0]
i1 = amax(A1[i0, :], axis=1) - A1[i0, 0]
i2 = amax(A2[i0, :], axis=1) - A2[i0, 0]
i3 = multiply(i1, i2)
index_common = argsort(i3)[::-1][:ncommon]
other = delete(arange(parameters['N']), index_common)
index = random.permutation(hstack((index_common, other[:n - ncommon])))

n2 = 200
t2 = dt * arange(-n2, parameters['nsteps'])
C1 = empty((n, parameters['nsteps'] + n2));
C2 = empty((n, parameters['nsteps'] + n2))

for i in range(n):
    v1 = hstack((A1[i, 0] * ones(n2), A1[i, :]))
    v2 = hstack((A2[i, 0] * ones(n2), A2[i, :]))
    C1[i, :] = v1;
    C2[i, :] = v2

centers = [1, 30]
dx, = np.diff(centers) / (C1.shape[0] - 1)
extent = [t2[0], t2[-1], centers[0] - dx / 2, centers[1] + dx / 2]

figure(figsize=(2., 1.55))
imshow(C1, cmap=cmm, aspect='auto', extent=extent, clim=(-.1, .1))
colorbar(ticks=[-.1, 0, .1]).set_label(r'Firing rate (a.u.)')
plot([0, 0], [.5, 30.5], c='k', lw=0.7, ls='--')
xlabel('Time (a.u.)')
ylabel('Neurons')
xticks([0, 2, 4])
yticks([10, 20, 30])
tight_layout()
show()

figure(figsize=(2., 1.55))
imshow(C2, cmap=cmm, aspect='auto', extent=extent, clim=(-.1, .1))
colorbar(ticks=[-.1, 0, .1]).set_label(r'Firing rate (a.u.)')
plot([0, 0], [.5, 30.5], c='k', lw=0.7, ls='--')
xlabel('Time (a.u.)')
ylabel('Neurons')
xticks([0, 2, 4])
yticks([10, 20, 30])
tight_layout()
show()

## Figure 6D
for neuron in range(10):
    figure(figsize=(2.3, 1.75))
    Min = min(amin(C1[neuron, :]), amin(C2[neuron, :]))
    Max = max(amax(C1[neuron, :]), amax(C2[neuron, :]))
    plot(t2, C1[neuron, :], lw=1.5, color=col1, label='Stim 1')
    plot(t2, C2[neuron, :], lw=1.5, color=col2, label='Stim 2')
    xlabel('Time (a.u.)')
    ylabel('Firing rate (a.u.)')
    xlim(dt * array([-n2, parameters['nsteps']]))
    tight_layout()

## Figure 6B
n2 = 200
nsteps = parameters['nsteps']
N = parameters['N']
C1 = empty((N, nsteps + n2));
C2 = empty((N, nsteps + n2))
t2 = dt * arange(-n2, nsteps)
for i in range(N):
    v1 = hstack((A1[i, 0] * ones(n2), A1[i, :]))
    v2 = hstack((A2[i, 0] * ones(n2), A2[i, :]))
    C1[i, :] = v1;
    C2[i, :] = v2

figure(figsize=(2, 1.5))
plot(t2, norm(C1, axis=0), color=col1, lw=1.5)
a = max(norm(C1, axis=0))
# xlim(0,400)
xlabel('Time (a.u.)')
ylabel('Distance from \n baseline')
# xticks([70,270,470],['$0$','$200$','$400$'])
plot([0, 0], [0, a], c='grey', lw=1, ls='--')
xlim(dt * array([-n2, nsteps]))
tight_layout()

figure(figsize=(2, 1.5))
plot(t2, norm(C2, axis=0), color=col2, lw=1.5)
a = max(norm(C2, axis=0))
# xlim(0,400)
xlabel('Time (a.u.)')
ylabel('Distance from \n baseline')
# xticks([70,270,470],['$0$','$200$','$400$'])
plot(array([0, 0]), [0, a], c='grey', lw=1, ls='--')
xlim(dt * array([-n2, nsteps]))
tight_layout()

## Figure 6E

parameters = {'T': 4, 'nsteps': 2000, 'N': 1000, 'n_trajectories': 5, 'connectivity': 'rotational',
              'noise': 0.0, 'g': 0, 'rank': 20, 'Delta': 5, 'initial_conditions': 'amplified', 'tau_sigma': 0.,
              'Delta_rot': [1, 7], 'tau': 1
              }

Data0, w1, w2 = simulated_data(parameters)
dt = parameters['T'] / float(parameters['nsteps'])
ncells = Data0.shape[0]
nstim = Data0.shape[2]

nsubsamplings = 100
f = .5
corr_ic_peak = empty((nsubsamplings, nstim))
for i_stim in range(nstim):
    Data = Data0[:, :, i_stim]
    for i_sub in range(nsubsamplings):
        cells = random.choice(arange(ncells), int(f * ncells), replace=False)
        D = Data[cells, :]
        ic = D[:, 0]
        ic = ic / norm(ic)
        Norm = norm(D, axis=0)
        peak = D[:, argmax(Norm)]
        peak = peak / norm(peak)
        corr_ic_peak[i_sub, i_stim] = ic @ peak


fig, ax = subplots(figsize=(2.4, 1.8))
m = mean(corr_ic_peak, 0)
s = std(corr_ic_peak, 0)
errorbar(arange(1, 1 + nstim), m - s, m + s, fmt='.-', lw=1.2, color='#2a2b2b', markersize=5)
plot(arange(1, 1 + nstim), zeros(nstim), '--', lw=1, color='grey')
ylim([-1, 1])
xticks([1, 2, 3, 4, 5])
xlabel('Stimulus')
ylabel('Initial state - peak\n correlation')
tight_layout()
##

