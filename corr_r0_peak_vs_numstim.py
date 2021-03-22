from myimports import*

path='./data'
Data = load(path+'/off_responses_trialavg.npy')
ncells = Data.shape[0]
nsteps = Data.shape[1]
nstim = Data.shape[2]
dt = 108*0.031731/1000.

# Compute mean and std of ic/peak corr across stimuli

ic_peak_corr = empty(nstim)
for i in range(nstim):
    Activity = Data[:, :, i]
    tpeak = argmax(norm(Activity, axis=0))
    r0 = Activity[:, 0]
    r0 = r0 / norm(r0)
    rpeak = Activity[:, tpeak]
    rpeak = rpeak / norm(rpeak)
    ic_peak_corr[i] = abs(dot(r0, rpeak))

mData = mean(ic_peak_corr)
sData = std(ic_peak_corr)

# %%
nstim_tot = 16
stimuli = range(nstim_tot)
percentage = .4
s = select_most_responsive_neurons(Data, stimuli, percentage)
Data_sel = Data[:, :, stimuli];
Data_sel = Data_sel[s, :, :]

nstim = arange(1, 17)
n = len(nstim)
nsubsamplings = 2

##
nbasis = 10
l = 1

ic_peak_corr_SingleCellModel = empty((nsubsamplings, n))
for i_n in range(n):
    print(i_n)
    for i_sub in range(nsubsamplings):
        stimuli_selection = random.choice(range(nstim_tot), nstim[i_n], replace=False)
        D = Data_sel[:, :, stimuli_selection]
        Y = copy(D)
        Y, R = normalize_by_range(Y)

        f = basis_gaussian(D.shape[1], .1, nbasis)
        X = repeat(expand_dims(f, axis=2), nstim[i_n], axis=2)

        _, _, [Y1, Yrec], _ = LinReg_basis_functions(Y, X, l, dt)
        Yrec = Yrec[:, :D.shape[1]]  # select first stimulus
        Y1 = Y1[:, :D.shape[1]]  # select first stimulus

        Yrec_renorm = normalize_back(Yrec[:, :, None], R[:, 0, None])
        Yrec_renorm = Yrec_renorm[:, :, 0]
        Y1_renorm = normalize_back(Y1[:, :, None], R[:, 0, None])
        Y1_renorm = Y1_renorm[:, :, 0]

        r0 = Yrec_renorm[:, 0]
        r0 = r0 / norm(r0)
        tpeak = argmax(norm(D[:, :, 0], axis=0))
        rpeak = Yrec_renorm[:, tpeak]
        rpeak = rpeak / norm(rpeak)
        ic_peak_corr_SingleCellModel[i_sub, i_n] = abs(dot(r0, rpeak))

##
l = 5
constraint = 'none'
nrandsampl = 1
nchunks = 1
ic_peak_corr_RecurrentModel = empty((nsubsamplings, n))
for i_n in range(n):
    print(i_n)
    for i_sub in range(nsubsamplings):
        stimuli_selection = random.choice(range(nstim_tot), nstim[i_n], replace=False)
        D = Data_sel[:, :, stimuli_selection];
        Y = copy(D)
        d, _ = PCA(D)
        dims = where(cumsum(d / sum(d)) > 0.9)[0][0]
        dims = 100
        rank = 6 * nstim[i_n]
        par = ['kfold', nchunks, nrandsampl, l, dims * dims, constraint, rank, dt]
        statistics, MT, R = fit_dynamical_system(Y, dims, False, par)
        print(statistics)

        time_steps = int(R[2].shape[1] / nstim[i_n])
        X_PC_P = empty((dims, time_steps))
        X_PC_P[:, 0] = R[5][:, 0]
        for j in range(time_steps - 1):
            X_PC_P[:, j + 1] = dot(expm(dt * (j + 1) * MT.T), X_PC_P[:, 0])

        r0 = X_PC_P[:, 0]
        r0 = r0 / norm(r0)
        Y1 = R[2][:, :time_steps]
        tpeak = argmax(norm(D[:, :, 0], axis=0))
        rpeak = X_PC_P[:, tpeak]
        rpeak = rpeak / norm(rpeak)

        ic_peak_corr_RecurrentModel[i_sub, i_n] = abs(dot(r0, rpeak))

## Plot

figure(figsize=(2.2, 2))

fill_between([nstim[0] - .5, nstim[-1] + .5], (mData - sData) * ones(2), (mData + sData) * ones(2), color='#e8e8e8')
plot([nstim[0] - .5, nstim[-1] + .5], (mData) * ones(2), lw=1., color='#819299', label='Data')

m = mean(ic_peak_corr_SingleCellModel, 0)
s = std(ic_peak_corr_SingleCellModel, 0) / sqrt(nsubsamplings)
fill_between(nstim, m - s, m + s, facecolor='#BEF4F9')
plot(nstim, m, '.-', lw=1.3, color='#64ADCE', markersize=5, label='Basis set')

m = mean(ic_peak_corr_RecurrentModel, 0)
s = std(ic_peak_corr_RecurrentModel, 0) / sqrt(nsubsamplings)
fill_between(nstim, m - s, m + s, facecolor='#F4CECD')
plot(nstim, m, '.-', lw=1.3, color='#F45B69', markersize=5, label='Dyn. system')

xlim([nstim[0] - .5, nstim[-1] + .5])
ylim([-0, .6])
xticks([1, 4, 8, 12, 16])
xlabel('Number of stimuli')
ylabel('Initial state - peak\n correlation')
tight_layout()