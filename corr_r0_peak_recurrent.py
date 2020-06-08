from myimports import*

parameters = {
    'T' : 10,
    'nsteps' : 200,
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

Data, J, _ = simulated_data(parameters)
dt = parameters['T']/float(parameters['nsteps'])
N = parameters['N']

nstim = 5
nresampling = 100
frac = 0.5
corr = empty((nresampling,nstim))

for res in range(nresampling):
    resample = random.choice(range(N),int(frac*N),replace=False)
    A2 = Data[resample,:,:]
    for i in range(nstim):
        ic = A2[:,0,i]
        ic = ic/norm(ic)
        t = argmax(norm(Data[:,:,i],axis=0))
        peak = mean(A2[:,t-5:t+5,i],axis=1)
        peak = peak/norm(peak);
        corr[res,i] = ic.dot(peak)

m = mean(corr,0)
s = std(corr,0)

figure(figsize=(2.3,1.8))
plot([0,5.5],[0,0],'--',lw=1,color='grey')
plot([0,5.5],[1/e,1/e],'--',lw=.6,color='grey')
plot([0,5.5],[-1/e,-1/e],'--',lw=.6,color='grey')
errorbar(arange(1,6),m,yerr=s,fmt='o-',lw=1.2,color='#595653',markersize=4,\
         markeredgewidth=.0,markeredgecolor='#3D3B39')

ylim(-1.05,1.05)
xlim(0.5,5.5)
xticks([1,2,3,4,5])
yticks([-1,0,1])
xlabel('Stimulus')
ylabel('Initial state - peak\n correlation')
tight_layout()
