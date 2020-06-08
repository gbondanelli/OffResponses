from myimports import*

path    = './data'
Data   = load(path+'/off_responses_trialavg.npy')
ncells  = Data.shape[0]
nstim   = Data.shape[2]

nresampling = 2000
frac = 0.5
corr = empty((nresampling,nstim))
for res in range(nresampling):
    resample = random.choice(range(ncells),int(frac*ncells),replace=False)
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
figure(figsize = (2.3,1.8))
plot([0,17],[0,0],'--',lw=1,color='grey')
col1 = '#D32E4A'
col1_fill = '#fac0ca'
col2 = '#629BC1'
col2_fill = '#cceaff'

M = m[:4]
S = s[:4]
fill_between(arange(1,5),M-S,M+S,color=col1_fill)
plot(arange(1,5),M,'o-',lw=1.,color=col1,markersize=2.5)

M = m[4:8]
S = s[4:8]
fill_between(arange(5,9),M-S,M+S,color=col2_fill)
plot(arange(5,9),M,'o-',lw=1.,color=col2,markersize=2.5)

M = m[8:12]
S = s[8:12]
fill_between(arange(9,13),M-S,M+S,color=col1_fill)
plot(arange(9,13),M,'o-',lw=1.,color=col1,markersize=2.5)

M = m[12:16]
S = s[12:16]
fill_between(arange(13,17),M-S,M+S,color=col2_fill)
plot(arange(13,17),M,'o-',lw=1.,color=col2,markersize=2.5)

plot([1,16],[1/e,1/e],'--',color='grey', lw=.7)
plot([1,16],[-1/e,-1/e],'--',color='grey', lw=.7)

for i in range(16):
    p = sum(abs(corr[:,i])>1/e)/float(nresampling)
    if p<0.05:
        plot([i+1],[.95],'x', markersize=2, color='#fc5203')

ylim(-1,1)
xlim(0.5,16.5)
xticks([1,4,8,12,16])
yticks([-1,0,1])
xlabel('Stimuli')
ylabel('Initial state - peak\n correlation')
tight_layout()
