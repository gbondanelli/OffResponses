from myimports import*

path = './data'
OrigData = load(path+'/off_responses_single_trials.npy')
infos = load(path+'/infos.npy')
ncells = OrigData.shape[0]
nsteps = OrigData.shape[1]
nstim = OrigData.shape[2]
dt = 108*0.031731/300.

i_stim = 5
i_session = 0
OrigData = OrigData[:,:32,i_stim,:]
nsteps = OrigData.shape[1]
ntrials = OrigData.shape[2]

Davg = mean(OrigData[:,:,:],2)
a = argmax(norm(Davg,axis=0))

idx_session = where(infos[:,2]==i_session+1)[0]
ncells_session = len(idx_session)
Data = OrigData[idx_session,:,:]

r0 = Davg[:,a][idx_session]
r0 = r0/norm(r0)
r_rand = random.normal(0,1,len(idx_session))
r_rand = r_rand/norm(r_rand)

VA_ampl = empty((ntrials,nsteps))
VA_rand = empty((ntrials, nsteps))

for i_trial in range(ntrials):
    selection_trials = delete(arange(ntrials), i_trial)
    Data_temp = Data[:,:,selection_trials]
    for i_step in range(nsteps):
        VA_ampl[i_trial,i_step] = var( dot( r0,Data_temp[:,i_step,:]  ) )
        VA_rand[i_trial,i_step] = var( dot( r_rand,Data_temp[:,i_step,:]  ) )

figure(figsize=(2.3,1.8))
m_ampl = mean(VA_ampl,0)
s_ampl = std(VA_ampl,0)
m_rand = mean(VA_rand,0)
s_rand = std(VA_rand,0)
t =  dt*arange(nsteps)

fill_between(t,m_ampl-s_ampl,m_ampl+s_ampl, color='#cae2fa')
plot(t,m_ampl, lw = 1.5, color = '#77caf7', label='Amplified dir.')

fill_between(t,m_rand-s_rand,m_rand+s_rand, color='#cae2fa')
plot(t,m_rand, '--', lw = 1.8, color ='#77caf7' , label='Random dir.')

plot([0.05,0.05],[0,max(m_ampl)],'--', lw=.7, color='#636363')
plot([a*dt,a*dt],[0,max(m_ampl)],'--', lw=1, color='#636363')
xticks([0.05,.15,.25,.35],['$0$','$0.1$','$0.2$','$0.3$'])
ylabel('Variability')
xlabel('Time(s)')
tight_layout()

##

