from myimports import*

colors = ['#FFFFFF', '#FFBABA', '#F75656', '#7C0D0D']
cm = LinearSegmentedColormap.from_list(
        'new_cm', colors, N=100)
path    = './data'
Data    = load(path+'/off_responses_trialavg.npy')
ncells  = Data.shape[0]
nstim   = Data.shape[2]
dt = 108*0.031731/1000

nchunks = 10
nrandsampl = 10
l = 2.
cv = True
cv_type = 'kfold'
dims = 150
rank = 70
n_free_params = dims*dims

par    = [cv_type,nchunks,nrandsampl,l,n_free_params,'none',rank,dt]
R2_J,_,_,R2_JTOT, R2_sh = fit_dynamical_system(Data,dims,cv,par)


data = [R2_J, R2_JTOT, R2_sh]
labels = ['Full','Sum', 'Shuffle']
col1 = '#439dd9'; col2 = '#f0657c'; col3 = '#c9c9c9'
facecolor = [col1, col2, col3]
colorwhisk = 2*[col1]+2*[col2]+2*[col3]
colorcaps = colorwhisk
colorfliers = 'w'
ax=my_boxplot((1.9,2.05), data, labels, 40, facecolor, colorwhisk, colorcaps, colorfliers, .8)
plot([0.5,3.5],[0,0],lw=0.5,color='grey')
ylabel('Goodness of fit')
ax.spines['left'].set_bounds(-2.,1)
ax.spines['bottom'].set_bounds(1,3)
yticks([-2,-1,0,1])
tight_layout()
##

