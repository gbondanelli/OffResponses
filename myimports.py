from numpy import *
from numpy import random
from random import sample
from numpy.linalg import eigh,norm,inv,svd,multi_dot,qr,cond
from scipy.linalg import eig,eigvals,svdvals,expm,block_diag
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from scipy.io import loadmat
import scipy
from scipy.stats import pearsonr,ttest_ind
import timeit, time
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from scipy.io import loadmat,savemat

from modules.DimRedTools import*
from modules.StatTools import *
from modules.rc_parameters import*
from modules.dynamicsTools import*
from modules.select_neurons import*
from modules.fit_basis_functions import*
from modules.rc_parameters import*
from modules.plottingtools import*
set_graphic_par()

from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
