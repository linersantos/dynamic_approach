#!/usr/bin/env python
# coding: utf-8

# In[1]:


#from IPython.core.display import display, HTML

#display(HTML("<style>.container { width:100% !important; }</style>"))


# In[2]:


#### Block 1 #### Please refer to this number in your questions
import sys
import os
import pickle
import numpy as np
import seaborn as sns
import pandas as pd
import math
import matplotlib.pyplot as plt

import subprocess
#from multiprocessing import Pool, cpu_count

from sklearn.decomposition import PCA
from numpy.linalg import inv
import sklearn, matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.gaussian_process import kernels as krnl
import scipy.stats as st
from scipy import optimize

#import emcee
import ptemcee
import h5py
from scipy.linalg import lapack
from multiprocessing import Pool, cpu_count
from multiprocessing import cpu_count
import time

from pyDOE import lhs
import emcee

sns.set("notebook")
print('ola\n''este é o codigo')


# In[3]:


#### Block 2 #### Please refer to this number in your questions

name="JETSCAPE_bayes"
#Saved emulator name
EMU='PbPb2760_emulators_scikit.dat'
# Where to save the figures and data files
PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results/FigureFiles"
DATA_ID = "JETSCAPE_bayes/Data/"


# In[4]:


#### Block 3 #### Please refer to this number in your questions

# Define folder structure

if not os.path.exists(PROJECT_ROOT_DIR):
    os.mkdir(PROJECT_ROOT_DIR)

if not os.path.exists(FIGURE_ID):
    os.makedirs(FIGURE_ID)

if not os.path.exists(DATA_ID):
    os.makedirs(DATA_ID)

def image_path(fig_id):
    return os.path.join(FIGURE_ID, fig_id)

def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)

def save_fig(fig_id):
    plt.savefig(image_path(fig_id) + ".png", format='png')

#def save_data(dat_id):
    #file.write(file_path(dat_id)+".dat")


# In[5]:


#### Block 4 #### Please refer to this number in your questions
# Design points
#design = pd.read_csv(filepath_or_buffer=data_path("design"))
design = pd.read_csv(filepath_or_buffer=data_path("design_points_main_PbPb-2760.dat"))
design2 = pd.read_csv(filepath_or_buffer=data_path("design_points_main_PbPb-2760.dat"))


# In[6]:


design.shape


# In[7]:


design2.shape


# In[8]:


design.head()


# In[9]:


#### Block 5 #### Please refer to this number in your questions

#Simulation outputs at the design points
#simulation = pd.read_csv(filepath_or_buffer=data_path("saida-real.txt"))
#simulation2 = pd.read_csv(filepath_or_buffer=data_path("saida-real.txt"))
simulation = pd.read_csv(filepath_or_buffer=data_path("saida-atlas.csv"))
simulation2 = pd.read_csv(filepath_or_buffer=data_path("saida-atlas.csv"))


# In[10]:


simulation.shape


# In[11]:


simulation2.shape


# In[ ]:





# In[12]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
simulation.head()


# In[13]:


#### Block 6 #### Please refer to this number in your questions

X = design.values
X2 = design2.values
Y = simulation.values
Y2 = simulation2.values
for i in range(0,1):
    print(X[i,:])
    #print(X2[i,:])
print( "X.shape : "+ str(X.shape) )
print( "Y.shape : "+ str(Y.shape) )

#Model parameter names in Latex compatble form
model_param_dsgn = ['$N$[$2.76$TeV]',
 '$p$',
 '$\\sigma_k$',
 '$w$ [fm]',
 '$d_{\\mathrm{min}}$ [fm]',
 '$\\tau_R$ [fm/$c$]',
 '$\\alpha$',
 '$T_{\\eta,\\mathrm{kink}}$ [GeV]',
 '$a_{\\eta,\\mathrm{low}}$ [GeV${}^{-1}$]',
 '$a_{\\eta,\\mathrm{high}}$ [GeV${}^{-1}$]',
 '$(\\eta/s)_{\\mathrm{kink}}$',
 '$(\\zeta/s)_{\\max}$',
 '$T_{\\zeta,c}$ [GeV]',
 '$w_{\\zeta}$ [GeV]',
 '$\\lambda_{\\zeta}$',
 '$b_{\\pi}$',
 '$T_{\\mathrm{sw}}$ [GeV]']
#print(X)


# In[14]:


#### Block 7 #### Please refer to this number in your questions
#Scaling the data to be zero mean and unit variance for each observables
SS  =  StandardScaler(copy=True)
#Singular Value Decomposition
u, s, vh = np.linalg.svd(SS.fit_transform(Y), full_matrices=True)
print(f'shape of u {u.shape} shape of s {s.shape} shape of vh {vh.shape}')


# In[15]:


#### Block 8 #### Please refer to this number in your questions

# print the explained raito of variance
# https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(7,4))
#importance = pca_analysis.explained_variance_
importance = np.square(s[:6]/math.sqrt(u.shape[0]-1)) #linha importante: diz qtos pc's serao usados
cumulateive_importance = np.cumsum(importance)/np.sum(importance)
idx = np.arange(1,1+len(importance))
ax1.bar(idx,importance)
ax1.set_xlabel("PC index")
ax1.set_ylabel("Variance")
ax2.bar(idx,cumulateive_importance)
ax2.set_xlabel(r"The first $n$ PC")
ax2.set_ylabel("Fraction of total variance")
#plt.tight_layout(True)
print(cumulateive_importance)


# In[16]:


#### Block 9 #### Please refer to this number in your questions
#whiten and project data to principal component axis (only keeping first 10 PCs)
pc_tf_data=u[:,0:6] * math.sqrt(u.shape[0]-1) #alterar essa linha para o numero de PCs desejado u[:,0:6], se quiser 6 pcs
print(f'Shape of PC transformed data {pc_tf_data.shape}')
#Scale Transformation from PC space to original data space
inverse_tf_matrix= np.diag(s[0:6]) @ vh[0:6,:] * SS.scale_.reshape(1,8)/ math.sqrt(u.shape[0]-1)


# In[17]:


#### Block 11 #### Please refer to this number in your questions


# We will use the first data point in our validation set as the pseudo experimental data

#validation_design = pd.read_csv(filepath_or_buffer=data_path("design"))
validation_design = pd.read_csv(filepath_or_buffer=data_path("design_points_main_PbPb-2760.dat"))
#validation_simulation = pd.read_csv(filepath_or_buffer=data_path("saida-real.txt"))
#validation_simulation_sd = pd.read_csv(filepath_or_buffer=data_path("erro-real.txt"))
validation_simulation = pd.read_csv(filepath_or_buffer=data_path("saida-atlas.csv"))
validation_simulation_sd = pd.read_csv(filepath_or_buffer=data_path("erro-atlas.csv"))


# In[18]:


print(f'Validation data design points have the shape {validation_design.shape} and validation simulation outputs have the shape {validation_simulation.shape}')


# In[19]:


validation_design.head()


# In[20]:


pd.set_option('display.max_columns', None)
validation_simulation.head()


# In[21]:


pd.set_option('display.max_columns', None)
validation_simulation_sd.head()


# In[22]:


#### Block 12 #### Please refer to this number in your questions

#This is how you can load the actual experimental data instead of pseudo experimental data
#experiment=pd.read_csv(filepath_or_buffer="JETSCAPE_bayes/Data/PbPb2760_experiment",index_col=0)
experiment=pd.read_csv(filepath_or_buffer="JETSCAPE_bayes/Data/PbPb2760_experiment",index_col=0)
experiment.head()


# In[23]:


#### Block 13 #### Please refer to this number in your questions


y_exp= validation_simulation.values[0,:]
y_exp_variance= np.square(validation_simulation_sd.values[0,:])
print(f'Shape of the experiment observables {y_exp.shape} and shape of the experimental error variance{y_exp_variance.shape}')


# In[24]:


#### Block 10 #### Please refer to this number in your questions

# Bounds for parametrs in the emulator are same as prior ranges so
#prior_df = pd.read_csv(filepath_or_buffer=data_path("PbPb2760_prior"), index_col=0)
prior_df = pd.read_csv(filepath_or_buffer=data_path("PbPb2760_prior"), index_col=0)


# In[25]:


prior_df.head()


# In[26]:


prior_df.shape


# In[27]:


design_max=prior_df.loc['max'].values
design_min=prior_df.loc['min'].values
print(design_min,design_max)


# In[28]:


#### Block 19 #### Please refer to this number in your questions

# If false, uses pre-trained emulators.
# If true, retrain emulators.
train_emulators = True
import time
design=X
input_dim=len(design_max)
print(input_dim)
ptp = design_max - design_min
bound=zip(design_min,design_max)
if (os.path.exists(data_path(EMU))) and (train_emulators==False):
    print('Saved emulators exists and overide is prohibited')
    with open(data_path(EMU),"rb") as f:
        Emulators=pickle.load(f)
else:
    Emulators=[]
    for i in range(0,6):
        start_time = time.time()
        kernel=1*krnl.RBF(length_scale=ptp,length_scale_bounds=np.outer(ptp, (4e-1, 1e2)))+ krnl.WhiteKernel(noise_level=.1, noise_level_bounds=(1e-2, 1e2))
        GPR=gpr(kernel=kernel,n_restarts_optimizer=4,alpha=0.0000000001)
        GPR.fit(design,pc_tf_data[:,i].reshape(-1,1))
        print(f'GPR score is {GPR.score(design,pc_tf_data[:,i])} \n')
        print(f'GPR log_marginal likelihood {GPR.log_marginal_likelihood()} \n')
        print("--- %s seconds ---" % (time.time() - start_time))
        Emulators.append(GPR)

if (train_emulators==True) or not(os.path.exists(data_path(EMU))):
    with open(data_path(EMU),"wb") as f:
        pickle.dump(Emulators,f)
print(X)


# In[29]:


#### Block 20 #### Please refer to this number in your questions


def predict_observables(model_parameters):
    """Predicts the observables for any model parameter value using the trained emulators.

    Parameters
    ----------
    Theta_input : Model parameter values. Should be an 1D array of 17 model parametrs.

    Return
    ----------
    Mean value and full error covaraiance matrix of the prediction is returened. """

    mean=[]
    variance=[]
    theta=np.array(model_parameters).flatten()

    if len(theta)!=17:
        raise TypeError('The input model_parameters array does not have the right dimensions')
    else:
        theta=np.array(theta).reshape(1,17) #17
        for i in range(0,6):
            mn,std=Emulators[i].predict(theta,return_std=True)
            mean.append(mn)
            variance.append(std**2)
    mean=np.array(mean).reshape(1,-1)
    inverse_transformed_mean=mean @ inverse_tf_matrix + np.array(SS.mean_).reshape(1,-1)
    variance_matrix=np.diag(np.array(variance).flatten())
    A_p=inverse_tf_matrix
    inverse_transformed_variance=np.einsum('ik,kl,lj-> ij', A_p.T, variance_matrix, A_p, optimize=False)
    return inverse_transformed_mean, inverse_transformed_variance



#### Block 21 #### Please refer to this number in your questions
#Este bloco é um teste para verificar se os parametros do modelo estão dentro d>
def log_prior(model_parameters):
    """
    Uniform Prior. Evaluvate prior for model.

    Parameters
    ----------
    model_parameters : 17 dimensional list of floats

    Return
    ----------
    unnormalized probability : float

    If all parameters are inside bounds function will return 0 otherwise -inf"""
    X = np.array(model_parameters).reshape(1,-1)
    lower = np.all(X >= design_min)
    upper = np.all(X <= design_max)
    if (lower and upper):
        lp=0
    else:
        lp = -np.inf
    return lp
print('log-prior')


#### Block 22 #### Please refer to this number in your questions
def mvn_loglike(y, cov):
    """
    Evaluate the multivariate-normal log-likelihood for difference vector `y`
    and covariance matrix `cov`:

        log_p = -1/2*[(y^T).(C^-1).y + log(det(C))] + const.

    The likelihood is NOT NORMALIZED, since this does not affect MCMC.  The
    normalization const = -n/2*log(2*pi), where n is the dimensionality.

    Arguments `y` and `cov` MUST be np.arrays with dtype == float64 and shapes
    (n) and (n, n), respectively.  These requirements are NOT CHECKED.

    The calculation follows algorithm 2.1 in Rasmussen and Williams (Gaussian
    Processes for Machine Learning).

    """
    # Compute the Cholesky decomposition of the covariance.
    # Use bare LAPACK function to avoid scipy.linalg wrapper overhead.
    L, info = lapack.dpotrf(cov, clean=False)

    if info < 0:
        raise ValueError(
            'lapack dpotrf error: '
            'the {}-th argument had an illegal value'.format(-info)
        )
    elif info < 0:
        raise np.linalg.LinAlgError(
            'lapack dpotrf error: '
            'the leading minor of order {} is not positive definite'
            .format(info)
        )

    # Solve for alpha = cov^-1.y using the Cholesky decomp.
    alpha, info = lapack.dpotrs(L, y)

    if info != 0:
        raise ValueError(
            'lapack dpotrs error: '
            'the {}-th argument had an illegal value'.format(-info)
        )
  #  print(L.diagonal())
    a=np.ones(len(L.diagonal()))*1e-10
    #print(a)
    #print(L)
   # L=L+np.diag(a)
    if np.all(L.diagonal()>0):
        return -.5*np.dot(y, alpha) - np.log(L.diagonal()).sum()
    else:
        print(L.diagonal())
        raise ValueError(
            'L has negative values on diagonal {}'.format(L.diagonal())
        )
print('mvn-log-like-block22')


#### Block 24 #### Please refer to this number in your questions
# Covariance truncation error from keeping subset of PC is not included
def log_like(model_parameters):
    """
        Parameters
    ----------
    model_parameters : 17 dimensional list of floats

    Return
    ----------
    unnormalized probability : float

    """
    mn,var=predict_observables(model_parameters)
    delta_y=mn-y_exp.reshape(1,-1)
    delta_y=delta_y.flatten()

    exp_var=np.diag(y_exp_variance)

    total_var=var + exp_var
    #only_diagonal=np.diag(total_var.diagonal())
    return mvn_loglike(delta_y,total_var)
print('log-like-block24')

#### Block 23 #### Please refer to this number in your questions
# Covariance truncation error from keeping subset of PC is not included
def log_posterior(model_parameters):
    """
        Parameters
    ----------
    model_parameters : 17 dimensional list of floats

    Return
    ----------
    unnormalized probability : float
    """

    mn,var=predict_observables(model_parameters)
    delta_y=mn-y_exp.reshape(1,-1)
    delta_y=delta_y.flatten()

    exp_var=np.diag(y_exp_variance)

    total_var=var + exp_var
    #only_diagonal=np.diag(total_var.diagonal())
    return log_prior(model_parameters) + mvn_loglike(delta_y,total_var)
print('log-posterior-block23')

#### Block 23 #### Please refer to this number in your questions


v2 = 0.0236
v3 = 0.0253
v4 = 0.0136
v5 = 0.0050
sigma2 = 0.00118
sigma3 = 0.00101
sigma4 = 0.00068
sigma5 = 0.00060

v2min = 1
v3min = 1
v4min = 1
v5min = 1
eps2min = 1
eps3min = 1
eps4min = 1
chi2min = 10000
do_mcmc = False
ntemps = 20
Tmax = np.inf

Cmin = 0
Cmax = 100
nwalkers = 100 #1000000 #2000000 #3*Xdim  # number of MCMC walkers
ndim = 17
Xdim = 17
nburnin = 500 #500 # The number of steps it takes for the walkers to thermalize
niterations= 1000 #1000 # The number of samples to draw once thermalized
nthin = 10 # Record every nthin-th iteration
nthreads = 3 # Easy parallelization!

nburn = 500 # "burn-in" period to let chains stabilize
nsteps = 2000  # number of MCMC steps to take
filename = data_path(name+".h5")
nu = 4
print(nwalkers)
#ref_arquivo = open("/home/liner/doutorado/visc/0/0/energia.dat","r")
min_theta = [1.625] # Lower bound for initializing walkers
max_theta = [24.79] # Upper bound for initializing walkers



if do_mcmc==True:

    starting_guesses = design_min + (design_max - design_min) * np.random.rand(nwalkers, Xdim)
    cte_vec = Cmin + (Cmax - Cmin)*np.random.rand(nwalkers,1)
    pe = np.random.rand(nwalkers,1)



    for i in range(0,nwalkers):
        #ref_arquivo = open("media.dat","r")
        ref_arquivo = open("energia.dat","r")
        chi2 = 0
        #print(starting_guesses[i,:])
        #print(starting_guesses.shape)
        media,var = predict_observables(starting_guesses[i,:])
        C = float(cte_vec[i,:])
        p = float(pe[i,:])
        num2 = 0
        den2 = 0
        num3 = 0
        den3 = 0
        num4 = 0
        den4 = 0
        num5 = 0
        den5 = 0
        for linha in ref_arquivo:
            valores = linha.split()
            #print(valores[0], valores[1], valores[2] )
            x = float(valores[0])
            y = float(valores[1])
            e = float(valores[2])
            k2 = pow(e,2*p)
            num2 += pow(pow(x,2)+pow(y,2),2)*pow(e,2*p)*pow(0.2016,2)
            den2 += pow(pow(x,2)+pow(y,2),1)*e*pow(0.2016,2)
            num3 += pow(pow(x,2)+pow(y,2),3)*pow(e,2*p)*pow(0.2016,2)
            den3 += pow(pow(x,2)+pow(y,2),1.5)*e*pow(0.2016,2)
            num4 += pow(pow(x,2)+pow(y,2),4)*pow(e,2*p)*pow(0.2016,2)
            den4 += pow(pow(x,2)+pow(y,2),2)*e*pow(0.2016,2)
            num5 += pow(pow(x,2)+pow(y,2),5)*pow(e,2*p)*pow(0.2016,2)
            den5 += pow(pow(x,2)+pow(y,2),2.5)*e*pow(0.2016,2)
        #print(den2)
        r2 = np.sqrt(num2)/den2
        r3 = np.sqrt(num3)/den3
        r4 = np.sqrt(num4)/den4
        r5 = np.sqrt(num5)/den5
        #print(e)
        ref_arquivo.close()
        eps2 = C*r2
        eps3 = C*r3
        eps4 = C*r4
        eps5 = C*r5
        k2 = media.flatten()[4,]
        k3 = media.flatten()[5,]
        k4 = media.flatten()[6,]
        k5 = media.flatten()[7,]

        v2calc = k2*eps2
        v3calc = k3*eps3
        v4calc = k4*eps4
        v5calc = k5*eps5

        #chi2 = (pow((v2calc - v2),2)/pow(sigma2,2)+pow((v3calc - v3),2)/pow(sigma3,2)        +pow((v4calc - v4),2)/pow(sigma4,2)+pow((v5calc - v5),2)/pow(sigma5,2))/nu
        #print(p,C,v2calc,chi2)
        vark2 = var.flatten()[4,]
        vark3 = var.flatten()[5,]
        vark4 = var.flatten()[6,]
        vark5 = var.flatten()[7,]
        sigma_k2 = np.sqrt(np.abs(vark2))
        sigma_k3 = np.sqrt(np.abs(vark3))
        sigma_k4 = np.sqrt(np.abs(vark4))
        sigma_k5 = np.sqrt(np.abs(vark5))

        sigma_v2 = v2calc*(sigma_k2/k2)
        sigma_v3 = v3calc*(sigma_k3/k3)
        sigma_v4 = v4calc*(sigma_k4/k4)
        sigma_v5 = v5calc*(sigma_k5/k5)

        #chi2 = (pow((v2calc - v2),2)/(pow(sigma2,2)) #+pow(sigma_v2,2))+(pow((v3calc - v3),2)/(pow(sigma3,2)+pow(sigma_v3,2))+(pow((v4calc - v4),2)/(pow(sigma4,2)+pow(sigma_v4,2))+(pow((v5calc - v5),2)/(pow(sigma5,2)+pow(sigma_v5,2)))/nu
        chi2 = (pow((v2calc - v2),2)/(pow(sigma2,2)+pow(sigma_v2,2)) + pow((v3calc - v3),2)/(pow(sigma3,2)+pow(sigma_v3,2)) + pow((v4calc - v4),2)/(pow(sigma4,2)+pow(sigma_v4,2)) + pow((v5calc - v5),2)/(pow(sigma5,2)+pow(sigma_v5,2)))/nu
        #var5 = var.flatten()[3,]
        al = float(starting_guesses[i,8:9])
        ah = float(starting_guesses[i,9:10])

        if (chi2 < chi2min): #and (v2 - sigma2) < v2calc and v2calc < (v2 + sigma2) and (v3 - sigma3) < v3calc and v3calc < (v3 + sigma3) \
        #and (v4 - sigma4) < v4calc and v4calc < (v4 + sigma4) and (v5 - sigma5) < v5calc and v5calc < (v5 + sigma5)): #and al < -0.01 and al > -1 and ah > 0.1 and ah < 1):]
            print('nwalk= ',i)
            chi2min = chi2
            pmin = p
            cmin = C
            v2min = v2calc
            v3min = v3calc
            v4min = v4calc
            v5min = v5calc
            eps2min = eps2
            eps3min = eps3
            eps4min = eps4
            eps5min = eps5
            k2min = k2
            k3min = k3
            k4min = k4
            k5min = k5
            stdk2 = np.sqrt(np.abs(vark2))
            stdk3 = np.sqrt(np.abs(vark3))
            stdk4 = np.sqrt(np.abs(vark4))
            stdk5 = np.sqrt(np.abs(vark5))
            stdv2 = v2min*stdk2/k2min
            stdv3 = v3min*stdk3/k3min
            stdv4 = v4min*stdk4/k4min
            stdv5 = v5min*stdk5/k5min
            indice = i
            teta = float(starting_guesses[i,7:8])
            alow = float(starting_guesses[i,8:9])
            ahigh = float(starting_guesses[i,9:10])
            etakink = float(starting_guesses[i,10:11])
            zetamax = float(starting_guesses[i,11:12])
            tzeta = float(starting_guesses[i,12:13])
            wzeta = float(starting_guesses[i,13:14])
            lambdazeta = float(starting_guesses[i,14:15])
            if(v3min > v2min):
               print('v2,v3,v4,v5= ',v2min,v3min,v4min,v5min,k2min,k3min,k4min,k5min,eps2min,eps3min,eps4min,eps5min,chi2min)
               print('v2,v3,v4,v5(exp)= ',v2,sigma2,v3,sigma3,v4,sigma4,v5,sigma5)
               print(' teta = ',teta,'\n','alow = ',alow, '\n','ahigh = ',ahigh,'\n','etakink = ',etakink,'\n','zetamax = ',zetamax,'\n','tzeta = ',tzeta,'\n','wzeta = ',wzeta,'\n','lambdazeta = ',lambdazeta)
    arq = open('har-pl-atlas.dat','a')
    print('n & $v_n^{exp}$ & $\sigma_{v_n}$ & $v_n^{calc}$ & $\sigma_{v_n}^{calc} & $k_n$ & $\sigma_{k_n}$ & $\epsilon_n$',file=arq)
    print(2, v2, sigma2, "%.4f" %v2min, "%.4f" %stdv2, "%.4f" %k2min, "%.4f" %stdk2, "%.4f" %eps2min,file=arq)
    print(3, v3, sigma3, "%.4f" %v3min, "%.4f" %stdv3, "%.4f" %k3min, "%.4f" %stdk3, "%.4f" %eps3min,file=arq)
    print(4, v4, sigma4, "%.4f" %v4min, "%.4f" %stdv4, "%.4f" %k4min, "%.4f" %stdk4, "%.4f" %eps4min,file=arq)
    print(5, v5, sigma5, "%.4f" %v5min, "%.4f" %stdv5, "%.4f" %k5min, "%.4f" %stdk5, "%.4f" %eps5min,file=arq)

    print("c = %.4f" %cmin,"p = %.4f" %pmin,"chi^2 = %.4f" %chi2min,indice,file=arq)
    print('Coeficientes das viscosidades:',file=arq)
    print(' teta = ',teta,'\n','alow = ',alow, '\n','ahigh = ',ahigh,'\n','etakink = ',etakink,'\n','zetamax = ',zetamax,'\n','tzeta = ',tzeta,'\n','wzeta = ',wzeta,'\n','lambdazeta = ',lambdazeta,file=arq)
    #print( "%.4f" %stdv2, "%.4f" %stdv3, "%.4f" %stdv4,"%.4f" %stdk2, "%.4f" %stdk3, "%.4f" %stdk4,file=arq)
    print('parameters which minimizes the chi^2',file=arq)
    print(starting_guesses[indice,:],file=arq)
    arq.close()




#### Teste para ver quais devem ser os melhores valores de C e p se K_n mudarem em X% dentro. E.g. se K_n' = 0.8*K_n,
### quais devem ser os valores otimos de C e p e quanto eles diferem dos valores originais.
v2 = 0.0236
v3 = 0.0253
v4 = 0.0136
v5 = 0.0050
sigma2 = 0.00118
sigma3 = 0.00101
sigma4 = 0.00068
sigma5 = 0.00060

v2min = 1
v3min = 1
v4min = 1
v5min = 1
chi2min = 10000
do_mcmc = True
Xdim = 17
Cmin = 3.00 #4.61
Cmax = 8.00 #4.65
pmin = 0.1
pmax = 0.2
nwalkers = 10000 #3*Xdim  # number of MCMC walkers
nburn = 500 # "burn-in" period to let chains stabilize
nsteps = 2000  # number of MCMC steps to take
filename = data_path(name+".h5")
nu = 4
print(nwalkers)
#ref_arquivo = open("/home/liner/doutorado/visc/0/0/energia.dat","r")
arquivo = open('resultado.dat','a')
'''
minimum = [1.24239778e+01 , 3.35188736e-01,  1.20526818e+00,  5.53561449e-01,
  2.82726929e+00,  8.69175196e-01, -1.91641212e-01,  1.94580825e-01,
 -8.14329067e-01,  1.95633137e-01,  8.27711674e-02,  1.11008172e-02,
  2.20498300e-01,  4.12015012e-02,  3.01388129e-01,  6.74816032e+00,
  1.30040234e-01]
'''

minimum = [14.95143572,  0.66654217,  1.21787487,  0.58664701,  2.68084598,  1.78618626,
 -0.08235095,  0.16025858, -0.26421382, -0.41939633,  0.02269186,  0.19373546,
  0.27743034,  0.14912181, -0.08674251,  4.81665725,  0.13052684]

#minimum2 = [[14.95143572  0.66654217  1.21787487  0.58664701  2.68084598  1.78618626]]
print('min2',np.array([minimum]))

if do_mcmc==True:

    #starting_guesses = design_min + (design_max - design_min) * np.random.rand(nwalkers, Xdim)
    cte_vec = Cmin + (Cmax - Cmin)*np.random.rand(nwalkers,1)
    pe = pmin + (pmax - pmin)*np.random.rand(nwalkers,1)

    for fator_k in range(10,15):
        #cte_vec = Cmin + (Cmax - Cmin)*np.random.rand(nwalkers,1)
        #pe = pmin + (pmax - pmin)*np.random.rand(nwalkers,1)
        print(fator_k/10)
        chi2min = 10000
        print(chi2min)
        if(fator_k >= 13):
          fator_k = (fator_k-5)

        for i in range(0,nwalkers):
            ref_arquivo = open("energia.dat","r")
            chi2 = 0
        #print(starting_guesses[i,:])
        #print(starting_guesses.shape)
        #media,var = predict_observables(starting_guesses[i,:])
            media,var = predict_observables(minimum)
            C = float(cte_vec[i,:])
            p = float(pe[i,:])
        #C = 4.6252*0.85
        #p = 0.1495*0.85 #ci^2 = 0.2979

        #c = 2.1345 p = 0.5675 chi^2 = 0.2399 -> sem correcao da incerteza
        #c = 4.6252 p = 0.1495 chi^2 = 0.2979 -> com correcao da incerteza
        #c = 2.1345 p = 0.5675

            num2 = 0
            den2 = 0
            num3 = 0
            den3 = 0
            num4 = 0
            den4 = 0
            num5 = 0
            den5 = 0
            for linha in ref_arquivo:
                valores = linha.split()
            #print(valores[0], valores[1], valores[2] )
                x = float(valores[0])
                y = float(valores[1])
                e = float(valores[2])
                kappa2 = pow(e,2*p)
                num2 += pow(pow(x,2)+pow(y,2),2)*pow(e,2*p)*pow(0.2016,2)
                den2 += pow(pow(x,2)+pow(y,2),1)*e*pow(0.2016,2)
                num3 += pow(pow(x,2)+pow(y,2),3)*pow(e,2*p)*pow(0.2016,2)
                den3 += pow(pow(x,2)+pow(y,2),1.5)*e*pow(0.2016,2)
                num4 += pow(pow(x,2)+pow(y,2),4)*pow(e,2*p)*pow(0.2016,2)
                den4 += pow(pow(x,2)+pow(y,2),2)*e*pow(0.2016,2)
                num5 += pow(pow(x,2)+pow(y,2),5)*pow(e,2*p)*pow(0.2016,2)
                den5 += pow(pow(x,2)+pow(y,2),2.5)*e*pow(0.2016,2)
        #print(den2)
            r2 = np.sqrt(num2)/den2
            r3 = np.sqrt(num3)/den3
            r4 = np.sqrt(num4)/den4
            r5 = np.sqrt(num5)/den5
        #print(e)
            ref_arquivo.close()
            eps2 = C*r2
            eps3 = C*r3
            eps4 = C*r4
            eps5 = C*r5
            k2 = media.flatten()[4,]*(fator_k/10)
            k3 = media.flatten()[5,]*(fator_k/10)
            k4 = media.flatten()[6,]*(fator_k/10)
            k5 = media.flatten()[7,]*(fator_k/10)
        #print('eps3= ',eps3)
            v2calc = k2*eps2
            v3calc = k3*eps3
            v4calc = k4*eps4
            v5calc = k5*eps5
        #print(k2)
            chi2 = (pow((v2calc - v2),2)/pow(sigma2,2)+pow((v3calc - v3),2)/pow(sigma3,2)\
            +pow((v4calc - v4),2)/pow(sigma4,2)+pow((v5calc - v5),2)/pow(sigma5,2))/nu
        #print(p,C,v2calc,chi2)
            var2 = var.flatten()[4,]
            var3 = var.flatten()[5,]
            var4 = var.flatten()[6,]
            var5 = var.flatten()[7,]
        #var5 = var.flatten()[3,]
            sigma_v2 = v2calc*np.sqrt(np.abs(var2))/k2
            sigma_v3 = v3calc*np.sqrt(np.abs(var3))/k3
            sigma_v4 = v4calc*np.sqrt(np.abs(var4))/k4
            sigma_v5 = v5calc*np.sqrt(np.abs(var5))/k5

            chi2_modificado = (pow((v2calc - v2),2)/(pow(sigma2,2)+pow(sigma_v2,2))+pow((v3calc - v3),2)/(pow(sigma3,2)\
            +pow(sigma_v3,2))+pow((v4calc - v4),2)/(pow(sigma4,2)+pow(sigma_v4,2))\
            +pow((v5calc - v5),2)/(pow(sigma5,2)+pow(sigma_v5,2)))

        #print('chi2_sem = ',chi2, 'chi2_com= ',chi2_modificado)

            #alow = starting_guesses[i,8:9]
            if (chi2_modificado < chi2min):
                if(fator_k == 10):
                  p1 = p
                  c1 = C
                chi2min = chi2_modificado
                i2 = i
                pmin = p
                cmin = C
                v2min = v2calc
                v3min = v3calc
                v4min = v4calc
                v5min = v5calc
                eps2min = eps2
                k2min = k2
                k3min = k3
                k4min = k4
                k5min = k5
                std2min = v2min*np.sqrt(np.abs(var2))/k2min
                std3min = v3min*np.sqrt(np.abs(var3))/k3min
                std4min = v4min*np.sqrt(np.abs(var4))/k4min
                std5min = v5min*np.sqrt(np.abs(var5))/k5min
                #alowmin = float(starting_guesses[i,15:16])#comeca em 0. O primeiro numero indica o valor a ser exivido
                eps2min = eps2
                indice = i
                 #c = 4.6252 p = 0.1495 chi^2 = 0.2979 -> com correcao da incerteza
        erro_relativo_p = (pmin - p1)/p1
        erro_relativo_C = (cmin - c1)/c1
        print(k2min,k3min,k4min,k5min,file=arquivo)
        print('',k2min, v2min,'\n',k3min, v3min,'\n',k4min, v4min,'\n',k5min, v5min,file=arquivo)
        print('p=',pmin,'c= ',cmin,'erro_p= ',erro_relativo_p,'erro_C= ',erro_relativo_C,fator_k/10,i2,file=arquivo)
        print('chi2_com= ',chi2min,file=arquivo)
        if(fator_k<10):
          fator_k = fator_k+5
