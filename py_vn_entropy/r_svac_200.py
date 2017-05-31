
# coding: utf-8

# **Entropy of Laser with Fixed Average Photon Numbers**
# 
# - author: Longfei Fan
# - created: 05/10/2017
# - modified: 05/17/2017

# **Abstract**
# 
# In this note, I study how the entropy of laser changes with respect to time 
# given on differnet A/C values and fixed avergage photon numbers.

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scipy.stats import poisson

from qutip import *
import laser, entropy_utils_svac

# In[3]:

G = 0.001
KAPPA = 0.001

NBAR = 20
N_max = 100
n_list = np.arange(N_max)


# vacuum
vacu = fock(N_max, 0)

# squeezed vacuum
s = 1
s_op = squeeze(N_max, s)
svac = s_op * vacu

# thermal state
# n_thml = 1
# thml = thermal_dm(N_max, n_thml)


init_psi = svac
solver = 'rho'


# In[4]:

print('Initial entropy: {:f}'.format(entropy_vn(init_psi)))


# In[5]:

print('Initial average photon numbers: {:f}'.format(expect(create(N_max) * destroy(N_max), init_psi)))


# The entropy calculated given on the photon statistics of a **coherent state**

# In[7]:

pns_cohe = [poisson.pmf(n, NBAR) for n in n_list]
ENTR_COHE = - sum([pn * np.log(pn) for pn in pns_cohe if pn > 0])
print('ENTROPY COHERENT: {:.4f}'.format(ENTR_COHE))

# The entropy calculated given on the photon statistics a **thermal state**

# In[9]:

ratio = 20/21.0
pns_thml = laser.boltzmann(ratio, N_max + 1000)
ENTR_THML = - sum([pn * np.log(pn) for pn in pns_thml if pn > 0])
print('ENTROPY THERMAL: {:.4f}\n'.format(ENTR_THML))


# ### Small A/C Ratios --------------------------------------------------------

# In[12]:

ratios1 = (1.05, 1.1, 1.2)
t_list1 = np.linspace(0, 10000, 101)
l1, n1, entr1 = entropy_utils_svac.entropy_vs_ratio( \
    ratios1, t_list1, G, KAPPA, NBAR, N_max, init_psi, solver)


# In[13]:

n1_df = pd.DataFrame(n1, columns=n1.keys() )
entr1_df = pd.DataFrame(entr1, columns=entr1.keys())

n1_df.to_csv('./data/200_svac_n1_df.csv', index=False)
entr1_df.to_csv('./data/200_svac_entr1_df.csv', index=False)
np.savez('./data/200_svac_l1.npz', lasers=l1)


# ### Medium Ratios -----------------------------------------------------------

# In[16]:

ratios2 = (1.6, 2, 4)
t_list2 = np.linspace(0, 4000, 101)
l2, n2, entr2 = entropy_utils_svac.entropy_vs_ratio( \
    ratios2, t_list2, G, KAPPA, NBAR, N_max, init_psi, solver)


# # In[17]:

n2_df = pd.DataFrame(n2, columns=sorted(n2.keys()))
entr2_df = pd.DataFrame(entr2, columns=sorted(entr2.keys()))

n2_df.to_csv('./data/200_svac_n2_df.csv', index=False)
entr2_df.to_csv('./data/200_svac_entr2_df.csv', index=False)
np.savez('./data/200_svac_l2.npz', lasers=l2)


# ### Large Ratios ------------------------------------------------------------

# In[20]:

ratios3 = (8, 16, 64, 256)
t_list3 = np.linspace(0, 2000, 101)
l3, n3, entr3 = entropy_utils_svac.entropy_vs_ratio( \
    ratios3, t_list3, G, KAPPA, NBAR, N_max, init_psi, solver)


# # In[21]:

n3_df = pd.DataFrame(n3, columns=n3.keys() )
entr3_df = pd.DataFrame(entr3, columns=entr3.keys())

n3_df.to_csv('./data/200_svac_n3_df.csv', index=False)
entr3_df.to_csv('./data/200_svac_entr3_df.csv', index=False)
np.savez('./data/200_svac_l3.npz', lasers=l3)
