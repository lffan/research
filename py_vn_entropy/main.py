
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
import laser, entropy_utils

# get_ipython().magic('matplotlib inline')
# get_ipython().magic('reload_ext autoreload')
# get_ipython().magic('autoreload 1')
# get_ipython().magic('aimport laser, entropy_utils')


# In[2]:

# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats('pdf', 'png')


# In[3]:

G = 0.001
KAPPA = 0.001
NBAR = 20

N_max = 100
n_list = np.arange(N_max)
s_op = squeeze(N_max, 1)
vac = fock(N_max, 0)
init_psi = ket2dm(s_op * vac) # initial cavity state## Average Photon Number $\bar{n} = 200$


# In[4]:

entropy_vn(init_psi), entropy_vn(s_op * vac)


# In[5]:

expect(create(N_max) * destroy(N_max), init_psi), expect(create(N_max) * destroy(N_max), s_op * vac)


# The entropy estimated by the paper under the apprximation of large average photon numbers

# In[6]:

ENTR_APPROX = 0.5 * np.log(NBAR) + 0.5 * np.log(2 * np.pi)
print('ENTROPY APPROXIMATION: {:.4f}'.format(ENTR_APPROX))


# The entropy calculated given on the photon statistics of a **coherent state**

# In[7]:

pns_cohe = [poisson.pmf(n, NBAR) for n in n_list]
ENTR_COHE = - sum([pn * np.log(pn) for pn in pns_cohe if pn > 0])
print('ENTROPY COHERENT: {:.4f}'.format(ENTR_COHE))


# In[8]:

# fig, ax = plt.subplots(figsize=(12, 4))
# ax.bar(n_list, pns_cohe, width=0.5)
# # ax.set_xlim(100, 300)
# ax.set_title('Possion Distribution with an Average of 200', fontsize=14);


# The entropy calculated given on the photon statistics a **thermal state**

# In[9]:

ratio = 20/21.0
pns_thml = laser.boltzmann(ratio, N_max + 1000)
ENTR_THML = - sum([pn * np.log(pn) for pn in pns_thml if pn > 0])
print('ENTROPY THERMAL: {:.4f}'.format(ENTR_THML))


# In[10]:

# fig, ax = plt.subplots(figsize=(12, 4))
# ax.bar(n_list, pns_thml[:N_max], width=0.5)
# ax.set_title('Thermal Distribution with an Average of 200', fontsize=14);


# In[11]:

# sum(i * pns_thml[i] for i in n_list)


# ### Small A/C Ratios

# In[12]:

ratios1 = (1.05, 1.2, 1.4, 2, 8)
t_list1 = np.linspace(0, 20000, 11)
l1, n1, entr1 = entropy_utils.entropy_vs_ratio(ratios1, t_list1, G, KAPPA, NBAR, N_max, init_psi, solver='rho')


# In[13]:

n1_df = pd.DataFrame(n1, columns=n1.keys() )
entr1_df = pd.DataFrame(entr1, columns=entr1.keys())

n1_df.to_csv('./data/sv_n1_df_test.csv', index=False)
entr1_df.to_csv('./data/sv_entropy1_df_test.csv', index=False)
# np.savez('./data/sv_l1.npz', lasers=l1)


# In[30]:

# n1_df = pd.read_csv('./data/sv_n1_df.csv')
# entropy_utils.df_plot(n1_df, xlim=(-2, 42), ylim=(-2, 25), \ 
#     style = ['-', '-.', ':', '--'], \
#     xlabel=r'$gt$', ylabel=r'$\bar{n}$ (mean photon number)')
# plt.title(r'Evolution of $\bar{n}$ under Different $A/C$', fontsize=14);


# In[31]:

# entr1_df = pd.read_csv('./data/sv_entropy1_df.csv')
# entropy_utils.df_plot(entr1_df, xlim=(-2, 42), ylim=(-0.5, 4.5), \ 
#     xlabel=r'$gt$', ylabel=r'$S$ (entropy)', \
#     style = ['-', '-.', ':', '--'], entr_cohe=ENTR_COHE, entr_thml=False)
# plt.title(r'Evolution of $S$ under Different $A/C$', fontsize=14);


# ### Large Ratios

# In[16]:

# ratios2 = (1.6, 2, 4, 8)
# t_list2 = np.linspace(0, 200, 11)
# l2, n2, entr2 = entropy_utils.entropy_vs_ratio(ratios2, t_list2, G, KAPPA, NBAR, N_max, init_psi, solver='rho')


# # In[17]:

# n2_df = pd.DataFrame(n2, columns=sorted(n2.keys()))
# entr2_df = pd.DataFrame(entr2, columns=sorted(entr2.keys()))

# n2_df.to_csv('./data/sv_n2_df.csv', index=False)
# entr2_df.to_csv('./data/sv_entropy2_df.csv', index=False)
# np.savez('./data/sv_l2.npz', lasers=l2)


# In[32]:

# n2_df = pd.read_csv('./data/sv_n3_df.csv')
# entropy_utils.df_plot(n2_df, xlim=(-1, 10), ylim=(-2, 22), \ 
#     style = ['-', '-.', ':', '--'], \ 
#     xlabel=r'$gt$', ylabel=r'$\bar{n}$ (mean photon number)')
# plt.title(r'Evolution of $\bar{n}$ under Different $A/C$', fontsize=14);


# In[33]:

# entr2_df = pd.read_csv('./data/sv_entropy3_df.csv')
# entropy_utils.df_plot(entr2_df, xlim=(-2, 22), ylim=(0, 3.8), \ 
#     xlabel=r'$gt$', ylabel=r'$S$ (entropy)', \ 
#     style = ['-', '-.', ':', '--'], entr_cohe=False)
# plt.title(r'Evolution of $S$ under Different $A/C$', fontsize=14);


# ### Extremely Large Ratios

# In[20]:

# ratios3 = (16, 64, 256, 1042)
# t_list3 = np.linspace(0, 100, 11)
# l3, n3, entr3 = entropy_utils.entropy_vs_ratio(ratios3, t_list3, G, KAPPA, NBAR, N_max, init_psi, solver='rho')


# In[21]:

# n3_df = pd.DataFrame(n3, columns=n3.keys() )
# entr3_df = pd.DataFrame(entr3, columns=entr3.keys())

# n3_df.to_csv('./data/sv_n3_df.csv', index=False)
# entr3_df.to_csv('./data/sv_entropy3_df.csv', index=False)
# np.savez('./data/sv_l3.npz', lasers=l3)
