import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from qutip import *
import laser

def entropy_vs_ratio(ratios, t_list, g, kappa, nbar, N_max, init_psi, solver='pn'):
    """ simulate lasers with different A/C ratios
    """
    def get_para(ratio, nbar, kappa, g):
        """ calculate parameters given on ratio, nbar, kappa, and g
        """
        gamma = np.sqrt(nbar / (alpha - 1)) * 2 * g
        ra = 2 * kappa * nbar * alpha / (alpha - 1)
        return {'g': g, 'gamma': gamma, 'C': kappa, 'ra': ra,
                'A': 2 * ra * g**2 / gamma**2, 'B': 8 * ra * g**4 / gamma**4}
    
    step = round(len(t_list) / 100)
    n_dict = {'gt': t_list[::step] * g}
    entr_dict = {'gt': t_list[::step] * g}
    l_dict = {}

    for alpha in ratios:
        paras = get_para(alpha, nbar, kappa, g)
        g, ra, gamma, kappa = paras['g'], paras['ra'], paras['gamma'], paras['C']
        print('ratio: {:>5.2f}, ra: {:3.4f}, A: {:.3e}, C: {:.3e}, B: {:.3e}\n'. \
              format(alpha, ra, paras['A'], kappa, paras['B']))
        l = laser.LaserOneMode(g, ra, gamma, kappa)
        if solver == 'pn':
            l.pn_evolve(init_psi, N_max, t_list)
        elif solver == 'rho':
            l.rho_evolve(init_psi, N_max, t_list)
        
        key = '{:.2f}'.format(alpha)
        # l_dict[key] = l
        n_dict[key] = l.get_ns()
        entr_dict[key] = l.get_entrs()

    return l_dict, n_dict, entr_dict


def df_plot(df, xlim, ylim, xlabel, ylabel, style, \
            entr_cohe=False, entr_thml=False):
    """ df plot
    """
    df.plot(x='gt', xlim=xlim, ylim=ylim, style=style,
             figsize=(6, 4), fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    if entr_thml:
        plt.axhline(y=entr_thml, color='black', linewidth=0.5, \
                    linestyle='--', label='thermal')
    if entr_cohe:
        plt.axhline(y=entr_cohe, color='red', linewidth=0.5, \
                    linestyle='-', label='coherent')
    plt.legend(fontsize=14, loc=4)