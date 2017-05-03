import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.integrate import odeint, complex_ode
from scipy.linalg import solve
# from scipy.sparse import coo_matrix

from qutip import *

__author__ = "Longfei Fan"

class LaserOneMode(object):
    """ Numerical simulation of laser given on the equation of motion
        for the density matrix of the cavity field in the book of
        Qunatum Optics (Scully and Zubairy) Page? Chapter 11.)
    """
    
    def __init__(self, w_c, w_a, g, ra, gamma, kappa):
        """ w_c: cavity frequency
            w_a: atom frequency
            g: atom-cavity interation strength
            ra: pumping rate
            gamma: atom damping rate
            kappa: cavity damping rate
            init_state: initial cavity state (an qutip.Qobj object)
            the atom is assumed in the ground state at the beginning
        """
        self.w_c = w_c
        self.w_a = w_a
        self.g = g
        
        self.ra = ra
        self.gamma = gamma
        self.kappa = kappa
        
        self.A = 2 * ra * g**2 / gamma**2
        self.B = 4 * self.A * g**2 / gamma**2
        self.BdA = 4 * g**2 / gamma**2
        
        self.N_max = None
        self.t_list = []
        self.init_state = None
        self.rho_vs_t = []
        self.pn_vs_t = []
        self.n_vs_t = []
        self.entr_vs_t = []
        
        self.steady_pn = None
        self.steady_n = None
        self.steady_entr = None
        
    
    def set_N_max(self, N_max):
        """ set the truncated photon numbers for numerical calcualtions
        """
        self.N_max = N_max
        
    
    def get_atom_cavity_args(self):
        """ return the setup parameters for the atom and cavity
        """
        return {'w_c': self.w_c, 'w_a': self.w_a, 'g': self.g, 
                'ra': self.ra, 'gamma': self.gamma, 'kapa': self.kappa}
    
    
    def get_abc(self):
        """ return A, B, kappa
        """
        return {'A': self.A, 'B': self.B, 'C': self.kappa}
    
    
    def get_pns(self):
        """ return diagonal terms vs. time
        """
        return self.pn_vs_t
    
    
    def get_rhos(self):
        """ return the whole denstiy matrix vs. time
        """
        return self.rho_vs_t
    
    def get_ns(self):
        """ return average photon number vs. time
        """
        return self.n_vs_t
    
    
#     def get_steady_state(self):
#         """ return the last elements of pn_vs_t, rho_vs_t, and n_vs_t
#         """
#         steady_pn, steady_rho, steady_avern = None, None, None
#         if len(self.pn_vs_t) > 0:
#             steady_pn = self.pn_vs_t[-1]
#         if len(self.rho_vs_t) > 0:
#             steady_rho = self.rho_vs_t[-1]
#         if len(self.n_vs_t):
#             steady_avern = self.n_vs_t[-1]
#         return steady_pn, steady_rho, steady_avern
    
    
    def steady_n_above(self):
        """ Calculate the average photon number in steady state
            for a laser operated **above threshold**
        """
        return self.A * (self.A - self.kappa) / self.kappa / self.B
    
        
    # solve the ode for pn, if rho only contains diagonal terms, reconstruct rho
    def pn_evolve(self, init_state, N_max, t_list, diag=False):
        """ **ode solver for pn**
            init_pn: an array, initial diagonal terms of rho
            N_max: truncted photon numbers
            t_list: a list of time points to be calculated on
            diag: if rho only has diagonal terms, reconstruct rho
        """
        self.N_max = N_max
        self.t_list = t_list
        n_list = np.arange(N_max)
        
        # parameters
        f = np.array([self._fnm(n, n) for n in n_list]) 
        g = np.array([self._gnm(n, n) for n in n_list])
        h = np.array([self._hnm(n, n) for n in n_list])
        
        # find diagonal terms
        if init_state.type is 'ket':
            init_state = ket2dm(init_state)
        init_pn = np.real(np.diag(init_state.data.toarray()))
        
        # solve the ode for pn
        self.pn_vs_t = odeint(self._pn_dot, init_pn, t_list, args=(f, g, h,))
        
        # reconstruct rho from pn if only the main diagonal terms exist
        self.rho_vs_t = np.array([Qobj(np.diag(pn)) for pn in self.pn_vs_t])
        
        # find average photon numbers
        self.n_vs_t = np.array([sum(pn * n_list) for pn in self.pn_vs_t])
        
    
    # solve the ode for rho
    def rho_evolve(self, init_rho, N_max, t_list):
        """ **ode solver for the density matrix rho**
            init_rho: initial density matrix given as a quitp.Qobj
            N_max: truncted photon numbers
            t_list: a list of time points to be calculated on
        """
        self.N_max = N_max
        self.t_list = t_list
        self.init_state = init_rho
        n_list = np.arange(N_max)
        
        # if ket state, convert to density matrix, then to a 1-d array
        if init_rho.type == 'ket':
            init_rho = ket2dm(init_rho)
        init_array = np.real(init_rho.data.toarray().reshape(-1))          
        
         # parameters
        f = np.array([self._fnm(i, j) for i in n_list          
                      for j in n_list]).reshape(N_max, N_max)
        g = np.array([self._gnm(i, j) for i in n_list 
                      for j in n_list]).reshape(N_max, N_max)
        h = np.array([self._hnm(i, j) for i in n_list 
                      for j in n_list]).reshape(N_max, N_max)

        # sovle the ode
        self.arrays = odeint(self._rho_nm_dot, init_array, t_list, args=(f, g, h, ))

        # convert arrays back to density matrices (rhos)
        self.rho_vs_t = np.array([Qobj(a.reshape(self.N_max, self.N_max)) 
                              for a in self.arrays])
        
        # find diagonal terms
        self.pn_vs_t = np.array([np.real(np.diag(rho.data.toarray())) for rho in self.rho_vs_t])
        
        # find average photon numbers
        self.n_vs_t = np.array([sum(pn * n_list) for pn in self.pn_vs_t])
        
    
    def plot_n_vs_time(self):
        """ Plot average photon numbers with respect to time
        """
        if len(self.n_vs_t) == 0:
            print "Solve the evolution equation first to obtain average photon numbers!"
            return
        fig, ax = plt.subplots(figsize=(5,3))
        ax.plot(self.t_list, self.n_vs_t)
        ax.set_xlabel("time", fontsize=14)
        ax.set_ylabel("average photon number", fontsize=14)
        ax.set_title("Average Photon Number vs. Time", fontsize=14)
        return fig, ax
    
    
    def _calc_entropy(self):
        """ Calculate von Neumann entropy given on the rho list
        """
        if len(self.rho_vs_t) == 0:
            print "Solve the evolution equation first to obtain entropy!"
            return
        print "calculating von Neuman entropy ..."
        self.entr_vs_t = np.array([entropy_vn(rho, 2) for rho in self.rho_vs_t])
    
    
    def plot_entropy_vs_time(self):
        """ Plot von Neumann entropy of the cavity field with respect to time
        """
        if len(self.entr_vs_t) == 0:
            self._calc_entropy()
        fig, ax = plt.subplots(figsize=(5,3))
        ax.plot(self.t_list, self.entr_vs_t)
        ax.set_xlabel("time", fontsize=12)
        ax.set_ylabel("von Neumann entropy", fontsize=12)
        ax.set_title("von Neumann Entropy vs. Time", fontsize=12)
        return fig, ax
    
    
    def plot_steady_wigner(self):
        """ Plot 2D wigner function of the steady state
        """
        l2_pn, l2_rho = laser_above.get_steady_state() 
        

    # coefficients of the equation of motion for rho
    def _M(self, n, m):
        return 0.5 * (n + m + 2) + (n - m)**2 * self.BdA / 8

    def _N(self, n, m):
        return 0.5 * (n + m + 2) + (n - m)**2 * self.BdA / 16

    def _fnm(self, n, m):
        f1 = self._M(n, m) * self.A / (1 + self._N(n, m) * self.BdA)
        f2 = 0.5 * self.kappa * (n + m)
        return - f1 - f2

    def _gnm(self, n, m):
        return np.sqrt(n * m) * self.A / (1 + self._N(n - 1, m - 1) * self.BdA)

    def _hnm(self, n, m):
        return self.kappa * np.sqrt((n + 1) * (m + 1)) 
    
        
    # ordinary differential equation for diagonal terms only
    def _pn_dot(self, pn, t, f, g, h):
        """ ode update rule for pn
        """
        pn_new = np.zeros(self.N_max)
        
        for n in xrange(self.N_max):
            pn_new[n] += f[n] * pn[n]
            if n > 0:
                pn_new[n] += g[n] * pn[n - 1]
            if n < self.N_max - 1:
                pn_new[n] += h[n] * pn[n + 1] 
                
        return pn_new
    

    # differential equation for the whole density matrix
    def _rho_nm_dot(self, rho_nm, t, f, g, h):
        """ ode update rul for rho_nm
        """
        rho = rho_nm.reshape(self.N_max, self.N_max)
        rho_new = np.zeros([self.N_max, self.N_max])

        ij = range(self.N_max)
        for i in ij:
            for j in ij:
                rho_new[i, j] += f[i, j] * rho[i, j]
                if i > 0 and j > 0:
                    rho_new[i, j] += g[i, j] * rho[i - 1, j - 1]
                if i < self.N_max - 1 and j < self.N_max - 1:
                    rho_new[i, j] += h[i, j] * rho[i + 1, j + 1]

        return rho_new.reshape(-1)
    
    
    def solve_steady_state(self, N_max):
        """ if the state is always diagonal during evolution
            get the diagonal terms of the steady state
        """
        eq = np.zeros([self.N_max, self.N_max])
        y = np.array([np.finfo(float).eps] * self.N_max)

        for k in range(self.N_max):
            eq[k, k] = self._fnm(k, k)
            if k < self.N_max - 1:
                eq[k, k + 1] = self._hnm(k, k)
            if k > 0:
                eq[k, k - 1] = self._gnm(k, k)        
        pn = solve(eq, y)
        
        pn = pn/sum(pn)
        n = sum(pn * range(self.N_max))
        entr = sum([- p * np.log2(p) for p in pn if p > 0])

        return pn, n, entr

    
def boltzmann(ratio, N_max):
    """ return an array of pn according to the boltzmann distribution
    """
    return np.array([(1 - ratio) * ratio ** n for n in np.arange(N_max)])


def poisson(l, N_max):
    """ return an array of pn according to the Poisson distribution
    """
    l = np.float(l)
    return np.array([np.exp(-l) * l**n / np.math.factorial(n) \
                     for n in np.arange(N_max)])
    