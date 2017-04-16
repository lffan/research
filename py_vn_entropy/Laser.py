import numpy as np
from scipy.integrate import odeint
from scipy.sparse import coo_matrix
from qutip import *

class Laser(w_c, w_a, g, ra, gamma, kappa, N_max, 
            init_state=None, t_list=None, states=None):
    """ Numerical simulation of laser given on the equation of motion
        for the density matrix of the cavity field in the book of
        Qunatum Optics (Scully and Zubairy) Page? Chapter 11.)
    """
    
    def __init__(w_c, w_a, g, ra, gamma, kappa, N_max, 
                 init_state, t_list, states):
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
        self.N_max = N_max
        
        self.init_state = init_state
        self.t_list = t_list
        self.states = states
        
        self.A = 2 * ra * g**2 / gamma**2
        self.B = 4 * self.A * g**2 / gamma**2
        self.BdA = 4 * g**2 / gamma**2
    
    def steady_average_n(self):
        """ Calculate the average photon number in steady state
            for a laser operated above threshold
        """
        return self.A (self.A - self.kappa) / self.kappa / self.B
        
    # ode solver wrapper
    def evolution(self, rho_0)
        

    # coefficients of the equation of motion for rho ---------------------------------
    def _M(self, n, m):
        return 0.5 * (n + m + 2) + (n - m)**2 * self.BdA / 8

    def _N(self, n, m):
        return 0.5 * (n + m + 2) + (n - m)**2 * self.Bda / 16

    def _fnm(self, n, m):
        f1 = self._M(n, m) * self.A / (1 + self._N(n, m) * self.BdA)
        f2 = 0.5 * self.kappa * (n + m)
        return - f1 - f2

    def _gnm(self, n, m):
        return np.sqrt(n * m) * self.fA / (1 + self._N(n - 1, m - 1) * self.BdA)

    def _hnm(self, n, m):
        return self.kappa * np.sqrt((n + 1) * (m + 1))

    
    # ordinary differential equation for diagonal terms only ----------------------- 
    def _pn_dot(self, pn, t, f, g, h):
        # pn_new = np.zeros(self.N_max)
        # for n in xrange(self.N_max):
        #     pn_new[n] += (- (n + 1) * self.A / (1 + (n + 1) * self.BdA)  - self.kappa * n) * pn[n]
        #     if n > 0:
        #         pn_new[n] += n * self.A / (1 + n * self.BdA) * pn[n - 1]
        #     if n < self.N_max - 1:
        #         pn_new[n] += self.kappa * (n + 1) * pn[n + 1]
        
        pn_new = np.zeros(self.N_max)
        for n in xrange(self.N_max):
            pn_new[n] += f[n, n] * pn[n]
            if n > 0:
                pn_new[n] += g[n, n] * pn[n - 1]
            if n < self.N_max - 1:
                pn_new[n] += h[n, n] * pn[n + 1]
                
        return pn_new

    # differential equation for the whole density matrix ---------------------------
    def _rho_nm_dot(self, rho_nm, t, f, g, h):
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
    
    
    