import numpy as np
from scipy.integrate import odeint, complex_ode
# from scipy.sparse import coo_matrix
from qutip import *

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
        self.t_list = None
        self.init_state = None
        self.rhos = []
        self.pns = []
        
    
    def get_atom_cavity_args(self):
        """ return the setup parameters for the atom and cavity
        """
        return {'w_c': self.w_c, 'w_a': self.w_a, 'g': self.g, 
                'ra': self.ra, 'gamma': self.gamma, 'kapa': self.kappa}
    
    
    def get_pns(self):
        """ return diagonal terms
        """
        return self.pns
    
    
    def get_rhos(self):
        """ return the whole denstiy matrix
        """
        return self.rhos
    
    
    def get_steady_state(self):
        """ return the last elements of pns and rhos
        """
        steady_pn, steady_rho = None, None
        if len(self.pns) > 0:
            steady_pn = self.pns[-1]
        if len(self.rhos) > 0:
            steady_rho = self.rhos[-1]
        return steady_pn, steady_rho
    
    
    def steady_average_n(self):
        """ Calculate the average photon number in steady state
            for a laser operated above threshold
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
        self.pns = odeint(self._pn_dot, init_pn, t_list, args=(f, g, h,))
        
        # reconstruct rho from pn if only the main diagonal terms exist
        self.rhos = np.array([Qobj(np.diag(pn)) for pn in self.pns])
        
    
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
        self.rhos = np.array([Qobj(a.reshape(self.N_max, self.N_max)) 
                              for a in self.arrays])
        
        # find diagonal terms
        self.pns = np.array([np.real(np.diag(rho.data.toarray())) for rho in self.rhos])
        

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
    