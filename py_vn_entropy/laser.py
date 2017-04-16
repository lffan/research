import numpy as np
from scipy.integrate import odeint
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
            init_rho: initial cavity state (an qutip.Qobj object)
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
        self.init_rho = None
        self.rhos = None      
    
    def get_atom_cavity_args(self):
        """ return the setup parameters for the atom and cavity
        """
        return {'w_c': self.w_c, 'w_a': self.w_a, 'g': self.g, 
                'ra': self.ra, 'gamma': self.gamma, 'kapa': self.kappa}
    
    
    def steady_average_n(self):
        """ Calculate the average photon number in steady state
            for a laser operated above threshold
        """
        return self.A * (self.A - self.kappa) / self.kappa / self.B
    
        
    # ode solver wrapper
    def evolution(self, init_rho, t_list, N_max, eq, diag):
        """ **ode solver wrapper**
            Return the solution to the equation of motion given on a
            time list and an initial state.
            init_rho: initial density matrix given as a quitp.Qobj
            t_list: a list of time points to be calculated on
            N_max: truncted photon numbers
            eq: method for ode solver
                - 'pn': calculate diagonal terms only
                - 'rho': calculate all the elements of the density matirx
            diag: indicate whether the matrix is diagonal or not
                - 'True': return rho given the diagonal terms
                - 'False': will not return rho as off-diagonal terms are unknown
        """
        self.N_max = N_max
        self.t_list = t_list
        self.init_rho = init_rho
        self.init_pn = init_pn
        n_list = np.arange(N_max)
        
        # calculate evolutions for diagonal terms only
        if eq is 'pn':
            init_pn = np.diag(init_rho.data.toarray()) # extract diagonal terms into an array
            
            f = np.array([self._fnm(n, n) for n in n_list]) # parameters
            g = np.array([self._gnm(n, n) for n in n_list])
            h = np.array([self._hnm(n, n) for n in n_list])
            
            # solve the ode
            self.pns = odeint(self._pn_dot, init_pn, t_list, args=(f, g, h,)) 
            
            if diag: # if the density matrix is diagonal, return an array of rhos
                self.rhos = [qdiags(diag, offset=0) for diag in self.pns]
        
        # calculate evolutions for the whole density matrix
        elif eq is 'rho':
            # convert qutip.Qobj into an matrix, then flat it to an array
            init_rho = inti_rho.data.toarray().reshape(-1)            
            
            f = np.array([self._fnm(i, j) for i in n_list          # parameters 
                          for j in n_list]).reshape(N_max, N_max)
            g = np.array([self._gnm(i, j) for i in n_list 
                          for j in n_list]).reshape(N_max, N_max)
            h = np.array([self._hnm(i, j) for i in n_list 
                          for j in n_list]).reshape(N_max, N_max)
            
            # sovle the ode
            self.arrays = odeint(self._rho_dot, init_array, t_list, args=(f, g, h, ))
            
            # convert arrays back to density matrices (rhos)
            self.rhos = [Qobj(a.reshape(self.N_max, self.N_max)) for a in self.arrays]
        
        return self.rhos
        

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
    