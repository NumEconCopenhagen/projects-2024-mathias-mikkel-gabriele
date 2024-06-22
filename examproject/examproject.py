import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

class CareerClass:
    '''
    
    '''
    def __init__(self):
        par = self.par = SimpleNamespace()

        par.J = 3
        par.N = 10
        par.K = 10000

        par.F = np.arange(1,par.N+1)
        par.sigma = 2

        par.v = np.array([1,2,3])
        par.c = 1

    
    def expec_utility(self, v, seed=None):
        '''
        Calculates the expected utility of given career choice
        '''
        par = self.par
        
        if seed is not None:
            np.random.seed(seed)

        return v + 1/par.K * self.simulate(seed=seed)
    
    def avg_real_utility(self, v, seed=None):
        '''
        Calculates the real utility of given career choice
        '''
        par = self.par
        real_utility = np.zeros(par.K)

        if seed is not None:
            np.random.seed(seed)

        for k in range(par.K):
            eps = np.random.normal(0, par.sigma)
            real_utility[k] = v + eps
        
        sum_real_utility = sum(real_utility)

        return 1/par.K*sum_real_utility

    def simulate(self, seed=None):
        '''
        Simulates the career choice
        '''
        par = self.par

        # Set seed
        if seed is not None:
            np.random.seed(seed)

        # Initialize the array of epsilons
        eps = np.random.normal(0, par.sigma, par.K)

        sum_eps = sum(eps)

        return sum_eps