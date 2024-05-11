import numpy as np
import scipy
from types import SimpleNamespace

class ASAD_class():

    def  __init__(self,do_print=True,exchange_rates='fixed'):
        """ create the model """

        # Option to print that the model is initialized
        if do_print: print('initializing the model:')

        # Initialize parameters, steadt state values and path values
        self.par = SimpleNamespace()
        self.path = SimpleNamespace()

        # Option to print that setup is being called
        if do_print: print('calling .setup()')

        # Option to print that allocate is being called
        if do_print: print('calling .allocate()')

        # Call setup function
        self.setup_allocate(exchange_rates)

    def setup_allocate(self,exchange_rates):
        """ baseline parameters """

        # Set parameters/conditions for model a) Households b) Firms and c) Initial conditions d) Optimizer + truncation horizon as Tpath
        par = self.par
        path = self.path

        par.betaone = 0.5
        par.gamma = 0.5
        par.Tpath = 20

        if exchange_rates == 'fixed':
            allvarnames = ['yhat','pihat','er','z','s']
            for varname in allvarnames:
                path.__dict__[varname] =  np.nan*np.ones(par.Tpath)
            path.er_minus = np.insert(path.er[:-1],0,0)
        
        if exchange_rates == 'floating':
            allvarnames = ['yhat','pihat','er','e','z','s']
            for varname in allvarnames:
                path.__dict__[varname] =  np.nan*np.ones(par.Tpath)
                path.er_minus = np.insert(path.er[:-1],0,0)
        
    def solve(self):

        par = self.par
        path = self.path

        initial_guess = np.zeros(2)

        for t in range(0,par.Tpath,1):
            result = scipy.optimize.root(self.eq_sys, initial_guess, args=(t,))
            path.yhat[t],path.pihat[t] = result.x
            if t < par.Tpath - 1:
                path.er[t] = path.er_minus[t] - path.pihat[t]
                path.er_minus[t+1] = path.er[t]
    
    def eq_sys(self, vars, t):
        
        par = self.par
        path = self.path

        yhat, pihat = vars

        gap_var = np.zeros(2)
        gap_var[0] = yhat-par.betaone*path.er_minus[t] + par.betaone*pihat - path.z[t]
        gap_var[1] = pihat-par.gamma*yhat-path.s[t]

        return gap_var
