import numpy as np
import scipy
from types import SimpleNamespace

class ASAD_class():

    def  __init__(self,do_print=True):
        """ create the model """

        if do_print: print('initializing the model:')

        self.par = SimpleNamespace()
        self.path = SimpleNamespace()

        if do_print: print('Setting up model with standard parameters')
        # Extra print with exact standard parameter values, note h = b = 0 resembles fixed exchange rates

        self.setup()

        if do_print: print('Allocating paths for output, inflation, nominal exchange rate and real exchange rate')

        self.allocate()

    def setup(self):
        """ baseline parameters """

        par = self.par
        path = self.path

        par.betaone = 0.5
        par.betatwo = 0.5
        par.gamma = 0.5
        par.Tpath = 500
        par.h = 0.5
        par.b = 0.5
        par.theta = 0.5

        par.betaonehat = par.betaone + par.betaone*(par.h/par.theta)+par.betatwo*par.h
        par.betatwohat = 1 + par.betaone*(par.b/par.theta)+par.betatwo*par.b

    def allocate(self):
        par = self.par
        path = self.path

        # Create paths for variables
        allvarnames = ['yhat','pihat','er','e','s','z','ihat']
        for varname in allvarnames:
                path.__dict__[varname] =  np.nan*np.ones(par.Tpath)

        # Dynamic variables with one foot in each period - the real exchange rate for period t is determined in period t and used to solve system i period t+1
        path.er_minus = np.insert(path.er,0,0)
        path.e_minus = np.insert(path.e,0,0)

        # Shock variables
        path.z = np.zeros(par.Tpath)
        path.s = np.zeros(par.Tpath)

    def solve(self):

        par = self.par
        path = self.path

        initial_guess = np.zeros(2)

        for t in range(0,par.Tpath,1):
            # a. Solving the equation system (essential operations)
            # Solve equation system of AS- and AD-curve for period t
            result = scipy.optimize.root(self.eq_sys, initial_guess, args=(t,))

            # Store solutions in the path-variables for period t for both yhat and pihat
            path.yhat[t],path.pihat[t] = result.x

            # Compute real exchange rate in period t
            path.er[t] = path.er_minus[t] - (1+par.h/par.theta)*path.pihat[t] - (par.b/par.theta)*path.yhat[t]

            # Pass on real exchange rate to the variable used in period t+1 to solve the equation system
            path.er_minus[t+1] = path.er[t]

            # b. Storing variables of interest (non-essential operations)
            path.ihat[t] = par.h*path.pihat[t] + par.b*path.yhat[t]
            path.e[t] = path.e_minus[t] - (par.h/par.theta)*path.pihat[t] - (par.b/par.theta)*path.yhat[t]
            path.e_minus[t+1] = path.e_minus[t]
    
    def eq_sys(self, vars, t):
        
        par = self.par
        path = self.path

        # Define yhat and pihat as the variables of interest
        yhat, pihat = vars

        # Define gap_var as the equation system
        gap_var = np.zeros(2)

        # AD-curve equal to zero
        gap_var[0] = yhat-(par.betaone/par.betatwohat)*path.er_minus[t] + (par.betaonehat/par.betatwohat)*pihat - (1/par.betatwohat)*path.z[t]

        # AS-curve equal to zero
        gap_var[1] = pihat-par.gamma*yhat-path.s[t]

        return gap_var

    def shocks(self,t0,t1,n,z=False,s=False):

        par = self.par
        path = self.path

        # Function that creates specified durationed shocks of equal value

        if z == True:
            for i in range(t0,t1,1):
                path.z[i] = n
        
        if s == True:
            for i in range(t0,t1,1):
                path.s[i] = n
    
    def stochastic_shocks(self,mean,sd,z=False,s=False):

        par = self.par
        path = self.path
        
        # Function that creates normal-distributed shocks

        np.random.seed(seed=2000)

        if z == True:
            path.z = np.random.normal(loc=mean,scale=sd,size=par.Tpath)
        
        if s == True:
            path.s = np.random.normal(loc=mean,scale=sd,size=par.Tpath)

    def AR_shocks(self,rho,sd,mean=0,z=False,s=False):

        par = self.par
        path = self.path
    
        # Function that creates an AR-process for shocks with normally distributed errors.

        np.random.seed(seed=2000)
        e = np.random.normal(mean,sd,size=par.Tpath)

        if z == True:
            for t in range(1,par.Tpath):
                path.z[t] = rho*path.z[t-1] + e[t]
        
        if s == True:
            for t in range(1,par.Tpath):
                path.s[t] = rho*path.s[t-1] + e[t]

    def compute_social_loss(self):
        
        par = self.par
        path = self.path

        loss = np.zeros(par.Tpath)

        loss = self.social_loss(type='quadratic')

        return np.sum(loss)
    
    def social_loss(self,type,kappa=1):

        par = self.par
        path = self.path

        if type == 'quadratic':
            return path.yhat**2 + kappa*path.pihat**2
        
        if type == 'numeric':
            return np.abs(path.yhat) + np.abs(path.pihat)
        
    