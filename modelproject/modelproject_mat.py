from types import SimpleNamespace
import numpy as np
from scipy import optimize

import numpy as np

import numpy as np

class RamseyModel:
    def __init__(self, T, sigma, gamma, upsilon, alpha, delta, r, A, K0, L_bar):
        self.T = T  # Number of periods
        self.sigma = sigma  # Coefficient of relative risk aversion
        self.gamma = gamma  # Elasticity of labor supply
        self.upsilon = upsilon  # Output elasticity of labor
        self.alpha = alpha  # Output elasticity of capital
        self.delta = delta  # Depreciation rate
        self.r = r  # Real interest rate
        self.A = A  # Total factor productivity
        self.K0 = K0  # Initial capital stock
        self.L_bar = L_bar  # Exogenous labor supply

    def utility(self, C, N):
        """
        Utility function: U(C, N) = (C^(1-sigma) - 1) / (1 - sigma) - upsilon * (N^(1+gamma) - 1) / (1 + gamma)
        """
        return (C**(1 - self.sigma) - 1) / (1 - self.sigma) - self.upsilon * (N**(1 + self.gamma) - 1) / (1 + self.gamma)

    def euler_equation(self, C, N, K, t):
        """
        Euler equation: U'(C, N) = (1 + r) * U'(C(+1), N(+1))
        """
        C_next = C[t + 1] if t < self.T - 1 else C[t]  # Assume consumption is constant in the last period
        N_next = N[t + 1] if t < self.T - 1 else N[t]  # Assume labor supply is constant in the last period
        
        # Calculate marginal utility
        U_prime_C = C[t]**(-self.sigma)
        U_prime_N = -self.upsilon * N[t]**self.gamma
        
        U_prime_C_next = C_next**(-self.sigma)
        U_prime_N_next = -self.upsilon * N_next**self.gamma
        
        return U_prime_C - (1 + self.r) * U_prime_C_next, U_prime_N - (1 + self.r) * U_prime_N_next

    def solve_model(self):
        # Initialize arrays to store optimal decisions
        C = np.zeros(self.T)
        N = np.zeros(self.T)
        K = np.zeros(self.T)

        # Initial guess for consumption and labor supply
        C[0] = 0.5  # Adjust initial consumption for better convergence
        N[0] = self.L_bar

        # Dynamic programming algorithm to solve for optimal decisions
        for t in range(self.T - 1):
            # Euler equation method to find optimal consumption and labor supply
            C[t + 1] = ???  # Calculate optimal consumption for period t+1
            N[t + 1] = ???  # Calculate optimal labor supply for period t+1
            K[t + 1] = (1 - self.delta) * K[t] + self.A * K[t]**self.alpha * N[t]**(1 - self.alpha) - C[t] + self.r * K[t]

        return C, N, K

# Example parameters
T = 50  # Number of periods
sigma = 2  # Coefficient of relative risk aversion
gamma = 1  # Elasticity of labor supply
upsilon = 1  # Output elasticity of labor
alpha = 0.3  # Output elasticity of capital
delta = 0.05  # Depreciation rate
r = 0.05  # Real interest rate
A = 1  # Total factor productivity
K0 = 1  # Initial capital stock
L_bar = 0.3  # Exogenous labor supply

# Create instance of RamseyModel
model = RamseyModel(T, sigma, gamma, upsilon, alpha, delta, r, A, K0, L_bar)

# Solve the model
C, N, K = model.solve_model()

# Print optimal paths of consumption, labor supply, and capital
print("Optimal consumption path:", C)
print("Optimal labor supply path:", N)
print("Optimal capital path:", K)



class RamseyLabor():
    def __init__(self, do_print=True):
        if do_print:
            print('Initializing the model:')
        self.par = SimpleNamespace()
        self.ss = SimpleNamespace()
        self.path = SimpleNamespace()
        if do_print:
            print('Calling .setup()')
        self.setup()
        if do_print:
            print('Calling .allocate()')
        self.allocate()

    def setup(self):
        par = self.par

        #utility function baseline parameters
        par.sigma = 2.0 # CRRA coefficient
        par.gamma = 2.0 # IES
        par.beta = np.nan # discount factor
        par.upsilon = 1.0

        # b. firms
        par.Gamma = np.nan
        par.production_function = 'cobb-douglas'
        par.alpha = 0.33 # capital weight
        par.theta = 0.05 # substitution parameter (CES case)     
        par.delta = 0.05 # depreciation rate

        # Initial values of variables
        par.K_lag_ini = 1.0
        par.N_ini = 0.5

        par.solver = 'broyden' # solver for the equation system, 'broyden' or 'scipy'
        par.Tpath = 500 # length of transition path, "truncation horizon"

    def allocate(self):
        """ allocate arrays for transition path """
        
        par = self.par
        path = self.path

        allvarnames = ['B','K','C','N','rk','w','r','Y','K_lag','Gamma']
        for varname in allvarnames:
            path.__dict__[varname] =  np.nan*np.ones(par.Tpath)

    def production(self, K, N):
        
        par = self.par 

        if par.production_function == "ces":
            Y = par.Gamma * (par.alpha * K**(-par.theta) + (1 - par.alpha) * N**(-par.theta))**(-1/par.theta)
            rk = par.Gamma * par.alpha * K**(-par.theta - 1) * (Y / par.Gamma)**(1 + par.theta)
            w = par.Gamma * (1 - par.alpha) * N**(-par.theta - 1) * (Y / par.Gamma)**(1 + par.theta)

        elif par.production_function == 'cobb-douglas':

            # a. production
            Y = Gamma*K_lag**par.alpha *N**(1-par.alpha)

            # b. factor prices
            rk = Gamma*par.alpha * K_lag**(par.alpha-1) * N**(1-par.alpha)
            w = Gamma*(1-par.alpha) * K_lag**(par.alpha) * N**(-par.alpha)

        else:
              
            raise Exception('unknown type of production function')

        return Y,rk,w 

    def utility(self, C, N):
        par = self.par
        return C**(1 - par.sigma) / (1 - par.sigma) - par.upsilon * N**(1 + par.gamma) / (1 + par.gamma)
    
    def euler(self, C, K, K_lag, N, N_lag, Y, Y_lag, rk, w, r, Gamma):
        par = self.par
        return C**(-par.sigma) - par.beta * (1 + r) * C_lag**(-par.sigma)
    
    def capital_accumulation(self, K, K_lag, Y, C):
        par = self.par
        return K - (1 - par.delta) * K_lag + (Y - C)
    
    def labor_supply(self, N, C, w):
        par = self.par
        return N + (C * w / par.upsilon)**(-1 / par.gamma)
       
class RamseyModelClass():

    def __init__(self,do_print=True):
        """ create the model """

        if do_print: print('initializing the model:')

        self.par = SimpleNamespace()
        self.ss = SimpleNamespace()
        self.path = SimpleNamespace()

        if do_print: print('calling .setup()')
        self.setup()

        if do_print: print('calling .allocate()')
        self.allocate()
    
    def setup(self):
        """ baseline parameters """

        par = self.par

        # a. household
        par.sigma = 2.0 # CRRA coefficient
        par.gamma = 2.0 # IES
        par.beta = np.nan # discount factor
        par.upsilon = 1.0

        # b. firms
        par.Gamma = np.nan
        par.production_function = 'cobb-douglas'
        par.alpha = 0.30 # capital weight
        par.theta = 0.05 # substitution parameter        
        par.delta = 0.05 # depreciation rate

        # c. initial
        par.K_lag_ini = 1.0
        par.N_ini = 0.5

        # d. misc
        par.solver = 'broyden' # solver for the equation system, 'broyden' or 'scipy'
        par.Tpath = 500 # length of transition path, "truncation horizon"

    def allocate(self):
        """ allocate arrays for transition path """
        
        par = self.par
        path = self.path

        allvarnames = ['B','K','C','N','rk','w','r','Y','K_lag','Gamma']
        for varname in allvarnames:
            path.__dict__[varname] =  np.nan*np.ones(par.Tpath)

    def find_steady_state(self,KY_ss,N,do_print=True):
        """ find steady state """

        par = self.par
        ss = self.ss

        # a. find A
        ss.K = KY_ss
        ss.N = N
        Y,_,_ = production(par,1.0,ss.K, ss.N)
        ss.Gamma = 1/Y

        # b. factor prices
        ss.Y,ss.rk,ss.w = production(par,ss.Gamma,ss.K, ss.N)
        assert np.isclose(ss.Y,1.0)

        ss.r = ss.rk-par.delta
        
        # c. implied discount factor
        par.beta = 1/(1+ss.r)

        # d. consumption
        ss.C = ss.Y - par.delta*ss.K

        # e. hours worked
        #ss.N = (1-par.sigma)/(par.sigma + par.gamma)
        ss.N = ((C*path.w)/par.upsilon)**(-(1/par.gamma))

        if do_print:

            print(f'Y_ss = {ss.Y:.4f}')
            print(f'K_ss/Y_ss = {ss.K/ss.Y:.4f}')
            print(f'rk_ss = {ss.rk:.4f}')
            print(f'r_ss = {ss.r:.4f}')
            print(f'w_ss = {ss.w:.4f}')
            print(f'ss_N = {ss.N:.4f}')
            print(f'Gamma = {ss.Gamma:.4f}')
            print(f'beta = {par.beta:.4f}')

    def evaluate_path_errors(self):
        """ evaluate errors along transition path """

        par = self.par
        ss = self.ss
        path = self.path

        # a. consumption        
        C = path.C
        C_plus = np.append(path.C[1:],ss.C)
        
        # b. capital
        K = path.K
        K_lag = path.K_lag = np.insert(K[:-1],0,par.K_lag_ini)
        
        N = path.N
        N_plus = np.append(path.N[1:],par.N_ini)

        # c. production and factor prices
        path.Y,path.rk,path.w = production(par,path.Gamma,K_lag, N)
        path.r = path.rk-par.delta
        r_plus = np.append(path.r[1:],ss.r)

        # d. errors (also called H)
        errors = np.nan*np.ones((3,par.Tpath))
        errors[0,:] = C**(-par.sigma) - par.beta*(1+r_plus)*C_plus**(-par.sigma)
        errors[1,:] = K - ((1-par.delta)*K_lag + (path.Y - C))
        errors[2,:] = N + ((C*path.w)/par.upsilon)**(-(1/par.gamma))
        
        return errors.ravel()
        
    def calculate_jacobian(self,h=1e-6):
        """ calculate jacobian """
        
        par = self.par
        ss = self.ss
        path = self.path
        
        # a. allocate
        Njac = 3*par.Tpath
        jac = self.jac = np.nan*np.ones((Njac,Njac))
        
        x_ss = np.nan*np.ones((3,par.Tpath))
        x_ss[0,:] = ss.C
        x_ss[1,:] = ss.K
        x_ss[2,:] = ss.N
        x_ss = x_ss.ravel()

        # b. baseline errors
        path.C[:] = ss.C
        path.K[:] = ss.K
        path.N[:] = ss.N
        base = self.evaluate_path_errors()

        # c. jacobian
        for i in range(Njac):
            
            # i. add small number to a single x (single K or C) 
            x_jac = x_ss.copy()
            x_jac[i] += h
            x_jac = x_jac.reshape((3,par.Tpath))
            
            # ii. alternative errors
            path.C[:] = x_jac[0,:]
            path.K[:] = x_jac[1,:]
            path.N[:] = x_jac[2,:]
            alt = self.evaluate_path_errors()

            # iii. numerical derivative
            jac[:,i] = (alt-base)/h
        
    def solve(self,do_print=True):
        """ solve for the transition path """

        par = self.par
        ss = self.ss
        path = self.path
        
        # a. equation system
        def eq_sys(x):
            
            # i. update
            x = x.reshape((3,par.Tpath))
            path.C[:] = x[0,:]
            path.K[:] = x[1,:]
            path.N[:] = x[2,:]
            
            # ii. return errors
            return self.evaluate_path_errors()

        # b. initial guess
        x0 = np.nan*np.ones((3,par.Tpath))
        x0[0,:] = ss.C
        x0[1,:] = ss.K
        x0[2,:] = ss.N
        x0 = x0.ravel()

        # c. call solver
        if par.solver == 'broyden':

            x = broyden_solver(eq_sys,x0,self.jac,do_print=do_print)
        
        elif par.solver == 'scipy':
            
            root = optimize.root(eq_sys,x0,method='hybr',options={'factor':1.0})
            # the factor determines the size of the initial step
            #  too low: slow
            #  too high: prone to errors
             
            x = root.x

        else:

            raise NotImplementedError('unknown solver')
            

        # d. final evaluation
        eq_sys(x)
            
def production(par,Gamma,K_lag, N):
    """ production and factor prices """

    # a. production and factor prices
    if par.production_function == 'ces':

        # a. production
        Y = Gamma*( par.alpha*K_lag**(-par.theta) + (1-par.alpha)*N**(-par.theta) )**(-1.0/par.theta)

        # b. factor prices
        rk = Gamma*par.alpha*K_lag**(-par.theta-1) * (Y/Gamma)**(1.0+par.theta)
        w = Gamma*(1-par.alpha)*N**(-par.theta-1) * (Y/Gamma)**(1.0+par.theta)

    elif par.production_function == 'cobb-douglas':

        # a. production
        Y = Gamma*K_lag**par.alpha *N**(1-par.alpha)

        # b. factor prices
        rk = Gamma*par.alpha * K_lag**(par.alpha-1) * N**(1-par.alpha)
        w = Gamma*(1-par.alpha) * K_lag**(par.alpha) * N**(-par.alpha)

    else:

        raise Exception('unknown type of production function')

    return Y,rk,w            

def broyden_solver(f,x0,jac,tol=1e-8,maxiter=100,do_print=False):
    """ numerical equation system solver using the broyden method 
    
        f (callable): function return errors in equation system
        jac (ndarray): initial jacobian
        tol (float,optional): tolerance
        maxiter (int,optional): maximum number of iterations
        do_print (bool,optional): print progress

    """

    # a. initial
    x = x0.ravel()
    y = f(x)

    # b. iterate
    for it in range(maxiter):
        
        # i. current difference
        abs_diff = np.max(np.abs(y))
        if do_print: print(f' it = {it:3d} -> max. abs. error = {abs_diff:12.8f}')

        if abs_diff < tol: return x
        
        # ii. new x
        dx = np.linalg.solve(jac,-y)
        assert not np.any(np.isnan(dx))
        
        # iii. evaluate
        ynew = f(x+dx)
        dy = ynew-y
        jac = jac + np.outer(((dy - jac @ dx) / np.linalg.norm(dx)**2), dx)
        y = ynew
        x += dx
            
    else:

        raise ValueError(f'no convergence after {maxiter} iterations')        
    

