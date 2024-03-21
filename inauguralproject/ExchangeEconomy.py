from types import SimpleNamespace
from scipy import optimize
import numpy as np

class ExchangeEconomyClass:

    def __init__(self):

        par = self.par = SimpleNamespace()

        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3

        # b. endowments
        par.w1A = 0.8
        par.w2A = 0.3

        # parameter for storing results
        sol = self.sol = SimpleNamespace()
        sol.x1 = np.nan
        sol.x2 = np.nan
        sol.u = np.nan
        sol.p = np.nan

    def utility_A(self,x1A,x2A):
        par = self.par
        return x1A**(par.alpha)*x2A**(1-par.alpha)

    def utility_B(self,x1B,x2B):
        par = self.par
        return x1B**(par.beta)*x2B**(1-par.beta)

    def demand_A(self,p1):
        par = self.par
        x1A = par.alpha*((p1*par.w1A+par.w2A)/p1)
        x2A = (1-par.alpha)*(p1*par.w1A+par.w2A)
        return x1A,x2A

    def demand_B(self,p1):
        par = self.par
        x1B = par.beta*((p1*(1-par.w1A)+(1-par.w2A))/p1)
        x2B = (1-par.beta)*(p1*(1-par.w1A)+(1-par.w2A))
        return x1B,x2B

    def check_market_clearing(self,p1):

        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1, eps2
    
    # Function that solves
    def solve(self, type="central"):
        par = self.par
        sol = self.sol

        # objective function to be minimized and constraints, depending on type chosen
        if type == "central":
            obj_fun = lambda x: -(self.utility_A(x[0],x[1])+self.utility_B((1-x[0]),(1-x[1])))       # The social planner maximises aggregate utility subject to available supply
            constraints = ({'type': 'ineq', 'fun': lambda x: x[0]-par.w1A+(1-x[0])-(1-par.w1A)})
        elif type == "mm":
            obj_fun = lambda x: -(self.utility_A(x[0],x[1]))        # A maximises their own utility, subject to the available supply and that B is not worse of than at the beginning
            constraints = ({'type': 'ineq', 'fun': lambda x: x[0]-par.w1A+(1-x[0])-(1-par.w1A)},{'type': 'ineq', 'fun': lambda x: self.utility_B(1-x[0],1-x[1])-self.utility_B(1-par.w1A,1-par.w2A)})
        elif type == 'market':                                      # The market finds the efficient allocation that makes the goods markets clear
            obj_fun = lambda x: np.sum(np.abs(self.check_market_clearing(x)))
        else:
            print('no type chosen')
    
        # bounds
        bounds = ((0,1),(0,1))
        
        # call solver
        if type == 'market':        # Market maker uses different solver as constraints are not needed, while additional code implemting p1 is.
            initial_prices = [1.0]
            res = optimize.minimize(obj_fun, initial_prices, method='Nelder-Mead')
            # store results
            p1 = res.x[0]
            x1A, x2A = self.demand_A(p1)
            x1B, x2B = self.demand_B(p1)
            sol.p = p1
            sol.x1 = x1A
            sol.x2 = x2A
            sol.u = self.utility_A(x1A, x2A) + self.utility_B(x1B, x2B)
            #print solution
            print(f'x1A = {sol.x1:.3f} x2A = {sol.x2:.3f}, U_{type} = {sol.u:.3f}, u_A = {self.utility_A(sol.x1,sol.x2):.3f}, u_B = {self.utility_B(1-sol.x1,1-sol.x2):.3f}, p = {sol.p:.3f}')
        else:       # Bot social planner and market maker optimizes utility under constraints
            initial_guess = [par.w1A,par.w2A]
            res = optimize.minimize(obj_fun,initial_guess,method='SLSQP',bounds=bounds,constraints=constraints)
            # save and print solution
            sol.x1 = res.x[0]
            sol.x2 = res.x[1]
            sol.u = -obj_fun((res.x[0],res.x[1]))
            print(f'x1A = {sol.x1:.3f} x2A = {sol.x2:.3f}, U_{type} = {sol.u:.3f}, u_A = {self.utility_A(sol.x1,sol.x2):.3f}, u_B = {self.utility_B(1-sol.x1,1-sol.x2):.3f}')
