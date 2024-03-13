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

        return eps1,eps2
    
    # Social planner solver
    def solve_central(self):
        par = self.par
        
        # objective function to be minimized
        obj_fun_central = lambda x: -(self.utility_A(x[0],x[1])+self.utility_B((1-x[0]),(1-x[1])))
        
        # bounds and counstraints
        constraints = ({'type': 'ineq', 'fun': lambda x: x[0]-par.w1A+(1-x[0])-(1-par.w1A)})
        bounds = ((0,1),(0,1))
        
        # call solver
        initial_guess = [par.w1A,par.w2A]
        sol = optimize.minimize(obj_fun_central,initial_guess,method='SLSQP',bounds=bounds,constraints=constraints)
        
        # print solution
        return print(f'x1A = {sol.x[0]} x2A = {sol.x[1]}, U_central = {-obj_fun_central((sol.x[0],sol.x[1]))}')