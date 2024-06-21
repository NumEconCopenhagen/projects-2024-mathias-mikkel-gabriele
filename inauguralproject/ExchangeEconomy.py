from types import SimpleNamespace
from scipy import optimize
import numpy as np

class ExchangeEconomyClass:

    def __init__(self):
        """
        Initializes the model by creating namespaces for parameter values and solutions, and setting initial parameter values.
        """

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
        """
        Returns the utility for consumer A, given endowments of each good.
        Args: quantities of good 1 and 2 for consumer A
        """
        par = self.par
        return x1A**(par.alpha)*x2A**(1-par.alpha)

    def utility_B(self,x1B,x2B):
        """
        Returns the utility for consumer B, given endowments of each good.
        Args: quantities of good 1 and 2 for consumer B
        """
        par = self.par
        return x1B**(par.beta)*x2B**(1-par.beta)

    def demand_A(self,p1):
        """
        Returns consumer A's demand for good 1 and 2, given their prices
        Args: Price of good 1, p1. Price of good 2 is numeraire p2=1.
        """
        par = self.par
        x1A = par.alpha*((p1*par.w1A+par.w2A)/p1)
        x2A = (1-par.alpha)*(p1*par.w1A+par.w2A)
        return x1A,x2A

    def demand_B(self,p1):
        """
        Returns consumer B's demand for good 1 and 2, given their prices
        Args: Price of good 1, p1. Price of good 2 is numeraire p2=1.
        """
        par = self.par
        x1B = par.beta*((p1*(1-par.w1A)+(1-par.w2A))/p1)
        x2B = (1-par.beta)*(p1*(1-par.w1A)+(1-par.w2A))
        return x1B,x2B

    def check_market_clearing(self,p1):
        """
        Returns market clearing errors for good 1 and 2, given prices and the associated demands of each consumer
        Args: Price of good 1, p1.
        """
        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1, eps2
    
    # Function that solves
    """
    Solve function, that solves the equation systems of the exchange economy under different solution regimes.
    
    Args: Only 'type'. Initial endowments or preferences can be changed, by overwriting the values set when initializing the model.
    By chosing 'type' the function can take different regimes into account.
    "Central" solution refers to a social planner, trying to maximize aggregate utility
    "mm" is when consumer A is a market maker, and optimizes their own utility without making B worse off.
    "market" is the (Walras) market equilibrium, that solves by minimizing the market clearing errors, with prices and quantities respecting the initial endowments.

    Each solution is stored in the sol. parameter to be plotted later.

    Args
    """
    def solve(self, type="central"):
        par = self.par
        sol = self.sol

        # objective function to be minimized and constraints, depending on type chosen
        if type == "central":
            obj_fun = lambda x: -(self.utility_A(x[0],x[1])+self.utility_B((1-x[0]),(1-x[1])))       # The social planner maximises aggregate utility subject to available supply
            constraints = ({'type': 'ineq', 'fun': lambda x: x[0]-par.w1A+(1-x[0])-(1-par.w1A)}) # Constraint makes sure that quantity doesn't exceed available supply
        elif type == "mm":
            obj_fun = lambda x: -(self.utility_A(x[0],x[1]))        # A maximises their own utility, subject to the available supply and that B is not worse of than at the beginning
            constraints = ({'type': 'ineq', 'fun': lambda x: x[0]-par.w1A+(1-x[0])-(1-par.w1A)},{'type': 'ineq', 'fun': lambda x: self.utility_B(1-x[0],1-x[1])-self.utility_B(1-par.w1A,1-par.w2A)})  # Constraint makes sure that quantity doesn't exceed available supply and that B is not worse off that given their initial endowment.
        elif type == 'market':                                      # The market finds the efficient allocation that makes the goods markets clear
            obj_fun = lambda x: np.sum(np.abs(self.check_market_clearing(x))) # Objective function is now the absolute sum of the market clearing errors for good 1 and 2.
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
        else:       # Both social planner and market maker optimizes utility under constraints
            initial_guess = [par.w1A,par.w2A]
            res = optimize.minimize(obj_fun,initial_guess,method='SLSQP',bounds=bounds,constraints=constraints)
            # save and print solution
            sol.x1 = res.x[0]
            sol.x2 = res.x[1]
            sol.u = -obj_fun((res.x[0],res.x[1]))
            print(f'x1A = {sol.x1:.3f} x2A = {sol.x2:.3f}, U_{type} = {sol.u:.3f}, u_A = {self.utility_A(sol.x1,sol.x2):.3f}, u_B = {self.utility_B(1-sol.x1,1-sol.x2):.3f}')
