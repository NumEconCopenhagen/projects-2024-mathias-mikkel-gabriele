from types import SimpleNamespace
import numpy as np
import scipy as sp

class ProductionCO2Class():
    """
    Class to solve the production economy model with C02 tax in problem 1.
    """
    def __init__(self):

        """
        Initialize class and set parameter values
        """

        par = self.par = SimpleNamespace()
        
        # firms
        par.A = 1.0
        par.gamma = 0.5

        # households
        par.alpha = 0.3
        par.nu = 1.0
        par.epsilon = 2.0

        # government
        par.tau = 0.0
        par.T = 0.0

        # question 3
        par.kappa = 0.1

        # solution
        sol = self.sol = SimpleNamespace()


    def firm1(self, p1):
        """
        Finds the optimal behavior of firm 1

        Args: price of good 1
        """
        
        par = self.par
        l1 = (p1*par.A*par.gamma)**(1/(1-par.gamma)) # labor demand
        y1 = par.A*l1**par.gamma # production of good 1
        pi1 = (1-par.gamma)/par.gamma*(p1*par.A*par.gamma)**(1/(1-par.gamma)) # implied profits
        return l1,y1,pi1

    def firm2(self,p2):
        """
        Finds the optimal behavior of firm 2

        Args: price of good 2
        """

        par = self.par
        l2 = (p2*par.A*par.gamma)**(1/(1-par.gamma)) # labor demand
        y2 = par.A*l2**par.gamma # production of good 2
        pi2 = (1-par.gamma)/par.gamma*(p2*par.A*par.gamma)**(1/(1-par.gamma)) # implied profits
        return l2,y2,pi2

    def optimal_c1(self,l,p1,pi):
        """
        Finds the consumer's optimal demand for good 1 

        Args: labor supply, price of good 1 and (total) profits
        """
        
        par = self.par
        return par.alpha*(l+par.T+pi)/p1

    def optimal_c2(self,l,p2,pi):
        """
        Finds the consumer's optimal demand for good 2 

        Args: labor supply, price of good 2 and (total) profits
        """

        par = self.par
        return (1-par.alpha)*(l+par.T+pi)/(p2+par.tau)

    def optimal_l(self, p1, p2, pi):
        """
        Finds the consumer's optimal behavior

        Args: price of good 1 and good 2 and (total) profits
        """

        par = self.par
        obj = lambda l: -(np.log(self.optimal_c1(l, p1, pi)**par.alpha * self.optimal_c2(l, pi, p2)**(1 - par.alpha)) - par.nu * (l**(1 + par.epsilon) / (1 + par.epsilon)))
        res = sp.optimize.minimize_scalar(obj, bounds=(0, 1), method="bounded") # minimize objective function
        l_star = res.x # optimal labor supply
        c1_star = self.optimal_c1(l_star, p1, pi) # optimal consumption of good 1
        c2_star = self.optimal_c2(l_star, p2, pi) # optimal consumption of good 2
        return l_star, c1_star, c2_star
    
    def utility_function(self,l_star, c1_star, c2_star):
        """
        Consumer's utility function (used in question 3)

        Args: Optimal labor supply, consumption of good 1 and good 2
        """
        par = self.par
        U = np.log(c1_star**par.alpha * c2_star**(1 - par.alpha)) - par.nu * (l_star**(1 + par.epsilon) / (1 + par.epsilon))
        return U
    
    def evaluate_equilibrium(self,prices,do_print=False):
        """
        Evaluates equilibrium by finding market clearing errors

        Args: prices (a vector including p1 and p2)
        """

        p1, p2 = prices
        par = self.par

        # Optimal firm behavior
        l1_star,y1_star,pi1_star=self.firm1(p1)
        l2_star,y2_star,pi2_star=self.firm2(p2)
        pi_star=pi1_star+pi2_star

        # Optimal consumer behavior
        l_star,c1_star,c2_star = self.optimal_l(p1,p2,pi_star)

        # Market clearing
        good1_clearing = c1_star-y1_star
        good2_clearing = c2_star-y2_star
        labor_market_clearing = l_star-(l1_star+l2_star)

        # Print statement
        if do_print is True:
            print(f'For p1={p1:.4f}, p2={p2:.4f}')
            print(f'Market clearing errors are:')
            print(f'Good1: {good1_clearing:.6f}')
            print(f'Good2: {good2_clearing:.6f}')
            print(f'Labor market: {labor_market_clearing:.6f}')
            print('------------------------------------')
        return np.abs(good1_clearing)+np.abs(good2_clearing)+np.abs(labor_market_clearing)
    
    def find_prices(self, do_print=False):
        """
        Finds prices that clears markets
        """
        sol = self.sol
        objective = lambda prices: self.evaluate_equilibrium(prices)

        result = sp.optimize.minimize(objective, [1.0, 1.0], method='Nelder-Mead')

        # Store results
        if result.success:
            sol.prices = result.x
            if do_print:
                self.evaluate_equilibrium([sol.prices[0], sol.prices[1]], do_print=True)
        else:
            raise ValueError('Price optimization failed:', result.message)
        
    def social_welfare_function(self, tau):
        """
        Social welfare function, that takes into account the negative utility of the social cost of carbon associated with the production of good 2.
        
        Args: tau (the tax rate on production of good 2)
        """

        par = self.par
        sol = self.sol
        par.tau = tau
        print(f'Evaluating social welfare function at tau={tau:.4f}')

        # Find equilibrium prices
        try:
            self.find_prices()
            p1, p2 = sol.prices
        except ValueError as e:
            return np.inf

        # Optimal firm and consumer behavior
        l1_star, y1_star, pi1_star = self.firm1(p1)
        l2_star, y2_star, pi2_star = self.firm2(p2)
        pi_star = pi1_star + pi2_star

        l_star, c1_star, c2_star = self.optimal_l(p1, p2, pi_star)

        # Update T and optimal consumption
        par.T = par.tau * c2_star
        l_star, c1_star, c2_star = self.optimal_l(p1, p2, pi_star)

        # Calculate utility and social welfare function
        U = self.utility_function(l_star, c1_star, c2_star)
        
        SWF = U - par.kappa * y2_star
        print(f'SWF={SWF:.2f}')
        return -SWF  # Minimize the negative of SWF to maximize SWF

    def find_optimal_tau(self):
        """
        Function that optimizes tau, to maximize the social welfare function
        """
        par = self.par
        result = sp.optimize.minimize_scalar(self.social_welfare_function, bounds=(0.0, 1.0), method='bounded')
    
        if result.success:
            optimal_tau = result.x
            print(f'Optimal tau: {optimal_tau:.4f}')
            print(f'Associated T: {par.T:.4f}')
        else:
            print('Optimization failed:', result.message)
            return None