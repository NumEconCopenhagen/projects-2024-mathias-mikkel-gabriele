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
        par.i = np.array([1,2,3,4,5,6,7,8,9,10])
        par.c = 1

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
    
    def prior_expec1(self, seed=None):
        '''
        Calculates prior expected utility based on friends
        '''
        par = self.par

        if seed is not None:
            np.random.seed(seed)

        eps = {}
        eps_self = {}
        prior_expec_util = np.zeros(par.N)
        choose_career = np.zeros(par.N)
        real_util = np.zeros(par.N)

        for k in range(par.K):
            for i in range(1,11,1):
                # do the J*i random draws for friends noise
                eps[i] = np.random.normal(0, par.sigma, par.J*i)
                # do the J random draws for own noise
                eps_self[i] = np.random.normal(0, par.sigma, par.J)

                # calculate the prior expected utility
                for j in range(par.J):
                    prior_expec_util[i,j] = 1/i * (i*par.v[j]+sum(eps[i]))

                # choose career based on max prior expected utility
                choose_career[i] = np.argmax(prior_expec_util[i])

                # calculate the real utility of the chosen career
                real_util[i] = choose_career[i] + eps_self[i]

        return prior_expec_util, choose_career, real_util
    
    def prior_expec(self, seed=None):
        par = self.par

        if seed is not None:
            np.random.seed(seed)
        
        eps_self = np.zeros((par.N, par.J))
        prior_expec_util = np.zeros((par.N, par.J))
        choose_career = np.zeros(par.N, dtype=int)
        real_util = np.zeros(par.N)
        friends_noise = np.zeros(par.J)

        for i in range(par.N):
            friends_noise = np.random.normal(0, par.sigma, par.J * (i + 1))
            eps_self[i] = np.random.normal(0, par.sigma, par.J)

            for j in range(par.J):
                prior_expec_util[i, j] = par.v[j] + np.mean(np.random.normal(0, par.sigma, i+1))
            
            choose_career[i] = np.argmax(prior_expec_util[i])
            real_util[i] = par.v[choose_career[i]] + eps_self[i, choose_career[i]]
        return prior_expec_util, choose_career, real_util
    
    def visualize_results(self, choices, avg_subjective_utilities, avg_realized_utilities):
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))

        # Share of graduates choosing each career
        for j in range(self.par.J):
            axs[0].bar(choices[:, j], range(1, self.par.N + 1), label=f'Career {j + 1}')
        axs[0].set_title('Share of Graduates Choosing Each Career')
        axs[0].set_xlabel('Graduate Index')
        axs[0].set_ylabel('Number of Graduates')
        axs[0].legend()

        # Average subjective expected utility and realized utility
        axs[1].plot(range(1, self.par.N + 1), avg_subjective_utilities, label='Avg. Subjective Expected Utility')
        axs[1].plot(range(1, self.par.N + 1), avg_realized_utilities, label='Avg. Realized Utility')
        axs[1].set_title('Average Subjective Expected Utility and Realized Utility')
        axs[1].set_xlabel('Graduate Index')
        axs[1].set_ylabel('Utility')
        axs[1].legend()

        plt.tight_layout()
        plt.show()