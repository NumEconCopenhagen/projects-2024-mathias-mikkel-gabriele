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

        return prior_expec_util, choose_career, real_util, eps_self
    
    def visualize_results(self, choices, avg_subjective_utilities, avg_realized_utilities):
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))

        career_counts = np.bincount(choices)

        career_labels = ['Career 1', 'Career 2', 'Career 3']

        axs[0].bar(career_labels, career_counts, color=['skyblue', 'green', 'red'])
        axs[0].set_ylabel('Number of Graduates')
        axs[0].set_title('Number of Graduates in Each Career Path')

        # Average subjective expected utility and realized utility
        axs[1].plot(range(1, self.par.N + 1), avg_subjective_utilities, label='Avg. Subjective Expected Utility')
        axs[1].plot(range(1, self.par.N + 1), avg_realized_utilities, label='Avg. Realized Utility')
        axs[1].set_title('Average Subjective Expected Utility and Realized Utility')
        axs[1].set_xlabel('Graduate Index')
        axs[1].set_ylabel('Utility')
        axs[1].legend()

        plt.tight_layout()
        plt.show()
    
    def career_switching(self, seed=None):
        par = self.par
        if seed is not None:
            np.random.seed(seed)
        new_chosen_career = np.zeros(par.N, dtype=int)
        new_real_util = np.zeros(par.N)
        switch_decisions = np.zeros(par.N)
        adjusted_priors = np.zeros((par.N, par.J))

        prior_expec_util, choose_career, real_util, eps_self = self.prior_expec()
        for i in range(par.N):
            initial_career = choose_career[i]
            initial_realized_utility = real_util[i]
            for j in range(par.J):
                initial_expected_utility = prior_expec_util[i, j]
                if j != initial_career:
                    adjusted_priors[i, j] = initial_expected_utility - par.c
                else:
                    adjusted_priors[i, j] = initial_realized_utility

            # Choose the career with the highest expected utility
            new_chosen_career[i] = np.argmax(adjusted_priors[i])
            new_real_util[i] = par.v[new_chosen_career[i]] + eps_self[i, new_chosen_career[i]]
            # Record if the graduate decided to switch careers
            switch_decisions[i] = 1 if new_chosen_career[i] != initial_career else 0
    
        return new_chosen_career, switch_decisions, adjusted_priors, new_real_util