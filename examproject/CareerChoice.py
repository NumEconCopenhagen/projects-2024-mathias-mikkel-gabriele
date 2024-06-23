import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

class CareerClass:
    '''
    Class to solve the career choice model in problem 2.
    '''
    def __init__(self):
        '''
        Initialize class and set parameter values
        '''
        par = self.par = SimpleNamespace()

        # Set baseline parameters
        par.J = 3
        par.N = 10
        par.K = 10000
        par.i = np.array([1,2,3,4,5,6,7,8,9,10])
        par.c = 1

        par.F = np.arange(1,par.N+1,1)
        par.sigma = 2

        par.v = np.arange(1,par.J+1, 1)
        par.c = 1

    def expec_and_avg_utility(self, seed=None):
        '''
        Calculates the expected and average utility of a given career choice. 
        It is simulated K times, in the sense that we draw random noise 
        for each career K times.

        Args: Seed for random draws
        '''
        # use class initialization parameters
        par = self.par

        # set seed if provided, for random draws
        if seed is not None:
            np.random.seed(seed)

        # initialize arrays to store expected and average utility
        expec_utility = np.zeros(par.J)
        avg_utility = np.zeros(par.J)

        # draw random noise for each career choice and calculate expected and average utility
        for j in range(par.J):
            eps = np.random.normal(0, par.sigma, par.K)
            expec_utility[j] = par.v[j] + np.mean(eps)
            avg_utility[j] = par.v[j] + np.mean(eps)

        # print results        
        for j in range(par.J):
            print(f'Career {j+1} - Expected Utility: {expec_utility[j]:.4f}, Average Utility: {avg_utility[j]:.4f}')
    
    def simulate_career_choices(self, seed=None):
        '''
        Simulates and output the career choice of each graduate type K times using prior expectation based on friends. 

        Calculates and outputs prior expected utility based on friends and realized utility based on own noise term.

        Graphs the results as requested in question 2.2, though they are also returned as arrays for completeness.
        
        Args: Seed for random draws
        '''

        # use class initialization parameters
        par = self.par

        # set seed if provided, for random draws
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize arrays for storing results
        choices = np.zeros((par.K, par.N), dtype=int)
        prior_expected_utilities = np.zeros((par.K, par.N))
        realized_utilities = np.zeros((par.K, par.N))

        # Simulate career choice for each graduate type, k times and store results appropriately
        for k in range(par.K):
            for i in range(par.N):
                # Number of friends
                Fi = i + 1
                # Calculate prior expected utilities based on friends, for each career
                prior_utilities = np.array([np.mean(par.v[j] + np.random.normal(0, par.sigma, Fi)) for j in range(par.J)])
                # Choose career with highest expected utility
                chosen_career = np.argmax(prior_utilities)
                # Store results
                choices[k, i] = chosen_career
                prior_expected_utilities[k, i] = prior_utilities[chosen_career]
                # Calculate and store the realized utility based on their own noise term for the chosen career
                noise_terms = np.random.normal(0, par.sigma, par.J)
                realized_utilities[k, i] = par.v[chosen_career] + noise_terms[chosen_career]

        # Calculate averages
        average_prior_expected_utilities = np.mean(prior_expected_utilities, axis=0)
        average_realized_utilities = np.mean(realized_utilities, axis=0)

        # Plot results
        fig, axs = plt.subplots(2, 1, figsize=(7, 10))

        # Share of graduates choosing each career
        for j in range(par.J):
            share = np.mean(choices == j, axis=0)
            axs[0].plot(range(1, par.N+1), share, label=f'Career {j+1}')
        axs[0].set_xlabel('Graduate type & Number of Friends')
        axs[0].set_ylabel('Share Choosing Career')
        axs[0].legend()
        axs[0].set_title('Share of Graduates Choosing Each Career')

        # Average prior expected utility vs Average realized utility
        axs[1].plot(range(1, par.N+1), average_prior_expected_utilities, label='Average Prior Expected Utility')
        axs[1].plot(range(1, par.N+1), average_realized_utilities, label='Average Realized Utility')
        axs[1].set_xlabel('Graduate type & Number of Friends')
        axs[1].set_ylabel('Utility')
        axs[1].legend()
        axs[1].set_title('Average Prior Expected Utility vs. Average Realized Utility')

        plt.tight_layout()
        plt.show()

        # return key results for question 2.3
        return choices, average_prior_expected_utilities, average_realized_utilities, realized_utilities
    
    def simulate_switching_career(self, choices, realized_utilities, seed=None):
        '''
        Calculates the career choice of each graduate type K times with switching cost. 

        Outputs share of graduates switching careers and average new prior expected and realized utilities.

        Graphs results as requested in question 2.3, though they are also returned as arrays for completeness.
        
        Args: Previous choices and previous realized utilities. Could be implemented as a call. 
        '''

        # use class initialization parameters
        par = self.par

        # initialize arrays for storing new results
        new_choices = np.zeros((par.K, par.N), dtype=int)
        new_prior_expected_utilities = np.zeros((par.K, par.N))
        new_realized_utilities = np.zeros((par.K, par.N))
        switch_decisions = np.zeros((par.K, par.N), dtype=int)

        # loop over each graduate type and simulate career choice with switching cost
        for k in range(par.K):
            for i in range(par.N):
                # Number of friends
                Fi = i + 1
                # Initial career choice and realized utility from 2.2 simulation
                initial_choice = choices[k, i]
                initial_realized_utility = realized_utilities[k, i]

                # Calculate new prior expected utilities including switching cost
                new_prior_utilities = np.array([
                    initial_realized_utility if j == initial_choice else np.mean(par.v[j] + np.random.normal(0, par.sigma, Fi)) - par.c
                    for j in range(par.J)
                ])

                # Choose new career with highest expected utility
                new_chosen_career = np.argmax(new_prior_utilities)
                new_choices[k, i] = new_chosen_career
                new_prior_expected_utilities[k, i] = new_prior_utilities[new_chosen_career]

                # Realized utility for the new chosen career
                if new_chosen_career == initial_choice:
                    new_realized_utilities[k, i] = initial_realized_utility
                else:
                    new_noise_terms = np.random.normal(0, par.sigma, par.J)
                    new_realized_utilities[k, i] = par.v[new_chosen_career] + new_noise_terms[new_chosen_career]-par.c

                # Record if the graduate decided to switch careers
                switch_decisions[k, i] = (new_chosen_career != initial_choice)

        # Calculate averages
        average_new_prior_expected_utilities = np.mean(new_prior_expected_utilities, axis=0)
        average_new_realized_utilities = np.mean(new_realized_utilities, axis=0)
        # Calculate share of graduates switching careers which is the mean since its a binary variable
        average_switch_decisions = np.mean(switch_decisions, axis=0)

        # Plot new results
        fig, axs = plt.subplots(2, 1, figsize=(7, 10))

        # Share of graduates switching careers
        axs[0].plot(range(1, par.N+1), average_switch_decisions)
        axs[0].set_xlabel('Graduate type & Number of Friends')
        axs[0].set_ylabel('Share Switching Careers')
        axs[0].set_title('Share of Graduates Switching Careers')

        # Average expected and realized utilities after switching
        axs[1].plot(range(1, par.N+1), average_new_prior_expected_utilities, label='Average New Prior Expected Utility')
        axs[1].plot(range(1, par.N+1), average_new_realized_utilities, label='Average New Realized Utility')
        axs[1].set_xlabel('Graduate type & Number of Friends')
        axs[1].set_ylabel('Utility')
        axs[1].legend()
        axs[1].set_title('Average New Prior Expected Utility vs. Average New Realized Utility')

        plt.tight_layout()
        plt.show()

        # Return new results for fun, doesnt serve a purpose other than testing
        return new_choices, average_new_prior_expected_utilities, average_new_realized_utilities, switch_decisions