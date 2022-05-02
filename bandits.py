import numpy as np 

class eps_greedy_bandit:
    '''
    epsilon-greedy k-bandit problem
    
    Inputs
    =====================================================
    k: number of arms (int)
    eps: probability of random action 0 < eps < 1 (float)
    iters: number of steps (int)
    mu: set the average rewards for each of the k-arms.
        Set to "random" for the means to be selected from
        a normal distribution with mean = 0. 
        Set to "sequence" for the means to be ordered from 
        0 to k-1 by 0.
        Set to "sequence2" for the means to be orderes from
        0 to (k-1)/2 by 0.5.
        Pass a list or array of length = k for user-defined
        values.
    '''
    
    def __init__(self, k, eps, iters, mu='random'):
        # Number of arms
        self.k = k
        # Search probability
        self.eps = eps
        # Number of iterations
        self.iters = iters
        # Step count
        self.n = 0
        # Step count for each arm
        self.k_n = np.zeros(k)
        # Total mean reward
        self.mean_reward = 0
        # Reward
        self.reward = np.zeros(iters)
        # Mean reward for each arm
        self.k_reward = np.zeros(k)

        # Define mean for each arm
        if type(mu) == list or type(mu).__module__ == np.__name__:
            # User-defined averages            
            self.mu = np.array(mu)
        elif mu == 'random':
            # Draw means from probability distribution
            self.mu = np.random.normal(0, 1, k)
        elif mu == 'sequence':
            # Increase the mean for each arm by one
            self.mu = np.linspace(0, k-1, k)
        elif mu=='sequence2':
            # Increase the mean for each arm by 0.5
            self.mu = np.linspace(0, k-1, k)/2
        
    def pull(self):
        # Generate random number
        p = np.random.rand()

        if self.eps == 0 and self.n == 0:
            # First move and 0-greedy strategy
            a = np.random.choice(self.k)
        elif p < self.eps:
            # Randomly select an action with probability eps
            a = np.random.choice(self.k)
        else:
            # Take greedy action with probability 1-eps
            a = np.argmax(self.k_reward)
            
        reward = np.random.normal(self.mu[a], 1)
        
        # Update counts
        self.n += 1
        self.k_n[a] += 1
        
        # Update total
        self.mean_reward = self.mean_reward + (
            reward - self.mean_reward) / self.n
        
        # Update results for a_k
        self.k_reward[a] = self.k_reward[a] + (
            reward - self.k_reward[a]) / self.k_n[a]
        
    def run(self):
        for i in range(self.iters):
            self.pull()
            self.reward[i] = self.mean_reward
            
    def reset(self):
        # Resets results while keeping settings
        self.n = 0
        self.k_n = np.zeros(k)
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        self.k_reward = np.zeros(k)


class eps_decay_bandit:
    '''
    epsilon-decay k-bandit problem
    
    Inputs
    =====================================================
    k: number of arms (int)
    iters: number of steps (int)
    beta: scaling parameter (float)
    mu: set the average rewards for each of the k-arms.
        Set to "random" for the means to be selected from
        a normal distribution with mean = 0. 
        Set to "sequence" for the means to be ordered from 
        0 to k-1 by 0.
        Set to "sequence2" for the means to be orderes from
        0 to (k-1)/2 by 0.5.
        Pass a list or array of length = k for user-defined
        values.
    '''
    
    def __init__(self, k, iters, beta=0.9, mu='random'):
        # Number of arms
        self.k = k
        # Scalaing parameter
        self.beta = beta
        # Number of iterations
        self.iters = iters
        # Step count
        self.n = 0
        # Step count for each arm
        self.k_n = np.zeros(k)
        # Total mean reward
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        # Mean reward for each arm
        self.k_reward = np.zeros(k)
        
        if type(mu) == list or type(mu).__module__ == np.__name__:
            # User-defined averages            
            self.mu = np.array(mu)
        elif mu == 'random':
            # Draw means from probability distribution
            self.mu = np.random.normal(0, 1, k)
        elif mu == 'sequence':
            # Increase the mean for each arm by one
            self.mu = np.linspace(0, k-1, k)
        elif mu == 'sequence2':
            # Increase the mean for each arm by one
            self.mu = np.linspace(0, k-1, k)/2
        
    def pull(self):
        # Generate random number
        p = np.random.rand()
        
        if p < 1 / (1 + self.n*self.beta):
            # Randomly select an action
            a = np.random.choice(self.k)
        else:
            # Take greedy action
            a = np.argmax(self.k_reward)
            
        reward = np.random.normal(self.mu[a], 1)
        
        # Update counts
        self.n += 1
        self.k_n[a] += 1
        
        # Update total
        self.mean_reward = self.mean_reward + (
            reward - self.mean_reward) / self.n
        
        # Update results for a_k
        self.k_reward[a] = self.k_reward[a] + (
            reward - self.k_reward[a]) / self.k_n[a]
        
    def run(self):
        for i in range(self.iters):
            self.pull()
            self.reward[i] = self.mean_reward
            
    def reset(self):
        # Resets results while keeping settings
        self.n = 0
        self.k_n = np.zeros(k)
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        self.k_reward = np.zeros(k)


class softmax_bandit:
    '''
    SoftMax k-bandit problem
    
    Inputs
    =====================================================
    k: number of arms (int)
    iters: number of steps (int)
    T: temperature 0 <= T (float)
    mu: set the average rewards for each of the k-arms.
        Set to "random" for the means to be selected from
        a normal distribution with mean = 0. 
        Set to "sequence" for the means to be ordered from 
        0 to k-1 by 0.
        Set to "sequence2" for the means to be orderes from
        0 to (k-1)/2 by 0.5.
        Pass a list or array of length = k for user-defined
        values.
    '''
    
    def __init__(self, k, iters, T=0.1, mu='random'):
        # Number of arms
        self.k = k
        # Number of iterations
        self.iters = iters
        # Step count
        self.n = 0
        # Step count for each arm
        self.k_n = np.zeros(k)
        # Total mean reward
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        # Mean reward for each arm
        self.k_reward = np.zeros(k)
        # Temperature
        self.T = T
        # Selection probabilities
        self.probabilities = np.zeros(k)

        # Define mean for each arm
        if type(mu) == list or type(mu).__module__ == np.__name__:
            # User-defined averages            
            self.mu = np.array(mu)
        elif mu == 'random':
            # Draw means from probability distribution
            self.mu = np.random.normal(0, 1, k)
        elif mu == 'sequence':
            # Increase the mean for each arm by one
            self.mu = np.linspace(0, k-1, k)
        elif mu == 'sequence2':
            # Increase the mean for each arm by one
            self.mu = np.linspace(0, k-1, k)/2
        
    def pull(self):
        # Probabilities distributed according to mean reward
        self.probabilities= np.exp(self.k_reward/self.T)
        self.probabilities = self.probabilities/np.sum(self.probabilities)
        probabilities_cmsm = np.cumsum(self.probabilities)

        # With Boltzman probability select an arm
        p=np.random.rand()
        probabilities_cmsm_b = probabilities_cmsm[probabilities_cmsm>p][0]
        a = np.argmax(probabilities_cmsm == probabilities_cmsm_b)
        
        reward = np.random.normal(self.mu[a], 1)
        
        # Update counts
        self.n += 1
        self.k_n[a] += 1
        
        # Update total
        self.mean_reward = self.mean_reward + (
            reward - self.mean_reward) / self.n
        
        # Update results for a_k
        self.k_reward[a] = self.k_reward[a] + (
            reward - self.k_reward[a]) / self.k_n[a]
        
    def run(self):
        for i in range(self.iters):
            self.pull()
            self.reward[i] = self.mean_reward
            
    def reset(self):
        # Resets results while keeping settings
        self.n = 0
        self.k_n = np.zeros(k)
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        self.k_reward = np.zeros(k)
        self.probabilities = np.zeros(k)


class annealing_softmax_bandit:
    '''
    SoftMax k-bandit problem using Simulated 
    
    Inputs
    =====================================================
    k: number of arms (int)
    iters: number of steps (int)
    T: temperature 0 <= T (float)
    mu: set the average rewards for each of the k-arms.
        Set to "random" for the means to be selected from
        a normal distribution with mean = 0. 
        Set to "sequence" for the means to be ordered from 
        0 to k-1 by 0.
        Set to "sequence2" for the means to be orderes from
        0 to (k-1)/2 by 0.5.
        Pass a list or array of length = k for user-defined
        values.
    '''
    
    def __init__(self, k, iters, mu='random',cooling='log'):
        # Number of arms
        self.k = k
        # Number of iterations
        self.iters = iters
        # Step count
        self.n = 0
        # Step count for each arm
        self.k_n = np.zeros(k)
        # Total mean reward
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        # Mean reward for each arm
        self.k_reward = np.zeros(k)
        # Temperature
        self.T = 1.2
        # Selection probabilities
        self.probabilities = np.zeros(k)
        # Colling the temperature
        self.cooling = 'log'

        # Define mean for each arm
        if type(mu) == list or type(mu).__module__ == np.__name__:
            # User-defined averages            
            self.mu = np.array(mu)
        elif mu == 'random':
            # Draw means from probability distribution
            self.mu = np.random.normal(0, 1, k)
        elif mu == 'sequence':
            # Increase the mean for each arm by one
            self.mu = np.linspace(0, k-1, k)
        elif mu == 'sequence2':
            # Increase the mean for each arm by one
            self.mu = np.linspace(0, k-1, k)/2
        
    def pull(self):

        # Update temperature
        self.update_temperature()

        # Probabilities distributed according to mean reward
        self.probabilities= np.exp(self.k_reward/self.T)
        self.probabilities = self.probabilities/np.sum(self.probabilities)
        probabilities_cmsm = np.cumsum(self.probabilities)

        # With Boltzman probability select an arm
        p=np.random.rand()
        probabilities_cmsm_b = probabilities_cmsm[probabilities_cmsm>p][0]
        a = np.argmax(probabilities_cmsm == probabilities_cmsm_b)
        
        reward = np.random.normal(self.mu[a], 1)
        
        # Update counts
        self.n += 1
        self.k_n[a] += 1
        
        # Update total
        self.mean_reward = self.mean_reward + (
            reward - self.mean_reward) / self.n
        
        # Update results for a_k
        self.k_reward[a] = self.k_reward[a] + (
            reward - self.k_reward[a]) / self.k_n[a]
        
    def run(self):
        for i in range(self.iters):
            self.pull()
            self.reward[i] = self.mean_reward
            
    def reset(self):
        # Resets results while keeping settings
        self.n = 0
        self.k_n = np.zeros(k)
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        self.k_reward = np.zeros(k)
        self.probabilities = np.zeros(k)
        self.T = 1.2

    def update_temperature(self):
        t = 1 + self.n

        if self.cooling == 'log':
            self.T = self.T + 1/np.log(t+0.0000001)
        elif self.coooling == 'abs':
            self.T = self.T+ 1/t
        elif self.cooling == 'random':
            r= np.random.rand(1)
            self.T = self.T +1/(r+0.0000001)