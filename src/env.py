import numpy as np
from numba import njit

@njit
def fast_monopolist_quantity(a_0, a_agent, state, mu):
    """
    """

    numerator = np.exp((a_agent - state) / mu)
    denom = np.exp(a_0 / mu) + numerator

    return numerator / denom

@njit
def fast_quantity(a_arr, prices, agent, mu):
    """ calculates demand for a given good based on params 
    
    Params
    ------
    a_arr : array
        array of a values for each agent. "Product quality indices 
        that capture vertical integration." From economics literature.
    
    state : array
        prices at time t for each agent 
    
    agent : int
        agent for whom quantity is being calculated.
    
    mu : float
        index of horizontal differentiation. Perfect substitutes 
        obtained as mu -> 0.

    Returns
    -------
    quantity : float?
        quantity of good i at time t demanded (sold).
    """

    numerator = np.exp((a_arr[agent + 1] - prices[agent]) / mu)

    # optimizing this for Numba speed, explicit computation = fewer calls than np.sum
    denom = np.exp(a_arr[0] / mu)
    for i in range(len(prices)): 
        denom += np.exp((a_arr[i + 1] - prices[i]) / mu)

    return numerator / denom

@njit
def fast_reward(a_arr, state, agent, costs, mu, r_matrix):

    # r_matrix is a matrix of prices for each action, in this case.
    # dim are num agents x actions. 

    prices = np.zeros(len(state))
    for i in range(len(state)):
        prices[i] = r_matrix[i, state[i]]
    
    return (r_matrix[agent, state[agent]] - costs[agent]) * fast_quantity(a_arr, prices, agent, mu)

@njit 
def fast_monopolist_reward(a_arr, state, agent, costs, mu, r_matrix):
    """ calculates reward for each agent separately as though they were
         monopolists in differing markets 
         
        """
    
    a_arr = np.array([a_arr[0], a_arr[agent + 1]])

    return (r_matrix[agent, state[agent]] - costs[agent]) * fast_monopolist_quantity(a_arr[0], a_arr[agent], state[agent], mu)

class StandardMarketEnv:
    """ Market environment that agents act in. 
    Start: Random price initialization for each agent. 
    Goal: Run for # steps determined in training script. 
    Actions: [Propose some price in discrete price range]
    Space: <Price range, some # steps>
    """

    def __init__(self, a_arr, costs, mu, num_agents, actions, rewards_matrix):
        """ initializes the market env. 
        
        Params
        ------
        a_arr : array
            array of product quality indices intended to represent 
            vertical differentiation

        costs : array
            array of marginal costs for each agent 
        
        mu : float
            float representing the substitutabilty of each agent's product
            
        actions : array
            array of arrays representing actions available to each agent
        """

        self.num_agents = num_agents
        self.actions = actions
        self.rewards_matrix = rewards_matrix
        self.a_arr = a_arr
        self.costs = costs
        self.mu = mu
        self.market = np.ones(num_agents, dtype=int)

        # initializing random values for starting prices 
        for agent in range(num_agents):
            self.market[agent] = np.random.choice(self.actions[agent])
        

    def reset(self):
        for agent in range(self.num_agents):
            self.market[agent] = np.random.choice(self.actions[agent], 1)
        

    def step(self, agent, action):

        if action in self.actions[agent]:
            self.market[agent] = action

        # calculating reward here
        r = fast_reward(self.a_arr, self.market, agent, self.costs, self.mu, r_matrix=self.rewards_matrix)

        return self.market, r
        

    def get_available_actions(self):
        return self.actions


class DirectionalMarketEnv:
    """ Market environment whose state space is characterized by directional 
        price adjustment mechanisms 
        
    Start: Random price init. for each agent 
    Goal: Run for # of steps in training script
    Actions: [raise, hold, lower]
    Space: < Price range, some discrete # steps> 
    """

    def __init__(self):
        pass

    def reset(self):
        pass

    def step(self, agent, action):
        pass

    def get_available_actions(self):
        pass
