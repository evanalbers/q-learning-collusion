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
# 
@njit
def fast_reward(a_arr, state, agent, costs, mu, r_matrix):

    # r_matrix is a matrix of prices for each action, in this case.
    # dim are num agents x actions. 

    prices = np.zeros(len(state))
    for i in range(len(state)):
        prices[i] = r_matrix[i, state[i]]
    
    return (r_matrix[agent, state[agent]] - costs[agent]) * fast_quantity(a_arr, prices, agent, mu), r_matrix[agent, state[agent]] - costs[agent]

@njit 
def fast_monopolist_reward(a_arr, state, agent, costs, mu, r_matrix):
    """ calculates reward for each agent separately as though they were
         monopolists in differing markets 
         
        """
    
    a_arr = np.array([a_arr[0], a_arr[agent + 1]])

    return (r_matrix[agent, state[agent]] - costs[agent]) * fast_monopolist_quantity(a_arr[0], a_arr[agent], state[agent], mu)



