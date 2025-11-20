import numpy as np
import random
from numba import njit

@njit
def fast_choose_action(q_table, state, epsilon, actions):
    """ numba optimized for speed """
    if  np.random.random() < epsilon:
            return random.randint(0, actions - 1)
    else: 

        return np.argmax(q_table[state[0], state[1]])
    
@njit 
def fast_update(q_table, state, action, reward, next_state, lr, discount):
    """ numba optimized for speed. """

    current_q = q_table[state[0], state[1], action]

    target_q = reward + discount * np.max(q_table[next_state[0], next_state[1]])

    q_table[state[0], state[1], action] = current_q + lr * (target_q - current_q)

    return q_table

class Agent:
    """ code file containing Q-learning code. """
    def __init__(self, state_shape, n_actions, learning_rate=0.1, discount=0.95, beta=0.1):
        """ Initializes q-learning agent
        
        Params
        ------
        n_states : int
            Number of states in the environment
        
        n_actions : int
            number of possible actions in the env.

        learning_rate : float
            (Alpha) - how much to update Q matrix values by each step

        discount : float
            (Gamma) discount rate for future rewards

        beta : float
            exploration decay rate, epsilon is function of timestep and beta

        Returns
        -------
        agent : Agent
            Learning agent
        """

        self.n_states = state_shape
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount
        self.beta = beta
        self.epsilon = np.exp(beta * 0)

        self.q_table = np.random.uniform(low=-0.01, high=0.01, size=(*state_shape, n_actions))

        

    def choose_action(self, state, training=True):
        """ choose an action according to the Q-matrix """

        if training:
            return fast_choose_action(self.q_table, state, self.epsilon, self.n_actions)
        else: 

            return np.argmax(self.q_table[tuple(state)])
        
    def update(self, state, action, reward, next_state):

        self.q_table = fast_update(self.q_table, state, action, 
                                   reward, next_state, self.lr, self.gamma)

    def set_epsilon(self, timestep):
        """ sets epsilon value based on the timestep

        Params
        ------
        timestep : int
            timestep in the simulation

        Returns
        -------
        None 
        """

        self.epsilon = np.exp(-1 * self.beta * timestep)





