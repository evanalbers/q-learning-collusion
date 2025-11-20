""" file containing code for running experiments 

General flow:

1. design and initialize agents 
2. design, enumerate environment params
3. Run train_agents, passing agents and env to func. 
4. conduct visualization using data. 

"""
from agent import Agent
from env import StandardMarketEnv
from train import train_agents, fast_episode
import numpy as np
from tqdm import tqdm

alphas = [0.25, 0.25]
betas = [0.2e-5, 0.2e-5]
deltas = [0.95, 0.95]
a_arr = [0, 2, 2]
costs = [1, 1]
mu = 0.25
exts = [0.1, 0.1]
nash = [1.47293, 1.47293]
coop = [1.92498, 1.92498]
m = 15



def calc_price_ranges(nash, coop, ext, m):
    """ calculates price ranges (action space). 
    Params
    ------
    nash : array
        array of Bertrand Nash equilibrium prices for agents in sim. 
    coop : array
        array of perfect cooperation prices for agents in sim.
    ext : float
        array of extension parameters - determines upper and lower bounds 
    """

    price_ranges = []

    for agent in range(nash.size):
        price_ranges.append(np.linspace(start=nash[agent] - (ext[agent] * (coop[agent] - nash[agent])),
                                      stop=coop[agent] + (ext[agent] * (coop[agent] - nash[agent])),
                                      num=m))
    
    return np.array(price_ranges)


def run_session(num_agents, alphas, betas, deltas,
                a_arr, costs, mu, exts, nash, coop, m):
    """ A script to replicate the results of the initial paper """

    # calc prices for reward function
    prices = calc_price_ranges(np.array(nash),
                                np.array(coop),
                                np.array(exts),
                                m=m)

    actions = [np.arange(start=0, stop=15, step=1, dtype=int),
               np.arange(start=0, stop=15, step=1, dtype=int)]
    

    # assembling a_arr, cost array
    a_arr = np.array(a_arr)
    costs = np.array(costs)

    # setting up agents 

    n_states = (15, 15) # 15 possible prices, therefore 15 possible 
                       # for each agents
    n_actions = 15 # 15 possible prices for each agent

    agents = []

    # other parameters are default
    for a in range(num_agents):
        agents.append(Agent(n_states, n_actions, learning_rate=alphas[a],
                            discount=deltas[a], beta=betas[a]))
    
    # setting up env. 

    env = StandardMarketEnv(a_arr, costs, mu, 
                            num_agents=2, actions=actions, rewards_matrix=prices)

    # Training agents in env. 

    train_agents(agents, env)

def fast_session(num_agents, n_actions, n_states, alphas, betas, deltas,
                a_arr, costs, mu, exts, nash, coop, m):
    """ leverages fast training loop for speed """

    # initializing agents as q_tables 
    q_tables = []
    for agent in range(num_agents):
        q_tables.append(np.random.uniform(low=-0.01, high=0.01,   
                                            size=(*n_states, n_actions)))
    q_tables = np.array(q_tables)

    print(q_tables.shape)

    # r_matrix
    prices = calc_price_ranges(np.array(nash),
                                np.array(coop),
                                np.array(exts),
                                m=m)


    q_tables, step_conv = fast_episode(q_tables, num_agents, n_actions, betas, a_arr, costs,
                mu, prices, deltas, alphas, 1000000000, 1e5)
    
    return q_tables, step_conv



if __name__ == "__main__":

    # run_session(2, alphas, betas, deltas, a_arr, costs, mu, exts, nash, coop, m)

    session_pbar = tqdm(range(1000), desc="Number of Sessions")
    for episode in session_pbar:

        q_tables, step_conv = fast_session(2, 15, (15, 15), alphas, betas, deltas, a_arr,
                    costs, mu, exts, nash, coop, m)
        
        print("Converged in " + str(step_conv) + " steps")