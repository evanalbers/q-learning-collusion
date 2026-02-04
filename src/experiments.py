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
import h5py

alphas = [0.05, 0.05]
betas = [0.5e-5, 0.5e-5]
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

def fast_session(num_agents, n_actions, n_demands, alphas, betas, deltas,
                a_arr, costs, mu, exts, nash, coop, m):
    """ leverages fast training loop for speed """

    # initializing agents as q_tables 
    q_tables = []
    for agent in range(num_agents):
        q_tables.append(np.random.uniform(low=-0.01, high=0.01,   
                                            size=(n_actions, n_actions, n_demands)))
    q_tables = np.array(q_tables)

    # print(q_tables.shape)

    # r_matrix
    prices = calc_price_ranges(np.array(nash),
                                np.array(coop),
                                np.array(exts),
                                m=m)



    q_tables, step_conv, action_data, reward_data, demand_data = fast_episode(q_tables, num_agents, n_actions, betas, a_arr, costs,
            mu, prices, deltas, alphas, 10000000, 5)
        
    for opp in [0, 7, 14]:
        actions_by_demand = [np.argmax(q_tables[0, opp, d, :]) for d in range(5)]
        print(f"opp={opp}: {actions_by_demand}")

    
    return q_tables, step_conv, action_data, reward_data, demand_data



if __name__ == "__main__":

    # run_session(2, alphas, betas, deltas, a_arr, costs, mu, exts, nash, coop, m)

    with h5py.File('agentdata.h5', 'w') as f:    

        session_pbar = tqdm(range(10), desc="Number of Sessions")
        for episode in session_pbar:

            q_tables, step_conv, action_data, reward_data, demand_data = fast_session(2, 15, 5, alphas, betas, deltas, a_arr,
                    costs, mu, exts, nash, coop, m)
                
            f.create_dataset(f"actions_{episode}", data=action_data.T)
            f.create_dataset(f"rewards_{episode}", data=reward_data.T)
            f.create_dataset(f"demands_{episode}", data = demand_data.T)
        # print("Converged in " + str(step_conv) + " steps")