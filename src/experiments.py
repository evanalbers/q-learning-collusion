""" file containing code for running experiments 

General flow:

1. design and initialize agents 
2. design, enumerate environment params
3. Run train_agents, passing agents and env to func. 
4. conduct visualization using data. 

"""

from train import fast_session
import numpy as np
from tqdm import tqdm
import h5py
from collections import namedtuple

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

Experiment_Params = namedtuple('Params', ('num_agents', 'num_actions', 'num_demands', 'alphas', 
                                          'betas', 'deltas', 'a_arr', 'costs', 'mu', 'exts',
                                           'nash', 'coop', 'steps_per_episode',
                                            'episodes_per_session', 'r_matrix'))


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

def generate_parameters():
    """ generates set of parameters for testing """

    num_actions = 15

    parameter_set = []

    prices = calc_price_ranges(np.array(nash),
                                np.array(coop),
                                np.array(exts),
                                m=num_actions)
    
    alpha_range = np.linspace(0.025, 0.25, 100)
    beta_range = np.linspace(0, 2e-5, 100)

    for alpha in alpha_range:
        for beta in beta_range: 
            params = Experiment_Params(
            num_agents=2,
            num_actions=num_actions,
            num_demands=5,
            alphas=[alpha, alpha],
            betas=[beta, beta],
            deltas=[0.95, 0.95],
            a_arr = [0, 2, 2],
            costs=[1, 1],
            mu=0.25,
            exts=[0.1, 0.1],
            nash=[1.47293, 1.47293],
            coop=[1.92498, 1.92498],
            steps_per_episode=25000,
            episodes_per_session=10000,
            r_matrix=prices)
            parameter_set.append(params)

    return parameter_set

if __name__ == "__main__":

    parameter_set = generate_parameters()

    with h5py.File('agentdata.h5', 'w') as f:   

        experiment_pbar = tqdm(range(5000, len(parameter_set)), desc="Parameter set #...", position=0)

        for experiment in experiment_pbar:

            session_pbar = tqdm(range(10), desc="Number of Sessions", position=1, leave=False)
            for session in session_pbar:

                params = parameter_set[experiment]

                # initializing agents as q_tables 
                q_tables = []
                for agent in range(params.num_agents):
                    q_tables.append(np.random.uniform(low=-0.01, high=0.01,   
                                                        size=(params.num_actions, params.num_actions, params.num_demands)))
                q_tables = np.array(q_tables)

                q_tables, step_conv, action_data, reward_data, demand_data = fast_session(q_tables, params)
                    
                f.create_dataset(f"actions_{session}", data=action_data.T)
                f.create_dataset(f"rewards_{session}", data=reward_data.T)
                f.create_dataset(f"demands_{session}", data = demand_data.T)