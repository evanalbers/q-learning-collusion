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
from tqdm_joblib import tqdm_joblib
from data_processing import conditional_mutual_info, rolling_conditional_mutual_information
from joblib import delayed, Parallel


exts = [0.1, 0.1]
nash = [1.47293, 1.47293]
coop = [1.92498, 1.92498]


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
    
    alpha_range = np.linspace(0.1, 0.2, 5)
    beta_range = np.linspace(2e-6, 1e-5, 5)

    # for alpha in alpha_range:
    #     for beta in beta_range: 
    params = Experiment_Params(
    num_agents=2,
    num_actions=num_actions,
    num_demands=5,
    alphas=[0.05, 0.05],
    betas=[0.5e-5, 0.5e-5],
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

    print("Price matrix (r_matrix):")
    print(params.r_matrix)

    return parameter_set

def run_session(params):
    """ runs session and returns results """

    # initializing agents as q_tables 
    q_tables = []
    for agent in range(params.num_agents):
        q_tables.append(np.random.uniform(low=-0.01, high=0.01,   
                                            size=(params.num_actions, params.num_demands, params.num_actions)))
    q_tables = np.array(q_tables)

    q_tables, step_conv, action_data, reward_data, demand_data = fast_session(q_tables, params)

    average_profit = reward_data[-10000:].mean()

    delta = (average_profit - params.nash[0]) / (params.coop[0] - params.nash[0])

    cmi_beginning = conditional_mutual_info(action_data[:10000, 0].astype(int), action_data[:10000, 1].astype(int), demand_data[:10000].astype(int))
    cmi_end = conditional_mutual_info(action_data[-10000:, 0].astype(int), action_data[-10000:, 1].astype(int), demand_data[-10000:].astype(int))
    print(action_data.shape)
    cmi_delta = cmi_end - cmi_beginning
    print(cmi_beginning)
    print(cmi_end)
    
    x = action_data[140000:150000, 0].astype(int)
    y = action_data[140000:150000, 1].astype(int)
    z = demand_data[140000:150000].astype(int)

    # Call conditional_mutual_info twice on the SAME data
    cmi_1 = conditional_mutual_info(x, y, z)
    cmi_2 = conditional_mutual_info(x, y, z)

    print(f"First call: {cmi_1}")
    print(f"Second call: {cmi_2}")

    cmi_rolling, centers = rolling_conditional_mutual_information(
    action_data[:, 0].astype(int),
    action_data[:, 1].astype(int),
    demand_data.astype(int),
    window_size=10000,
    step=10000  # Large step so we only get a few values
    )

    print(f"Centers: {centers}")
    print(f"Rolling CMIs: {cmi_rolling}")

    # Manually compute for first center
    c = centers[0]
    x_man = action_data[c-5000:c+5000, 0].astype(int)
    y_man = action_data[c-5000:c+5000, 1].astype(int)
    z_man = demand_data[c-5000:c+5000].astype(int)
    cmi_man = conditional_mutual_info(x_man, y_man, z_man)

    print(f"Manual at center {c}: {cmi_man}")
    print(f"Rolling at center {c}: {cmi_rolling[0]}")

    print(f"demand_data[:10]: {demand_data[:10]}")
    print(f"demand_data[-10:]: {demand_data[-10:]}")

    # Check if action_data has weird values at the end
    print(f"action_data[:10, :]: {action_data[:10, :]}")
    print(f"action_data[-10:, :]: {action_data[-10:, :]}")

    # Are there zeros or NaNs?
    print(f"Zeros in demand_data[-10000:]: {np.sum(demand_data[-10000:] == 0)}")
    print(f"Zeros in action_data[-10000:, 0]: {np.sum(action_data[-10000:, 0] == 0)}")
    print(f"Zeros in action_data[-10000:, 1]: {np.sum(action_data[-10000:, 1] == 0)}")

    x = action_data[-10000:, 0].copy()
    y = action_data[-10000:, 1].copy()
    z = demand_data[-10000:].copy()

    print(f"x: {x[:5]} ... {x[-5:]}")
    print(f"y: {y[:5]} ... {y[-5:]}")
    print(f"z: {z[:5]} ... {z[-5:]}")

    cmi_direct = conditional_mutual_info(x, y, z)
    print(f"Direct CMI: {cmi_direct}")


    return action_data, reward_data, demand_data

    # {
    #     'profit_delta' : delta,
    #     'cmi_delta' : cmi_delta,
    #     'converged' : 1 - (step_conv == 5000000 - 1)
    # }

if __name__ == "__main__":

    parameter_set = generate_parameters()

    with h5py.File('agentdata.h5', 'w') as f:   

        results = {}

        experiment_pbar = tqdm(range(len(parameter_set)), desc="Parameter set #...", position=0)

        for experiment in experiment_pbar:

                params = parameter_set[experiment]

                action_data, reward_data, demand_data = run_session(params)

                # session_generator = Parallel(n_jobs=8, verbose=0, return_as='generator')(
                #     delayed(run_session)(params)
                #     for num in range(50)
                # )

                # session_results = list(tqdm(
                #     session_generator,
                #     total=50,
                #     desc='Sessions',
                #     position=1,
                #     leave=False
                # ))

                # # session results 
                # cmi_delta = [r['cmi_delta'] for r in session_results]
                # profit_delta = [r['profit_delta'] for r in session_results]
                # converged = [r['converged'] for r in session_results]
                # results[(tuple(params.alphas), tuple(params.betas))] = {
                #     'profit_delta_mean' : np.mean(profit_delta),
                #     'profit_delta_std' : np.std(profit_delta),
                #     'profit_deltas' : profit_delta,
                #     'cmi_delta_mean' : np.mean(cmi_delta),
                #     'cmi_delta_std' : np.std(cmi_delta),
                #     'cmi_delta' : cmi_delta,
                #     'fraction_converged' : np.mean(converged)
                # }

                # print(results[((tuple(params.alphas), tuple(params.betas)))]['cmi_delta_mean'])
                # print(f"Fraction converged {results[((tuple(params.alphas), tuple(params.betas)))]['fraction_converged']}")

                
                    
                f.create_dataset(f"actions_{0}", data=action_data.T)
                f.create_dataset(f"rewards_{0}", data=reward_data.T)
                f.create_dataset(f"demands_{0}", data = demand_data.T)