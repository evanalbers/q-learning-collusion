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
    
    alpha_range = np.linspace(0.025, 0.25, 50)
    beta_range = np.linspace(0, 2e-5, 50)

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
            episodes_per_session=400,
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

    q_tables, step_conv, action_data, reward_data, demand_data, profits = fast_session(q_tables, params)

    average_profit = profits[-10000:, 0].mean()
    # print(profits[-10000:, 0])
    # print(f"Average profit: {average_profit}")

    delta = (average_profit - (params.nash[0] - params.costs[0])) / ((params.coop[0] - params.costs[0]) - (params.nash[0] - params.costs[0]))
    # print(f"Delta: {delta}")

    agent_one_actions = action_data[1:, 0]
    agent_two_actions = action_data[:-1, 1]
    demand_data = demand_data[1:]

    cmi_beginning = conditional_mutual_info(agent_one_actions[:10000], agent_two_actions[:10000], demand_data[:10000])
    cmi_end = conditional_mutual_info(agent_one_actions[-10000:], agent_two_actions[-10000:], demand_data[-10000:])
    cmi_delta = cmi_end - cmi_beginning
    # print(cmi_beginning)
    # print(cmi_end)
    # print(cmi_delta)

    return {
        'profit_delta' : delta,
        'cmi_delta' : cmi_delta,
        'converged' : 1 - (step_conv == params.steps_per_episode * params.episodes_per_session - 1),
        'params' : params
    }

if __name__ == "__main__":

    parameter_set = generate_parameters()

    with h5py.File('testdata.h5', 'w') as f:   

        results = {}

        experiment_pbar = tqdm(range(len(parameter_set)), desc="Parameter set #...", position=0)

        for experiment in experiment_pbar:

            params = parameter_set[experiment]

            session_generator = Parallel(n_jobs=64, verbose=0, return_as='generator')(
                delayed(run_session)(params)
                for num in range(1000)
            )

            session_results = list(tqdm(
                session_generator,
                total=1000,
                desc='Sessions',
                position=1,
                leave=False
            ))

            # session results 
            cmi_delta = [r['cmi_delta'] for r in session_results]
            profit_delta = [r['profit_delta'] for r in session_results]
            converged = [r['converged'] for r in session_results]
            results[(tuple(params.alphas), tuple(params.betas))] = {
                'profit_delta_mean' : np.mean(profit_delta),
                'profit_delta_std' : np.std(profit_delta),
                'profit_deltas' : profit_delta,
                'cmi_delta_mean' : np.mean(cmi_delta),
                'cmi_delta_std' : np.std(cmi_delta),
                'cmi_delta' : cmi_delta,
                'fraction_converged' : np.mean(converged)
            }
            # print(results[(tuple(params.alphas), tuple(params.betas))]['cmi_delta'])
            # print(results[((tuple(params.alphas), tuple(params.betas)))]['cmi_delta_mean'])
            # print(f"Fraction converged {results[((tuple(params.alphas), tuple(params.betas)))]['fraction_converged']}")

            f.create_dataset(f"cmi_deltas_{experiment}", data=results[(tuple(params.alphas), tuple(params.betas))]['cmi_delta'])
            f.create_dataset(f"profit_deltas_{experiment}", data=results[(tuple(params.alphas), tuple(params.betas))]['profit_deltas'])
            f.create_dataset(f"converged_{experiment}", data=results[(tuple(params.alphas), tuple(params.betas))]['fraction_converged'])

        params_group = f.create_group("params_set")

        # For the first param set, get field names
        field_names = parameter_set[0]._fields

        for field in field_names:
            data = [getattr(p, field) for p in parameter_set]
            try:
                params_group.create_dataset(field, data=data)
            except (ValueError, TypeError):
                # For complex fields like r_matrix, handle separately
                params_group.create_dataset(field, data=np.array(data, dtype=object))
        