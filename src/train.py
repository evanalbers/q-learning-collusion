""" Script for training Q-learning agents """
from tqdm import tqdm
import numpy as np
from numba import njit
from env import fast_reward

@njit
def argmax_1d(arr, n):
    best_idx = 0
    best_val = arr[0]
    for i in range(1, n):
        if arr[i] > best_val:
            best_val = arr[i]
            best_idx = i
    return best_idx

@njit
def max_1d(arr, n):
    best_val = arr[0]
    for i in range(1, n):
        if arr[i] > best_val:
            best_val = arr[i]
    return best_val

@njit
def get_greedy_policies(num_agents, num_actions, num_demands, q_tables):
    
    # Extract current greedy policies
    current_greedy_policies = np.zeros((num_agents, num_actions, num_demands), dtype=np.int32)
    for agent in range(num_agents):
        for s in range(num_actions):
            for d in range(num_demands):
                current_greedy_policies[agent, s, d] = argmax_1d(q_tables[agent, s, d], num_actions)
    
    return current_greedy_policies

@njit
def fast_convergence_check(num_agents, num_actions, num_demands, q_tables,
                            step, last_check_step, prev_greedy_policies,
                            conv, conv_threshold=100000):
    """ checks convergence according to Calvano convention """

    current_greedy_policies = get_greedy_policies(num_agents, num_actions, num_demands, q_tables)

    if np.array_equal(current_greedy_policies, prev_greedy_policies):
        # Policies unchanged, increment by steps since last check
        conv += (step - last_check_step)
        if conv >= conv_threshold:
            return True, prev_greedy_policies, conv
    else:    
        conv = 0
        prev_greedy_policies = current_greedy_policies.copy()

    return False, prev_greedy_policies, conv


@njit
def fast_session(q_tables, params):
    
    max_steps = params.steps_per_episode * params.episodes_per_session

    a_arr = np.array(params.a_arr, dtype=np.float64)

    # initializing arrays to keep track of data
    action_data = np.zeros((max_steps, params.num_agents))
    reward_data = np.zeros((max_steps, params.num_agents))
    demand_data = np.zeros(max_steps, dtype=np.int32)
    profits_data = np.zeros((max_steps, params.num_agents))

    CHECK_INTERVAL = 1000  # Check every 1000 steps
    
    # to assist with determining convergence 
    prev_greedy_policies = np.zeros((params.num_agents, params.num_actions, params.num_demands), dtype=np.int32)
    conv = 0
    last_check_step = 0
    
    # Initialize market state
    state = np.zeros(params.num_agents, dtype=np.int32)
    for i in range(params.num_agents):
        state[i] = np.random.randint(params.num_actions)

    # initializing demand here
    next_demand = np.random.randint(0, 5)

    # during each step
    for step in range(max_steps):

        # Only check convergence periodically
        if step % CHECK_INTERVAL == 0 and step > 0:
            converged, prev_greedy_policies, conv = fast_convergence_check(params.num_agents, params.num_actions, 
                                                                           params.num_demands, q_tables, step,
                                                                           last_check_step,
                                                                           prev_greedy_policies, conv)
            last_check_step = step
            if converged: 
                break

        # if it's a new episode, random previous action
        # if step % params.steps_per_episode == 0:
        #     for i in range(params.num_agents):
        #         state[i] = np.random.randint(params.num_actions)
        
        # random draw for the demand state
        demand_state = next_demand
        a_arr[0] = demand_state
        demand_data[step] = demand_state
        next_demand = np.random.randint(0, 5)

        # initialize for this step's actions, rewards 
        actions = np.zeros(params.num_agents, dtype=np.int32)
        rewards = np.zeros(params.num_agents)
        profits = np.zeros(params.num_agents)
        
        # for each agent, determine action, then save actions
        for agent in range(params.num_agents):
            opp = 1 - agent
            # epsilon selection by agent
            if np.random.random() < np.exp(-params.betas[agent] * step):
                actions[agent] = np.random.randint(params.num_actions)
            else:
                actions[agent] = argmax_1d(q_tables[agent, state[opp], demand_state], params.num_actions) # hard coded for two agents here
        action_data[step] = actions

        # for each agent, determine reward then save rewards
        for agent in range(params.num_agents):
            rewards[agent], profits[agent] = fast_reward(a_arr, actions, agent, params.costs, params.mu, params.r_matrix)
        reward_data[step] = rewards
        profits_data[step] = profits

        # for each agent, calculate highest value next move, update Q tables accordingly
        old_state = state.copy()
        for agent in range(params.num_agents):
            opp = 1 - agent
            current_q = q_tables[agent, old_state[opp], demand_state, actions[agent]]
            max_next_q = max_1d(q_tables[agent, actions[opp], next_demand], params.num_actions)
            target = rewards[agent] + params.deltas[agent] * max_next_q
            q_tables[agent, old_state[opp], demand_state, actions[agent]] += params.alphas[agent] * (target - current_q)
        state = actions.copy()
    
    
    return q_tables, step, action_data[:step, :].copy(), reward_data[:step, :].copy(), demand_data[:step].copy(), profits_data[:step, :].copy()








