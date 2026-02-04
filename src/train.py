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
def fast_episode(q_tables, num_agents, num_actions, 
                 betas, a_arr, costs, mu, r_matrix,
                 gamma, lr, max_steps, num_demands):

    action_data = np.zeros((max_steps, num_agents))
    reward_data = np.zeros((max_steps, num_agents))
    demand_data = np.zeros(max_steps, dtype=np.int32)

    CONV_THRESHOLD = 100000
    CHECK_INTERVAL = 1000  # Check every 1000 steps
    
    prev_greedy_policies = np.zeros((num_agents, num_actions, num_demands), dtype=np.int32)
    conv = 0
    last_check_step = 0
    
    # Initialize market state
    state = np.zeros(num_agents, dtype=np.int32)
    for i in range(num_agents):
        state[i] = np.random.randint(num_actions)

    # initializing demand here
    next_demand = np.random.randint(0, 5)

    # during each step
    for step in range(max_steps):

        # Only check convergence periodically
        if step % CHECK_INTERVAL == 0 and step > 0:
            # Extract current greedy policies
            current_greedy_policies = np.zeros((num_agents, num_actions, num_demands), dtype=np.int32)
            for agent in range(num_agents):
                for s in range(num_actions):
                    for d in range(num_demands):
                        current_greedy_policies[agent, s, d] = argmax_1d(q_tables[agent, s, d], num_actions)
            
            if np.array_equal(current_greedy_policies, prev_greedy_policies):
                # Policies unchanged, increment by steps since last check
                conv += (step - last_check_step)
                if conv >= CONV_THRESHOLD:
                    break
            else:
                conv = 0
                prev_greedy_policies = current_greedy_policies.copy()
            
            last_check_step = step

        # random draw for the demand state
        demand_state = next_demand
        a_arr[0] = demand_state
        demand_data[step] = demand_state
        next_demand = np.random.randint(0, 5)

        actions = np.zeros(num_agents, dtype=np.int32)
        
        # for each agent
        for agent in range(num_agents):

            opp = 1 - agent

            # epsilon selection by agent
            if np.random.random() < np.exp(-betas[agent] * step):
                actions[agent] = np.random.randint(num_actions)
            else:
                actions[agent] = argmax_1d(q_tables[agent, state[opp], demand_state], num_actions) # hard coded for two agents here

        action_data[step] = actions

        rewards = np.zeros(num_agents)
        for agent in range(num_agents):
            rewards[agent] = fast_reward(a_arr, actions, agent, costs, mu, r_matrix)

        reward_data[step] = rewards

        old_state = state.copy()
        for agent in range(num_agents):
            opp = 1 - agent
            current_q = q_tables[agent, old_state[opp], demand_state, actions[agent]]
            max_next_q = max_1d(q_tables[agent, actions[opp], next_demand], num_actions)
            target = rewards[agent] + gamma[agent] * max_next_q
            q_tables[agent, old_state[opp], demand_state, actions[agent]] += lr[agent] * (target - current_q)

        state = actions.copy()
    
    
    return q_tables, step, action_data[:step, :].copy(), reward_data[:step, :].copy(), demand_data[:step].copy()

def train_agents(agents, env, num_episodes=5000, steps_per_episode=25000):
    """ """

    episode_pbar = tqdm(range(num_episodes), desc="Number of Episodes")
    for episode in episode_pbar:

        state = env.reset()

        step_pbar = tqdm(range(steps_per_episode), desc="Step number in episode...", disable=False)

        for step in step_pbar:
            for agent in range(len(agents)):
                # choose action, step in env. 
                
                state = env.market
                action = agents[agent].choose_action(state)
                next_state, reward = env.step(agent, action)

                # update agent Q-table 
                agents[agent].update(state, action, reward, next_state)

                # update the agent epsilon value
                agents[agent].set_epsilon(step)






