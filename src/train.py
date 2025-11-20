""" Script for training Q-learning agents """
from tqdm import tqdm
import numpy as np
from numba import njit
from env import fast_reward


@njit
def fast_episode(q_tables, num_agents, num_actions, 
                 betas, a_arr, costs, mu, r_matrix,
                 gamma, lr, max_steps, conv_cutoff):

      
    CONV_THRESHOLD = 100_000
    CHECK_INTERVAL = 1000  # Check every 1000 steps
    
    prev_greedy_policies = np.zeros((num_agents, num_actions, num_actions), dtype=np.int32)
    conv = 0
    last_check_step = 0
    
    # Initialize market state
    state = np.zeros(num_agents, dtype=np.int32)
    for i in range(num_agents):
        state[i] = np.random.randint(num_actions)
    
    # during each step
    for step in range(max_steps):

        # Only check convergence periodically
        if step % CHECK_INTERVAL == 0 and step > 0:
            # Extract current greedy policies
            current_greedy_policies = np.zeros((num_agents, num_actions, num_actions), dtype=np.int32)
            for agent in range(num_agents):
                for s1 in range(num_actions):
                    for s2 in range(num_actions):
                        current_greedy_policies[agent, s1, s2] = np.argmax(q_tables[agent, s1, s2, :])
            
            if np.array_equal(current_greedy_policies, prev_greedy_policies):
                # Policies unchanged, increment by steps since last check
                conv += (step - last_check_step)
                if conv >= CONV_THRESHOLD:
                    break
            else:
                conv = 0
                prev_greedy_policies[:] = current_greedy_policies
            
            last_check_step = step
        
        # for each agent
        for agent in range(num_agents):
            # epsilon selection by agent
            if np.random.random() < np.exp(-betas[agent] * step):
                action = np.random.randint(num_actions)
            else:
                action = np.argmax(q_tables[agent, state[0], state[1], :]) # hard coded for two agents here

            old_state = state.copy()

            # execute action
            state[agent] = action

            r = fast_reward(a_arr, state, agent, costs, mu, r_matrix)
            current_q = q_tables[agent, old_state[0], old_state[1], old_state[agent]]
            max_next_q = np.max(q_tables[agent, state[0], state[1], :])
            target = r + gamma[agent] * max_next_q
            q_tables[agent, old_state[0], old_state[1], old_state[agent]] += lr[agent] * (target - current_q)

    return q_tables, step

def train_agents(agents, env, num_episodes=5000, steps_per_episode=25000):
    """ """

    episode_pbar = tqdm(range(num_episodes), desc="Number of Episodes")
    for episode in episode_pbar:

        state = env.reset()

        step_pbar = tqdm(range(steps_per_episode), desc="Step number in episode...", disable=True)

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






