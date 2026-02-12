import numpy as np
import matplotlib.pyplot as plt
import h5py
from data_processing import rolling_mutual_entropy, rolling_entropy, rolling_conditional_mutual_information, conditional_mutual_info


def plot_agent_data(data_filepath):
    """Plot data for a single agent"""
    
    # Load agent price, volume data
    with h5py.File(data_filepath, 'r') as f:
        agent_one_actions = f['actions_0'][0]
        print(agent_one_actions[:100])
        
        agent_two_actions = f['actions_0'][1]
        print(agent_two_actions[:100])
        demands = f['demands_0'][:]
        print(demands.shape)
        # agent_one_rewards = f['rewards_0'][0][:200000]
        # agent_two_rewards = f['rewards_0'][1][:200000]

    # print("Conditional Mutual Information: ", conditional_mutual_info(agent_one_actions, prices, signals))
    print("Mutual Info. - Market Context, Purchase ")

    mutual_entropy = rolling_mutual_entropy(agent_one_actions, agent_two_actions, 10, step=1000)
    # print(mutual_entropy)

    print(agent_one_actions.shape)

    entropy = rolling_entropy(agent_one_actions, 10, step=1000)
    print(entropy[:100])
    print(mutual_entropy[0][:100])



    print(f"MI range: {mutual_entropy[0].min()} to {mutual_entropy[0].max()}")
    print(f"Entropy range: {entropy.min()} to {entropy.max()}")

    cmi = conditional_mutual_info(agent_one_actions[1:], agent_two_actions[:-1], demands[1:])
    print(f"CMI of last 200k steps: {cmi}")
    

    # rolling_cmi_1, centers_1 = rolling_conditional_mutual_information(agent_one_actions, agent_two_actions, demands, window_size=700000, step=5000)
    rolling_cmi_2, centers_2 = rolling_conditional_mutual_information(agent_one_actions[1:], agent_two_actions[:-1], demands[1:], window_size=10000, step=500)

    # Create timesteps array
    timesteps = np.arange(len(agent_one_actions))
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 1, figsize=(12, 10), sharex=True)
    # fig.suptitle(f'Agent: {0}', fontsize=14, fontweight='bold')
    
    # # Plot Capital
    # axes[0].plot(timesteps, agent_one_actions, color='green', linewidth=1.5, alpha=0.8)
    # axes[0].plot(timesteps, agent_one_rewards, color='red', linewidth=1.5, alpha=0.8)
    # axes[0].set_ylabel('Purchase Volume', fontsize=12)
    # axes[0].grid(True, alpha=0.3)
    # axes[0].set_title('Purchase Volume over Time')
    
    # # Plot Prices
    # axes[1].plot(timesteps, agent_two_actions, color='green', linewidth=1.5, alpha=0.8)
    # axes[1].plot(timesteps, agent_two_rewards, color='red', linewidth=1.5, alpha=0.8)

    # axes[1].set_ylabel('Price', fontsize=12)
    # axes[1].grid(True, alpha=0.3)
    # axes[1].set_title('Prices over Time')

    

    # Plot Entropy of Decisions
    # axes.plot(centers_1, rolling_cmi_1, color='blue', linewidth=1.5, alpha=0.8)
    axes.plot(centers_2, rolling_cmi_2, color='red', linewidth=1.5, alpha=0.8)
    # axes.plot(timesteps, entropy, color='blue', linewidth=1.5, alpha=0.8)
    axes.set_ylabel('Information', fontsize=12)
    axes.set_xlabel('Timestep', fontsize=12)
    axes.grid(True, alpha=0.3)
    axes.set_title('Mutual Information Between Agent Interactions Conditioned on Demand Versus Time')

    plt.tight_layout()
    return fig, axes


def heatmap_cmi():

    # Load the data
    with h5py.File("testdata.h5", "r") as f:
        # Get unique alpha and beta values
        alphas_all = f["params_set/alphas"][:]
        betas_all = f["params_set/betas"][:]
        
        # Extract first value from each (since they're pairs like [0.05, 0.05])
        alphas = np.array([a[0] for a in alphas_all])
        betas = np.array([b[0] for b in betas_all])
        
        unique_alphas = np.sort(np.unique(alphas))
        unique_betas = np.sort(np.unique(betas))
        
        # Create 2D array for heatmap
        heatmap_data = np.zeros((len(unique_alphas), len(unique_betas)))
        
        # Fill in the heatmap
        for experiment in range(len(alphas)):
            cmi_data = f[f"cmi_deltas_{experiment}"]
            avg_cmi = np.mean(cmi_data)
            print(avg_cmi)
            
            alpha_idx = np.where(unique_alphas == alphas[experiment])[0][0]
            beta_idx = np.where(unique_betas == betas[experiment])[0][0]
            
            heatmap_data[alpha_idx, beta_idx] = avg_cmi

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(heatmap_data, aspect='auto', origin='lower', cmap='viridis')

    num_ticks = 10
    alpha_tick_indices = np.linspace(3, len(unique_alphas) - 1, num_ticks, dtype=int)
    beta_tick_indices = np.linspace(0, len(unique_betas) - 1, num_ticks, dtype=int)
    beta_labels = [unique_betas[i] for i in beta_tick_indices]
    alpha_labels = [unique_alphas[i] for i in alpha_tick_indices]

    # Set axis labels and ticks
    ax.set_xticks(beta_tick_indices)
    ax.set_yticks(alpha_tick_indices)
    ax.set_xticklabels([f"{b:.1e}" for b in beta_labels], rotation=45, ha='right')
    ax.set_yticklabels([f"{a:.3f}" for a in alpha_labels])

    ax.set_xlabel("Beta")
    ax.set_ylabel("Alpha")
    ax.set_title("Average CMI Delta by Alpha and Beta")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Average CMI Delta")

    plt.tight_layout()
    plt.savefig("cmi_heatmap.png", dpi=150)
    plt.show()

# Main execution
if __name__ == "__main__":
    import sys
    
    # Get directory from command line or use current directory
    output_directory = sys.argv[1] if len(sys.argv) > 1 else "."
    
    # Plot individual agent (example)
    # fig1, ax1 = plot_agent_data("agentdata.h5")

    heatmap_cmi()

    # fig2, ax2 = information_histogram(output_directory)
    
    plt.show()
