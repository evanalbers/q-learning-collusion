import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def fast_quantity(a_arr, state, agent, mu):
    """Calculate market share for given agent"""
    numerator = np.exp((a_arr[agent] - state[agent]) / mu)
    denom = np.exp(a_arr[0] / mu)
    for i in range(1, len(a_arr)): 
        denom += np.exp((a_arr[i] - state[i]) / mu)
    return numerator / denom

def calculate_symmetric_demand(n_agents, a0=0, a=2, mu=0.1):
    """
    Calculate demand curves when all agents have same quality and price
    
    Parameters:
    -----------
    n_agents : int
        Number of inside agents (products)
    a0 : float
        Outside good utility (inverse aggregate demand)
    a : float
        Quality parameter for all inside goods
    mu : float
        Differentiation parameter
    
    Returns:
    --------
    prices : array
        Price range
    q_individual : array
        Individual agent demand at each price
    q_total : array
        Total inside market share at each price
    """
    prices = np.linspace(1, 3, 100)
    q_individual = np.zeros(len(prices))
    q_total = np.zeros(len(prices))
    
    for i, p in enumerate(prices):
        # Create state and a_arr for n agents
        a_arr = np.zeros(n_agents + 1)
        a_arr[0] = a0  # Outside good
        a_arr[1:] = a  # All inside goods have same quality
        
        state = np.zeros(n_agents + 1)
        state[0] = 0  # Outside good price is 0
        state[1:] = p  # All inside goods have same price
        
        # Calculate quantity for agent 1 (representative)
        q_individual[i] = fast_quantity(a_arr, state, 1, mu)
        q_total[i] = q_individual[i] * n_agents
    
    return prices, q_individual, q_total

def create_demand_contour_plots():
    """Create visualization of demand curves for different numbers of agents"""
    
    # Parameters
    a0 = 0  # Outside good utility
    a = 2   # Inside good quality
    mu = 0.25 # Differentiation
    
    # Different numbers of agents to compare
    n_agents_list = [2, 3, 5, 10, 25, 50, 100]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(n_agents_list)))
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 1, figsize=(15, 10))
    
    # # Plot 1: Individual demand curves
    # ax1 = axes[0, 0]
    # for n, color in zip(n_agents_list, colors):
    #     prices, q_ind, _ = calculate_symmetric_demand(n, a0, a, mu)
    #     ax1.plot(q_ind, prices, label=f'N={n}', color=color, linewidth=2)
    # ax1.set_xlabel('Quantity per Agent', fontsize=11)
    # ax1.set_ylabel('Price', fontsize=11)
    # ax1.set_title('Individual Agent Demand Curves', fontsize=12, fontweight='bold')
    # ax1.legend(loc='upper right')
    # ax1.grid(True, alpha=0.3)
    # ax1.set_xlim([0, 0.6])
    
    # # Plot 2: Total market demand curves
    # ax2 = axes[0, 1]
    # for n, color in zip(n_agents_list, colors):
    #     prices, _, q_total = calculate_symmetric_demand(n, a0, a, mu)
    #     ax2.plot(q_total, prices, label=f'N={n}', color=color, linewidth=2)
    # ax2.set_xlabel('Total Quantity (All Agents)', fontsize=11)
    # ax2.set_ylabel('Price', fontsize=11)
    # ax2.set_title('Total Inside Market Demand', fontsize=12, fontweight='bold')
    # ax2.legend(loc='upper right')
    # ax2.grid(True, alpha=0.3)
    # ax2.set_xlim([0, 1])
    
    # # Plot 3: Contour plot - Individual quantity
    # ax3 = axes[0, 2]
    # n_agents_cont = np.arange(2, 11, 1)
    # prices_cont = np.linspace(1, 3, 50)
    # N_grid, P_grid = np.meshgrid(n_agents_cont, prices_cont)
    # Q_ind_grid = np.zeros_like(N_grid, dtype=float)
    
    # for i, n in enumerate(n_agents_cont):
    #     _, q_ind, _ = calculate_symmetric_demand(int(n), a0, a, mu)
    #     # Interpolate to match prices_cont
    #     Q_ind_grid[:, i] = np.interp(prices_cont, np.linspace(0, 3, 100), q_ind)
    
    # contour1 = ax3.contourf(Q_ind_grid, P_grid, N_grid, levels=15, cmap='viridis')
    # ax3.set_xlabel('Quantity per Agent', fontsize=11)
    # ax3.set_ylabel('Price', fontsize=11)
    # ax3.set_title('Individual Demand Contours', fontsize=12, fontweight='bold')
    # cbar1 = fig.colorbar(contour1, ax=ax3)
    # cbar1.set_label('Number of Agents', fontsize=10)
    
    # Plot 4: Revenue per agent
    ax4 = axes #[1, 0]
    for n, color in zip(n_agents_list, colors):
        prices, q_ind, _ = calculate_symmetric_demand(n, a0, a, mu)
        revenue = (prices - np.ones(prices.shape)) * q_ind
        ax4.plot(prices, revenue, label=f'N={n}', color=color, linewidth=2)

    ax4.axvline(x=1.8, color='red', linestyle='--', linewidth=2, label='P=1.8', alpha=0.7)
    
    ax4.set_xlabel('Price', fontsize=11)
    ax4.set_ylabel('Reward per Agent', fontsize=11)
    ax4.set_title('Individual Agent Reward Curves', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # # Plot 5: Market share loss rate
    # ax5 = axes[1, 1]
    # base_n = 2
    # prices_base, q_base, _ = calculate_symmetric_demand(base_n, a0, a, mu)
    
    # for n in n_agents_list[1:]:  # Skip n=2 since it's the baseline
    #     _, q_ind, _ = calculate_symmetric_demand(n, a0, a, mu)
    #     loss_rate = (q_base - q_ind) / q_base * 100
    #     ax5.plot(prices, loss_rate, label=f'N={n} vs N=2', linewidth=2)
    
    # ax5.set_xlabel('Price', fontsize=11)
    # ax5.set_ylabel('Market Share Loss (%)', fontsize=11)
    # ax5.set_title('Individual Share Loss vs 2-Agent Case', fontsize=12, fontweight='bold')
    # ax5.legend(loc='lower right')
    # ax5.grid(True, alpha=0.3)
    
    # # Plot 6: Elasticity comparison
    # ax6 = axes[1, 2]
    # for n, color in zip([2, 4, 8], ['blue', 'green', 'red']):
    #     prices, q_ind, _ = calculate_symmetric_demand(n, a0, a, mu)
    #     # Calculate approximate elasticity
    #     dq = np.gradient(q_ind)
    #     dp = np.gradient(prices)
    #     elasticity = (dq / dp) * (prices / (q_ind + 1e-10))
    #     # Only plot where q > 0.01 to avoid numerical issues
    #     mask = q_ind > 0.01
    #     ax6.plot(prices[mask], elasticity[mask], label=f'N={n}', color=color, linewidth=2)
    
    # ax6.set_xlabel('Price', fontsize=11)
    # ax6.set_ylabel('Price Elasticity', fontsize=11)
    # ax6.set_title('Demand Elasticity by Number of Agents', fontsize=12, fontweight='bold')
    # ax6.legend(loc='lower left')
    # ax6.grid(True, alpha=0.3)
    # ax6.set_ylim([-10, 0])
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("=" * 60)
    print("DEMAND CURVE ANALYSIS - SYMMETRIC EQUILIBRIUM")
    print("=" * 60)
    print(f"Parameters: a0={a0}, a={a}, μ={mu}")
    print("-" * 60)
    
    # Compare quantities at different price points
    test_prices = [0.5, 1.0, 1.5, 2.0]
    print("\nPer-Agent Quantity at Different Prices:")
    print("-" * 60)
    print("N Agents | P=0.5 | P=1.0 | P=1.5 | P=2.0")
    print("-" * 60)
    
    for n in [2, 3, 4, 5, 8]:
        quantities = []
        for p in test_prices:
            a_arr = np.array([a0] + [a] * n)
            state = np.array([0] + [p] * n)
            q = fast_quantity(a_arr, state, 1, mu)
            quantities.append(q)
        print(f"   {n:2d}    | {quantities[0]:5.3f} | {quantities[1]:5.3f} | "
              f"{quantities[2]:5.3f} | {quantities[3]:5.3f}")
    
    print("=" * 60)

# Run the visualization
if __name__ == "__main__":
    create_demand_contour_plots()