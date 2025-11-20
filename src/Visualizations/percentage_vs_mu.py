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

def find_optimal_revenue(n_agents, a0, a, mu):
    """Find the maximum revenue and optimal price for symmetric case"""
    price_range = np.linspace(0, 5, 200)
    
    revenues = []
    for p in price_range:
        a_arr = np.array([a0] + [a] * n_agents)
        state = np.array([0] + [p] * n_agents)
        q = fast_quantity(a_arr, state, 1, mu)
        revenue = p * q
        revenues.append(revenue)
    
    max_idx = np.argmax(revenues)
    return price_range[max_idx], revenues[max_idx]

def analyze_revenue_loss_vs_mu():
    """Graph percentage revenue loss from 2 to 3 agents as function of μ"""
    
    # Parameters
    a0 = 0  # Outside good utility
    a = 2   # Inside good quality
    
    # Range of μ values to test (more granular for smooth curve)
    mu_values = np.linspace(0.0005, 2.5, 100)
    
    # Calculate revenues for 2 and 3 agents
    revenue_2_agents = []
    revenue_3_agents = []
    optimal_price_2 = []
    optimal_price_3 = []
    
    for mu in mu_values:
        # 2 agents case
        p2, r2 = find_optimal_revenue(2, a0, a, mu)
        revenue_2_agents.append(r2)
        optimal_price_2.append(p2)
        
        # 3 agents case
        p3, r3 = find_optimal_revenue(5, a0, a, mu)
        revenue_3_agents.append(r3)
        optimal_price_3.append(p3)
    
    # Calculate percentage loss
    percentage_loss = [(r2 - r3) / r2 * 100 for r2, r3 in zip(revenue_2_agents, revenue_3_agents)]
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Main plot - Revenue loss vs μ
    ax1 = axes[0, 0]
    ax1.plot(mu_values, percentage_loss, 'b-', linewidth=2.5)
    ax1.fill_between(mu_values, 0, percentage_loss, alpha=0.3)
    ax1.set_xlabel('μ (Differentiation Parameter)', fontsize=12)
    ax1.set_ylabel('Revenue Loss (%)', fontsize=12)
    ax1.set_title('Revenue Loss: 2 → 3 Agents', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add reference lines
    ax1.axhline(y=33.33, color='r', linestyle='--', alpha=0.5, label='Perfect competition baseline (33.3%)')
    ax1.axvline(x=0.25, color='g', linestyle='--', alpha=0.5, label='Your current μ=0.25')
    
    # Highlight key points
    mu_025_idx = np.argmin(np.abs(mu_values - 0.25))
    ax1.plot(0.25, percentage_loss[mu_025_idx], 'go', markersize=8)
    ax1.annotate(f'{percentage_loss[mu_025_idx]:.1f}%', 
                xy=(0.25, percentage_loss[mu_025_idx]),
                xytext=(0.4, percentage_loss[mu_025_idx] + 2),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10)
    
    ax1.legend(loc='upper right')
    
    # Plot 2: Absolute revenues
    ax2 = axes[0, 1]
    ax2.plot(mu_values, revenue_2_agents, 'b-', linewidth=2, label='2 agents')
    ax2.plot(mu_values, revenue_3_agents, 'r-', linewidth=2, label='3 agents')
    ax2.set_xlabel('μ (Differentiation Parameter)', fontsize=12)
    ax2.set_ylabel('Maximum Revenue per Agent', fontsize=12)
    ax2.set_title('Absolute Revenue Comparison', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Shade the difference
    ax2.fill_between(mu_values, revenue_2_agents, revenue_3_agents, 
                     alpha=0.2, color='gray', label='Revenue loss')
    
    # Plot 3: Optimal prices
    ax3 = axes[1, 0]
    ax3.plot(mu_values, optimal_price_2, 'b-', linewidth=2, label='2 agents')
    ax3.plot(mu_values, optimal_price_3, 'r-', linewidth=2, label='3 agents')
    ax3.set_xlabel('μ (Differentiation Parameter)', fontsize=12)
    ax3.set_ylabel('Optimal Price', fontsize=12)
    ax3.set_title('Profit-Maximizing Prices', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Revenue retention rate
    ax4 = axes[1, 1]
    retention_rate = [r3/r2 * 100 for r2, r3 in zip(revenue_2_agents, revenue_3_agents)]
    ax4.plot(mu_values, retention_rate, 'g-', linewidth=2.5)
    ax4.fill_between(mu_values, retention_rate, 100, alpha=0.2, color='red')
    ax4.set_xlabel('μ (Differentiation Parameter)', fontsize=12)
    ax4.set_ylabel('Revenue Retention (%)', fontsize=12)
    ax4.set_title('Revenue Retained After Entry (3 vs 2 agents)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=66.67, color='r', linestyle='--', alpha=0.5, label='2/3 baseline')
    ax4.axhline(y=80, color='orange', linestyle='--', alpha=0.5, label='80% target')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print analysis at key μ values
    print("=" * 70)
    print("REVENUE IMPACT OF THIRD ENTRANT AT DIFFERENT μ VALUES")
    print("=" * 70)
    print(f"Setup: Outside option a0={a0}, Product quality a={a}")
    print("-" * 70)
    print("\n{:<6} | {:<12} | {:<12} | {:<12} | {:<12}".format(
        "μ", "Rev(n=2)", "Rev(n=3)", "Loss(%)", "Retention(%)"))
    print("-" * 70)
    
    # Key μ values to highlight
    key_mus = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    for mu_target in key_mus:
        idx = np.argmin(np.abs(mu_values - mu_target))
        mu = mu_values[idx]
        r2 = revenue_2_agents[idx]
        r3 = revenue_3_agents[idx]
        loss = percentage_loss[idx]
        retention = retention_rate[idx]
        print(f"{mu:6.2f} | {r2:12.4f} | {r3:12.4f} | {loss:11.2f}% | {retention:11.2f}%")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("-" * 70)
    
    # Find μ for specific targets
    target_losses = [25, 20, 15, 10]
    print("\nμ values for target revenue losses:")
    for target in target_losses:
        idx = np.argmin(np.abs(np.array(percentage_loss) - target))
        if idx < len(mu_values):
            print(f"  • For {target}% loss: μ ≈ {mu_values[idx]:.2f}")
    
    # Find point of diminishing returns
    d_loss_d_mu = np.gradient(percentage_loss, mu_values)
    inflection_idx = np.argmin(np.abs(d_loss_d_mu + 5))  # Where slope ≈ -5
    
    print(f"\n• Current setting (μ=0.25): {percentage_loss[mu_025_idx]:.1f}% revenue loss")
    print(f"• Increasing μ to 1.0 reduces loss to {percentage_loss[np.argmin(np.abs(mu_values - 1.0))]:.1f}%")
    print(f"• Diminishing returns start around μ ≈ {mu_values[inflection_idx]:.2f}")
    print("\n• In perfect competition (μ→0), loss approaches 33.3% (1/3)")
    print("• With high differentiation (μ→∞), loss approaches 0%")
    print("=" * 70)

# Run the analysis
if __name__ == "__main__":
    analyze_revenue_loss_vs_mu()