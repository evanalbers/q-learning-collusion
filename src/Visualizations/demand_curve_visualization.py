from env import fast_quantity
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def demand_curve_visualization():
    """Visualizing the demand curve in 3D"""
    
    # Fixed: np.array instead of np.arr
    a_arr = np.array([0, 2, 2])
    mu = 0.25
    
    # Fixed: np.arange needs start parameter, and stop should be larger than step
    p1 = np.arange(0, 2, 0.02)
    p2 = np.arange(0, 2, 0.02)
    
    # Create meshgrid for 3D plotting
    P1, P2 = np.meshgrid(p1, p2)
    
    # Initialize Q array with proper shape
    Q1 = np.zeros_like(P1)
    Q2 = np.zeros_like(P2)

    
    
    # Calculate quantities for each price combination
    for i in range(len(p1)):
        for j in range(len(p2)):
            # Fixed: np.array instead of np.arr
            Q1[j, i] = fast_quantity(a_arr, np.array([p1[i], p2[j]]), 1, mu)

    for i in range(len(p1)):
        for j in range(len(p2)):
            # Fixed: np.array instead of np.arr
            Q2[j, i] = fast_quantity(a_arr, np.array([p1[i], p2[j]]), 2, mu)
    
    print(Q1)

    # Create 3D visualization
    fig = plt.figure(figsize=(12, 9))
    
    # Surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(P1, P2, Q1, cmap='viridis', alpha=0.8, edgecolor='none')
    surf2 = ax1.plot_surface(P1, P2, Q2, cmap='coolwarm', alpha=0.8, edgecolor='none')

    ax1.set_xlabel('Price 1 (p1)', fontsize=10)
    ax1.set_ylabel('Price 2 (p2)', fontsize=10)
    ax1.set_zlabel('Quantity (q)', fontsize=10)
    ax1.set_title('3D Demand Surface', fontsize=12)
    ax1.view_init(elev=30, azim=45)
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    
    # Contour plot view
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.contour3D(P1, P2, Q1, 50, cmap='plasma')
    ax2.set_xlabel('Price 1 (p1)', fontsize=10)
    ax2.set_ylabel('Price 2 (p2)', fontsize=10)
    ax2.set_zlabel('Quantity (q)', fontsize=10)
    ax2.set_title('3D Demand Contours', fontsize=12)
    ax2.view_init(elev=20, azim=-60)
    
    plt.tight_layout()
    plt.show()
    
    # Additional 2D contour plot for better understanding
    fig2, ax3 = plt.subplots(figsize=(8, 6))
    contour = ax3.contourf(P1, P2, Q1, levels=20, cmap='coolwarm')
    ax3.contour(P1, P2, Q1, levels=20, colors='black', alpha=0.2, linewidths=0.5)
    ax3.set_xlabel('Price 1 (p1)', fontsize=11)
    ax3.set_ylabel('Price 2 (p2)', fontsize=11)
    ax3.set_title('Demand Curve - 2D Contour View', fontsize=13)
    fig2.colorbar(contour, ax=ax3, label='Quantity (q)')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return P1, P2, Q1

# Run the visualization
if __name__ == "__main__":
    P1, P2, Q = demand_curve_visualization()
    
    # Print some statistics
    print(f"Price 1 range: [{P1.min():.2f}, {P1.max():.2f}]")
    print(f"Price 2 range: [{P2.min():.2f}, {P2.max():.2f}]")
    print(f"Quantity range: [{Q.min():.2f}, {Q.max():.2f}]")
    print(f"Shape of output: {Q.shape}")