"""
Part 1: Basic Nonlinear Dynamical System
Implements equations (25.64) and (25.65) from the paper
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')

class NonlinearDynamicalSystem:
    def __init__(self, n_states=2, m_inputs=1, l_outputs=1):
        """
        Initialize the nonlinear dynamical system G from equations (25.64-25.65):
        ẋ(t) = f(x(t)) + G(x(t))u(t)
        y(t) = h(x(t)) + J(x(t))u(t)
        
        Args:
            n_states: dimension of state vector x
            m_inputs: dimension of control input u  
            l_outputs: dimension of output y
        """
        self.n = n_states
        self.m = m_inputs
        self.l = l_outputs
        
    def f(self, x):
        """Natural dynamics f(x) - example: damped nonlinear oscillator"""
        if self.n == 2:
            # Van der Pol-like dynamics
            x1, x2 = x
            mu = 0.5  # damping parameter
            return np.array([
                x2,
                -x1 - mu * (x1**2 - 1) * x2
            ])
        else:
            # General nonlinear dynamics with stable tendency
            return -0.5 * x - 0.1 * x**3
    
    def G(self, x):
        """Control input matrix G(x)"""
        if self.n == 2 and self.m == 1:
            # State-dependent control effectiveness
            return np.array([[0], [1 + 0.1 * x[0]**2]]).flatten()
        else:
            return np.ones((self.n, self.m))
    
    def h(self, x):
        """Output function h(x)"""
        if self.l == 1 and self.n == 2:
            # Observe first state
            return np.array([x[0]])
        else:
            # Full state observation
            return x[:self.l]
    
    def J(self, x):
        """Direct feedthrough J(x)"""
        return np.zeros((self.l, self.m))
    
    def dynamics(self, x, t, u_func):
        """Complete dynamics for integration"""
        u = u_func(t, x) if callable(u_func) else u_func
        u = np.atleast_1d(u)
        
        dx = self.f(x) + self.G(x).reshape(self.n, self.m) @ u
        return dx
    
    def simulate(self, x0, t_span, u_func=None):
        """
        Simulate the system
        
        Args:
            x0: initial state
            t_span: time vector
            u_func: control input function u(t,x) or constant
        """
        if u_func is None:
            u_func = lambda t, x: np.zeros(self.m)
        
        # Integrate the dynamics
        states = odeint(self.dynamics, x0, t_span, args=(u_func,))
        
        # Compute outputs
        outputs = []
        for t, x in zip(t_span, states):
            if callable(u_func):
                u = u_func(t, x)
            else:
                u = u_func
            u = np.atleast_1d(u)
            
            # Ensure u has the right shape
            if u.shape[0] != self.m:
                u = np.ones(self.m) * u[0] if len(u) > 0 else np.zeros(self.m)
            
            y = self.h(x) + self.J(x).reshape(self.l, self.m) @ u
            outputs.append(y)
        
        outputs = np.array(outputs)
        
        return states, outputs

# Demonstration 1: Phase Portrait and Trajectories
def demo_phase_portrait():
    """Visualize the phase portrait and system trajectories"""
    sys = NonlinearDynamicalSystem(n_states=2)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Vector field (no control)
    ax = axes[0]
    x1_range = np.linspace(-3, 3, 20)
    x2_range = np.linspace(-3, 3, 20)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    DX1 = np.zeros_like(X1)
    DX2 = np.zeros_like(X2)
    
    for i in range(len(x1_range)):
        for j in range(len(x2_range)):
            state = np.array([X1[j,i], X2[j,i]])
            dx = sys.f(state)
            DX1[j,i] = dx[0]
            DX2[j,i] = dx[1]
    
    # Normalize arrows for better visualization
    M = np.sqrt(DX1**2 + DX2**2)
    M[M == 0] = 1
    
    ax.quiver(X1, X2, DX1/M, DX2/M, M, cmap='viridis', alpha=0.6)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('Vector Field $f(x)$ (No Control)')
    ax.grid(True, alpha=0.3)
    
    # Add equilibrium point
    ax.plot(0, 0, 'ro', markersize=10, label='Equilibrium')
    ax.legend()
    
    # 2. Uncontrolled trajectories
    ax = axes[1]
    t_span = np.linspace(0, 20, 1000)
    
    # Multiple initial conditions
    initial_conditions = [
        [2, 0], [-2, 0], [0, 2], [0, -2],
        [1.5, 1.5], [-1.5, -1.5], [1.5, -1.5], [-1.5, 1.5]
    ]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(initial_conditions)))
    
    for x0, color in zip(initial_conditions, colors):
        states, _ = sys.simulate(x0, t_span)
        ax.plot(states[:, 0], states[:, 1], color=color, alpha=0.7, linewidth=1.5)
        ax.plot(x0[0], x0[1], 'o', color=color, markersize=8)
    
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('Uncontrolled Trajectories')
    ax.grid(True, alpha=0.3)
    ax.plot(0, 0, 'ro', markersize=10, label='Equilibrium')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    
    # 3. Controlled trajectory with feedback
    ax = axes[2]
    
    # Define a stabilizing feedback control
    K = np.array([[-1, -2]])  # Feedback gain
    u_feedback = lambda t, x: -K @ x
    
    # Simulate controlled system
    states_ctrl, _ = sys.simulate([2, 1], t_span, u_feedback)
    states_no_ctrl, _ = sys.simulate([2, 1], t_span, None)
    
    ax.plot(states_no_ctrl[:, 0], states_no_ctrl[:, 1], 
            'r--', alpha=0.5, linewidth=2, label='No Control')
    ax.plot(states_ctrl[:, 0], states_ctrl[:, 1], 
            'b-', alpha=0.7, linewidth=2, label='With Feedback Control')
    
    # Mark initial and final points
    ax.plot(2, 1, 'go', markersize=10, label='Initial State')
    ax.plot(0, 0, 'ro', markersize=10, label='Target Equilibrium')
    
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('Feedback Control: $u = -Kx$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-1, 1.5)
    
    plt.tight_layout()
    plt.suptitle('Nonlinear Dynamical System Behavior', y=1.02, fontsize=14)
    plt.show()

# Demonstration 2: Time evolution
def demo_time_evolution():
    """Show state and output evolution over time"""
    sys = NonlinearDynamicalSystem(n_states=2)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    t_span = np.linspace(0, 30, 1000)
    x0 = [2, 1]
    
    # No control
    states_no_ctrl, outputs_no_ctrl = sys.simulate(x0, t_span, None)
    
    # Constant control
    u_const = 0.5
    states_const, outputs_const = sys.simulate(x0, t_span, u_const)
    
    # Feedback control
    K = np.array([[-0.5, -1]])
    u_feedback = lambda t, x: -K @ x
    states_fb, outputs_fb = sys.simulate(x0, t_span, u_feedback)
    
    # Sinusoidal control
    u_sin = lambda t, x: 0.5 * np.sin(2*t)
    states_sin, outputs_sin = sys.simulate(x0, t_span, u_sin)
    
    # Plot state evolution
    ax = axes[0, 0]
    ax.plot(t_span, states_no_ctrl[:, 0], 'r-', label='No Control', alpha=0.7)
    ax.plot(t_span, states_const[:, 0], 'g-', label='Constant u=0.5', alpha=0.7)
    ax.plot(t_span, states_fb[:, 0], 'b-', label='Feedback', alpha=0.7)
    ax.plot(t_span, states_sin[:, 0], 'm-', label='Sinusoidal', alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('$x_1$')
    ax.set_title('State $x_1$ Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(t_span, states_no_ctrl[:, 1], 'r-', alpha=0.7)
    ax.plot(t_span, states_const[:, 1], 'g-', alpha=0.7)
    ax.plot(t_span, states_fb[:, 1], 'b-', alpha=0.7)
    ax.plot(t_span, states_sin[:, 1], 'm-', alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('$x_2$')
    ax.set_title('State $x_2$ Evolution')
    ax.grid(True, alpha=0.3)
    
    # Plot output evolution
    ax = axes[1, 0]
    ax.plot(t_span, outputs_no_ctrl[:, 0], 'r-', label='No Control', alpha=0.7)
    ax.plot(t_span, outputs_const[:, 0], 'g-', label='Constant u=0.5', alpha=0.7)
    ax.plot(t_span, outputs_fb[:, 0], 'b-', label='Feedback', alpha=0.7)
    ax.plot(t_span, outputs_sin[:, 0], 'm-', label='Sinusoidal', alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('$y$')
    ax.set_title('Output Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot control inputs
    ax = axes[1, 1]
    u_fb_vals = []
    for t, x in zip(t_span, states_fb):
        u = u_feedback(t, x)
        u_fb_vals.append(u[0] if isinstance(u, np.ndarray) else u)
    
    u_sin_vals = []
    for t in t_span:
        u = u_sin(t, None)
        u_sin_vals.append(u[0] if isinstance(u, np.ndarray) else u)
    
    ax.axhline(y=0, color='r', linestyle='-', alpha=0.7, label='No Control')
    ax.axhline(y=u_const, color='g', linestyle='-', alpha=0.7, label='Constant')
    ax.plot(t_span, u_fb_vals, 'b-', alpha=0.7, label='Feedback')
    ax.plot(t_span, u_sin_vals, 'm-', alpha=0.7, label='Sinusoidal')
    ax.set_xlabel('Time')
    ax.set_ylabel('Control Input $u$')
    ax.set_title('Control Signals')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('System Evolution Under Different Control Strategies', y=1.02, fontsize=14)
    plt.show()

# Run demonstrations
if __name__ == "__main__":
    print("Demonstrating Basic Nonlinear Dynamical System")
    print("=" * 50)
    print("\n1. Phase Portrait and Trajectories:")
    demo_phase_portrait()
    print("\n2. Time Evolution Under Different Controls:")
    demo_time_evolution()