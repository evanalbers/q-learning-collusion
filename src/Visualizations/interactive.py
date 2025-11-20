"""
Interactive Nonlinear Dynamical System Visualization
Allows real-time parameter adjustment to explore system behavior
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

# Use interactive backend
plt.ion()
plt.style.use('seaborn-v0_8-darkgrid')

class InteractiveDynamicalSystem:
    def __init__(self):
        """
        Interactive nonlinear dynamical system with adjustable parameters
        """
        self.n = 2  # 2D system for visualization
        self.m = 1  # Single control input
        self.l = 1  # Single output
        
        # Adjustable parameters
        self.damping = 0.5
        self.nonlinearity = 0.1
        self.control_gain = 1.0
        self.feedback_gain = np.array([[-1.0, -2.0]])
        
        # Initial conditions
        self.x0 = np.array([2.0, 1.0])
        
        # Control mode
        self.control_mode = 'none'
        
        # Time parameters
        self.t_max = 30
        self.dt = 0.01
        
        # Initialize the figure
        self.setup_figure()
        
    def f(self, x):
        """Natural dynamics with adjustable parameters"""
        x1, x2 = x
        return np.array([
            x2,
            -x1 - self.damping * (x1**2 - 1) * x2 - self.nonlinearity * x1**3
        ])
    
    def G(self, x):
        """Control input matrix with adjustable gain"""
        return np.array([[0], [self.control_gain + 0.1 * x[0]**2]]).flatten()
    
    def h(self, x):
        """Output function"""
        return np.array([x[0]])
    
    def J(self, x):
        """Direct feedthrough"""
        return np.zeros((self.l, self.m))
    
    def control_input(self, t, x):
        """Control input based on selected mode"""
        if self.control_mode == 'none':
            return np.zeros(1)
        elif self.control_mode == 'feedback':
            return -self.feedback_gain @ x
        elif self.control_mode == 'sinusoidal':
            return 0.5 * np.sin(2 * t)
        elif self.control_mode == 'constant':
            return 0.5 * np.ones(1)
        else:
            return np.zeros(1)
    
    def dynamics(self, x, t):
        """Complete dynamics for integration"""
        u = self.control_input(t, x)
        u = np.atleast_1d(u)
        dx = self.f(x) + self.G(x).reshape(self.n, self.m) @ u
        return dx
    
    def simulate(self):
        """Simulate the system with current parameters"""
        t_span = np.linspace(0, self.t_max, int(self.t_max/self.dt))
        states = odeint(self.dynamics, self.x0, t_span)
        
        # Calculate outputs and control signals
        outputs = []
        controls = []
        for t, x in zip(t_span, states):
            y = self.h(x)
            u = self.control_input(t, x)
            outputs.append(y)
            controls.append(u)
        
        return t_span, states, np.array(outputs), np.array(controls)
    
    def setup_figure(self):
        """Setup the interactive figure with plots and controls"""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Interactive Nonlinear Dynamical System Explorer', fontsize=16)
        
        # Create grid for plots
        gs = self.fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.3], 
                                   width_ratios=[1, 1, 1],
                                   hspace=0.3, wspace=0.3)
        
        # Main plots
        self.ax_phase = self.fig.add_subplot(gs[0, 0])
        self.ax_vector = self.fig.add_subplot(gs[0, 1])
        self.ax_3d = self.fig.add_subplot(gs[0, 2], projection='3d')
        self.ax_states = self.fig.add_subplot(gs[1, 0])
        self.ax_output = self.fig.add_subplot(gs[1, 1])
        self.ax_control = self.fig.add_subplot(gs[1, 2])
        
        # Control panel area
        self.ax_controls = self.fig.add_subplot(gs[2, :])
        self.ax_controls.axis('off')
        
        # Create sliders
        self.create_sliders()
        
        # Create buttons and radio buttons
        self.create_buttons()
        
        # Initial plot
        self.update_plots()
        
    def create_sliders(self):
        """Create parameter sliders"""
        # Define slider positions [left, bottom, width, height]
        slider_color = 'lightblue'
        
        # Damping slider
        ax_damping = plt.axes([0.15, 0.22, 0.3, 0.02])
        self.slider_damping = Slider(ax_damping, 'Damping', 0.0, 2.0, 
                                     valinit=self.damping, color=slider_color)
        self.slider_damping.on_changed(self.update_damping)
        
        # Nonlinearity slider
        ax_nonlin = plt.axes([0.15, 0.19, 0.3, 0.02])
        self.slider_nonlin = Slider(ax_nonlin, 'Nonlinearity', 0.0, 1.0, 
                                    valinit=self.nonlinearity, color=slider_color)
        self.slider_nonlin.on_changed(self.update_nonlinearity)
        
        # Control gain slider
        ax_control = plt.axes([0.15, 0.16, 0.3, 0.02])
        self.slider_control = Slider(ax_control, 'Control Gain', 0.0, 3.0, 
                                     valinit=self.control_gain, color=slider_color)
        self.slider_control.on_changed(self.update_control_gain)
        
        # Feedback gain sliders
        ax_k1 = plt.axes([0.55, 0.22, 0.3, 0.02])
        self.slider_k1 = Slider(ax_k1, 'Feedback K1', -5.0, 0.0, 
                               valinit=self.feedback_gain[0, 0], color=slider_color)
        self.slider_k1.on_changed(self.update_k1)
        
        ax_k2 = plt.axes([0.55, 0.19, 0.3, 0.02])
        self.slider_k2 = Slider(ax_k2, 'Feedback K2', -5.0, 0.0, 
                               valinit=self.feedback_gain[0, 1], color=slider_color)
        self.slider_k2.on_changed(self.update_k2)
        
        # Initial condition sliders
        ax_x0 = plt.axes([0.55, 0.16, 0.3, 0.02])
        self.slider_x0 = Slider(ax_x0, 'Initial x1', -3.0, 3.0, 
                               valinit=self.x0[0], color='lightgreen')
        self.slider_x0.on_changed(self.update_x0)
        
        ax_y0 = plt.axes([0.55, 0.13, 0.3, 0.02])
        self.slider_y0 = Slider(ax_y0, 'Initial x2', -3.0, 3.0, 
                               valinit=self.x0[1], color='lightgreen')
        self.slider_y0.on_changed(self.update_y0)
        
        # Time span slider
        ax_time = plt.axes([0.15, 0.13, 0.3, 0.02])
        self.slider_time = Slider(ax_time, 'Time Span', 5, 100, 
                                 valinit=self.t_max, color='lightyellow')
        self.slider_time.on_changed(self.update_time)
    
    def create_buttons(self):
        """Create control buttons and radio buttons"""
        # Control mode radio buttons
        ax_radio = plt.axes([0.05, 0.05, 0.15, 0.12])
        self.radio = RadioButtons(ax_radio, 
                                  ('No Control', 'Feedback', 'Sinusoidal', 'Constant'),
                                  active=0)
        self.radio.on_clicked(self.update_control_mode)
        
        # Reset button
        ax_reset = plt.axes([0.88, 0.22, 0.08, 0.03])
        self.btn_reset = Button(ax_reset, 'Reset', color='lightcoral')
        self.btn_reset.on_clicked(self.reset_parameters)
        
        # Animation button
        ax_animate = plt.axes([0.88, 0.18, 0.08, 0.03])
        self.btn_animate = Button(ax_animate, 'Animate', color='lightgreen')
        self.btn_animate.on_clicked(self.animate_trajectory)
        
        # Options checkboxes
        ax_check = plt.axes([0.25, 0.05, 0.15, 0.08])
        self.check = CheckButtons(ax_check, 
                                  ['Show Vector Field', 'Show Equilibria', 'Show Energy'],
                                  [True, True, False])
        self.check.on_clicked(self.update_plots)
    
    def update_plots(self, val=None):
        """Update all plots with current parameters"""
        # Clear all axes
        self.ax_phase.clear()
        self.ax_vector.clear()
        self.ax_3d.clear()
        self.ax_states.clear()
        self.ax_output.clear()
        self.ax_control.clear()
        
        # Simulate system
        t_span, states, outputs, controls = self.simulate()
        
        # Get checkbox states
        show_vector = self.check.get_status()[0]
        show_equilibria = self.check.get_status()[1]
        show_energy = self.check.get_status()[2]
        
        # 1. Phase portrait
        self.ax_phase.plot(states[:, 0], states[:, 1], 'b-', linewidth=2, alpha=0.7)
        self.ax_phase.plot(self.x0[0], self.x0[1], 'go', markersize=10, label='Start')
        self.ax_phase.plot(states[-1, 0], states[-1, 1], 'ro', markersize=10, label='End')
        
        if show_equilibria:
            self.ax_phase.plot(0, 0, 'k*', markersize=15, label='Equilibrium')
        
        if show_energy:
            # Add energy contours
            x_range = np.linspace(-3, 3, 30)
            y_range = np.linspace(-3, 3, 30)
            X, Y = np.meshgrid(x_range, y_range)
            Z = 0.5 * (X**2 + Y**2)
            CS = self.ax_phase.contour(X, Y, Z, levels=10, colors='gray', alpha=0.3)
            self.ax_phase.clabel(CS, inline=True, fontsize=6)
        
        self.ax_phase.set_xlabel('$x_1$')
        self.ax_phase.set_ylabel('$x_2$')
        self.ax_phase.set_title('Phase Portrait')
        self.ax_phase.legend()
        self.ax_phase.grid(True, alpha=0.3)
        self.ax_phase.set_xlim(-3, 3)
        self.ax_phase.set_ylim(-3, 3)
        
        # 2. Vector field
        if show_vector:
            x1_range = np.linspace(-3, 3, 15)
            x2_range = np.linspace(-3, 3, 15)
            X1, X2 = np.meshgrid(x1_range, x2_range)
            
            DX1 = np.zeros_like(X1)
            DX2 = np.zeros_like(X2)
            
            for i in range(len(x1_range)):
                for j in range(len(x2_range)):
                    state = np.array([X1[j,i], X2[j,i]])
                    dx = self.f(state)
                    if self.control_mode == 'feedback':
                        u = self.control_input(0, state)
                        dx += self.G(state).reshape(2, 1).flatten() * u
                    DX1[j,i] = dx[0]
                    DX2[j,i] = dx[1]
            
            M = np.sqrt(DX1**2 + DX2**2)
            M[M == 0] = 1
            
            self.ax_vector.quiver(X1, X2, DX1/M, DX2/M, M, cmap='viridis', alpha=0.6)
            self.ax_vector.plot(states[:, 0], states[:, 1], 'r-', linewidth=2, alpha=0.7)
        else:
            self.ax_vector.plot(states[:, 0], states[:, 1], 'b-', linewidth=2)
            
        self.ax_vector.set_xlabel('$x_1$')
        self.ax_vector.set_ylabel('$x_2$')
        self.ax_vector.set_title('Vector Field & Trajectory')
        self.ax_vector.grid(True, alpha=0.3)
        self.ax_vector.set_xlim(-3, 3)
        self.ax_vector.set_ylim(-3, 3)
        
        # 3. 3D trajectory with time
        self.ax_3d.plot(states[:, 0], states[:, 1], t_span, 'b-', linewidth=2)
        self.ax_3d.scatter(self.x0[0], self.x0[1], 0, s=100, c='g', marker='o')
        self.ax_3d.scatter(states[-1, 0], states[-1, 1], t_span[-1], s=100, c='r', marker='o')
        self.ax_3d.set_xlabel('$x_1$')
        self.ax_3d.set_ylabel('$x_2$')
        self.ax_3d.set_zlabel('Time')
        self.ax_3d.set_title('Space-Time Trajectory')
        
        # 4. State evolution
        self.ax_states.plot(t_span, states[:, 0], 'b-', label='$x_1$', linewidth=2)
        self.ax_states.plot(t_span, states[:, 1], 'r-', label='$x_2$', linewidth=2)
        self.ax_states.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        self.ax_states.set_xlabel('Time')
        self.ax_states.set_ylabel('States')
        self.ax_states.set_title('State Evolution')
        self.ax_states.legend()
        self.ax_states.grid(True, alpha=0.3)
        
        # 5. Output
        self.ax_output.plot(t_span, outputs[:, 0], 'g-', linewidth=2)
        self.ax_output.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        self.ax_output.set_xlabel('Time')
        self.ax_output.set_ylabel('Output $y$')
        self.ax_output.set_title('System Output')
        self.ax_output.grid(True, alpha=0.3)
        
        # 6. Control signal
        self.ax_control.plot(t_span, controls, 'm-', linewidth=2)
        self.ax_control.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        self.ax_control.set_xlabel('Time')
        self.ax_control.set_ylabel('Control $u$')
        self.ax_control.set_title(f'Control Signal ({self.control_mode})')
        self.ax_control.grid(True, alpha=0.3)
        
        plt.draw()
    
    def animate_trajectory(self, event):
        """Animate the trajectory (simplified version)"""
        # Clear phase portrait
        self.ax_phase.clear()
        
        # Simulate system
        t_span, states, _, _ = self.simulate()
        
        # Plot background
        self.ax_phase.plot(states[:, 0], states[:, 1], 'b-', alpha=0.2, linewidth=1)
        self.ax_phase.plot(0, 0, 'k*', markersize=15)
        
        # Animate
        for i in range(0, len(states), 10):
            if i > 0:
                self.ax_phase.plot(states[:i, 0], states[:i, 1], 'b-', linewidth=2, alpha=0.7)
                self.ax_phase.plot(states[i, 0], states[i, 1], 'ro', markersize=8)
                plt.pause(0.01)
        
        self.ax_phase.plot(self.x0[0], self.x0[1], 'go', markersize=10, label='Start')
        self.ax_phase.plot(states[-1, 0], states[-1, 1], 'ro', markersize=10, label='End')
        self.ax_phase.set_xlabel('$x_1$')
        self.ax_phase.set_ylabel('$x_2$')
        self.ax_phase.set_title('Phase Portrait (Animated)')
        self.ax_phase.legend()
        self.ax_phase.grid(True, alpha=0.3)
        plt.draw()
    
    # Callback functions for sliders
    def update_damping(self, val):
        self.damping = self.slider_damping.val
        self.update_plots()
    
    def update_nonlinearity(self, val):
        self.nonlinearity = self.slider_nonlin.val
        self.update_plots()
    
    def update_control_gain(self, val):
        self.control_gain = self.slider_control.val
        self.update_plots()
    
    def update_k1(self, val):
        self.feedback_gain[0, 0] = self.slider_k1.val
        self.update_plots()
    
    def update_k2(self, val):
        self.feedback_gain[0, 1] = self.slider_k2.val
        self.update_plots()
    
    def update_x0(self, val):
        self.x0[0] = self.slider_x0.val
        self.update_plots()
    
    def update_y0(self, val):
        self.x0[1] = self.slider_y0.val
        self.update_plots()
    
    def update_time(self, val):
        self.t_max = self.slider_time.val
        self.update_plots()
    
    def update_control_mode(self, label):
        mode_map = {
            'No Control': 'none',
            'Feedback': 'feedback',
            'Sinusoidal': 'sinusoidal',
            'Constant': 'constant'
        }
        self.control_mode = mode_map[label]
        self.update_plots()
    
    def reset_parameters(self, event):
        """Reset all parameters to defaults"""
        self.damping = 0.5
        self.nonlinearity = 0.1
        self.control_gain = 1.0
        self.feedback_gain = np.array([[-1.0, -2.0]])
        self.x0 = np.array([2.0, 1.0])
        self.t_max = 30
        
        # Update slider positions
        self.slider_damping.set_val(self.damping)
        self.slider_nonlin.set_val(self.nonlinearity)
        self.slider_control.set_val(self.control_gain)
        self.slider_k1.set_val(self.feedback_gain[0, 0])
        self.slider_k2.set_val(self.feedback_gain[0, 1])
        self.slider_x0.set_val(self.x0[0])
        self.slider_y0.set_val(self.x0[1])
        self.slider_time.set_val(self.t_max)
        
        self.update_plots()

def main():
    """Main function to run the interactive system"""
    print("=" * 60)
    print("Interactive Nonlinear Dynamical System Explorer")
    print("=" * 60)
    print("\nInstructions:")
    print("- Use sliders to adjust system parameters")
    print("- Select control mode with radio buttons")
    print("- Toggle display options with checkboxes")
    print("- Click 'Animate' to see trajectory animation")
    print("- Click 'Reset' to restore default parameters")
    print("\nExperiments to try:")
    print("1. Vary damping to see transition from stable to limit cycle")
    print("2. Adjust feedback gains to stabilize unstable systems")
    print("3. Compare different control strategies")
    print("4. Explore how nonlinearity affects system behavior")
    print("5. Change initial conditions to see basins of attraction")
    print("=" * 60)
    
    # Create and show the interactive system
    interactive_sys = InteractiveDynamicalSystem()
    plt.show(block=True)

if __name__ == "__main__":
    main()