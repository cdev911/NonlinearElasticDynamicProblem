import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

df = pd.read_csv('time_history.csv')

# Create comprehensive figure
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. Displacement trajectory (tip orbit)
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(df['disp_x_tip'], df['disp_y_tip'], linewidth=2, alpha=0.7)
ax1.set_xlabel('X Displacement')
ax1.set_ylabel('Y Displacement')
ax1.set_title('Tip Trajectory')
ax1.axis('equal')
ax1.grid(True, alpha=0.3)

# 2. Displacement vs time
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(df['time'], df['disp_x_tip'], label='X displacement', linewidth=2)
ax2.plot(df['time'], df['disp_y_tip'], label='Y displacement', linewidth=2)
ax2.set_xlabel('Time')
ax2.set_ylabel('Displacement')
ax2.set_title('Displacement History')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Velocity vs time
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(df['time'], df['vel_x_tip'], label='X velocity', linewidth=2)
ax3.plot(df['time'], df['vel_y_tip'], label='Y velocity', linewidth=2)
ax3.set_xlabel('Time')
ax3.set_ylabel('Velocity')
ax3.set_title('Velocity History')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Energy vs time
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(df['time'], df['kinetic_energy'], label='Kinetic', linewidth=2)
ax4.plot(df['time'], df['potential_energy'], label='Potential', linewidth=2)
ax4.plot(df['time'], df['total_energy'], label='Total', linewidth=2, linestyle='--')
ax4.set_xlabel('Time')
ax4.set_ylabel('Energy')
ax4.set_title('Energy Evolution')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Load angle vs time
ax5 = fig.add_subplot(gs[2, 0])
ax5.plot(df['time'], df['load_angle_deg'], linewidth=2)
ax5.set_xlabel('Time')
ax5.set_ylabel('Load Angle (degrees)')
ax5.set_title('Applied Load Direction')
ax5.grid(True, alpha=0.3)

# 6. Newton iterations vs time
ax6 = fig.add_subplot(gs[2, 1])
ax6.plot(df['time'], df['newton_iterations'], linewidth=2)
ax6.set_xlabel('Time')
ax6.set_ylabel('Iterations')
ax6.set_title('Newton-Raphson Convergence Rate')
ax6.grid(True, alpha=0.3)

plt.savefig('comprehensive_response.png', dpi=300)
