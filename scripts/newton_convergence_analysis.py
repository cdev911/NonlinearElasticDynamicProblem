import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('newton_convergence.csv')

# Group by time step
time_steps = df['time_step'].unique()

# Find problematic time steps (>6 iterations)
problematic = df.groupby('time_step')['newton_iter'].max()
problematic = problematic[problematic > 6]

print(f"Time steps with slow convergence: {len(problematic)}")
print(f"Max iterations: {problematic.max()}")

# Plot convergence for a few representative time steps
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sample_steps = [10, 500, 1000, 1500]  # Different phases
for idx, step in enumerate(sample_steps):
    ax = axes[idx // 2, idx % 2]
    data = df[df['time_step'] == step]
    
    ax.semilogy(data['newton_iter'], data['normalized_residual'], 'o-', linewidth=2)
    ax.set_xlabel('Newton Iteration')
    ax.set_ylabel('Normalized Residual')
    ax.set_title(f'Time step {step} (t = {step * 0.01:.2f})')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1e-8, color='r', linestyle='--', label='Convergence threshold')
    ax.legend()

plt.tight_layout()
plt.savefig('newton_convergence_samples.png', dpi=300)

# Average iterations vs time
avg_iters = df.groupby('time_step')['newton_iter'].max()
plt.figure(figsize=(12, 6))
plt.plot(avg_iters.index * 0.01, avg_iters.values, linewidth=2)
plt.xlabel('Time')
plt.ylabel('Newton Iterations per Time Step')
plt.title('Solver Performance Throughout Simulation')
plt.grid(True, alpha=0.3)
plt.savefig('newton_iterations_vs_time.png', dpi=300)
