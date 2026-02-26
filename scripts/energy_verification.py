import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('energy_balance.csv')

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Energy components
ax = axes[0]
ax.plot(df['time'], df['KE'], label='Kinetic Energy', linewidth=2)
ax.plot(df['time'], df['PE'], label='Potential Energy', linewidth=2)
ax.plot(df['time'], df['E_total'], label='Total Energy (KE+PE)', linewidth=2, linestyle='--')
ax.plot(df['time'], df['W_external'], label='External Work', linewidth=2)
ax.plot(df['time'], df['D_viscous'], label='Dissipated Energy', linewidth=2)
ax.set_xlabel('Time')
ax.set_ylabel('Energy')
ax.set_title('Energy Components vs Time')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Energy balance error
ax = axes[1]
ax.semilogy(df['time'], np.abs(df['energy_error']), linewidth=2, color='red')
ax.set_xlabel('Time')
ax.set_ylabel('Absolute Energy Error')
ax.set_title('Energy Conservation Error (should be < 1e-6)')
ax.grid(True, alpha=0.3)
ax.axhline(y=1e-6, color='k', linestyle='--', label='Target threshold')
ax.legend()

plt.tight_layout()
plt.savefig('energy_verification.png', dpi=300)

# Statistics
print(f"Max energy error: {df['energy_error'].abs().max():.2e}")
print(f"Mean energy error: {df['energy_error'].abs().mean():.2e}")
print(f"Max relative error: {df['relative_error_percent'].abs().max():.2f}%")
