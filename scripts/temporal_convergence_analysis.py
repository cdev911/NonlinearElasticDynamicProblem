import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load results from different time steps
results = {}
time_steps = [0.04, 0.02, 0.01, 0.005, 0.0025]

for dt in time_steps:
    df = pd.read_csv(f'results_dt_{dt}/time_history.csv')
    results[dt] = df

# Compare displacement at fixed time (t = 10.0)
target_time = 10.0
errors = []

# Use finest as reference
reference = results[0.0025]
ref_disp = reference[reference['time'].between(target_time-0.001, target_time+0.001)]['disp_x_tip'].values[0]

for dt in time_steps[:-1]:  # Exclude reference
    df = results[dt]
    disp = df[df['time'].between(target_time-dt, target_time+dt)]['disp_x_tip'].values[0]
    error = np.abs(disp - ref_disp)
    errors.append(error)

# Convergence plot
plt.figure(figsize=(8, 6))
plt.loglog(time_steps[:-1], errors, 'o-', label='Computed error')
plt.loglog(time_steps[:-1], np.array(time_steps[:-1])**2 * errors[0] / time_steps[0]**2, 
           '--', label='2nd order reference')
plt.xlabel('Time step Δt')
plt.ylabel('Error in displacement at t=10.0')
plt.title('Temporal Convergence Study')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.savefig('temporal_convergence.png', dpi=300)

# Compute convergence rate
log_dt = np.log(time_steps[:-1])
log_err = np.log(errors)
slope, _ = np.polyfit(log_dt, log_err, 1)
print(f"Observed convergence rate: {slope:.2f} (expected: 2.0)")
