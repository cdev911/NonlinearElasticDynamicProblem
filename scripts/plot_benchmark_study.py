#!/usr/bin/env python3
"""
Plotting script for 2D Neo-Hookean cantilever benchmark results
Generates comparison plots for time step and mesh refinement studies
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy.interpolate import interp1d

# Set publication-quality plot defaults
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def read_time_history(csv_path):
    """Read time history CSV file"""
    try:
        df = pd.read_csv(csv_path)
        print(f"  Loaded: {csv_path} ({len(df)} points)")
        return df
    except Exception as e:
        print(f"  ERROR loading {csv_path}: {e}")
        return None

def read_energy_balance(csv_path):
    """Read energy balance CSV file"""
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"  ERROR loading {csv_path}: {e}")
        return None

def extract_peak_displacement(df):
    """Extract peak displacement and time"""
    idx_min = df['disp_y'].idxmin()
    peak_disp = df.loc[idx_min, 'disp_y']
    peak_time = df.loc[idx_min, 'time']
    return peak_disp, peak_time

def compute_rms_error(y1, y2):
    """Compute RMS error between two signals"""
    return np.sqrt(np.mean((y1 - y2)**2))

def interpolate_to_common_time(time_ref, data_ref, time_new):
    """Interpolate data to a common time grid"""
    # Only interpolate where new times are within bounds
    valid_mask = (time_new >= time_ref.min()) & (time_new <= time_ref.max())
    time_interp = time_new[valid_mask]
    
    interp_func = interp1d(time_ref, data_ref, kind='linear')
    data_interp = interp_func(time_interp)
    
    return time_interp, data_interp

# ============================================================================
# TIME STEP CONVERGENCE STUDY
# ============================================================================

def plot_time_step_study():
    """Generate all plots for time step convergence study"""
    
    print("\n" + "="*70)
    print("TIME STEP CONVERGENCE STUDY")
    print("="*70)
    
    # Define cases
    cases = {
        'dt=0.020': {'path': 'time_step_study/dt_0.020', 'dt': 0.020, 'label': r'$\Delta t = 0.020$ s'},
        'dt=0.010': {'path': 'time_step_study/dt_0.010', 'dt': 0.010, 'label': r'$\Delta t = 0.010$ s'},
        'dt=0.005': {'path': 'time_step_study/dt_0.005', 'dt': 0.005, 'label': r'$\Delta t = 0.005$ s'},
        'dt=0.001': {'path': 'time_step_study/dt_0.001', 'dt': 0.001, 'label': r'$\Delta t = 0.001$ s (ref)'},
    }
    
    # Load data
    data = {}
    for key, case in cases.items():
        csv_file = Path(case['path']) / 'time_history.csv'
        df = read_time_history(csv_file)
        if df is not None:
            data[key] = df
            case['df'] = df
    
    if len(data) == 0:
        print("ERROR: No data loaded. Check paths.")
        return
    
    # Create output directory
    output_dir = Path('plotting/time_step_study')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== Plot 1: Displacement vs Time ==========
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']  # Red to blue
    
    for i, (key, case) in enumerate(cases.items()):
        if key in data:
            df = data[key]
            ax.plot(df['time'], df['disp_y'], label=case['label'], 
                   color=colors[i], linewidth=2, alpha=0.8)
    
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Load end')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Vertical Displacement $u_y$ (m)')
    ax.set_title('Time Step Convergence: Tip Displacement History')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'displacement_vs_time.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'displacement_vs_time.pdf', bbox_inches='tight')
    print(f"  Saved: {output_dir / 'displacement_vs_time.png'}")
    plt.close()
    
    # ========== Plot 2: Zoomed view around peak ==========
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (key, case) in enumerate(cases.items()):
        if key in data:
            df = data[key]
            mask = (df['time'] >= 0.2) & (df['time'] <= 1.0)
            ax.plot(df.loc[mask, 'time'], df.loc[mask, 'disp_y'], 
                   label=case['label'], color=colors[i], linewidth=2, alpha=0.8)
    
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Vertical Displacement $u_y$ (m)')
    ax.set_title('Time Step Convergence: Peak Region (Zoomed)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'displacement_zoom.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'displacement_zoom.png'}")
    plt.close()
    
    # ========== Plot 3: Energy Conservation ==========
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    for i, (key, case) in enumerate(cases.items()):
        if key in data:
            df = data[key]
            # Total energy and work
            ax1.plot(df['time'], df['E_total'], label=f"{case['label']} (KE+PE)", 
                    color=colors[i], linewidth=2, alpha=0.8)
            ax1.plot(df['time'], df['W_ext'], '--', color=colors[i], 
                    linewidth=1.5, alpha=0.6)
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Energy (J)')
    ax1.set_title('Energy Conservation: Total Energy vs External Work')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Energy error percentage
    for i, (key, case) in enumerate(cases.items()):
        energy_file = Path(case['path']) / 'energy_balance.csv'
        df_energy = read_energy_balance(energy_file)
        if df_energy is not None:
            ax2.semilogy(df_energy['time'], np.abs(df_energy['error_pct']), 
                        label=case['label'], color=colors[i], linewidth=2, alpha=0.8)
    
    ax2.axhline(y=1.0, color='r', linestyle='--', linewidth=1, alpha=0.5, label='1% error')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Energy Error (%)')
    ax2.set_title('Energy Conservation Error (log scale)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'energy_conservation.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'energy_conservation.png'}")
    plt.close()
    
    # ========== Plot 4: Convergence Analysis ==========
    
    # Extract metrics
    dt_values = []
    peak_disps = []
    peak_times = []
    
    for key, case in cases.items():
        if key in data:
            df = data[key]
            peak_disp, peak_time = extract_peak_displacement(df)
            dt_values.append(case['dt'])
            peak_disps.append(peak_disp)
            peak_times.append(peak_time)
    
    dt_values = np.array(dt_values)
    peak_disps = np.array(peak_disps)
    peak_times = np.array(peak_times)
    
    # Reference is finest resolution
    ref_idx = np.argmin(dt_values)
    ref_disp = peak_disps[ref_idx]
    ref_time = peak_times[ref_idx]
    
    # Compute errors
    disp_errors = np.abs(peak_disps - ref_disp)
    time_errors = np.abs(peak_times - ref_time)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Peak displacement convergence
    ax1.loglog(dt_values, disp_errors, 'o-', markersize=8, linewidth=2, label='Computed')
    
    # Add reference lines for convergence orders
    if len(dt_values) > 1:
        dt_plot = np.array([dt_values.min(), dt_values.max()])
        # O(dt^2) reference line
        C2 = disp_errors[-2] / dt_values[-2]**2
        ax1.loglog(dt_plot, C2 * dt_plot**2, 'k--', alpha=0.5, label=r'$O(\Delta t^2)$')
        # O(dt) reference line  
        C1 = disp_errors[-2] / dt_values[-2]
        ax1.loglog(dt_plot, C1 * dt_plot, 'k:', alpha=0.5, label=r'$O(\Delta t)$')
    
    ax1.set_xlabel(r'Time Step $\Delta t$ (s)')
    ax1.set_ylabel(r'Peak Displacement Error $|u_{y,peak} - u_{y,ref}|$ (m)')
    ax1.set_title('Temporal Convergence: Peak Displacement')
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')
    
    # Peak time convergence
    ax2.loglog(dt_values, time_errors, 's-', markersize=8, linewidth=2, 
              color='tab:orange', label='Computed')
    ax2.set_xlabel(r'Time Step $\Delta t$ (s)')
    ax2.set_ylabel(r'Peak Time Error $|t_{peak} - t_{ref}|$ (s)')
    ax2.set_title('Temporal Convergence: Peak Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'convergence_analysis.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'convergence_analysis.png'}")
    plt.close()
    
    # ========== Summary Table ==========
    print("\n" + "-"*70)
    print("SUMMARY: Time Step Study")
    print("-"*70)
    print(f"{'dt (s)':<10} {'Peak u_y (m)':<15} {'Time (s)':<12} {'Error vs ref':<15}")
    print("-"*70)
    for i, (key, case) in enumerate(cases.items()):
        if key in data:
            err = disp_errors[i] if i < len(disp_errors) else 0
            err_pct = 100 * err / abs(ref_disp) if ref_disp != 0 else 0
            print(f"{case['dt']:<10.4f} {peak_disps[i]:<15.6f} {peak_times[i]:<12.4f} {err_pct:<15.4f}%")
    print("-"*70)

# ============================================================================
# MESH REFINEMENT STUDY
# ============================================================================

def plot_mesh_study():
    """Generate all plots for mesh refinement study"""
    
    print("\n" + "="*70)
    print("MESH REFINEMENT STUDY")
    print("="*70)
    
    # Define cases
    cases = {
        'ref_2': {'path': 'mesh_study/ref_2', 'level': 2, 'elements': 640, 'label': 'Level 2 (640 elem)'},
        'ref_3': {'path': 'mesh_study/ref_3', 'level': 3, 'elements': 1280, 'label': 'Level 3 (1280 elem)'},
        'ref_4': {'path': 'mesh_study/ref_4', 'level': 4, 'elements': 2560, 'label': 'Level 4 (2560 elem)'},
        'ref_5': {'path': 'mesh_study/ref_5', 'level': 5, 'elements': 5120, 'label': 'Level 5 (5120 elem, ref)'},
    }
    
    # Load data
    data = {}
    for key, case in cases.items():
        csv_file = Path(case['path']) / 'time_history.csv'
        df = read_time_history(csv_file)
        if df is not None:
            data[key] = df
            case['df'] = df
    
    if len(data) == 0:
        print("ERROR: No data loaded. Check paths.")
        return
    
    # Create output directory
    output_dir = Path('plotting/mesh_study')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== Plot 1: Displacement vs Time ==========
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
    
    for i, (key, case) in enumerate(cases.items()):
        if key in data:
            df = data[key]
            ax.plot(df['time'], df['disp_y'], label=case['label'], 
                   color=colors[i], linewidth=2, alpha=0.8)
    
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Load end')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Vertical Displacement $u_y$ (m)')
    ax.set_title('Mesh Convergence: Tip Displacement History')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'displacement_vs_time.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'displacement_vs_time.pdf', bbox_inches='tight')
    print(f"  Saved: {output_dir / 'displacement_vs_time.png'}")
    plt.close()
    
    # ========== Plot 2: Convergence Analysis ==========
    
    # Extract metrics
    h_values = []  # Characteristic element size
    peak_disps = []
    peak_times = []
    
    for key, case in cases.items():
        if key in data:
            df = data[key]
            peak_disp, peak_time = extract_peak_displacement(df)
            # h ~ 1 / sqrt(elements)
            h = 1.0 / np.sqrt(case['elements'])
            h_values.append(h)
            peak_disps.append(peak_disp)
            peak_times.append(peak_time)
    
    h_values = np.array(h_values)
    peak_disps = np.array(peak_disps)
    peak_times = np.array(peak_times)
    
    # Reference is finest mesh
    ref_idx = np.argmin(h_values)
    ref_disp = peak_disps[ref_idx]
    
    # Compute errors
    disp_errors = np.abs(peak_disps - ref_disp)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Remove zero error for reference case
    plot_h = h_values[disp_errors > 0]
    plot_errors = disp_errors[disp_errors > 0]
    
    ax.loglog(plot_h, plot_errors, 'o-', markersize=10, linewidth=2, label='Computed')
    
    # Add reference lines
    if len(plot_h) > 1:
        h_plot = np.array([plot_h.min(), plot_h.max()])
        # O(h^2) reference line
        C2 = plot_errors[0] / plot_h[0]**2
        ax.loglog(h_plot, C2 * h_plot**2, 'k--', alpha=0.5, label=r'$O(h^2)$')
        # O(h) reference line
        C1 = plot_errors[0] / plot_h[0]
        ax.loglog(h_plot, C1 * h_plot, 'k:', alpha=0.5, label=r'$O(h)$')
    
    ax.set_xlabel('Characteristic Element Size $h$')
    ax.set_ylabel(r'Peak Displacement Error $|u_{y,peak} - u_{y,ref}|$ (m)')
    ax.set_title('Spatial Convergence: Peak Displacement (Q1 Elements)')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'convergence_analysis.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'convergence_analysis.png'}")
    plt.close()
    
    # ========== Summary Table ==========
    print("\n" + "-"*70)
    print("SUMMARY: Mesh Refinement Study")
    print("-"*70)
    print(f"{'Level':<8} {'Elements':<12} {'Peak u_y (m)':<15} {'Error vs ref':<15}")
    print("-"*70)
    for i, (key, case) in enumerate(cases.items()):
        if key in data:
            err = disp_errors[i]
            err_pct = 100 * err / abs(ref_disp) if ref_disp != 0 else 0
            print(f"{case['level']:<8} {case['elements']:<12} {peak_disps[i]:<15.6f} {err_pct:<15.4f}%")
    print("-"*70)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("2D NEO-HOOKEAN CANTILEVER BENCHMARK - PLOTTING SCRIPT")
    print("="*70)
    
    # Generate time step study plots
    try:
        plot_time_step_study()
    except Exception as e:
        print(f"\nERROR in time step study: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate mesh refinement study plots
    # try:
    #     plot_mesh_study()
    # except Exception as e:
    #     print(f"\nERROR in mesh study: {e}")
    #     import traceback
    #     traceback.print_exc()
    
    print("\n" + "="*70)
    print("PLOTTING COMPLETE")
    print("="*70)
    print("\nOutput locations:")
    print("  - plotting/time_step_study/")
    print("  - plotting/mesh_study/")
    print("="*70 + "\n")