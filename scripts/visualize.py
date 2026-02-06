#!/usr/bin/env python3
"""
Tokamak Disruption Simulation - Visualization and Analysis Tools

Reads diagnostic CSV output and field snapshot data files to produce
publication-quality plots of the disruption cascade.

Usage:
    python3 visualize.py [output_dir]
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import sys
import struct
import glob


def read_diagnostics_csv(filepath):
    """Read the diagnostics CSV file into a dictionary of arrays."""
    data = {}
    with open(filepath, 'r') as f:
        header = f.readline().strip().split(',')
        for col in header:
            data[col] = []

        for line in f:
            values = line.strip().split(',')
            for col, val in zip(header, values):
                try:
                    data[col].append(float(val))
                except ValueError:
                    data[col].append(val)

    # Convert numeric columns to numpy arrays
    for col in data:
        try:
            data[col] = np.array(data[col], dtype=float)
        except (ValueError, TypeError):
            pass

    return data


def read_field_snapshot(filepath):
    """Read a binary field snapshot (.dat file)."""
    with open(filepath, 'rb') as f:
        nr = struct.unpack('i', f.read(4))[0]
        nz = struct.unpack('i', f.read(4))[0]
        bounds = struct.unpack('4d', f.read(32))
        R_min, R_max, Z_min, Z_max = bounds

        data = np.frombuffer(f.read(nr * nz * 8), dtype=np.float64)
        data = data.reshape(nr, nz)

    R = np.linspace(R_min, R_max, nr)
    Z = np.linspace(Z_min, Z_max, nz)

    return R, Z, data


def plot_diagnostics_timeseries(data, output_dir):
    """Plot diagnostic time series showing the disruption cascade."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Tokamak Disruption Cascade - Time Series Diagnostics',
                 fontsize=16, fontweight='bold')

    time = data.get('time_ms', np.array([]))
    if len(time) == 0:
        print("No time series data found")
        return

    # 1. Temperature evolution
    ax = axes[0, 0]
    Te_avg = data.get('Te_avg_keV', np.zeros_like(time))
    Te_max = data.get('Te_max_keV', np.zeros_like(time))
    ax.semilogy(time, np.maximum(Te_avg, 1e-4), 'b-', label='Average Te', linewidth=2)
    ax.semilogy(time, np.maximum(Te_max, 1e-4), 'r--', label='Peak Te', linewidth=1.5)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Electron Temperature [keV]')
    ax.set_title('Temperature Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Plasma current
    ax = axes[0, 1]
    Ip = data.get('Ip_A', np.zeros_like(time))
    ax.plot(time, Ip / 1e6, 'g-', linewidth=2)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Plasma Current [MA]')
    ax.set_title('Current Decay')
    ax.grid(True, alpha=0.3)

    # 3. Magnetic island width
    ax = axes[0, 2]
    iw = data.get('island_width_m', np.zeros_like(time))
    ax.plot(time, iw, 'm-', linewidth=2)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Island Width [m]')
    ax.set_title('Magnetic Island Growth')
    ax.grid(True, alpha=0.3)

    # 4. Electric field
    ax = axes[1, 0]
    E = data.get('E_field_Vm', np.zeros_like(time))
    ax.semilogy(time, np.maximum(np.abs(E), 1e-10), 'orange', linewidth=2)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Electric Field [V/m]')
    ax.set_title('Induced Electric Field')
    ax.grid(True, alpha=0.3)

    # 5. Runaway electron current
    ax = axes[1, 1]
    I_RE = data.get('I_RE_A', np.zeros_like(time))
    ax.semilogy(time, np.maximum(np.abs(I_RE), 1e-1), 'r-', linewidth=2)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('RE Current [A]')
    ax.set_title('Runaway Electron Current')
    ax.grid(True, alpha=0.3)

    # 6. Radiated power
    ax = axes[1, 2]
    P_rad = data.get('P_rad_W', np.zeros_like(time))
    ax.semilogy(time, np.maximum(np.abs(P_rad), 1e-1), 'c-', linewidth=2)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Radiated Power [W]')
    ax.set_title('Radiation Losses')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = os.path.join(output_dir, 'disruption_timeseries.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


def plot_field_snapshot(R, Z, data, title, filename, output_dir,
                        cmap='inferno', log_scale=False):
    """Plot a 2D field snapshot on the tokamak cross-section."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    R_mesh, Z_mesh = np.meshgrid(R, Z, indexing='ij')

    plot_data = data.copy()
    if log_scale:
        plot_data = np.maximum(np.abs(plot_data), 1e-30)
        im = ax.pcolormesh(R_mesh, Z_mesh, plot_data, cmap=cmap,
                           norm=LogNorm(), shading='auto')
    else:
        im = ax.pcolormesh(R_mesh, Z_mesh, plot_data, cmap=cmap,
                           shading='auto')

    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xlabel('R [m]', fontsize=12)
    ax.set_ylabel('Z [m]', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')

    # Draw tokamak outline (simplified circular cross-section)
    theta = np.linspace(0, 2 * np.pi, 100)
    R0, a = 6.2, 2.0
    ax.plot(R0 + a * np.cos(theta), a * np.sin(theta), 'w--',
            linewidth=1.5, alpha=0.7, label='Plasma boundary')
    ax.legend(loc='upper right')

    outpath = os.path.join(output_dir, filename)
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


def plot_all_snapshots(output_dir):
    """Find and plot all field snapshots in the output directory."""
    patterns = {
        'psi_*.dat': ('Poloidal Flux ψ', 'RdBu_r', False),
        'pressure_*.dat': ('Plasma Pressure', 'inferno', False),
        'Te_*.dat': ('Electron Temperature [keV]', 'hot', True),
        'E_field_*.dat': ('Electric Field [V/m]', 'viridis', True),
        'n_RE_*.dat': ('Runaway Electron Density [m⁻³]', 'plasma', True),
    }

    for pattern, (title, cmap, log_scale) in patterns.items():
        files = sorted(glob.glob(os.path.join(output_dir, pattern)))
        for filepath in files:
            try:
                R, Z, data = read_field_snapshot(filepath)
                basename = os.path.splitext(os.path.basename(filepath))[0]
                plot_field_snapshot(R, Z, data, title,
                                   f'{basename}.png', output_dir,
                                   cmap=cmap, log_scale=log_scale)
            except Exception as e:
                print(f"  Warning: Could not plot {filepath}: {e}")


def plot_disruption_phases(data, output_dir):
    """Create a single summary plot showing phase transitions."""
    fig, ax = plt.subplots(figsize=(14, 6))

    time = data.get('time_ms', np.array([]))
    phases = data.get('phase', [])

    if len(time) == 0:
        return

    # Color-code phases
    phase_colors = {
        'MHD_EQUILIBRIUM': '#2196F3',
        'MHD_UNSTABLE': '#FF9800',
        'THERMAL_QUENCH': '#F44336',
        'CURRENT_QUENCH': '#9C27B0',
        'RUNAWAY_PLATEAU': '#E91E63',
        'MITIGATED': '#4CAF50',
        'POST_DISRUPTION': '#607D8B',
    }

    # Plot temperature on primary axis
    Te = data.get('Te_avg_keV', np.zeros_like(time))
    ax.semilogy(time, np.maximum(Te, 1e-4), 'b-', linewidth=2,
                label='Temperature [keV]')

    # Shade phase regions
    if isinstance(phases, list) or isinstance(phases, np.ndarray):
        prev_phase = None
        start_t = time[0] if len(time) > 0 else 0
        for i, (t, phase) in enumerate(zip(time, phases)):
            if isinstance(phase, str) and phase != prev_phase:
                if prev_phase is not None:
                    color = phase_colors.get(str(prev_phase), '#CCCCCC')
                    ax.axvspan(start_t, t, alpha=0.15, color=color,
                               label=str(prev_phase) if i < 10 else '')
                start_t = t
                prev_phase = phase

    ax.set_xlabel('Time [ms]', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Disruption Cascade Phase Evolution', fontsize=14,
                 fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    outpath = os.path.join(output_dir, 'disruption_phases.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else 'output'

    print(f"\n=== Tokamak Disruption Visualization ===")
    print(f"Output directory: {output_dir}\n")

    # Plot diagnostics CSV
    csv_path = os.path.join(output_dir, 'diagnostics.csv')
    if os.path.exists(csv_path):
        print("Plotting diagnostic time series...")
        data = read_diagnostics_csv(csv_path)
        plot_diagnostics_timeseries(data, output_dir)
        plot_disruption_phases(data, output_dir)
    else:
        print(f"  Warning: {csv_path} not found")

    # Plot field snapshots
    print("\nPlotting field snapshots...")
    plot_all_snapshots(output_dir)

    print(f"\n=== Visualization complete ===\n")


if __name__ == '__main__':
    main()
