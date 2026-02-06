#!/usr/bin/env python3
"""
Tokamak Disruption Simulation — Publication-Quality Visualization Suite

Generates a comprehensive set of plasma physics visualizations from
simulation field dumps, targeting an audience of fusion physicists.

Produces:
  1.  Tokamak D-shaped geometry (Miller parameterization) with flux surfaces
  2.  Toroidal magnetic field B_φ(R,Z) with 1/R vacuum field
  3.  Poloidal flux ψ contours (magnetic equilibrium flux surfaces)
  4.  Plasma pressure p(R,Z) and density n(R,Z) profiles
  5.  Toroidal current density J_φ(R,Z)
  6.  Safety factor q(ρ) profile
  7.  MHD perturbation δψ (tearing mode structure)
  8.  Electron temperature Te(R,Z) — pre- and post-thermal quench
  9.  Thermal quench Te evolution sequence (6 time snapshots)
  10. Radiated power density P_rad(R,Z) during thermal quench
  11. Temperature and radiation time series during TQ
  12. Electric field E(R,Z) during current quench
  13. Plasma current decay and E-field time series during CQ
  14. Resistivity η(R,Z) evolution during CQ
  15. Runaway electron density n_RE(R,Z) growth
  16. Runaway electron current time series

Usage:
    python3 scripts/generate_visualizations.py [input_dir] [output_dir]
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec
import os
import sys
import struct
import csv
from pathlib import Path

# ─── Simulation timestep constants (must match dump_fields.cpp) ────────
DT_TQ_MS = 1e-5 * 1e3    # Thermal quench dt in ms (dt_tq = 1e-5 s)
DT_CQ_MS = 1e-5 * 1e3    # Current quench dt in ms (dt_cq = 1e-5 s)
DT_RE_MS = 1e-7 * 1e3    # Runaway electron dt in ms (dt_re = 1e-7 s)


# ─── Publication Style ────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'image.cmap': 'inferno',
})


# ─── I/O Helpers ──────────────────────────────────────────────────────

def read_binary_field(filepath):
    """Read a 2D scalar field from binary format."""
    with open(filepath, 'rb') as f:
        nr = struct.unpack('i', f.read(4))[0]
        nz = struct.unpack('i', f.read(4))[0]
        R_min, R_max, Z_min, Z_max = struct.unpack('4d', f.read(32))
        data = np.frombuffer(f.read(nr * nz * 8), dtype=np.float64).reshape(nr, nz)
    R = np.linspace(R_min, R_max, nr)
    Z = np.linspace(Z_min, Z_max, nz)
    return R, Z, data


def read_csv_timeseries(filepath):
    """Read a CSV time-series file."""
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = {h: [] for h in header}
        for row in reader:
            for h, v in zip(header, row):
                data[h].append(float(v))
    return {k: np.array(v) for k, v in data.items()}


def read_geometry(filepath):
    """Read Miller-parameterized plasma boundary points."""
    R_b, Z_b = [], []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split(',')
            if len(parts) == 3:
                R_b.append(float(parts[1]))
                Z_b.append(float(parts[2]))
    return np.array(R_b), np.array(Z_b)


def plasma_boundary_overlay(ax, R_b, Z_b, color='white', ls='--', lw=1.5, alpha=0.8, label=None):
    """Draw the plasma boundary on an axes."""
    ax.plot(R_b, Z_b, color=color, linestyle=ls, linewidth=lw, alpha=alpha, label=label)


def standard_axes(ax, xlabel='R [m]', ylabel='Z [m]'):
    """Apply standard tokamak cross-section formatting."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect('equal')


# ─── Individual Visualization Functions ───────────────────────────────

def fig01_geometry(indir, outdir, R_b, Z_b):
    """Figure 1: Tokamak D-shaped geometry with flux surfaces."""
    R, Z, rho = read_binary_field(os.path.join(indir, 'rho_norm.dat'))
    _, _, mask = read_binary_field(os.path.join(indir, 'plasma_mask.dat'))
    Rm, Zm = np.meshgrid(R, Z, indexing='ij')

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # (a) D-shaped cross-section with flux surfaces
    ax = axes[0]
    levels = np.linspace(0, 1, 11)
    cs = ax.contour(Rm, Zm, rho, levels=levels, cmap='coolwarm', linewidths=1.2)
    ax.clabel(cs, inline=True, fontsize=8, fmt='ρ=%.1f')
    ax.contour(Rm, Zm, rho, levels=[1.0], colors='red', linewidths=2.5)
    plasma_boundary_overlay(ax, R_b, Z_b, color='red', ls='-', lw=2, label='LCFS (ρ=1)')

    # Mark magnetic axis
    ax.plot(6.2, 0, 'k+', markersize=15, markeredgewidth=2, label='Magnetic axis')

    # Annotations
    ax.annotate('', xy=(8.2, 0), xytext=(6.2, 0),
                arrowprops=dict(arrowstyle='<->', color='navy', lw=1.5))
    ax.text(7.2, -0.3, 'a = 2.0 m', ha='center', fontsize=10, color='navy')
    ax.annotate('', xy=(6.2, 3.4), xytext=(6.2, 0),
                arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=1.5))
    ax.text(5.5, 1.7, 'κa = 3.4 m', ha='center', fontsize=10, color='darkgreen',
            rotation=90)

    ax.set_title('(a) ITER-like D-shaped Cross-Section\nMiller Parameterization: '
                 'κ=1.7, δ=0.33')
    standard_axes(ax)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(3.5, 9.0)
    ax.set_ylim(-5, 5)

    # (b) Plasma mask
    ax = axes[1]
    im = ax.pcolormesh(Rm, Zm, mask, cmap='Blues', shading='auto', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Inside plasma', shrink=0.8)
    plasma_boundary_overlay(ax, R_b, Z_b, color='red', lw=2)
    ax.set_title('(b) Computational Domain\n(128×128 grid)')
    standard_axes(ax)
    ax.set_xlim(3.5, 9.0)
    ax.set_ylim(-5, 5)

    fig.suptitle('Figure 1: Tokamak Geometry — D-shaped (Miller) Poloidal Cross-Section',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = os.path.join(outdir, 'fig01_geometry.png')
    plt.savefig(out)
    plt.close()
    print(f'  Saved: {out}')
    return out


def fig02_magnetic_field(indir, outdir, R_b, Z_b):
    """Figure 2: Toroidal magnetic field B_φ(R,Z)."""
    R, Z, Btor = read_binary_field(os.path.join(indir, 'Btor.dat'))
    Rm, Zm = np.meshgrid(R, Z, indexing='ij')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (a) 2D B_tor field
    ax = axes[0]
    im = ax.pcolormesh(Rm, Zm, Btor, cmap='viridis', shading='auto')
    cb = plt.colorbar(im, ax=ax, label='B_φ [T]', shrink=0.8)
    plasma_boundary_overlay(ax, R_b, Z_b, color='white', lw=2, label='LCFS')
    ax.set_title('(a) Toroidal Field B_φ(R,Z)')
    standard_axes(ax)
    ax.legend(loc='lower right')

    # (b) Midplane 1/R profile
    ax = axes[1]
    jmid = len(Z) // 2
    B_midplane = Btor[:, jmid]
    B_theory = 5.3 * 6.2 / R
    ax.plot(R, B_midplane, 'b-', linewidth=2, label='Simulation B_φ(R)')
    ax.plot(R, B_theory, 'r--', linewidth=1.5, label='Vacuum: B₀R₀/R')
    ax.axvline(6.2, color='gray', linestyle=':', alpha=0.5, label='R₀ = 6.2 m')
    ax.axvline(4.2, color='green', linestyle=':', alpha=0.5, label='R₀ − a')
    ax.axvline(8.2, color='green', linestyle=':', alpha=0.5, label='R₀ + a')
    ax.set_xlabel('R [m]')
    ax.set_ylabel('B_φ [T]')
    ax.set_title('(b) Midplane Profile: B_φ ∝ 1/R')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Figure 2: Toroidal Magnetic Field — Vacuum 1/R Dependence',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = os.path.join(outdir, 'fig02_magnetic_field.png')
    plt.savefig(out)
    plt.close()
    print(f'  Saved: {out}')
    return out


def fig03_equilibrium(indir, outdir, R_b, Z_b):
    """Figure 3: MHD Equilibrium — pressure, density, flux, current."""
    R, Z, psi = read_binary_field(os.path.join(indir, 'Bpol_psi.dat'))
    _, _, pressure = read_binary_field(os.path.join(indir, 'pressure.dat'))
    _, _, density = read_binary_field(os.path.join(indir, 'density.dat'))
    _, _, Jtor = read_binary_field(os.path.join(indir, 'Jtor.dat'))
    Rm, Zm = np.meshgrid(R, Z, indexing='ij')

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for ax_row in axes:
        for ax in ax_row:
            plasma_boundary_overlay(ax, R_b, Z_b, color='white', lw=1.5)
            standard_axes(ax)

    # (a) Poloidal flux ψ
    ax = axes[0, 0]
    cs = ax.contour(Rm, Zm, psi, levels=20, colors='white', linewidths=0.6, alpha=0.8)
    im = ax.pcolormesh(Rm, Zm, psi, cmap='RdBu_r', shading='auto')
    plt.colorbar(im, ax=ax, label='ψ [Wb]', shrink=0.8)
    ax.set_title('(a) Poloidal Flux ψ — Magnetic Surfaces')

    # (b) Plasma Pressure
    ax = axes[0, 1]
    im = ax.pcolormesh(Rm, Zm, pressure, cmap='inferno', shading='auto')
    plt.colorbar(im, ax=ax, label='p [normalized]', shrink=0.8)
    cs = ax.contour(Rm, Zm, pressure, levels=8, colors='white', linewidths=0.5, alpha=0.6)
    ax.set_title('(b) Plasma Pressure p(R,Z)')

    # (c) Mass Density
    ax = axes[1, 0]
    im = ax.pcolormesh(Rm, Zm, density, cmap='YlOrRd', shading='auto')
    plt.colorbar(im, ax=ax, label='ρ [normalized]', shrink=0.8)
    ax.set_title('(c) Mass Density ρ(R,Z)')

    # (d) Toroidal Current Density
    ax = axes[1, 1]
    Jmax = np.max(np.abs(Jtor)) * 0.8
    if Jmax > 0:
        im = ax.pcolormesh(Rm, Zm, Jtor, cmap='RdBu_r', shading='auto',
                           vmin=-Jmax, vmax=Jmax)
    else:
        im = ax.pcolormesh(Rm, Zm, Jtor, cmap='RdBu_r', shading='auto')
    plt.colorbar(im, ax=ax, label='J_φ [A/m²]', shrink=0.8)
    ax.set_title('(d) Toroidal Current Density J_φ(R,Z)')

    fig.suptitle('Figure 3: MHD Equilibrium (Grad-Shafranov) — D-shaped Tokamak',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = os.path.join(outdir, 'fig03_equilibrium.png')
    plt.savefig(out)
    plt.close()
    print(f'  Saved: {out}')
    return out


def fig04_safety_factor(indir, outdir):
    """Figure 4: Safety factor q(ρ) profile."""
    data = read_csv_timeseries(os.path.join(indir, 'safety_factor.csv'))
    rho = data['rho_norm']
    q = data['q']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rho, q, 'b-', linewidth=2.5)
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='q = 1 (sawtooth)')
    ax.axhline(2.0, color='orange', linestyle='--', alpha=0.7, label='q = 2 (m/n=2/1 tearing)')
    ax.axhline(3.0, color='green', linestyle='--', alpha=0.7, label='q = 3 (edge)')

    # Mark rational surfaces
    for q_rat, label in [(1.0, 'q=1'), (2.0, 'q=2')]:
        idx = np.argmin(np.abs(q - q_rat))
        ax.plot(rho[idx], q[idx], 'ko', markersize=8)
        ax.annotate(f'{label} at ρ={rho[idx]:.2f}', xy=(rho[idx], q[idx]),
                    xytext=(rho[idx]+0.1, q[idx]+0.2), fontsize=10,
                    arrowprops=dict(arrowstyle='->', color='black'))

    ax.set_xlabel('Normalized radius ρ = r/a')
    ax.set_ylabel('Safety factor q')
    ax.set_title('Figure 4: Safety Factor Profile q(ρ)\n'
                 'Rational surfaces drive tearing mode instabilities')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 4)

    out = os.path.join(outdir, 'fig04_safety_factor.png')
    plt.savefig(out)
    plt.close()
    print(f'  Saved: {out}')
    return out


def fig05_tearing_mode(indir, outdir, R_b, Z_b):
    """Figure 5: Tearing mode perturbation δψ."""
    R, Z, dpsi = read_binary_field(os.path.join(indir, 'psi_perturbation.dat'))
    _, _, psi_eq = read_binary_field(os.path.join(indir, 'Bpol_psi.dat'))
    _, _, psi_pert = read_binary_field(os.path.join(indir, 'Bpol_psi_perturbed.dat'))
    Rm, Zm = np.meshgrid(R, Z, indexing='ij')

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # (a) Equilibrium ψ
    ax = axes[0]
    ax.contour(Rm, Zm, psi_eq, levels=20, colors='navy', linewidths=0.8)
    plasma_boundary_overlay(ax, R_b, Z_b, color='red', lw=2)
    ax.set_title('(a) Equilibrium ψ\n(nested flux surfaces)')
    standard_axes(ax)

    # (b) Perturbed ψ
    ax = axes[1]
    ax.contour(Rm, Zm, psi_pert, levels=20, colors='navy', linewidths=0.8)
    plasma_boundary_overlay(ax, R_b, Z_b, color='red', lw=2)
    ax.set_title('(b) Perturbed ψ\n(with tearing mode)')
    standard_axes(ax)

    # (c) Perturbation δψ
    ax = axes[2]
    dpsi_max = np.max(np.abs(dpsi)) * 0.8
    if dpsi_max > 0:
        im = ax.pcolormesh(Rm, Zm, dpsi, cmap='RdBu_r', shading='auto',
                           vmin=-dpsi_max, vmax=dpsi_max)
    else:
        im = ax.pcolormesh(Rm, Zm, dpsi, cmap='RdBu_r', shading='auto')
    plt.colorbar(im, ax=ax, label='δψ [Wb]', shrink=0.8)
    plasma_boundary_overlay(ax, R_b, Z_b, color='black', lw=1.5)
    ax.set_title('(c) Tearing Mode Perturbation δψ\n(m=2/n=1 structure)')
    standard_axes(ax)

    fig.suptitle('Figure 5: MHD Tearing Mode Instability — Disruption Trigger',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = os.path.join(outdir, 'fig05_tearing_mode.png')
    plt.savefig(out)
    plt.close()
    print(f'  Saved: {out}')
    return out


def fig06_thermal_quench_evolution(indir, outdir, R_b, Z_b):
    """Figure 6: Thermal quench Te evolution — 6-panel time sequence."""
    steps = [0, 50, 200, 500, 1000, 2000]
    labels = ['t = 0 ms\n(Pre-TQ)', 't = 0.5 ms', 't = 2.0 ms',
              't = 5.0 ms', 't = 10 ms', 't = 20 ms\n(Post-TQ)']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes_flat = axes.flatten()

    for idx, (step, label) in enumerate(zip(steps, labels)):
        ax = axes_flat[idx]
        fname = os.path.join(indir, f'Te_tq_{step}.dat')
        if not os.path.exists(fname):
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            continue
        R, Z, Te = read_binary_field(fname)
        Rm, Zm = np.meshgrid(R, Z, indexing='ij')

        Te_plot = np.maximum(Te, 1e-3)
        im = ax.pcolormesh(Rm, Zm, Te_plot, cmap='hot', shading='auto',
                           norm=LogNorm(vmin=0.005, vmax=12.0))
        plt.colorbar(im, ax=ax, label='Te [keV]', shrink=0.75)
        plasma_boundary_overlay(ax, R_b, Z_b, color='cyan', lw=1.5)
        ax.set_title(label, fontsize=12)
        standard_axes(ax)

    fig.suptitle('Figure 6: Thermal Quench — Electron Temperature Collapse\n'
                 'Te drops from 10 keV to ~10 eV in milliseconds',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = os.path.join(outdir, 'fig06_thermal_quench_evolution.png')
    plt.savefig(out)
    plt.close()
    print(f'  Saved: {out}')
    return out


def fig07_radiation(indir, outdir, R_b, Z_b):
    """Figure 7: Radiation power density during thermal quench."""
    steps = [50, 200, 500, 1000]
    labels = ['t = 0.5 ms', 't = 2.0 ms', 't = 5.0 ms', 't = 10 ms']

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for idx, (step, label) in enumerate(zip(steps, labels)):
        ax = axes[idx]
        fname = os.path.join(indir, f'Prad_tq_{step}.dat')
        if not os.path.exists(fname):
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            continue
        R, Z, Prad = read_binary_field(fname)
        Rm, Zm = np.meshgrid(R, Z, indexing='ij')

        Prad_plot = np.maximum(np.abs(Prad), 1e-1)
        im = ax.pcolormesh(Rm, Zm, Prad_plot, cmap='magma', shading='auto',
                           norm=LogNorm())
        plt.colorbar(im, ax=ax, label='P_rad [W/m³]', shrink=0.8)
        plasma_boundary_overlay(ax, R_b, Z_b, color='cyan', lw=1.5)
        ax.set_title(label)
        standard_axes(ax)

    fig.suptitle('Figure 7: Radiative Power Density During Thermal Quench',
                 fontsize=15, fontweight='bold', y=1.05)
    plt.tight_layout()
    out = os.path.join(outdir, 'fig07_radiation.png')
    plt.savefig(out)
    plt.close()
    print(f'  Saved: {out}')
    return out


def fig08_tq_timeseries(indir, outdir):
    """Figure 8: Thermal quench time series — temperature and radiation."""
    tq_temp = read_csv_timeseries(os.path.join(indir, 'tq_temperature.csv'))
    tq_rad = read_csv_timeseries(os.path.join(indir, 'tq_radiation.csv'))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.semilogy(tq_temp['time_ms'], np.maximum(tq_temp['Te_avg_keV'], 1e-4),
                'r-', linewidth=2)
    ax.axhline(0.01, color='blue', linestyle='--', alpha=0.5, label='Wall temp (10 eV)')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Average Te [keV]')
    ax.set_title('(a) Electron Temperature Decay')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(tq_rad['time_ms'], np.maximum(tq_rad['P_rad_W'], 1e-1),
                'orange', linewidth=2)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Total Radiated Power [W]')
    ax.set_title('(b) Radiation Power Evolution')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Figure 8: Thermal Quench Dynamics',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = os.path.join(outdir, 'fig08_tq_timeseries.png')
    plt.savefig(out)
    plt.close()
    print(f'  Saved: {out}')
    return out


def fig09_current_quench(indir, outdir, R_b, Z_b):
    """Figure 9: Current quench — E-field and current density evolution."""
    steps = [0, 500, 1000, 3000]
    labels = ['t = 0 ms', 't = 5 ms', 't = 10 ms', 't = 30 ms']

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for idx, (step, label) in enumerate(zip(steps, labels)):
        # Top row: E-field
        ax = axes[0, idx]
        fname = os.path.join(indir, f'E_field_cq_{step}.dat')
        if os.path.exists(fname):
            R, Z, E = read_binary_field(fname)
            Rm, Zm = np.meshgrid(R, Z, indexing='ij')
            E_plot = np.maximum(np.abs(E), 1e-6)
            im = ax.pcolormesh(Rm, Zm, E_plot, cmap='plasma', shading='auto',
                               norm=LogNorm())
            plt.colorbar(im, ax=ax, label='|E| [V/m]', shrink=0.8)
        plasma_boundary_overlay(ax, R_b, Z_b, color='white', lw=1.5)
        ax.set_title(f'E-field\n{label}')
        standard_axes(ax)

        # Bottom row: Current density
        ax = axes[1, idx]
        fname = os.path.join(indir, f'Jtor_cq_{step}.dat')
        if os.path.exists(fname):
            R, Z, J = read_binary_field(fname)
            Rm, Zm = np.meshgrid(R, Z, indexing='ij')
            im = ax.pcolormesh(Rm, Zm, J, cmap='RdBu_r', shading='auto')
            plt.colorbar(im, ax=ax, label='J_φ [A/m²]', shrink=0.8)
        plasma_boundary_overlay(ax, R_b, Z_b, color='black', lw=1.5)
        ax.set_title(f'J_φ\n{label}')
        standard_axes(ax)

    fig.suptitle('Figure 9: Current Quench — Electric Field and Current Redistribution',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = os.path.join(outdir, 'fig09_current_quench.png')
    plt.savefig(out)
    plt.close()
    print(f'  Saved: {out}')
    return out


def fig10_cq_timeseries(indir, outdir):
    """Figure 10: Current quench time series."""
    cq_curr = read_csv_timeseries(os.path.join(indir, 'cq_current.csv'))
    cq_ef = read_csv_timeseries(os.path.join(indir, 'cq_efield.csv'))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(cq_curr['time_ms'], cq_curr['Ip_A'] / 1e6, 'g-', linewidth=2.5)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Plasma Current [MA]')
    ax.set_title('(a) Plasma Current Decay (L/R)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    ax = axes[1]
    ax.semilogy(cq_ef['time_ms'], np.maximum(np.abs(cq_ef['E_field_Vm']), 1e-6),
                'orange', linewidth=2.5)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Toroidal E-field [V/m]')
    ax.set_title('(b) Induced Electric Field')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Figure 10: Current Quench Dynamics — I_p(t) and E(t)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = os.path.join(outdir, 'fig10_cq_timeseries.png')
    plt.savefig(out)
    plt.close()
    print(f'  Saved: {out}')
    return out


def fig11_resistivity(indir, outdir, R_b, Z_b):
    """Figure 11: Spitzer resistivity map during CQ."""
    steps = [0, 1000, 3000]
    labels = ['CQ start', 'CQ +10 ms', 'CQ +30 ms']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, (step, label) in enumerate(zip(steps, labels)):
        ax = axes[idx]
        fname = os.path.join(indir, f'resistivity_cq_{step}.dat')
        if os.path.exists(fname):
            R, Z, eta = read_binary_field(fname)
            Rm, Zm = np.meshgrid(R, Z, indexing='ij')
            eta_plot = np.maximum(np.abs(eta), 1e-12)
            im = ax.pcolormesh(Rm, Zm, eta_plot, cmap='YlOrRd', shading='auto',
                               norm=LogNorm())
            plt.colorbar(im, ax=ax, label='η [Ω·m]', shrink=0.8)
        plasma_boundary_overlay(ax, R_b, Z_b, color='cyan', lw=1.5)
        ax.set_title(label)
        standard_axes(ax)

    fig.suptitle('Figure 11: Spitzer Resistivity η(R,Z) — Post-TQ Cold Plasma',
                 fontsize=15, fontweight='bold', y=1.05)
    plt.tight_layout()
    out = os.path.join(outdir, 'fig11_resistivity.png')
    plt.savefig(out)
    plt.close()
    print(f'  Saved: {out}')
    return out


def fig12_runaway_electrons(indir, outdir, R_b, Z_b):
    """Figure 12: Runaway electron density evolution."""
    steps = [0, 1000, 5000, 10000, 20000]
    labels = ['t = 0', 't = 0.1 ms', 't = 0.5 ms', 't = 1 ms', 't = 2 ms']

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))

    for idx, (step, label) in enumerate(zip(steps, labels)):
        ax = axes[idx]
        fname = os.path.join(indir, f'n_RE_re_{step}.dat')
        if os.path.exists(fname):
            R, Z, nRE = read_binary_field(fname)
            Rm, Zm = np.meshgrid(R, Z, indexing='ij')
            nRE_plot = np.maximum(np.abs(nRE), 1.0)
            im = ax.pcolormesh(Rm, Zm, nRE_plot, cmap='plasma', shading='auto',
                               norm=LogNorm())
            plt.colorbar(im, ax=ax, label='n_RE [m⁻³]', shrink=0.8)
        plasma_boundary_overlay(ax, R_b, Z_b, color='white', lw=1.5)
        ax.set_title(label, fontsize=11)
        standard_axes(ax)

    fig.suptitle('Figure 12: Runaway Electron Density n_RE(R,Z) — Avalanche Growth',
                 fontsize=15, fontweight='bold', y=1.05)
    plt.tight_layout()
    out = os.path.join(outdir, 'fig12_runaway_electrons.png')
    plt.savefig(out)
    plt.close()
    print(f'  Saved: {out}')
    return out


def fig13_re_timeseries(indir, outdir):
    """Figure 13: Runaway electron current time series."""
    data = read_csv_timeseries(os.path.join(indir, 're_current.csv'))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(data['time_ms'], np.maximum(np.abs(data['I_RE_A']), 1.0),
                'r-', linewidth=2.5)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Runaway Electron Current [A]')
    ax.set_title('Figure 13: Runaway Electron Current — Avalanche Growth\n'
                 'Rosenbluth-Putvinski model')
    ax.grid(True, alpha=0.3)

    out = os.path.join(outdir, 'fig13_re_current.png')
    plt.savefig(out)
    plt.close()
    print(f'  Saved: {out}')
    return out


def fig14_disruption_summary(indir, outdir, R_b, Z_b):
    """Figure 14: Grand summary — all disruption phases in one figure."""
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 4, hspace=0.35, wspace=0.35)

    # Row 1: Equilibrium
    ax = fig.add_subplot(gs[0, 0])
    R, Z, psi = read_binary_field(os.path.join(indir, 'Bpol_psi.dat'))
    Rm, Zm = np.meshgrid(R, Z, indexing='ij')
    ax.contour(Rm, Zm, psi, levels=15, colors='navy', linewidths=0.6)
    plasma_boundary_overlay(ax, R_b, Z_b, color='red', lw=2)
    ax.set_title('ψ Equilibrium', fontsize=11)
    standard_axes(ax)

    ax = fig.add_subplot(gs[0, 1])
    _, _, Btor = read_binary_field(os.path.join(indir, 'Btor.dat'))
    im = ax.pcolormesh(Rm, Zm, Btor, cmap='viridis', shading='auto')
    plt.colorbar(im, ax=ax, label='B_φ [T]', shrink=0.75)
    plasma_boundary_overlay(ax, R_b, Z_b, color='white', lw=1.5)
    ax.set_title('B_φ (1/R)', fontsize=11)
    standard_axes(ax)

    ax = fig.add_subplot(gs[0, 2])
    _, _, pressure = read_binary_field(os.path.join(indir, 'pressure.dat'))
    im = ax.pcolormesh(Rm, Zm, pressure, cmap='inferno', shading='auto')
    plt.colorbar(im, ax=ax, label='p', shrink=0.75)
    plasma_boundary_overlay(ax, R_b, Z_b, color='white', lw=1.5)
    ax.set_title('Pressure', fontsize=11)
    standard_axes(ax)

    ax = fig.add_subplot(gs[0, 3])
    _, _, dpsi = read_binary_field(os.path.join(indir, 'psi_perturbation.dat'))
    dpsi_max = np.max(np.abs(dpsi)) * 0.8
    if dpsi_max > 0:
        im = ax.pcolormesh(Rm, Zm, dpsi, cmap='RdBu_r', shading='auto',
                           vmin=-dpsi_max, vmax=dpsi_max)
    else:
        im = ax.pcolormesh(Rm, Zm, dpsi, cmap='RdBu_r', shading='auto')
    plt.colorbar(im, ax=ax, label='δψ', shrink=0.75)
    plasma_boundary_overlay(ax, R_b, Z_b, color='black', lw=1.5)
    ax.set_title('Tearing Mode δψ', fontsize=11)
    standard_axes(ax)

    # Row 2: Thermal Quench
    for col, step in enumerate([0, 200, 1000, 2000]):
        ax = fig.add_subplot(gs[1, col])
        fname = os.path.join(indir, f'Te_tq_{step}.dat')
        if os.path.exists(fname):
            _, _, Te = read_binary_field(fname)
            Te_plot = np.maximum(Te, 1e-3)
            im = ax.pcolormesh(Rm, Zm, Te_plot, cmap='hot', shading='auto',
                               norm=LogNorm(vmin=0.005, vmax=12))
            plt.colorbar(im, ax=ax, label='Te [keV]', shrink=0.75)
        plasma_boundary_overlay(ax, R_b, Z_b, color='cyan', lw=1.5)
        t_ms = step * DT_TQ_MS
        ax.set_title(f'TQ: t={t_ms:.1f} ms', fontsize=11)
        standard_axes(ax)

    # Row 3: Current Quench + RE
    for col, step in enumerate([0, 1000, 3000]):
        ax = fig.add_subplot(gs[2, col])
        fname = os.path.join(indir, f'E_field_cq_{step}.dat')
        if os.path.exists(fname):
            _, _, E = read_binary_field(fname)
            E_plot = np.maximum(np.abs(E), 1e-6)
            im = ax.pcolormesh(Rm, Zm, E_plot, cmap='plasma', shading='auto',
                               norm=LogNorm())
            plt.colorbar(im, ax=ax, label='|E| [V/m]', shrink=0.75)
        plasma_boundary_overlay(ax, R_b, Z_b, color='white', lw=1.5)
        t_ms = step * DT_CQ_MS
        ax.set_title(f'CQ E-field: t={t_ms:.0f} ms', fontsize=11)
        standard_axes(ax)

    ax = fig.add_subplot(gs[2, 3])
    fname = os.path.join(indir, 'n_RE_final.dat')
    if os.path.exists(fname):
        _, _, nRE = read_binary_field(fname)
        nRE_plot = np.maximum(np.abs(nRE), 1.0)
        im = ax.pcolormesh(Rm, Zm, nRE_plot, cmap='plasma', shading='auto',
                           norm=LogNorm())
        plt.colorbar(im, ax=ax, label='n_RE [m⁻³]', shrink=0.75)
    plasma_boundary_overlay(ax, R_b, Z_b, color='white', lw=1.5)
    ax.set_title('RE Density (final)', fontsize=11)
    standard_axes(ax)

    fig.suptitle('TOKAMAK DISRUPTION CASCADE — Full Multi-Physics Simulation\n'
                 'ITER-like: R₀=6.2m, a=2.0m, κ=1.7, δ=0.33, B₀=5.3T, I_p=15MA',
                 fontsize=16, fontweight='bold', y=1.01)
    out = os.path.join(outdir, 'fig14_disruption_summary.png')
    plt.savefig(out)
    plt.close()
    print(f'  Saved: {out}')
    return out


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    indir = sys.argv[1] if len(sys.argv) > 1 else 'viz_output'
    outdir = sys.argv[2] if len(sys.argv) > 2 else 'docs/figures'

    os.makedirs(outdir, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'  Tokamak Disruption Visualization Suite')
    print(f'  Input:  {indir}')
    print(f'  Output: {outdir}')
    print(f'{"="*60}\n')

    # Load geometry
    R_b, Z_b = read_geometry(os.path.join(indir, 'geometry.csv'))
    print(f'Loaded plasma boundary ({len(R_b)} points)\n')

    # Generate all figures
    figures = {}
    print('Generating visualizations...\n')

    figures['fig01'] = fig01_geometry(indir, outdir, R_b, Z_b)
    figures['fig02'] = fig02_magnetic_field(indir, outdir, R_b, Z_b)
    figures['fig03'] = fig03_equilibrium(indir, outdir, R_b, Z_b)
    figures['fig04'] = fig04_safety_factor(indir, outdir)
    figures['fig05'] = fig05_tearing_mode(indir, outdir, R_b, Z_b)
    figures['fig06'] = fig06_thermal_quench_evolution(indir, outdir, R_b, Z_b)
    figures['fig07'] = fig07_radiation(indir, outdir, R_b, Z_b)
    figures['fig08'] = fig08_tq_timeseries(indir, outdir)
    figures['fig09'] = fig09_current_quench(indir, outdir, R_b, Z_b)
    figures['fig10'] = fig10_cq_timeseries(indir, outdir)
    figures['fig11'] = fig11_resistivity(indir, outdir, R_b, Z_b)
    figures['fig12'] = fig12_runaway_electrons(indir, outdir, R_b, Z_b)
    figures['fig13'] = fig13_re_timeseries(indir, outdir)
    figures['fig14'] = fig14_disruption_summary(indir, outdir, R_b, Z_b)

    print(f'\n{"="*60}')
    print(f'  Generated {len(figures)} publication-quality figures')
    print(f'  Output directory: {outdir}/')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()
