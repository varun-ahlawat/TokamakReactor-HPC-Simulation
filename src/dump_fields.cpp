/// @file dump_fields.cpp
/// @brief Dumps all simulation field data for visualization.
///
/// Initializes each physics module in sequence and writes 2D field snapshots
/// as binary files that the Python visualization script reads.
/// This produces the complete set of fields across all disruption phases.

#include "core/grid.h"
#include "core/field.h"
#include "physics/mhd_solver.h"
#include "physics/thermal_quench.h"
#include "physics/current_quench.h"
#include "physics/runaway_electrons.h"
#include "physics/disruption_mitigation.h"
#include "utils/physical_constants.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <sys/stat.h>
#include <mpi.h>

using namespace tokamak;

/// Write a 2D scalar field as a binary file
void write_binary(const std::string& filepath, const Grid& grid,
                  const ScalarField& field) {
    std::ofstream out(filepath, std::ios::binary);
    int nr = grid.nr(), nz = grid.nz();
    double bounds[4] = {grid.R_min(), grid.R_max(), grid.Z_min(), grid.Z_max()};
    out.write(reinterpret_cast<char*>(&nr), sizeof(int));
    out.write(reinterpret_cast<char*>(&nz), sizeof(int));
    out.write(reinterpret_cast<char*>(bounds), 4 * sizeof(double));
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nz; ++j) {
            double v = field(i, j);
            out.write(reinterpret_cast<char*>(&v), sizeof(double));
        }
    out.close();
}

/// Write geometry data (plasma boundary points) for visualization
void write_geometry(const std::string& filepath, const MHDConfig& cfg) {
    std::ofstream out(filepath);
    out << "# Tokamak geometry: Miller-parameterized D-shaped cross-section\n";
    out << "# R0=" << cfg.R0 << " a=" << cfg.a
        << " kappa=" << cfg.elongation << " delta=" << cfg.triangularity << "\n";
    out << "# theta, R_boundary, Z_boundary\n";
    int npts = 360;
    for (int k = 0; k <= npts; ++k) {
        double theta = 2.0 * M_PI * k / npts;
        double R = cfg.R0 + cfg.a * std::cos(theta + cfg.triangularity * std::sin(theta));
        double Z = cfg.elongation * cfg.a * std::sin(theta);
        out << theta << "," << R << "," << Z << "\n";
    }
    out.close();
}

/// Write a CSV time-series from a vector of (time, value) pairs
void write_timeseries(const std::string& filepath,
                      const std::vector<std::pair<double,double>>& data,
                      const std::string& header) {
    std::ofstream out(filepath);
    out << header << "\n";
    for (const auto& [t, v] : data) out << t << "," << v << "\n";
    out.close();
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    std::string outdir = "viz_output";
    mkdir(outdir.c_str(), 0755);

    int nr = 128, nz = 128;
    Grid grid(nr, nz, 4.0, 8.5, -4.5, 4.5);

    // MHD Configuration with D-shaped geometry
    MHDConfig mhd_cfg;
    mhd_cfg.B0 = 5.3;
    mhd_cfg.R0 = 6.2;
    mhd_cfg.a = 2.0;
    mhd_cfg.elongation = 1.7;
    mhd_cfg.triangularity = 0.33;

    std::cout << "=== Dumping Tokamak Simulation Fields ===" << std::endl;
    std::cout << "Grid: " << nr << "x" << nz << std::endl;
    std::cout << "Geometry: R0=" << mhd_cfg.R0 << "m, a=" << mhd_cfg.a
              << "m, kappa=" << mhd_cfg.elongation
              << ", delta=" << mhd_cfg.triangularity << std::endl;

    // ========== 1. GEOMETRY ==========
    std::cout << "\n[1/6] Writing geometry..." << std::endl;
    write_geometry(outdir + "/geometry.csv", mhd_cfg);

    // Write a field marking inside/outside plasma
    ScalarField plasma_mask(grid, "plasma_mask");
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nz; ++j)
            plasma_mask(i, j) = grid.is_inside_plasma(grid.R(i), grid.Z(j),
                mhd_cfg.R0, mhd_cfg.a, mhd_cfg.elongation, mhd_cfg.triangularity) ? 1.0 : 0.0;
    write_binary(outdir + "/plasma_mask.dat", grid, plasma_mask);

    // Write normalized radius field
    ScalarField rho_norm(grid, "rho_norm");
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nz; ++j)
            rho_norm(i, j) = grid.normalized_radius(grid.R(i), grid.Z(j),
                mhd_cfg.R0, mhd_cfg.a, mhd_cfg.elongation, mhd_cfg.triangularity);
    write_binary(outdir + "/rho_norm.dat", grid, rho_norm);

    // ========== 2. MHD EQUILIBRIUM ==========
    std::cout << "[2/6] Computing MHD equilibrium..." << std::endl;
    MHDSolver mhd(grid, mhd_cfg);
    mhd.initialize_equilibrium();

    write_binary(outdir + "/Btor.dat", grid, mhd.state().Btor);
    write_binary(outdir + "/Bpol_psi.dat", grid, mhd.state().Bpol_psi);
    write_binary(outdir + "/pressure.dat", grid, mhd.state().pressure);
    write_binary(outdir + "/density.dat", grid, mhd.state().density);
    write_binary(outdir + "/Jtor.dat", grid, mhd.state().Jtor);
    write_binary(outdir + "/temperature_mhd.dat", grid, mhd.state().temperature);

    // Safety factor profile (1D along midplane)
    {
        std::ofstream qf(outdir + "/safety_factor.csv");
        qf << "rho_norm,q\n";
        for (int k = 0; k <= 100; ++k) {
            double rho = k / 100.0;
            double q = mhd_cfg.q0 + (mhd_cfg.q_edge - mhd_cfg.q0) * rho * rho;
            qf << rho << "," << q << "\n";
        }
    }

    std::cout << "  Magnetic energy: " << mhd.magnetic_energy() / 1e9 << " GJ" << std::endl;

    // ========== 3. MHD PERTURBATION & EVOLUTION ==========
    std::cout << "[3/6] Applying MHD perturbation and evolving..." << std::endl;
    mhd.apply_perturbation();

    // Evolve a few MHD steps to grow the perturbation
    double dt_mhd = 1e-7;
    std::vector<std::pair<double,double>> island_evolution;
    for (int step = 0; step < 200; ++step) {
        mhd.advance(dt_mhd);
        if (step % 10 == 0)
            island_evolution.push_back({step * dt_mhd * 1e3, mhd.island_width()});
    }
    write_binary(outdir + "/Bpol_psi_perturbed.dat", grid, mhd.state().Bpol_psi);
    write_binary(outdir + "/Jtor_perturbed.dat", grid, mhd.state().Jtor);

    // Compute perturbation delta-psi
    ScalarField psi_pert(grid, "psi_pert");
    MHDSolver mhd_eq(grid, mhd_cfg);
    mhd_eq.initialize_equilibrium();
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nz; ++j)
            psi_pert(i, j) = mhd.state().Bpol_psi(i, j) - mhd_eq.state().Bpol_psi(i, j);
    write_binary(outdir + "/psi_perturbation.dat", grid, psi_pert);

    write_timeseries(outdir + "/island_width.csv", island_evolution,
                     "time_ms,island_width_m");

    // ========== 4. THERMAL QUENCH ==========
    std::cout << "[4/6] Simulating thermal quench..." << std::endl;
    ThermalQuenchConfig tq_cfg;
    ThermalQuench tq(grid, tq_cfg);

    // Initialize with 10 keV plasma
    ScalarField Te_init(grid, "Te_init", 10.0);
    ScalarField ne_init(grid, "ne_init", 1.0e20);
    // Apply shaped profile to initial conditions
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nz; ++j) {
            double rho = grid.normalized_radius(grid.R(i), grid.Z(j),
                mhd_cfg.R0, mhd_cfg.a, mhd_cfg.elongation, mhd_cfg.triangularity);
            if (rho > 1.0) {
                Te_init(i, j) = 0.01;
                ne_init(i, j) = 1e18;
            } else {
                double profile = std::max(0.0, 1.0 - rho * rho);
                Te_init(i, j) = 10.0 * profile + 0.01;
                ne_init(i, j) = 1e20 * std::sqrt(std::max(0.01, profile));
            }
        }
    tq.initialize_from_mhd(Te_init, ne_init);

    write_binary(outdir + "/Te_pre_tq.dat", grid, tq.state().Te);
    write_binary(outdir + "/ne_pre_tq.dat", grid, tq.state().ne);

    // Evolve thermal quench and capture snapshots
    std::vector<std::pair<double,double>> tq_Te_avg, tq_Prad;
    double dt_tq = 1e-5;
    int tq_snapshots[] = {0, 50, 200, 500, 1000, 2000};
    int snap_idx = 0;
    for (int step = 0; step <= 2000; ++step) {
        if (snap_idx < 6 && step == tq_snapshots[snap_idx]) {
            std::string suffix = "_tq_" + std::to_string(step);
            write_binary(outdir + "/Te" + suffix + ".dat", grid, tq.state().Te);
            write_binary(outdir + "/Prad" + suffix + ".dat", grid, tq.state().radiation_power);
            snap_idx++;
        }
        tq_Te_avg.push_back({step * dt_tq * 1e3, tq.average_temperature()});
        tq_Prad.push_back({step * dt_tq * 1e3, tq.total_radiated_power()});
        if (step < 2000) tq.advance(dt_tq);
    }
    write_timeseries(outdir + "/tq_temperature.csv", tq_Te_avg, "time_ms,Te_avg_keV");
    write_timeseries(outdir + "/tq_radiation.csv", tq_Prad, "time_ms,P_rad_W");

    write_binary(outdir + "/Te_post_tq.dat", grid, tq.state().Te);
    write_binary(outdir + "/impurity_fraction.dat", grid, tq.state().impurity_fraction);

    std::cout << "  Final Te_avg: " << tq.average_temperature() << " keV" << std::endl;

    // ========== 5. CURRENT QUENCH ==========
    std::cout << "[5/6] Simulating current quench..." << std::endl;
    CurrentQuenchConfig cq_cfg;
    CurrentQuench cq(grid, cq_cfg);
    cq.initialize_from_thermal_quench(tq.state().Te, tq.state().ne, mhd.state().Jtor);

    write_binary(outdir + "/Jtor_pre_cq.dat", grid, cq.state().Jtor);

    std::vector<std::pair<double,double>> cq_current, cq_efield;
    double dt_cq = 1e-5;
    int cq_snapshots[] = {0, 100, 500, 1000, 2000, 3000};
    snap_idx = 0;
    for (int step = 0; step <= 3000; ++step) {
        if (snap_idx < 6 && step == cq_snapshots[snap_idx]) {
            std::string suffix = "_cq_" + std::to_string(step);
            write_binary(outdir + "/E_field" + suffix + ".dat", grid, cq.state().E_field);
            write_binary(outdir + "/Jtor" + suffix + ".dat", grid, cq.state().Jtor);
            write_binary(outdir + "/resistivity" + suffix + ".dat", grid, cq.state().resistivity);
            snap_idx++;
        }
        cq_current.push_back({step * dt_cq * 1e3, cq.state().total_current});
        cq_efield.push_back({step * dt_cq * 1e3, cq.state().total_E_field});
        if (step < 3000) cq.advance(dt_cq, tq.state().Te, tq.state().ne);
    }
    write_timeseries(outdir + "/cq_current.csv", cq_current, "time_ms,Ip_A");
    write_timeseries(outdir + "/cq_efield.csv", cq_efield, "time_ms,E_field_Vm");

    write_binary(outdir + "/E_field_post_cq.dat", grid, cq.state().E_field);

    std::cout << "  Final Ip: " << cq.state().total_current / 1e6 << " MA" << std::endl;

    // ========== 6. RUNAWAY ELECTRONS ==========
    std::cout << "[6/6] Simulating runaway electrons..." << std::endl;
    RunawayConfig re_cfg;
    RunawayElectrons re(grid, re_cfg);
    re.initialize_from_current_quench(tq.state().ne, tq.state().Te, cq.state().E_field);

    std::vector<std::pair<double,double>> re_current;
    double dt_re = 1e-7;
    int re_snapshots[] = {0, 1000, 5000, 10000, 20000};
    snap_idx = 0;
    for (int step = 0; step <= 20000; ++step) {
        if (snap_idx < 5 && step == re_snapshots[snap_idx]) {
            std::string suffix = "_re_" + std::to_string(step);
            write_binary(outdir + "/n_RE" + suffix + ".dat", grid, re.state().n_RE);
            snap_idx++;
        }
        if (step % 100 == 0)
            re_current.push_back({step * dt_re * 1e3, re.state().total_RE_current});
        if (step < 20000) re.advance(dt_re, tq.state().ne, tq.state().Te, cq.state().E_field);
    }
    write_timeseries(outdir + "/re_current.csv", re_current, "time_ms,I_RE_A");
    write_binary(outdir + "/n_RE_final.dat", grid, re.state().n_RE);

    std::cout << "  Final RE current: " << re.state().total_RE_current << " A" << std::endl;

    // ========== Write summary ==========
    {
        std::ofstream summary(outdir + "/summary.txt");
        summary << "Tokamak Disruption Simulation - Field Dump Summary\n";
        summary << "Grid: " << nr << " x " << nz << "\n";
        summary << "R range: [" << grid.R_min() << ", " << grid.R_max() << "] m\n";
        summary << "Z range: [" << grid.Z_min() << ", " << grid.Z_max() << "] m\n";
        summary << "R0 = " << mhd_cfg.R0 << " m\n";
        summary << "a = " << mhd_cfg.a << " m\n";
        summary << "kappa = " << mhd_cfg.elongation << "\n";
        summary << "delta = " << mhd_cfg.triangularity << "\n";
        summary << "B0 = " << mhd_cfg.B0 << " T\n";
        summary << "E_mag = " << mhd.magnetic_energy() / 1e9 << " GJ\n";
    }

    std::cout << "\n=== All fields written to " << outdir << "/ ===" << std::endl;

    MPI_Finalize();
    return 0;
}
