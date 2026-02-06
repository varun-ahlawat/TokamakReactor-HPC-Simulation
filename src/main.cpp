/// @file main.cpp
/// @brief Main entry point for the Tokamak Disruption Simulation.
///
/// Runs a complete disruption cascade simulation:
///   MHD instability → Thermal quench → Current quench → Runaway electrons
/// With optional Disruption Mitigation System (shattered pellet injection).

#include "coupling/multi_physics_coupler.h"
#include "io/diagnostics.h"
#include "io/data_writer.h"
#include <iostream>
#include <string>
#include <cstring>
#include <mpi.h>
#include <omp.h>

using namespace tokamak;

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --nr <N>          Grid points in R direction (default: 64)\n"
              << "  --nz <N>          Grid points in Z direction (default: 64)\n"
              << "  --tmax <T>        Maximum simulation time in ms (default: 10)\n"
              << "  --no-dms          Disable Disruption Mitigation System\n"
              << "  --output <dir>    Output directory (default: output)\n"
              << "  --diag-interval <N> Steps between diagnostics (default: 100)\n"
              << "  --help            Show this help message\n";
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Default configuration
    DisruptionSimConfig config;
    config.nr = 64;
    config.nz = 64;
    config.t_max = 0.010; // 10 ms
    config.enable_dms = true;
    std::string output_dir = "output";

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--nr") == 0 && i + 1 < argc) {
            config.nr = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--nz") == 0 && i + 1 < argc) {
            config.nz = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--tmax") == 0 && i + 1 < argc) {
            config.t_max = std::stod(argv[++i]) * 1.0e-3; // ms to s
        } else if (std::strcmp(argv[i], "--no-dms") == 0) {
            config.enable_dms = false;
        } else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--diag-interval") == 0 && i + 1 < argc) {
            config.diagnostic_interval = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--help") == 0) {
            if (rank == 0) print_usage(argv[0]);
            MPI_Finalize();
            return 0;
        }
    }

    if (rank == 0) {
        std::cout << "\n"
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║       TOKAMAK DISRUPTION SIMULATION (HPC)                  ║\n"
            "║       Multi-Physics Plasma Disruption Cascade              ║\n"
            "║                                                            ║\n"
            "║  Phases: MHD → Thermal Quench → Current Quench → RE       ║\n"
            "║  Parallelization: MPI + OpenMP                             ║\n"
            "╚══════════════════════════════════════════════════════════════╝\n"
            << std::endl;

        std::cout << "MPI processes: " << num_procs << std::endl;
        #pragma omp parallel
        {
            #pragma omp single
            std::cout << "OpenMP threads per process: " << omp_get_num_threads()
                      << std::endl;
        }
        std::cout << std::endl;
    }

    // Set up diagnostics
    Diagnostics diagnostics(output_dir);
    DataWriter writer(output_dir);

    // Create and initialize the multi-physics coupler
    MultiPhysicsCoupler coupler(config);
    coupler.initialize();

    // Write grid data
    if (rank == 0) {
        writer.write_grid(coupler.grid());
    }

    // Set up diagnostic callback
    coupler.set_diagnostic_callback(
        [&](DisruptionPhase phase, double time, int step) {
            if (rank != 0) return;

            DiagnosticPoint point;
            point.time = time;
            point.phase = phase;

            // Collect diagnostics based on current phase
            switch (phase) {
                case DisruptionPhase::MHD_EQUILIBRIUM:
                case DisruptionPhase::MHD_UNSTABLE:
                    point.temperature_avg = 0.0;
                    point.temperature_max = coupler.mhd().state().temperature.max_abs();
                    point.magnetic_energy = coupler.mhd().magnetic_energy();
                    point.kinetic_energy = coupler.mhd().kinetic_energy();
                    point.island_width = coupler.mhd().island_width();
                    break;

                case DisruptionPhase::THERMAL_QUENCH:
                    point.temperature_avg = coupler.thermal_quench().average_temperature();
                    point.temperature_max = coupler.thermal_quench().state().Te.max_abs();
                    point.radiated_power = coupler.thermal_quench().total_radiated_power();
                    break;

                case DisruptionPhase::CURRENT_QUENCH:
                    point.temperature_avg = coupler.thermal_quench().average_temperature();
                    point.plasma_current = coupler.current_quench().state().total_current;
                    point.E_field_avg = coupler.current_quench().state().total_E_field;
                    point.RE_current = coupler.runaway().state().total_RE_current;
                    break;

                case DisruptionPhase::RUNAWAY_PLATEAU:
                    point.RE_current = coupler.runaway().state().total_RE_current;
                    point.RE_fraction = coupler.runaway().state().total_RE_fraction;
                    break;

                default:
                    break;
            }

            diagnostics.record(point);

            // Write field snapshots periodically
            if (step % config.output_interval == 0) {
                switch (phase) {
                    case DisruptionPhase::MHD_EQUILIBRIUM:
                    case DisruptionPhase::MHD_UNSTABLE:
                        diagnostics.write_snapshot(coupler.grid(),
                            coupler.mhd().state().Bpol_psi, step, "psi");
                        diagnostics.write_snapshot(coupler.grid(),
                            coupler.mhd().state().pressure, step, "pressure");
                        break;
                    case DisruptionPhase::THERMAL_QUENCH:
                        diagnostics.write_snapshot(coupler.grid(),
                            coupler.thermal_quench().state().Te, step, "Te");
                        break;
                    case DisruptionPhase::CURRENT_QUENCH:
                        diagnostics.write_snapshot(coupler.grid(),
                            coupler.current_quench().state().E_field, step, "E_field");
                        break;
                    case DisruptionPhase::RUNAWAY_PLATEAU:
                        diagnostics.write_snapshot(coupler.grid(),
                            coupler.runaway().state().n_RE, step, "n_RE");
                        break;
                    default:
                        break;
                }
            }
        }
    );

    // Run simulation
    coupler.run();

    // Write final diagnostics
    if (rank == 0) {
        diagnostics.write_csv();

        // Print summary
        const auto& stats = coupler.stats();
        std::cout << "\nDiagnostic data written to: " << output_dir << "/diagnostics.csv"
                  << std::endl;
        std::cout << "Field snapshots written to: " << output_dir << "/"
                  << std::endl;
    }

    MPI_Finalize();
    return 0;
}
