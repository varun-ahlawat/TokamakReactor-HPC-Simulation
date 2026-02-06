/// @file multi_physics_coupler.cpp
/// @brief Multi-physics coupling framework implementation.
///
/// Orchestrates the full disruption cascade simulation, managing phase
/// transitions and data handoff between physics solvers.

#include "coupling/multi_physics_coupler.h"
#include "utils/physical_constants.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>

namespace tokamak {

const char* phase_to_string(DisruptionPhase phase) {
    switch (phase) {
        case DisruptionPhase::MHD_EQUILIBRIUM:  return "MHD_EQUILIBRIUM";
        case DisruptionPhase::MHD_UNSTABLE:     return "MHD_UNSTABLE";
        case DisruptionPhase::THERMAL_QUENCH:   return "THERMAL_QUENCH";
        case DisruptionPhase::CURRENT_QUENCH:   return "CURRENT_QUENCH";
        case DisruptionPhase::RUNAWAY_PLATEAU:  return "RUNAWAY_PLATEAU";
        case DisruptionPhase::MITIGATED:        return "MITIGATED";
        case DisruptionPhase::POST_DISRUPTION:  return "POST_DISRUPTION";
        default:                                return "UNKNOWN";
    }
}

MultiPhysicsCoupler::MultiPhysicsCoupler(const DisruptionSimConfig& config)
    : config_(config) {}

void MultiPhysicsCoupler::initialize() {
    // Create grid
    grid_ = std::make_unique<Grid>(config_.nr, config_.nz,
                                    config_.R_min, config_.R_max,
                                    config_.Z_min, config_.Z_max);

    // Create physics solvers
    mhd_ = std::make_unique<MHDSolver>(*grid_, config_.mhd_config);
    tq_ = std::make_unique<ThermalQuench>(*grid_, config_.tq_config);
    cq_ = std::make_unique<CurrentQuench>(*grid_, config_.cq_config);
    re_ = std::make_unique<RunawayElectrons>(*grid_, config_.re_config);
    dms_ = std::make_unique<DisruptionMitigation>(*grid_, config_.dms_config);

    // Initialize MHD equilibrium
    mhd_->initialize_equilibrium();
    mhd_->apply_perturbation();

    phase_ = DisruptionPhase::MHD_EQUILIBRIUM;
    time_ = 0.0;
    step_count_ = 0;

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::cout << "=== Tokamak Disruption Simulation Initialized ===" << std::endl;
        std::cout << "  Grid: " << config_.nr << " x " << config_.nz << std::endl;
        std::cout << "  Domain: R=[" << config_.R_min << ", " << config_.R_max
                  << "] Z=[" << config_.Z_min << ", " << config_.Z_max << "]" << std::endl;
        std::cout << "  Max time: " << config_.t_max * 1000.0 << " ms" << std::endl;
        std::cout << "  DMS enabled: " << (config_.enable_dms ? "yes" : "no") << std::endl;
        std::cout << "=================================================" << std::endl;
    }
}

void MultiPhysicsCoupler::run() {
    auto wall_start = std::chrono::high_resolution_clock::now();

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    while (step()) {
        // Periodic progress report
        if (rank == 0 && step_count_ % config_.diagnostic_interval == 0) {
            std::cout << std::fixed << std::setprecision(6)
                      << "  Step " << std::setw(8) << step_count_
                      << " | t = " << std::setw(12) << time_ * 1000.0 << " ms"
                      << " | Phase: " << phase_to_string(phase_)
                      << std::endl;
        }

        // Call diagnostic callback
        if (diag_callback_ && step_count_ % config_.diagnostic_interval == 0) {
            diag_callback_(phase_, time_, step_count_);
        }
    }

    auto wall_end = std::chrono::high_resolution_clock::now();
    stats_.wall_time = std::chrono::duration<double>(wall_end - wall_start).count();
    stats_.total_steps = step_count_;

    if (rank == 0) {
        std::cout << "\n=== Simulation Complete ===" << std::endl;
        std::cout << "  Total steps: " << stats_.total_steps << std::endl;
        std::cout << "  Wall time: " << std::fixed << std::setprecision(2)
                  << stats_.wall_time << " s" << std::endl;
        std::cout << "  Final phase: " << phase_to_string(phase_) << std::endl;
        std::cout << "  MHD steps: " << stats_.mhd_steps << std::endl;
        std::cout << "  TQ steps: " << stats_.tq_steps << std::endl;
        std::cout << "  CQ steps: " << stats_.cq_steps << std::endl;
        std::cout << "  RE steps: " << stats_.re_steps << std::endl;
        std::cout << "  Peak island width: " << stats_.peak_island_width << " m" << std::endl;
        std::cout << "  Min temperature: " << stats_.min_temperature << " keV" << std::endl;
        std::cout << "  Peak E field: " << stats_.peak_E_field << " V/m" << std::endl;
        std::cout << "  Peak RE current: " << stats_.peak_RE_current << " A" << std::endl;
        std::cout << "  Mitigated: " << (stats_.disruption_mitigated ? "YES" : "NO") << std::endl;
        std::cout << "===========================" << std::endl;
    }
}

bool MultiPhysicsCoupler::step() {
    if (phase_ == DisruptionPhase::POST_DISRUPTION) return false;
    if (time_ >= config_.t_max) {
        phase_ = DisruptionPhase::POST_DISRUPTION;
        return false;
    }

    double dt = current_dt();

    switch (phase_) {
        case DisruptionPhase::MHD_EQUILIBRIUM:
        case DisruptionPhase::MHD_UNSTABLE: {
            double dt_suggest = mhd_->advance(dt);
            stats_.mhd_steps++;

            double iw = mhd_->island_width();
            if (iw > stats_.peak_island_width)
                stats_.peak_island_width = iw;

            // Check for instability growth
            if (phase_ == DisruptionPhase::MHD_EQUILIBRIUM &&
                iw > 0.01 * config_.mhd_config.a) {
                phase_ = DisruptionPhase::MHD_UNSTABLE;
            }
            break;
        }

        case DisruptionPhase::THERMAL_QUENCH: {
            double dt_suggest = tq_->advance(dt);
            stats_.tq_steps++;

            double T_avg = tq_->average_temperature();
            if (T_avg < stats_.min_temperature)
                stats_.min_temperature = T_avg;

            // Trigger DMS if enabled
            if (config_.enable_dms && !dms_->is_triggered()) {
                dms_->trigger(time_);
            }

            // Evolve DMS alongside thermal quench
            if (dms_->is_triggered()) {
                dms_->advance(dt, time_, tq_->state().Te, tq_->state().ne);
            }
            break;
        }

        case DisruptionPhase::CURRENT_QUENCH: {
            double dt_suggest = cq_->advance(dt, tq_->state().Te, tq_->state().ne);
            stats_.cq_steps++;

            double E = cq_->state().total_E_field;
            if (E > stats_.peak_E_field) stats_.peak_E_field = E;

            // Also evolve REs during current quench
            if (cq_->E_over_Dreicer() > 1.0) {
                re_->advance(dt, tq_->state().ne, tq_->state().Te,
                             cq_->state().E_field);
                stats_.re_steps++;

                double I_RE = re_->state().total_RE_current;
                if (I_RE > stats_.peak_RE_current)
                    stats_.peak_RE_current = I_RE;
            }

            // Continue DMS
            if (dms_->is_triggered() && !dms_->is_complete()) {
                dms_->advance(dt, time_, tq_->state().Te, tq_->state().ne);
            }
            break;
        }

        case DisruptionPhase::RUNAWAY_PLATEAU: {
            re_->advance(dt, tq_->state().ne, tq_->state().Te,
                         cq_->state().E_field);
            stats_.re_steps++;

            double I_RE = re_->state().total_RE_current;
            if (I_RE > stats_.peak_RE_current)
                stats_.peak_RE_current = I_RE;
            break;
        }

        case DisruptionPhase::MITIGATED:
            stats_.disruption_mitigated = true;
            phase_ = DisruptionPhase::POST_DISRUPTION;
            return false;

        case DisruptionPhase::POST_DISRUPTION:
            return false;
    }

    time_ += dt;
    step_count_++;

    check_phase_transition();
    return true;
}

void MultiPhysicsCoupler::check_phase_transition() {
    switch (phase_) {
        case DisruptionPhase::MHD_UNSTABLE:
            // Transition to thermal quench when island exceeds threshold
            if (mhd_->island_width() >
                config_.mhd_island_threshold * config_.mhd_config.a) {
                int rank = 0;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                if (rank == 0) {
                    std::cout << "\n*** DISRUPTION TRIGGERED at t = "
                              << time_ * 1000.0 << " ms ***" << std::endl;
                    std::cout << "    Island width: " << mhd_->island_width()
                              << " m" << std::endl;
                }
                transfer_mhd_to_thermal_quench();
                phase_ = DisruptionPhase::THERMAL_QUENCH;
            }
            break;

        case DisruptionPhase::THERMAL_QUENCH:
            if (tq_->is_complete(config_.tq_temperature_threshold)) {
                int rank = 0;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                if (rank == 0) {
                    std::cout << "\n*** THERMAL QUENCH COMPLETE at t = "
                              << time_ * 1000.0 << " ms ***" << std::endl;
                    std::cout << "    Avg temperature: "
                              << tq_->average_temperature() << " keV" << std::endl;
                }
                transfer_thermal_quench_to_current_quench();
                phase_ = DisruptionPhase::CURRENT_QUENCH;
            }
            break;

        case DisruptionPhase::CURRENT_QUENCH:
            if (cq_->is_complete(config_.cq_current_threshold)) {
                // Check if DMS mitigated the disruption
                if (dms_->is_complete() &&
                    re_->state().total_RE_current < 1.0e3) {
                    stats_.disruption_mitigated = true;
                    phase_ = DisruptionPhase::MITIGATED;
                } else {
                    transfer_current_quench_to_runaway();
                    phase_ = DisruptionPhase::RUNAWAY_PLATEAU;
                }

                int rank = 0;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                if (rank == 0) {
                    std::cout << "\n*** CURRENT QUENCH COMPLETE at t = "
                              << time_ * 1000.0 << " ms ***" << std::endl;
                    std::cout << "    E/E_dreicer: " << cq_->E_over_Dreicer()
                              << std::endl;
                    std::cout << "    RE current: "
                              << re_->state().total_RE_current << " A"
                              << std::endl;
                }
            }
            break;

        case DisruptionPhase::RUNAWAY_PLATEAU:
            if (re_->is_saturated()) {
                phase_ = DisruptionPhase::POST_DISRUPTION;
            }
            break;

        default:
            break;
    }
}

void MultiPhysicsCoupler::transfer_mhd_to_thermal_quench() {
    // Transfer temperature and density from MHD to thermal quench
    // Scale MHD normalized quantities to physical units

    ScalarField Te_phys(mhd_->state().temperature);
    ScalarField ne_phys(*grid_, "ne_phys");

    // Convert MHD density to physical electron density
    double n0 = constants::default_n_e;
    int nr = grid_->nr();
    int nz = grid_->nz();
    for (int i = 0; i < nr; ++i) {
        for (int j = 0; j < nz; ++j) {
            ne_phys(i, j) = mhd_->state().density(i, j) * n0;
        }
    }

    tq_->initialize_from_mhd(Te_phys, ne_phys);
}

void MultiPhysicsCoupler::transfer_thermal_quench_to_current_quench() {
    // Transfer final TQ state to CQ initialization
    cq_->initialize_from_thermal_quench(
        tq_->state().Te, tq_->state().ne, mhd_->state().Jtor);
}

void MultiPhysicsCoupler::transfer_current_quench_to_runaway() {
    // Initialize RE model from CQ electric field
    re_->initialize_from_current_quench(
        tq_->state().ne, tq_->state().Te, cq_->state().E_field);
}

double MultiPhysicsCoupler::current_dt() const {
    switch (phase_) {
        case DisruptionPhase::MHD_EQUILIBRIUM:
        case DisruptionPhase::MHD_UNSTABLE:
            return config_.dt_mhd;
        case DisruptionPhase::THERMAL_QUENCH:
            return config_.dt_thermal;
        case DisruptionPhase::CURRENT_QUENCH:
            return config_.dt_current;
        case DisruptionPhase::RUNAWAY_PLATEAU:
            return config_.dt_runaway;
        default:
            return config_.dt_mhd;
    }
}

} // namespace tokamak
