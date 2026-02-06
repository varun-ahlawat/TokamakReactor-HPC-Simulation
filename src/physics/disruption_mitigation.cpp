/// @file disruption_mitigation.cpp
/// @brief Disruption Mitigation System (DMS) implementation.
///
/// Models the shattered pellet injection (SPI) response to a disruption.
/// Uses the Neutral Gas Shielding (NGS) model for pellet ablation.

#include "physics/disruption_mitigation.h"
#include "utils/physical_constants.h"
#include <cmath>
#include <algorithm>
#include <omp.h>

namespace tokamak {

void DMSState::initialize(const Grid& grid) {
    deposited_density = ScalarField(grid, "n_dep", 0.0);
    collisionality = ScalarField(grid, "nu_coll", 1.0);
    is_triggered = false;
    is_complete = false;
    trigger_time = 0.0;
    deposited_fraction = 0.0;
}

DisruptionMitigation::DisruptionMitigation(Grid& grid, const DMSConfig& config)
    : grid_(grid), config_(config) {
    state_.initialize(grid);
}

void DisruptionMitigation::trigger(double current_time) {
    state_.is_triggered = true;
    state_.trigger_time = current_time;
    state_.deposited_fraction = 0.0;
}

void DisruptionMitigation::advance(double dt, double current_time,
                                    const ScalarField& Te, ScalarField& ne) {
    if (!state_.is_triggered) return;

    double elapsed = current_time - state_.trigger_time;

    // Delay before pellet arrives
    if (elapsed < config_.trigger_delay) return;

    // Check if deposition is complete
    if (state_.deposited_fraction >= 1.0) {
        state_.is_complete = true;
        return;
    }

    // Compute deposition profile and update density
    compute_deposition_profile(Te, dt);

    // Update deposited fraction based on time
    double deposition_elapsed = elapsed - config_.trigger_delay;
    state_.deposited_fraction = std::min(1.0,
        deposition_elapsed / config_.assimilation_time);

    // Add deposited material to electron density and update collisionality
    int nr = grid_.nr();
    int nz = grid_.nz();

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nr; ++i) {
        for (int j = 0; j < nz; ++j) {
            // Add deposited density to electron density
            // Each neon atom contributes Z electrons when fully ionized
            ne(i, j) += state_.deposited_density(i, j) * config_.pellet_Z * dt /
                        config_.assimilation_time;

            // Enhanced collisionality factor
            double n_dep = state_.deposited_density(i, j);
            double n_e = ne(i, j);
            if (n_e > 1.0e15) {
                state_.collisionality(i, j) = 1.0 + config_.pellet_Z *
                    config_.pellet_Z * n_dep / n_e;
            }
        }
    }
}

double DisruptionMitigation::ablation_rate(double Te_keV, double ne,
                                            double r_pellet) const {
    // Neutral Gas Shielding (NGS) pellet ablation model:
    // dN/dt = C * n_e^{1/3} * T_e^{5/3} * r_p^{4/3}
    // C ≈ 1.12e16 for neon in SI units

    double T_eV = Te_keV * 1000.0;
    if (T_eV < 1.0 || ne < 1.0e15 || r_pellet < 1.0e-6) return 0.0;

    double C_ngs = 1.12e16; // NGS coefficient
    return C_ngs * std::pow(ne, 1.0/3.0) * std::pow(T_eV, 5.0/3.0) *
           std::pow(r_pellet, 4.0/3.0);
}

void DisruptionMitigation::compute_deposition_profile(const ScalarField& Te,
                                                        double dt) {
    int nr = grid_.nr();
    int nz = grid_.nz();
    double R0 = constants::default_major_radius;
    double a = constants::default_minor_radius;

    // Total particles in all fragments
    double V_fragment = (4.0 / 3.0) * M_PI * std::pow(config_.pellet_radius, 3);
    double N_total = config_.pellet_n_solid * V_fragment * config_.num_fragments;

    // Deposition profile: Gaussian centered on injection path
    // Fragments penetrate to different depths based on size and Te
    double sigma_R = 0.3 * a; // Radial width of deposition
    double sigma_Z = 0.2 * a; // Vertical width

    // Injection path center moves inward over time
    double penetration = config_.pellet_velocity * dt;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nr; ++i) {
        for (int j = 0; j < nz; ++j) {
            double R = grid_.R(i);
            double Z = grid_.Z(j);

            // Distance from injection path
            double dR = R - R0;
            double dZ = Z - config_.injection_Z;

            // Gaussian deposition profile (toroidally symmetric approximation)
            double r = std::sqrt(dR * dR + dZ * dZ);
            double rho_norm = r / a;

            // Peaked deposition near plasma center
            double profile = std::exp(-rho_norm * rho_norm / (2.0 * 0.3 * 0.3));

            // Rate-limited deposition
            double rate = N_total * profile / (config_.assimilation_time *
                          M_PI * a * a); // Volume-normalized

            state_.deposited_density(i, j) = rate * state_.deposited_fraction;
        }
    }
}

double DisruptionMitigation::total_deposited() const {
    double total = 0.0;
    double dr = grid_.dr();
    double dz = grid_.dz();

    #pragma omp parallel for collapse(2) reduction(+:total)
    for (int i = 0; i < grid_.nr(); ++i) {
        for (int j = 0; j < grid_.nz(); ++j) {
            double R = grid_.R(i);
            total += state_.deposited_density(i, j) * R * dr * dz * 2.0 * M_PI;
        }
    }
    return total;
}

} // namespace tokamak
