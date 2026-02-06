/// @file runaway_electrons.cpp
/// @brief Runaway electron avalanche model implementation.
///
/// Implements the Dreicer (primary) generation mechanism and the
/// Rosenbluth-Putvinski avalanche (secondary) multiplication model.
/// Based on the physics described in https://arxiv.org/abs/2403.04948.

#include "physics/runaway_electrons.h"
#include "utils/physical_constants.h"
#include <cmath>
#include <algorithm>
#include <omp.h>

namespace tokamak {

void RunawayState::initialize(const Grid& grid) {
    n_RE = ScalarField(grid, "n_RE", 0.0);
    RE_current = ScalarField(grid, "J_RE", 0.0);
    growth_rate = ScalarField(grid, "gamma_RE", 0.0);
    avg_energy = ScalarField(grid, "E_avg_RE", 0.0);
    total_RE_current = 0.0;
    total_RE_fraction = 0.0;
}

RunawayElectrons::RunawayElectrons(Grid& grid, const RunawayConfig& config)
    : grid_(grid), config_(config) {
    state_.initialize(grid);
}

void RunawayElectrons::initialize_from_current_quench(
    const ScalarField& ne, const ScalarField& Te, const ScalarField& E_field) {
    int nr = grid_.nr();
    int nz = grid_.nz();

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nr; ++i) {
        for (int j = 0; j < nz; ++j) {
            // Initial seed RE population (hot-tail remnant)
            state_.n_RE(i, j) = config_.seed_fraction * ne(i, j);
            state_.avg_energy(i, j) = 1.0; // 1 MeV initial energy
        }
    }
}

double RunawayElectrons::advance(double dt, const ScalarField& ne,
                                   const ScalarField& Te,
                                   const ScalarField& E_field) {
    int nr = grid_.nr();
    int nz = grid_.nz();

    // Compute generation sources
    ScalarField dreicer_source(grid_, "dreicer_src");
    ScalarField avalanche_growth(grid_, "aval_growth");
    ScalarField synch_loss(grid_, "synch_loss");

    compute_dreicer_generation(ne, Te, E_field, dreicer_source);
    compute_avalanche(ne, E_field, avalanche_growth);
    compute_synchrotron_losses(synch_loss);

    // Evolve RE density:
    // dn_RE/dt = S_dreicer + γ_avalanche * n_RE - ν_loss * n_RE
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < nr - 1; ++i) {
        for (int j = 1; j < nz - 1; ++j) {
            double n_re = state_.n_RE(i, j);
            double gamma_av = avalanche_growth(i, j);
            double s_drei = dreicer_source(i, j);
            double loss = synch_loss(i, j);

            state_.growth_rate(i, j) = gamma_av;

            // Semi-implicit update to handle exponential growth
            // dn/dt = S + (γ - ν) * n
            double net_rate = gamma_av - loss;
            double new_n_re;
            if (std::abs(net_rate * dt) < 0.01) {
                // Small growth: use Euler
                new_n_re = n_re + dt * (s_drei + net_rate * n_re);
            } else {
                // Large growth: use analytical exponential
                new_n_re = n_re * std::exp(net_rate * dt) +
                           s_drei / (net_rate + 1.0e-30) *
                           (std::exp(net_rate * dt) - 1.0);
            }

            // Floor and cap
            state_.n_RE(i, j) = std::max(0.0, std::min(new_n_re, ne(i, j)));

            // RE current density: J_RE = n_RE * e * c (relativistic REs at ~c)
            state_.RE_current(i, j) = state_.n_RE(i, j) *
                                       constants::e_charge * constants::c_light;

            // Average energy evolution (simplified)
            double E_acc = E_field(i, j) * constants::e_charge *
                           constants::c_light * dt; // Energy gain per step [J]
            double E_loss_sync = synch_loss(i, j) * state_.avg_energy(i, j) *
                                 constants::eV_to_J * 1.0e6 * dt;
            double E_gain_MeV = (E_acc - E_loss_sync) /
                                (constants::eV_to_J * 1.0e6);
            state_.avg_energy(i, j) += E_gain_MeV;
            state_.avg_energy(i, j) = std::clamp(state_.avg_energy(i, j),
                                                  0.1, config_.max_energy_MeV);
        }
    }

    // Compute total RE current
    double total_J_RE = 0.0;
    double total_J = 0.0;
    double dr = grid_.dr();
    double dz = grid_.dz();

    #pragma omp parallel for collapse(2) reduction(+:total_J_RE)
    for (int i = 0; i < nr; ++i) {
        for (int j = 0; j < nz; ++j) {
            double R = grid_.R(i);
            total_J_RE += state_.RE_current(i, j) * R * dr * dz * 2.0 * M_PI;
        }
    }
    state_.total_RE_current = total_J_RE;

    // Adaptive timestep: limit RE density change
    double max_growth = avalanche_growth.max_abs();
    double dt_suggest = dt;
    if (max_growth > 0) {
        dt_suggest = std::min(dt, 0.1 / max_growth);
    }
    return std::max(dt_suggest, 1.0e-9);
}

bool RunawayElectrons::is_saturated() const {
    // Saturated when RE current fraction > 50% or growth rate < threshold
    return state_.total_RE_fraction > 0.5 ||
           state_.growth_rate.max_abs() < 1.0;
}

double RunawayElectrons::dreicer_rate(double ne, double Te_keV, double E,
                                        double E_c, double Z_eff) {
    // Dreicer generation rate (Connor-Hastie formula, simplified):
    // S_D = C * ν_ee * n_e * (E/E_D)^(-h) * exp(-E_D/(4E) - sqrt(E_D/E))
    // where h = (Z_eff + 1)/16

    if (E <= 0 || E_c <= 0 || Te_keV < 1.0e-3) return 0.0;

    double E_over_Ec = E / E_c;
    if (E_over_Ec < 1.0) return 0.0; // Below critical field, no generation

    double ln_lambda = constants::coulomb_logarithm(ne, Te_keV);
    double nu_ee = constants::collision_frequency(ne, Te_keV, ln_lambda);

    double h = (Z_eff + 1.0) / 16.0;
    double alpha = E_c / E;
    double rate = 0.35 * nu_ee * ne * std::pow(E_over_Ec, -h) *
                  std::exp(-0.25 * alpha - std::sqrt(alpha));

    return std::max(0.0, rate);
}

double RunawayElectrons::avalanche_rate(double ne, double E, double E_c,
                                          double ln_lambda, double Z_eff) {
    // Rosenbluth-Putvinski avalanche growth rate:
    // γ_aval = (1/τ_c) * (E/E_c - 1) / (ln Λ * √(4 + ν_s/ν_D))
    // Simplified: γ ≈ (e*E)/(m_e*c*ln_lambda) * (1 - E_c/E + correction)

    if (E <= E_c || E_c <= 0) return 0.0;

    double tau_c = constants::m_electron * constants::c_light /
                   (constants::e_charge * E_c);
    double rate = (1.0 / tau_c) * (E / E_c - 1.0) /
                  (ln_lambda * std::sqrt(4.0 + Z_eff));

    return std::max(0.0, rate);
}

void RunawayElectrons::compute_dreicer_generation(
    const ScalarField& ne, const ScalarField& Te,
    const ScalarField& E_field, ScalarField& source) {

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < grid_.nr(); ++i) {
        for (int j = 0; j < grid_.nz(); ++j) {
            double n = ne(i, j);
            double T = Te(i, j);
            double E = E_field(i, j);

            if (T < 1.0e-3 || n < 1.0e15) {
                source(i, j) = 0.0;
                continue;
            }

            double ln_lambda = constants::coulomb_logarithm(n, T);
            double E_D = constants::dreicer_field(n, T, ln_lambda);
            double Z_eff = constants::default_Z_eff;

            source(i, j) = dreicer_rate(n, T, E, E_D, Z_eff);
        }
    }
}

void RunawayElectrons::compute_avalanche(const ScalarField& ne,
                                           const ScalarField& E_field,
                                           ScalarField& growth) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < grid_.nr(); ++i) {
        for (int j = 0; j < grid_.nz(); ++j) {
            double n = ne(i, j);
            double E = E_field(i, j);

            if (n < 1.0e15 || E < 1.0e-3) {
                growth(i, j) = 0.0;
                continue;
            }

            // Critical field (Connor-Hastie)
            double T_cold = 0.01; // Post-quench temperature ~10 eV
            double ln_lambda = constants::coulomb_logarithm(n, T_cold);
            double E_c = constants::dreicer_field(n, T_cold, ln_lambda) *
                         constants::default_T_e; // Scale to critical

            // Use simpler estimate: E_c ≈ n_e * e^3 * ln Λ / (4π ε₀² m_e c²)
            E_c = n * std::pow(constants::e_charge, 3) * ln_lambda /
                  (4.0 * M_PI * constants::epsilon_0 * constants::epsilon_0 *
                   constants::m_electron * constants::c_light * constants::c_light);

            growth(i, j) = config_.avalanche_coefficient *
                           avalanche_rate(n, E, E_c, ln_lambda,
                                         constants::default_Z_eff);
        }
    }
}

void RunawayElectrons::compute_synchrotron_losses(ScalarField& loss) {
    // Synchrotron radiation loss rate for relativistic electrons in B field
    // ν_synch = (2/3) * r_e * c * γ² * (B/B_crit)²
    // Simplified: use constant loss rate proportional to energy²

    double B = constants::default_B_toroidal;
    double B_crit = constants::m_electron * constants::m_electron *
                    constants::c_light * constants::c_light * constants::c_light /
                    (constants::e_charge * constants::e_charge * constants::e_charge);
    // Actually B_crit = m²c³/(e³) is wrong; use B_crit = m²c²/(eℏ) ≈ 4.4e9 T
    double B_crit_correct = 4.414e9; // Schwinger critical field [T]

    double synch_rate = config_.synchrotron_loss_factor *
                        (2.0 / 3.0) * constants::classical_electron_radius *
                        constants::c_light * (B / B_crit_correct) * (B / B_crit_correct);

    loss.fill(synch_rate);
}

} // namespace tokamak
