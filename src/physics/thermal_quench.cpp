/// @file thermal_quench.cpp
/// @brief Thermal quench implementation.
///
/// Models the catastrophic temperature collapse following MHD instability.
/// Includes radiative cooling via impurity line radiation, thermal diffusion
/// along stochastic field lines, and impurity influx.

#include "physics/thermal_quench.h"
#include "utils/physical_constants.h"
#include <cmath>
#include <algorithm>
#include <omp.h>

namespace tokamak {

void ThermalQuenchState::initialize(const Grid& grid) {
    Te = ScalarField(grid, "Te", 0.0);
    Ti = ScalarField(grid, "Ti", 0.0);
    ne = ScalarField(grid, "ne", 0.0);
    impurity_fraction = ScalarField(grid, "f_imp", 0.0);
    radiation_power = ScalarField(grid, "P_rad", 0.0);
}

ThermalQuench::ThermalQuench(Grid& grid, const ThermalQuenchConfig& config)
    : grid_(grid), config_(config) {
    state_.initialize(grid);
}

void ThermalQuench::initialize_from_mhd(const ScalarField& Te_initial,
                                          const ScalarField& ne_initial) {
    int nr = grid_.nr();
    int nz = grid_.nz();

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nr; ++i) {
        for (int j = 0; j < nz; ++j) {
            state_.Te(i, j) = Te_initial(i, j);
            state_.Ti(i, j) = Te_initial(i, j) * 0.8; // Ti slightly less than Te
            state_.ne(i, j) = ne_initial(i, j);
            state_.impurity_fraction(i, j) = config_.impurity_fraction_initial;
        }
    }
}

double ThermalQuench::advance(double dt) {
    int nr = grid_.nr();
    int nz = grid_.nz();

    // Step 1: Compute radiation power
    compute_radiation(state_.Te, state_.ne, state_.impurity_fraction,
                      state_.radiation_power);

    // Step 2: Compute thermal diffusion
    ScalarField dTe_dt(grid_, "dTe_dt");
    double chi_eff = config_.chi_perpendicular +
                     config_.stochasticity_factor * config_.chi_parallel;
    // Limit chi_eff by CFL condition for explicit diffusion: chi*dt/dx² < 0.25
    double dx_min = std::min(grid_.dr(), grid_.dz());
    double chi_cfl_max = 0.25 * dx_min * dx_min / dt;
    chi_eff = std::min(chi_eff, chi_cfl_max);
    compute_diffusion(state_.Te, dTe_dt, chi_eff);

    // Step 3: Evolve impurities
    evolve_impurities(dt);

    // Step 4: Update temperatures
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < nr - 1; ++i) {
        for (int j = 1; j < nz - 1; ++j) {
            double ne = state_.ne(i, j);
            if (ne < 1.0e15) ne = 1.0e15;

            // ∂Te/∂t = -P_rad/(1.5 * ne * keV_to_J) + χ∇²Te
            double cooling = state_.radiation_power(i, j) /
                             (1.5 * ne * constants::keV_to_J);

            // Limit cooling to not exceed available thermal energy per step
            // (enforces energy conservation: cannot radiate more than exists)
            double T_available = state_.Te(i, j) - config_.wall_temperature;
            if (T_available < 0.0) T_available = 0.0;
            double max_cooling_rate = T_available / dt;
            if (cooling > max_cooling_rate) {
                cooling = max_cooling_rate;
                // Update radiation_power to reflect actual energy radiated
                state_.radiation_power(i, j) = cooling * 1.5 * ne * constants::keV_to_J;
            }

            state_.Te(i, j) += dt * (dTe_dt(i, j) - cooling);

            // Floor temperature
            if (state_.Te(i, j) < config_.wall_temperature)
                state_.Te(i, j) = config_.wall_temperature;

            // Ion temperature equilibrates toward electron temperature
            double tau_eq = 1.0e-3; // Ion-electron equilibration time
            state_.Ti(i, j) += dt * (state_.Te(i, j) - state_.Ti(i, j)) / tau_eq;
            if (state_.Ti(i, j) < config_.wall_temperature)
                state_.Ti(i, j) = config_.wall_temperature;
        }
    }

    // Boundary conditions
    for (int j = 0; j < nz; ++j) {
        state_.Te(0, j) = config_.wall_temperature;
        state_.Te(nr-1, j) = config_.wall_temperature;
    }
    for (int i = 0; i < nr; ++i) {
        state_.Te(i, 0) = config_.wall_temperature;
        state_.Te(i, nz-1) = config_.wall_temperature;
    }

    // Adaptive timestep: limit temperature change per step
    double T_max = state_.Te.max_abs();
    double dt_suggest = dt;
    if (T_max > 0.0) {
        double max_cooling = state_.radiation_power.max_abs() /
                             (1.5 * 1.0e20 * constants::keV_to_J);
        if (max_cooling > 0) {
            dt_suggest = std::min(dt, 0.1 * T_max / max_cooling);
        }
    }
    return std::max(dt_suggest, 1.0e-7);
}

bool ThermalQuench::is_complete(double T_threshold_keV) const {
    return average_temperature() < T_threshold_keV;
}

double ThermalQuench::total_radiated_power() const {
    double total = 0.0;
    double dr = grid_.dr();
    double dz = grid_.dz();

    #pragma omp parallel for collapse(2) reduction(+:total)
    for (int i = 0; i < grid_.nr(); ++i) {
        for (int j = 0; j < grid_.nz(); ++j) {
            double R = grid_.R(i);
            total += state_.radiation_power(i, j) * R * dr * dz * 2.0 * M_PI;
        }
    }
    return total;
}

double ThermalQuench::average_temperature() const {
    double sum_T = 0.0;
    double sum_w = 0.0;
    double dr = grid_.dr();
    double dz = grid_.dz();

    #pragma omp parallel for collapse(2) reduction(+:sum_T,sum_w)
    for (int i = 0; i < grid_.nr(); ++i) {
        for (int j = 0; j < grid_.nz(); ++j) {
            double R = grid_.R(i);
            double w = state_.ne(i, j) * R * dr * dz;
            sum_T += state_.Te(i, j) * w;
            sum_w += w;
        }
    }
    return (sum_w > 0) ? sum_T / sum_w : 0.0;
}

void ThermalQuench::compute_radiation(const ScalarField& Te, const ScalarField& ne,
                                       const ScalarField& f_imp,
                                       ScalarField& P_rad) {
    // Radiative cooling: P_rad = n_e * n_imp * L_z(T)
    // where L_z(T) is the cooling function (simplified model)
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < grid_.nr(); ++i) {
        for (int j = 0; j < grid_.nz(); ++j) {
            double T = Te(i, j);
            double n = ne(i, j);
            double f = f_imp(i, j);

            // Simplified cooling function for neon impurities
            // L_z peaks around ~100 eV for neon
            double T_eV = T * 1000.0; // keV to eV
            double L_z;
            if (T_eV < 1.0) {
                L_z = 1.0e-35; // Very cold, minimal radiation
            } else if (T_eV < 100.0) {
                // Rising part: L_z ~ T^0.5 below peak
                L_z = config_.radiation_coefficient * std::sqrt(T_eV / 100.0);
            } else if (T_eV < 1000.0) {
                // Peak and declining
                L_z = config_.radiation_coefficient *
                      std::exp(-(T_eV - 100.0) / 500.0);
            } else {
                // High-T: bremsstrahlung ~ sqrt(T)
                L_z = config_.radiation_coefficient * 0.1 *
                      std::sqrt(T_eV / 1000.0);
            }

            P_rad(i, j) = n * n * f * L_z;
        }
    }
}

void ThermalQuench::compute_diffusion(const ScalarField& T, ScalarField& dTdt,
                                       double chi_eff) const {
    double dr = grid_.dr();
    double dz = grid_.dz();

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < grid_.nr() - 1; ++i) {
        for (int j = 1; j < grid_.nz() - 1; ++j) {
            double R = grid_.R(i);
            double d2T_dR2 = (T(i+1, j) - 2.0 * T(i, j) + T(i-1, j)) / (dr * dr);
            double dT_dR = (T(i+1, j) - T(i-1, j)) / (2.0 * dr);
            double d2T_dZ2 = (T(i, j+1) - 2.0 * T(i, j) + T(i, j-1)) / (dz * dz);

            // Cylindrical Laplacian
            dTdt(i, j) = chi_eff * (d2T_dR2 + dT_dR / R + d2T_dZ2);
        }
    }
}

void ThermalQuench::evolve_impurities(double dt) {
    double R0 = constants::default_major_radius;
    double a = constants::default_minor_radius;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < grid_.nr(); ++i) {
        for (int j = 0; j < grid_.nz(); ++j) {
            double R = grid_.R(i);
            double Z = grid_.Z(j);
            double r = std::sqrt((R - R0) * (R - R0) + Z * Z);
            double rho_norm = r / a;

            // Impurity influx from edge (diffusive-like)
            double source = 0.0;
            if (rho_norm > 0.7) {
                source = config_.impurity_influx_rate * dt *
                         (rho_norm - 0.7) / 0.3;
            }
            double n = state_.ne(i, j);
            if (n < 1.0e15) n = 1.0e15;
            state_.impurity_fraction(i, j) += source / n;

            // Cap impurity fraction
            if (state_.impurity_fraction(i, j) > 0.5)
                state_.impurity_fraction(i, j) = 0.5;
        }
    }
}

} // namespace tokamak
