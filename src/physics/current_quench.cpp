/// @file current_quench.cpp
/// @brief Current quench model implementation.
///
/// Simulates the plasma current decay as temperature drops, resistivity rises,
/// and an electric field is induced that can generate runaway electrons.

#include "physics/current_quench.h"
#include "utils/physical_constants.h"
#include <cmath>
#include <algorithm>
#include <omp.h>

namespace tokamak {

void CurrentQuenchState::initialize(const Grid& grid) {
    Jtor = ScalarField(grid, "Jtor_cq", 0.0);
    E_field = ScalarField(grid, "E_field", 0.0);
    resistivity = ScalarField(grid, "eta", 0.0);
    total_current = 0.0;
    total_E_field = 0.0;
}

CurrentQuench::CurrentQuench(Grid& grid, const CurrentQuenchConfig& config)
    : grid_(grid), config_(config), initial_current_(config.initial_current) {
    state_.initialize(grid);
}

void CurrentQuench::initialize_from_thermal_quench(const ScalarField& Te,
                                                     const ScalarField& ne,
                                                     const ScalarField& Jtor_initial) {
    int nr = grid_.nr();
    int nz = grid_.nz();

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nr; ++i) {
        for (int j = 0; j < nz; ++j) {
            state_.Jtor(i, j) = Jtor_initial(i, j);
        }
    }

    compute_resistivity(Te, ne);
    compute_electric_field();

    // Compute initial total current
    state_.total_current = config_.initial_current;
    initial_current_ = config_.initial_current;
    elapsed_time_ = 0.0;
}

double CurrentQuench::advance(double dt, const ScalarField& Te,
                                const ScalarField& ne) {
    int nr = grid_.nr();
    int nz = grid_.nz();

    elapsed_time_ += dt;

    // Update resistivity from current temperature
    compute_resistivity(Te, ne);

    // Compute volume-averaged resistivity for global L/R decay
    double avg_eta = 0.0;
    double total_volume = 0.0;
    double dr = grid_.dr();
    double dz = grid_.dz();

    #pragma omp parallel for collapse(2) reduction(+:avg_eta,total_volume)
    for (int i = 1; i < nr - 1; ++i) {
        for (int j = 1; j < nz - 1; ++j) {
            double R = grid_.R(i);
            double dV = R * dr * dz;
            avg_eta += state_.resistivity(i, j) * dV;
            total_volume += dV;
        }
    }
    avg_eta /= (total_volume > 0 ? total_volume : 1.0);

    // L/R decay: dI/dt = -R_eff * I / L = -(η_avg / (μ₀ * a²)) * I
    // Where effective R = η * (2πR₀) / (πa²) and L ≈ μ₀R₀(ln(8R₀/a) - 2)
    double R_eff = avg_eta * 2.0 * M_PI * constants::default_major_radius /
                   (M_PI * constants::default_minor_radius *
                    constants::default_minor_radius);
    double L = config_.plasma_inductance;
    if (L < 1.0e-9) L = 1.0e-9;

    double tau_LR = L / R_eff;
    double decay_factor = std::exp(-dt / tau_LR);
    state_.total_current *= decay_factor;

    quench_time_ = tau_LR;

    // Update current density profile (maintain shape, scale amplitude)
    double scale = (initial_current_ > 0) ?
                   state_.total_current / initial_current_ : 0.0;

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < nr - 1; ++i) {
        for (int j = 1; j < nz - 1; ++j) {
            // Scale current density with total current
            state_.Jtor(i, j) *= decay_factor;

            // Compute local electric field: E = η * J
            state_.E_field(i, j) = state_.resistivity(i, j) *
                                    std::abs(state_.Jtor(i, j));
        }
    }

    // Volume-averaged electric field
    double sum_E = 0.0;
    #pragma omp parallel for collapse(2) reduction(+:sum_E)
    for (int i = 1; i < nr - 1; ++i) {
        for (int j = 1; j < nz - 1; ++j) {
            sum_E += state_.E_field(i, j);
        }
    }
    state_.total_E_field = sum_E / ((nr - 2) * (nz - 2));

    // Compute E/E_dreicer
    double T_avg = 0.0;
    double n_avg = 0.0;
    int count = 0;
    for (int i = nr/4; i < 3*nr/4; ++i) {
        for (int j = nz/4; j < 3*nz/4; ++j) {
            T_avg += Te(i, j);
            n_avg += ne(i, j);
            count++;
        }
    }
    T_avg /= count;
    n_avg /= count;

    double ln_lambda = constants::coulomb_logarithm(n_avg, std::max(T_avg, 0.001));
    double E_D = constants::dreicer_field(n_avg, std::max(T_avg, 0.001), ln_lambda);
    E_over_Ed_ = (E_D > 0) ? state_.total_E_field / E_D : 0.0;

    // Adaptive timestep: limit current change to 10% per step
    double dt_suggest = dt;
    if (tau_LR > 0 && tau_LR < 1.0) {
        dt_suggest = 0.1 * tau_LR;
    }
    return std::max(dt_suggest, 1.0e-7);
}

bool CurrentQuench::is_complete(double I_threshold_fraction) const {
    return std::abs(state_.total_current) <
           I_threshold_fraction * std::abs(initial_current_);
}

void CurrentQuench::compute_resistivity(const ScalarField& Te,
                                          const ScalarField& ne) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < grid_.nr(); ++i) {
        for (int j = 0; j < grid_.nz(); ++j) {
            double T = std::max(Te(i, j), 1.0e-3); // Floor at 1 eV
            double n = std::max(ne(i, j), 1.0e15);
            double ln_lambda = constants::coulomb_logarithm(n, T);
            state_.resistivity(i, j) = constants::spitzer_resistivity(
                T, config_.Z_eff, ln_lambda);
        }
    }
}

void CurrentQuench::compute_electric_field() {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < grid_.nr(); ++i) {
        for (int j = 0; j < grid_.nz(); ++j) {
            state_.E_field(i, j) = state_.resistivity(i, j) *
                                    std::abs(state_.Jtor(i, j));
        }
    }
}

} // namespace tokamak
