#pragma once
/// @file thermal_quench.h
/// @brief Thermal quench model (Phase 2 of disruption cascade).
///
/// Models the catastrophic temperature collapse when MHD instability
/// causes loss of confinement. The hot plasma touches the cold wall,
/// impurities flood in, and radiation losses cause rapid cooling.
///
/// Physics modeled:
///   - Radiative cooling via impurity line radiation
///   - Parallel heat transport along stochastic field lines
///   - Impurity influx from plasma-wall interaction
///   - Temperature evolution: ∂T/∂t = -P_rad/(n_e) + χ∇²T

#include "core/grid.h"
#include "core/field.h"

namespace tokamak {

/// Configuration for thermal quench model
struct ThermalQuenchConfig {
    double impurity_fraction_initial = 0.01;  // Initial neon/argon fraction
    double impurity_influx_rate = 1.0e21;     // Impurity influx rate [m⁻³/s]
    double chi_parallel = 1.0e8;              // Parallel thermal diffusivity [m²/s]
    double chi_perpendicular = 1.0;           // Perpendicular thermal diffusivity [m²/s]
    double stochasticity_factor = 0.1;        // Degree of field line stochasticity [0-1]
    double radiation_coefficient = 1.0e-31;   // Radiative cooling rate coefficient [W·m³]
    double wall_temperature = 0.01;           // Wall temperature [keV]
};

/// State variables for the thermal quench phase
struct ThermalQuenchState {
    ScalarField Te;                // Electron temperature [keV]
    ScalarField Ti;                // Ion temperature [keV]
    ScalarField ne;                // Electron density [m⁻³]
    ScalarField impurity_fraction; // Impurity fraction
    ScalarField radiation_power;   // Radiated power density [W/m³]

    void initialize(const Grid& grid);
};

/// Thermal quench simulator
class ThermalQuench {
public:
    explicit ThermalQuench(Grid& grid,
                           const ThermalQuenchConfig& config = ThermalQuenchConfig());

    /// Initialize from MHD state at onset of thermal quench
    /// @param Te_initial Initial electron temperature field [keV]
    /// @param ne_initial Initial electron density field [m⁻³]
    void initialize_from_mhd(const ScalarField& Te_initial,
                             const ScalarField& ne_initial);

    /// Advance the thermal quench by one timestep
    /// @return Suggested next timestep
    double advance(double dt);

    /// Get current state
    const ThermalQuenchState& state() const { return state_; }
    ThermalQuenchState& state() { return state_; }

    /// Check if thermal quench is complete (T < threshold everywhere)
    bool is_complete(double T_threshold_keV = 0.1) const;

    /// Compute total radiated power [W]
    double total_radiated_power() const;

    /// Get average temperature [keV]
    double average_temperature() const;

private:
    Grid& grid_;
    ThermalQuenchConfig config_;
    ThermalQuenchState state_;

    /// Compute radiative cooling rate
    void compute_radiation(const ScalarField& Te, const ScalarField& ne,
                           const ScalarField& f_imp, ScalarField& P_rad);

    /// Compute thermal diffusion
    void compute_diffusion(const ScalarField& T, ScalarField& dTdt,
                           double chi_eff) const;

    /// Evolve impurity influx
    void evolve_impurities(double dt);
};

} // namespace tokamak
