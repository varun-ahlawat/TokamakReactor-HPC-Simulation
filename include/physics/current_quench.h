#pragma once
/// @file current_quench.h
/// @brief Current quench model (Phase 3 of disruption cascade).
///
/// As the plasma cools during the thermal quench, its resistivity increases
/// dramatically (Spitzer resistivity ~ T^{-3/2}). The plasma current decays
/// on an L/R timescale, inducing a large toroidal electric field that can
/// accelerate electrons to relativistic energies.
///
/// Physics modeled:
///   - Ohmic current decay with temperature-dependent resistivity
///   - Induced electric field computation
///   - Transition from Spitzer to cold-plasma resistivity

#include "core/grid.h"
#include "core/field.h"

namespace tokamak {

/// Configuration for current quench model
struct CurrentQuenchConfig {
    double Z_eff = 1.7;            // Effective charge number
    double plasma_inductance = 10.0e-6; // Plasma self-inductance [H]
    double wall_resistance = 1.0e-6;    // Vessel wall resistance [Ohm]
    double wall_time_constant = 0.005;  // Wall L/R time constant [s]
    double initial_current = 15.0e6;    // Initial plasma current [A]
};

/// State variables for the current quench phase
struct CurrentQuenchState {
    ScalarField Jtor;             // Toroidal current density [A/m²]
    ScalarField E_field;          // Toroidal electric field [V/m]
    ScalarField resistivity;      // Local resistivity [Ohm·m]
    double total_current;         // Total plasma current [A]
    double total_E_field;         // Volume-averaged electric field [V/m]

    void initialize(const Grid& grid);
};

/// Current quench simulator
class CurrentQuench {
public:
    explicit CurrentQuench(Grid& grid,
                           const CurrentQuenchConfig& config = CurrentQuenchConfig());

    /// Initialize from thermal quench end state
    /// @param Te Final electron temperature from thermal quench [keV]
    /// @param ne Electron density [m⁻³]
    /// @param Jtor_initial Current density from MHD state [A/m²]
    void initialize_from_thermal_quench(const ScalarField& Te,
                                         const ScalarField& ne,
                                         const ScalarField& Jtor_initial);

    /// Advance the current quench by one timestep
    /// @return Suggested next timestep
    double advance(double dt, const ScalarField& Te, const ScalarField& ne);

    /// Get current state
    const CurrentQuenchState& state() const { return state_; }
    CurrentQuenchState& state() { return state_; }

    /// Check if current quench is complete
    bool is_complete(double I_threshold_fraction = 0.01) const;

    /// Get the ratio E/E_dreicer (determines runaway generation)
    double E_over_Dreicer() const { return E_over_Ed_; }

    /// Get current quench time [s]
    double quench_time() const { return quench_time_; }

private:
    Grid& grid_;
    CurrentQuenchConfig config_;
    CurrentQuenchState state_;
    double initial_current_;
    double E_over_Ed_ = 0.0;
    double quench_time_ = 0.0;
    double elapsed_time_ = 0.0;

    /// Compute resistivity from temperature
    void compute_resistivity(const ScalarField& Te, const ScalarField& ne);

    /// Compute induced electric field from current decay rate
    void compute_electric_field();
};

} // namespace tokamak
