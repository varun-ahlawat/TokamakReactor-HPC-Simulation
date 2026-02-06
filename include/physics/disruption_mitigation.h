#pragma once
/// @file disruption_mitigation.h
/// @brief Disruption Mitigation System (DMS) model.
///
/// Models the emergency response to a disruption: firing shattered pellets
/// of frozen neon or argon into the plasma core. The massive material injection
/// rapidly increases density and collisionality, creating intense frictional
/// drag that stops runaway electrons.
///
/// Physics modeled:
///   - Pellet ablation (NGS model)
///   - Material deposition profile
///   - Rapid density increase and collisionality enhancement
///   - RE suppression through enhanced friction

#include "core/grid.h"
#include "core/field.h"

namespace tokamak {

/// Configuration for the Disruption Mitigation System
struct DMSConfig {
    // Pellet parameters
    double pellet_radius = 0.005;      // Pellet fragment radius [m]
    int num_fragments = 28;            // Number of shattered fragments
    double pellet_velocity = 200.0;    // Fragment velocity [m/s]
    double pellet_Z = 10.0;           // Atomic number (neon=10, argon=18)
    double pellet_A = 20.0;           // Mass number
    double pellet_n_solid = 4.0e28;   // Solid neon density [m⁻³]

    // Injection geometry
    double injection_R = 8.0;         // Injection R position [m]
    double injection_Z = 0.0;         // Injection Z position [m]
    double injection_angle = -45.0;   // Injection angle [degrees]

    // Timing
    double trigger_delay = 0.001;     // Delay after disruption detection [s]
    double assimilation_time = 0.001; // Time for full assimilation [s]
};

/// State of the DMS
struct DMSState {
    ScalarField deposited_density;    // Deposited impurity density [m⁻³]
    ScalarField collisionality;       // Enhanced collisionality factor
    bool is_triggered = false;
    bool is_complete = false;
    double trigger_time = 0.0;
    double deposited_fraction = 0.0;  // Fraction of pellet deposited [0-1]

    void initialize(const Grid& grid);
};

/// Disruption Mitigation System simulator
class DisruptionMitigation {
public:
    explicit DisruptionMitigation(Grid& grid,
                                   const DMSConfig& config = DMSConfig());

    /// Trigger the DMS
    void trigger(double current_time);

    /// Advance the DMS model by one timestep
    /// @param Te Current electron temperature [keV]
    /// @param ne Current electron density [m⁻³]
    void advance(double dt, double current_time,
                 const ScalarField& Te, ScalarField& ne);

    /// Get current state
    const DMSState& state() const { return state_; }
    DMSState& state() { return state_; }

    /// Check if DMS has been triggered
    bool is_triggered() const { return state_.is_triggered; }

    /// Check if material deposition is complete
    bool is_complete() const { return state_.is_complete; }

    /// Get total deposited particles
    double total_deposited() const;

private:
    Grid& grid_;
    DMSConfig config_;
    DMSState state_;

    /// Compute pellet ablation rate (Neutral Gas Shielding model)
    double ablation_rate(double Te_keV, double ne, double r_pellet) const;

    /// Compute deposition profile
    void compute_deposition_profile(const ScalarField& Te, double dt);
};

} // namespace tokamak
