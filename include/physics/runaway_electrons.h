#pragma once
/// @file runaway_electrons.h
/// @brief Runaway electron avalanche model (Phase 4 of disruption cascade).
///
/// When the electric field exceeds the critical (Dreicer) field, electrons
/// are continuously accelerated. The avalanche mechanism (knock-on collisions)
/// causes exponential growth of the runaway population.
///
/// Physics modeled:
///   - Dreicer (primary) generation rate
///   - Avalanche (secondary) multiplication
///   - Synchrotron radiation losses
///   - Simplified Fokker-Planck kinetics for the distribution function
///
/// Reference: arXiv:2403.04948v1 (physics-constrained surrogate model)

#include "core/grid.h"
#include "core/field.h"

namespace tokamak {

/// Configuration for runaway electron model
struct RunawayConfig {
    double E_critical_factor = 1.0;   // Multiplier on critical field
    double avalanche_coefficient = 1.0; // Rosenbluth-Putvinski coefficient
    double synchrotron_loss_factor = 1.0;
    double max_energy_MeV = 100.0;    // Maximum runaway energy [MeV]
    double seed_fraction = 1.0e-12;   // Initial seed RE fraction
    int num_energy_bins = 50;         // Energy grid resolution
};

/// State variables for runaway electron model
struct RunawayState {
    ScalarField n_RE;              // Runaway electron density [m⁻³]
    ScalarField RE_current;        // Runaway current density [A/m²]
    ScalarField growth_rate;       // Local avalanche growth rate [s⁻¹]
    ScalarField avg_energy;        // Average RE energy [MeV]
    double total_RE_current;       // Total runaway current [A]
    double total_RE_fraction;      // Fraction of current carried by REs

    void initialize(const Grid& grid);
};

/// Runaway electron simulator
class RunawayElectrons {
public:
    explicit RunawayElectrons(Grid& grid,
                              const RunawayConfig& config = RunawayConfig());

    /// Initialize from current quench state
    /// @param ne Electron density [m⁻³]
    /// @param Te Electron temperature [keV]
    /// @param E_field Electric field [V/m]
    void initialize_from_current_quench(const ScalarField& ne,
                                         const ScalarField& Te,
                                         const ScalarField& E_field);

    /// Advance the runaway electron model by one timestep
    /// @return Suggested next timestep
    double advance(double dt, const ScalarField& ne, const ScalarField& Te,
                   const ScalarField& E_field);

    /// Get current state
    const RunawayState& state() const { return state_; }
    RunawayState& state() { return state_; }

    /// Check if avalanche has saturated
    bool is_saturated() const;

    /// Compute Dreicer generation rate [m⁻³/s]
    static double dreicer_rate(double ne, double Te_keV, double E,
                                double E_c, double Z_eff);

    /// Compute avalanche growth rate [s⁻¹]
    /// Based on Rosenbluth-Putvinski formula
    static double avalanche_rate(double ne, double E, double E_c,
                                  double ln_lambda, double Z_eff);

private:
    Grid& grid_;
    RunawayConfig config_;
    RunawayState state_;

    /// Compute primary (Dreicer) generation
    void compute_dreicer_generation(const ScalarField& ne,
                                     const ScalarField& Te,
                                     const ScalarField& E_field,
                                     ScalarField& source);

    /// Compute avalanche multiplication
    void compute_avalanche(const ScalarField& ne,
                            const ScalarField& E_field,
                            ScalarField& growth);

    /// Compute synchrotron radiation losses
    void compute_synchrotron_losses(ScalarField& loss);
};

} // namespace tokamak
