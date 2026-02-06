#pragma once
/// @file mhd_solver.h
/// @brief 2D Resistive Magnetohydrodynamics (MHD) solver.
///
/// Solves the resistive MHD equations in the tokamak poloidal cross-section.
/// This is Phase 1 of the disruption cascade: the initial instability trigger
/// (tearing modes) that operates on microsecond timescales.
///
/// The equations solved (in 2D cylindrical (R,Z) coordinates):
///   ∂ρ/∂t + ∇·(ρv) = 0                     (mass conservation)
///   ρ(∂v/∂t + v·∇v) = J×B - ∇p + μ∇²v      (momentum)
///   ∂B/∂t = ∇×(v×B) + η∇²B                 (induction with resistivity)
///   ∂p/∂t + v·∇p + γp∇·v = (γ-1)ηJ²        (energy/pressure)
///   J = (1/μ₀)∇×B                           (current density)

#include "core/grid.h"
#include "core/field.h"
#include "core/time_integrator.h"

namespace tokamak {

/// Configuration for the MHD solver
struct MHDConfig {
    double gamma = 5.0 / 3.0;     // Adiabatic index
    double resistivity = 1.0e-6;  // Magnetic resistivity η [Ohm·m]
    double viscosity = 1.0e-3;    // Kinematic viscosity [m²/s]

    // Initial equilibrium parameters
    double B0 = 5.3;              // Toroidal field strength [T]
    double R0 = 6.2;              // Major radius [m]
    double a = 2.0;               // Minor radius [m]
    double elongation = 1.7;      // Plasma elongation κ (ITER: ~1.7)
    double triangularity = 0.33;  // Plasma triangularity δ (ITER: ~0.33)
    double q0 = 1.0;              // Safety factor on axis
    double q_edge = 3.0;          // Safety factor at edge

    // Perturbation for triggering instability
    double perturbation_amplitude = 1.0e-4;
    int perturbation_m = 2;       // Poloidal mode number
    int perturbation_n = 1;       // Toroidal mode number
};

/// MHD state variables at each grid point
struct MHDState {
    ScalarField density;     // ρ [kg/m³]
    ScalarField pressure;    // p [Pa]
    VectorField velocity;    // v [m/s]
    ScalarField Bpol_psi;    // Poloidal flux function ψ
    ScalarField Btor;        // Toroidal field Bφ [T]
    ScalarField Jtor;        // Toroidal current density Jφ [A/m²]
    ScalarField temperature; // T [keV] (derived from p, ρ)

    void initialize(const Grid& grid);
};

/// 2D Resistive MHD Solver
class MHDSolver {
public:
    explicit MHDSolver(Grid& grid, const MHDConfig& config = MHDConfig());

    /// Initialize equilibrium state (simplified Grad-Shafranov solution)
    void initialize_equilibrium();

    /// Apply perturbation to trigger instability
    void apply_perturbation();

    /// Compute one timestep
    /// @return Suggested next timestep based on CFL condition
    double advance(double dt);

    /// Get the current MHD state
    const MHDState& state() const { return state_; }
    MHDState& state() { return state_; }

    /// Compute maximum Alfvén speed (for CFL condition)
    double max_alfven_speed() const;

    /// Compute total magnetic energy
    double magnetic_energy() const;

    /// Compute total kinetic energy
    double kinetic_energy() const;

    /// Check if disruption has been triggered (based on island width)
    bool disruption_triggered() const;

    /// Get the current magnetic island width
    double island_width() const { return island_width_; }

    /// Get configuration
    const MHDConfig& config() const { return config_; }

private:
    Grid& grid_;
    MHDConfig config_;
    MHDState state_;
    double island_width_ = 0.0;

    /// Compute RHS of the MHD equations
    void compute_rhs(const MHDState& state, MHDState& rhs);

    /// Apply boundary conditions
    void apply_boundary_conditions();

    /// Update derived quantities (J, T from p and ρ)
    void update_derived();

    /// Compute Laplacian of a scalar field
    void laplacian(const ScalarField& f, ScalarField& lap) const;

    /// Compute current density from poloidal flux
    void compute_current_density();
};

} // namespace tokamak
