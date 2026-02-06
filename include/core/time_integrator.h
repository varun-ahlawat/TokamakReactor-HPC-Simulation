#pragma once
/// @file time_integrator.h
/// @brief Time integration schemes for the multi-physics simulation.
///
/// Supports adaptive multi-rate time-stepping where different physics
/// domains can evolve at their natural timescales.

#include <functional>
#include <vector>
#include <string>

namespace tokamak {

/// Time integration state
struct TimeState {
    double time = 0.0;           // Current simulation time [s]
    double dt = 1.0e-6;         // Current timestep [s]
    int step = 0;                // Current step number
    double t_end = 0.0;         // End time [s]
    double cfl_number = 0.5;    // CFL safety factor
};

/// RHS function type: computes du/dt given current state
using RHSFunction = std::function<void(const std::vector<double>& u,
                                        std::vector<double>& dudt,
                                        double t)>;

/// Time integrator base class
class TimeIntegrator {
public:
    virtual ~TimeIntegrator() = default;

    /// Advance state u by one timestep dt
    /// @param u State vector (modified in place)
    /// @param rhs Right-hand side function
    /// @param t Current time
    /// @param dt Timestep
    virtual void step(std::vector<double>& u, const RHSFunction& rhs,
                      double t, double dt) = 0;

    virtual std::string name() const = 0;
};

/// 4th-order Runge-Kutta integrator
class RK4Integrator : public TimeIntegrator {
public:
    void step(std::vector<double>& u, const RHSFunction& rhs,
              double t, double dt) override;
    std::string name() const override { return "RK4"; }
};

/// 2nd-order Runge-Kutta (Heun's method) — faster for stiff problems
class RK2Integrator : public TimeIntegrator {
public:
    void step(std::vector<double>& u, const RHSFunction& rhs,
              double t, double dt) override;
    std::string name() const override { return "RK2"; }
};

/// Forward Euler — simplest, for testing
class EulerIntegrator : public TimeIntegrator {
public:
    void step(std::vector<double>& u, const RHSFunction& rhs,
              double t, double dt) override;
    std::string name() const override { return "Euler"; }
};

/// Adaptive time-stepping controller
class AdaptiveTimeStepper {
public:
    /// @param dt_min Minimum allowed timestep
    /// @param dt_max Maximum allowed timestep
    /// @param safety CFL safety factor
    AdaptiveTimeStepper(double dt_min, double dt_max, double safety = 0.5);

    /// Compute timestep from CFL condition
    /// @param dx Minimum grid spacing
    /// @param v_max Maximum wave/flow speed
    double compute_dt(double dx, double v_max) const;

    /// Adjust timestep based on error estimate
    double adjust_dt(double dt_current, double error, double tolerance) const;

    double dt_min() const { return dt_min_; }
    double dt_max() const { return dt_max_; }

private:
    double dt_min_, dt_max_, safety_;
};

} // namespace tokamak
