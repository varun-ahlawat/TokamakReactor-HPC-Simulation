/// @file time_integrator.cpp
#include "core/time_integrator.h"
#include <algorithm>
#include <cmath>

namespace tokamak {

// --- RK4 Integrator ---

void RK4Integrator::step(std::vector<double>& u, const RHSFunction& rhs,
                          double t, double dt) {
    size_t n = u.size();
    std::vector<double> k1(n), k2(n), k3(n), k4(n), tmp(n);

    // k1 = f(t, u)
    rhs(u, k1, t);

    // k2 = f(t + dt/2, u + dt/2 * k1)
    for (size_t i = 0; i < n; ++i) tmp[i] = u[i] + 0.5 * dt * k1[i];
    rhs(tmp, k2, t + 0.5 * dt);

    // k3 = f(t + dt/2, u + dt/2 * k2)
    for (size_t i = 0; i < n; ++i) tmp[i] = u[i] + 0.5 * dt * k2[i];
    rhs(tmp, k3, t + 0.5 * dt);

    // k4 = f(t + dt, u + dt * k3)
    for (size_t i = 0; i < n; ++i) tmp[i] = u[i] + dt * k3[i];
    rhs(tmp, k4, t + dt);

    // u += dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    for (size_t i = 0; i < n; ++i) {
        u[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
}

// --- RK2 Integrator ---

void RK2Integrator::step(std::vector<double>& u, const RHSFunction& rhs,
                          double t, double dt) {
    size_t n = u.size();
    std::vector<double> k1(n), k2(n), tmp(n);

    // k1 = f(t, u)
    rhs(u, k1, t);

    // k2 = f(t + dt, u + dt * k1)
    for (size_t i = 0; i < n; ++i) tmp[i] = u[i] + dt * k1[i];
    rhs(tmp, k2, t + dt);

    // u += dt/2 * (k1 + k2)
    for (size_t i = 0; i < n; ++i) {
        u[i] += 0.5 * dt * (k1[i] + k2[i]);
    }
}

// --- Euler Integrator ---

void EulerIntegrator::step(std::vector<double>& u, const RHSFunction& rhs,
                            double t, double dt) {
    size_t n = u.size();
    std::vector<double> dudt(n);

    rhs(u, dudt, t);

    for (size_t i = 0; i < n; ++i) {
        u[i] += dt * dudt[i];
    }
}

// --- Adaptive Time Stepper ---

AdaptiveTimeStepper::AdaptiveTimeStepper(double dt_min, double dt_max, double safety)
    : dt_min_(dt_min), dt_max_(dt_max), safety_(safety) {}

double AdaptiveTimeStepper::compute_dt(double dx, double v_max) const {
    if (v_max < 1.0e-30) return dt_max_;
    double dt = safety_ * dx / v_max;
    return std::clamp(dt, dt_min_, dt_max_);
}

double AdaptiveTimeStepper::adjust_dt(double dt_current, double error,
                                       double tolerance) const {
    if (error < 1.0e-30) return std::min(2.0 * dt_current, dt_max_);

    double factor = 0.9 * std::pow(tolerance / error, 0.2);
    factor = std::clamp(factor, 0.1, 2.0);
    return std::clamp(factor * dt_current, dt_min_, dt_max_);
}

} // namespace tokamak
