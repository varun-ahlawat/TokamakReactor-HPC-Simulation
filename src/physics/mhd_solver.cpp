/// @file mhd_solver.cpp
/// @brief 2D Resistive MHD solver implementation.
///
/// Solves resistive MHD equations on a 2D (R,Z) grid using finite differences.
/// Implements a simplified Grad-Shafranov equilibrium and tearing mode instability.

#include "physics/mhd_solver.h"
#include "utils/physical_constants.h"
#include <cmath>
#include <algorithm>
#include <omp.h>

namespace tokamak {

void MHDState::initialize(const Grid& grid) {
    density     = ScalarField(grid, "density", 1.0);
    pressure    = ScalarField(grid, "pressure", 1.0);
    velocity    = VectorField(grid, "velocity");
    Bpol_psi    = ScalarField(grid, "Bpol_psi", 0.0);
    Btor        = ScalarField(grid, "Btor", 0.0);
    Jtor        = ScalarField(grid, "Jtor", 0.0);
    temperature = ScalarField(grid, "temperature", 0.0);
}

MHDSolver::MHDSolver(Grid& grid, const MHDConfig& config)
    : grid_(grid), config_(config) {
    state_.initialize(grid);
}

void MHDSolver::initialize_equilibrium() {
    double R0 = config_.R0;
    double a  = config_.a;
    double B0 = config_.B0;
    double q0 = config_.q0;
    double q_edge = config_.q_edge;
    double kappa = config_.elongation;
    double delta = config_.triangularity;

    // Simplified Grad-Shafranov equilibrium with D-shaped (Miller) geometry:
    //   R(ρ,θ) = R₀ + ρ·a·cos(θ + δ·sin(θ))
    //   Z(ρ,θ) = ρ·κ·a·sin(θ)
    // where ρ is the normalized minor radius, θ is the poloidal angle,
    // κ is elongation and δ is triangularity.
    //
    // Safety factor profile: q(ρ) = q₀ + (q_edge - q₀) × ρ²
    // Profiles use ρ from the shaped geometry for proper flux surface mapping.

    state_.density.fill([&](double R, double Z) -> double {
        double rho_norm = grid_.normalized_radius(R, Z, R0, a, kappa, delta);
        if (rho_norm > 1.0) return 0.1; // Low density outside plasma
        // Parabolic density profile: n(ρ) = n₀ × (1 - ρ²)^0.5
        return 1.0 * std::sqrt(std::max(0.0, 1.0 - rho_norm * rho_norm)) + 0.1;
    });

    state_.pressure.fill([&](double R, double Z) -> double {
        double rho_norm = grid_.normalized_radius(R, Z, R0, a, kappa, delta);
        if (rho_norm > 1.0) return 0.01;
        // Pressure profile: p(ρ) = p₀ × (1 - ρ²)²
        double x = 1.0 - rho_norm * rho_norm;
        return 1.0 * x * x + 0.01;
    });

    state_.Btor.fill([&](double R, double /*Z*/) -> double {
        // Toroidal field: B_φ = B₀ × R₀ / R (1/R dependence, shape-independent)
        return B0 * R0 / R;
    });

    state_.Bpol_psi.fill([&](double R, double Z) -> double {
        double rho_norm = grid_.normalized_radius(R, Z, R0, a, kappa, delta);
        if (rho_norm > 1.0) rho_norm = 1.0;
        // Poloidal flux function (simplified shaped equilibrium):
        // ψ(ρ) = ψ₀ × (1 - (1 - ρ²)²) with q-profile dependence
        // The flux is a function of ρ only (flux surfaces are labeled by ρ).
        double q_r = q0 + (q_edge - q0) * rho_norm * rho_norm;
        double psi = B0 * a * a / (2.0 * R0 * q_r) *
                     (1.0 - std::pow(1.0 - rho_norm * rho_norm, 2));
        return psi;
    });

    update_derived();
    apply_boundary_conditions();
}

void MHDSolver::apply_perturbation() {
    double R0 = config_.R0;
    double a  = config_.a;
    double kappa = config_.elongation;
    double delta = config_.triangularity;
    int m = config_.perturbation_m;
    double amp = config_.perturbation_amplitude;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < grid_.nr(); ++i) {
        for (int j = 0; j < grid_.nz(); ++j) {
            double R = grid_.R(i);
            double Z = grid_.Z(j);
            double rho_norm = grid_.normalized_radius(R, Z, R0, a, kappa, delta);
            if (rho_norm > 1.0) continue;

            double theta = std::atan2(Z, R - R0);

            // Perturbation localized near rational surface (q=m/n)
            double r_s = a * std::sqrt((config_.q0 - 1.0) /
                         (config_.q_edge - config_.q0 + 1e-10));
            r_s = std::min(r_s, 0.9 * a);
            double r_equiv = rho_norm * a; // Equivalent circular radius
            double width = 0.1 * a;
            double radial_env = std::exp(-(r_equiv - r_s) * (r_equiv - r_s) /
                                         (2.0 * width * width));

            state_.Bpol_psi(i, j) += amp * radial_env * std::cos(m * theta);
        }
    }
}

double MHDSolver::advance(double dt) {
    int nr = grid_.nr();
    int nz = grid_.nz();

    // Store previous state for computing RHS
    MHDState rhs;
    rhs.initialize(grid_);
    compute_rhs(state_, rhs);

    // Forward Euler update (for simplicity; RK4 used via coupler for accuracy)
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < nr - 1; ++i) {
        for (int j = 1; j < nz - 1; ++j) {
            state_.density(i, j)  += dt * rhs.density(i, j);
            state_.pressure(i, j) += dt * rhs.pressure(i, j);
            state_.Bpol_psi(i, j) += dt * rhs.Bpol_psi(i, j);
            state_.velocity.R_comp()(i, j) += dt * rhs.velocity.R_comp()(i, j);
            state_.velocity.Z_comp()(i, j) += dt * rhs.velocity.Z_comp()(i, j);

            // Enforce positivity
            if (state_.density(i, j) < 0.01) state_.density(i, j) = 0.01;
            if (state_.pressure(i, j) < 0.001) state_.pressure(i, j) = 0.001;
        }
    }

    apply_boundary_conditions();
    update_derived();

    // Compute island width for disruption detection
    // Simplified: measure max perturbation in ψ at rational surface
    double psi_max_pert = 0.0;
    double R0 = config_.R0;
    double a = config_.a;
    #pragma omp parallel for collapse(2) reduction(max:psi_max_pert)
    for (int i = 1; i < nr - 1; ++i) {
        for (int j = 1; j < nz - 1; ++j) {
            double R = grid_.R(i);
            double Z = grid_.Z(j);
            double r = std::sqrt((R - R0) * (R - R0) + Z * Z);
            if (r > 0.3 * a && r < 0.7 * a) {
                double pert = std::abs(state_.Bpol_psi(i, j));
                if (pert > psi_max_pert) psi_max_pert = pert;
            }
        }
    }
    // Island width ~ sqrt(ψ_pert) in normalized units
    island_width_ = 4.0 * std::sqrt(std::max(0.0, psi_max_pert) /
                    (config_.B0 / (config_.R0 * config_.q0)));

    // CFL condition: dt < dx / v_alfven
    double v_max = max_alfven_speed();
    double dx = std::min(grid_.dr(), grid_.dz());
    return 0.5 * dx / (v_max + 1.0e-30);
}

void MHDSolver::compute_rhs(const MHDState& s, MHDState& rhs) {
    int nr = grid_.nr();
    int nz = grid_.nz();
    double dr = grid_.dr();
    double dz = grid_.dz();
    double eta = config_.resistivity;
    double gamma = config_.gamma;
    double mu = config_.viscosity;

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < nr - 1; ++i) {
        for (int j = 1; j < nz - 1; ++j) {
            double R = grid_.R(i);
            double vR = s.velocity.R_comp()(i, j);
            double vZ = s.velocity.Z_comp()(i, j);
            double rho = s.density(i, j);
            double p = s.pressure(i, j);
            double psi = s.Bpol_psi(i, j);

            // Finite difference derivatives
            double dpsi_dR = (s.Bpol_psi(i+1, j) - s.Bpol_psi(i-1, j)) / (2.0 * dr);
            double dpsi_dZ = (s.Bpol_psi(i, j+1) - s.Bpol_psi(i, j-1)) / (2.0 * dz);

            double drho_dR = (s.density(i+1, j) - s.density(i-1, j)) / (2.0 * dr);
            double drho_dZ = (s.density(i, j+1) - s.density(i, j-1)) / (2.0 * dz);

            double dp_dR = (s.pressure(i+1, j) - s.pressure(i-1, j)) / (2.0 * dr);
            double dp_dZ = (s.pressure(i, j+1) - s.pressure(i, j-1)) / (2.0 * dz);

            double dvR_dR = (s.velocity.R_comp()(i+1, j) - s.velocity.R_comp()(i-1, j)) / (2.0 * dr);
            double dvZ_dZ = (s.velocity.Z_comp()(i, j+1) - s.velocity.Z_comp()(i, j-1)) / (2.0 * dz);
            double dvR_dZ = (s.velocity.R_comp()(i, j+1) - s.velocity.R_comp()(i, j-1)) / (2.0 * dz);
            double dvZ_dR = (s.velocity.Z_comp()(i+1, j) - s.velocity.Z_comp()(i-1, j)) / (2.0 * dr);

            // Laplacians
            double lap_psi = (s.Bpol_psi(i+1, j) - 2.0 * psi + s.Bpol_psi(i-1, j)) / (dr * dr)
                           + (s.Bpol_psi(i, j+1) - 2.0 * psi + s.Bpol_psi(i, j-1)) / (dz * dz)
                           + dpsi_dR / R; // Cylindrical correction

            double lap_vR = (s.velocity.R_comp()(i+1, j) - 2.0 * vR + s.velocity.R_comp()(i-1, j)) / (dr * dr)
                          + (s.velocity.R_comp()(i, j+1) - 2.0 * vR + s.velocity.R_comp()(i, j-1)) / (dz * dz);

            double lap_vZ = (s.velocity.Z_comp()(i+1, j) - 2.0 * vZ + s.velocity.Z_comp()(i-1, j)) / (dr * dr)
                          + (s.velocity.Z_comp()(i, j+1) - 2.0 * vZ + s.velocity.Z_comp()(i, j-1)) / (dz * dz);

            // Current density: J_tor = -Δ*ψ / (μ₀ R)
            double Jtor = -lap_psi / (constants::mu_0 * R);

            // B_R = -(1/R) ∂ψ/∂Z, B_Z = (1/R) ∂ψ/∂R
            double B_R = -dpsi_dZ / R;
            double B_Z = dpsi_dR / R;

            // Continuity: ∂ρ/∂t = -∇·(ρv) = -(vR ∂ρ/∂R + vZ ∂ρ/∂Z + ρ(∂vR/∂R + ∂vZ/∂Z + vR/R))
            rhs.density(i, j) = -(vR * drho_dR + vZ * drho_dZ +
                                   rho * (dvR_dR + dvZ_dZ + vR / R));

            // Momentum R: ρ ∂vR/∂t = -ρ(v·∇)vR + J×B_R - ∂p/∂R + μ∇²vR
            double JxB_R = Jtor * s.Btor(i, j) * constants::mu_0; // Simplified
            rhs.velocity.R_comp()(i, j) = (-vR * dvR_dR - vZ * dvR_dZ
                                            + JxB_R / rho - dp_dR / rho
                                            + mu * lap_vR);

            // Momentum Z: ρ ∂vZ/∂t = -ρ(v·∇)vZ + J×B_Z - ∂p/∂Z + μ∇²vZ
            rhs.velocity.Z_comp()(i, j) = (-vR * dvZ_dR - vZ * dvZ_dZ
                                            - dp_dZ / rho
                                            + mu * lap_vZ);

            // Induction: ∂ψ/∂t = R(v×B)_φ + η∇²ψ
            //                   = R(vR * B_Z - vZ * B_R) + η * lap_psi
            // Simplified: use -v·∇ψ + η∇²ψ (ideal + resistive terms)
            rhs.Bpol_psi(i, j) = -(vR * dpsi_dR + vZ * dpsi_dZ) + eta * lap_psi;

            // Pressure: ∂p/∂t = -v·∇p - γp∇·v + (γ-1)ηJ²
            double div_v = dvR_dR + dvZ_dZ + vR / R;
            rhs.pressure(i, j) = -(vR * dp_dR + vZ * dp_dZ)
                                  - gamma * p * div_v
                                  + (gamma - 1.0) * eta * Jtor * Jtor;
        }
    }
}

void MHDSolver::apply_boundary_conditions() {
    int nr = grid_.nr();
    int nz = grid_.nz();

    // Dirichlet BC on velocity (no-slip at walls)
    // Neumann BC on ψ and p at boundaries
    for (int j = 0; j < nz; ++j) {
        state_.velocity.R_comp()(0, j) = 0.0;
        state_.velocity.Z_comp()(0, j) = 0.0;
        state_.velocity.R_comp()(nr-1, j) = 0.0;
        state_.velocity.Z_comp()(nr-1, j) = 0.0;

        // Neumann: copy from interior
        state_.Bpol_psi(0, j) = state_.Bpol_psi(1, j);
        state_.Bpol_psi(nr-1, j) = state_.Bpol_psi(nr-2, j);
        state_.pressure(0, j) = state_.pressure(1, j);
        state_.pressure(nr-1, j) = state_.pressure(nr-2, j);
        state_.density(0, j) = state_.density(1, j);
        state_.density(nr-1, j) = state_.density(nr-2, j);
    }

    for (int i = 0; i < nr; ++i) {
        state_.velocity.R_comp()(i, 0) = 0.0;
        state_.velocity.Z_comp()(i, 0) = 0.0;
        state_.velocity.R_comp()(i, nz-1) = 0.0;
        state_.velocity.Z_comp()(i, nz-1) = 0.0;

        state_.Bpol_psi(i, 0) = state_.Bpol_psi(i, 1);
        state_.Bpol_psi(i, nz-1) = state_.Bpol_psi(i, nz-2);
        state_.pressure(i, 0) = state_.pressure(i, 1);
        state_.pressure(i, nz-1) = state_.pressure(i, nz-2);
        state_.density(i, 0) = state_.density(i, 1);
        state_.density(i, nz-1) = state_.density(i, nz-2);
    }
}

void MHDSolver::update_derived() {
    compute_current_density();

    // Temperature from p = n * k_B * T (in normalized units: T = p / ρ)
    // Converting to keV
    double T_scale = constants::default_T_e; // Scale factor
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < grid_.nr(); ++i) {
        for (int j = 0; j < grid_.nz(); ++j) {
            double rho = state_.density(i, j);
            if (rho < 0.01) rho = 0.01;
            state_.temperature(i, j) = T_scale * state_.pressure(i, j) / rho;
        }
    }
}

void MHDSolver::laplacian(const ScalarField& f, ScalarField& lap) const {
    double dr = grid_.dr();
    double dz = grid_.dz();

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < grid_.nr() - 1; ++i) {
        for (int j = 1; j < grid_.nz() - 1; ++j) {
            double R = grid_.R(i);
            double d2f_dR2 = (f(i+1, j) - 2.0 * f(i, j) + f(i-1, j)) / (dr * dr);
            double df_dR = (f(i+1, j) - f(i-1, j)) / (2.0 * dr);
            double d2f_dZ2 = (f(i, j+1) - 2.0 * f(i, j) + f(i, j-1)) / (dz * dz);
            // Cylindrical Laplacian: ∇²f = d²f/dR² + (1/R)df/dR + d²f/dZ²
            lap(i, j) = d2f_dR2 + df_dR / R + d2f_dZ2;
        }
    }
}

void MHDSolver::compute_current_density() {
    ScalarField lap_psi(grid_, "lap_psi");
    laplacian(state_.Bpol_psi, lap_psi);

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < grid_.nr() - 1; ++i) {
        for (int j = 1; j < grid_.nz() - 1; ++j) {
            double R = grid_.R(i);
            // J_tor = -Δ*ψ / (μ₀ R)  (Grad-Shafranov operator)
            state_.Jtor(i, j) = -lap_psi(i, j) / (constants::mu_0 * R);
        }
    }
}

double MHDSolver::max_alfven_speed() const {
    double v_max = 0.0;
    #pragma omp parallel for collapse(2) reduction(max:v_max)
    for (int i = 0; i < grid_.nr(); ++i) {
        for (int j = 0; j < grid_.nz(); ++j) {
            double rho = state_.density(i, j);
            if (rho < 0.01) rho = 0.01;
            double B2 = state_.Btor(i, j) * state_.Btor(i, j);
            // v_A = B / sqrt(μ₀ ρ)
            double v_A = std::sqrt(B2 / (constants::mu_0 * rho));
            if (v_A > v_max) v_max = v_A;
        }
    }
    return v_max;
}

double MHDSolver::magnetic_energy() const {
    double energy = 0.0;
    double dr = grid_.dr();
    double dz = grid_.dz();

    #pragma omp parallel for collapse(2) reduction(+:energy)
    for (int i = 0; i < grid_.nr(); ++i) {
        for (int j = 0; j < grid_.nz(); ++j) {
            double R = grid_.R(i);
            double B2 = state_.Btor(i, j) * state_.Btor(i, j);
            // E_mag = ∫ B²/(2μ₀) dV, dV = R dR dZ dφ (integrate over 2π)
            energy += B2 / (2.0 * constants::mu_0) * R * dr * dz * 2.0 * M_PI;
        }
    }
    return energy;
}

double MHDSolver::kinetic_energy() const {
    double energy = 0.0;
    double dr = grid_.dr();
    double dz = grid_.dz();

    #pragma omp parallel for collapse(2) reduction(+:energy)
    for (int i = 0; i < grid_.nr(); ++i) {
        for (int j = 0; j < grid_.nz(); ++j) {
            double R = grid_.R(i);
            double rho = state_.density(i, j);
            double vR = state_.velocity.R_comp()(i, j);
            double vZ = state_.velocity.Z_comp()(i, j);
            double v2 = vR * vR + vZ * vZ;
            energy += 0.5 * rho * v2 * R * dr * dz * 2.0 * M_PI;
        }
    }
    return energy;
}

bool MHDSolver::disruption_triggered() const {
    return island_width_ > config_.a * 0.1; // Island > 10% of minor radius
}

} // namespace tokamak
