/// @file grid.cpp
#include "core/grid.h"
#include <cmath>
#include <stdexcept>

namespace tokamak {

Grid::Grid(int nr, int nz, double R_min, double R_max, double Z_min, double Z_max)
    : nr_(nr), nz_(nz), R_min_(R_min), R_max_(R_max), Z_min_(Z_min), Z_max_(Z_max)
{
    if (nr < 2 || nz < 2)
        throw std::invalid_argument("Grid dimensions must be >= 2");
    if (R_max <= R_min || Z_max <= Z_min)
        throw std::invalid_argument("Invalid grid bounds");

    dr_ = (R_max_ - R_min_) / (nr_ - 1);
    dz_ = (Z_max_ - Z_min_) / (nz_ - 1);

    // Default: single-process decomposition
    decomp_.rank = 0;
    decomp_.num_procs = 1;
    decomp_.i_start = 0;
    decomp_.i_end = nr_;
    decomp_.j_start = 0;
    decomp_.j_end = nz_;
    decomp_.local_nr = nr_;
    decomp_.local_nz = nz_;
}

void Grid::setup_decomposition(MPI_Comm comm) {
    decomp_.comm = comm;
    MPI_Comm_rank(comm, &decomp_.rank);
    MPI_Comm_size(comm, &decomp_.num_procs);

    // 1D domain decomposition along R direction
    int base = nr_ / decomp_.num_procs;
    int remainder = nr_ % decomp_.num_procs;

    decomp_.i_start = decomp_.rank * base + std::min(decomp_.rank, remainder);
    decomp_.local_nr = base + (decomp_.rank < remainder ? 1 : 0);
    decomp_.i_end = decomp_.i_start + decomp_.local_nr;

    decomp_.j_start = 0;
    decomp_.j_end = nz_;
    decomp_.local_nz = nz_;
}

bool Grid::is_inside_plasma(double R_pos, double Z_pos, double R0, double a,
                             double elongation, double triangularity) const {
    return normalized_radius(R_pos, Z_pos, R0, a, elongation, triangularity) <= 1.0;
}

double Grid::normalized_radius(double R_pos, double Z_pos, double R0, double a,
                                double elongation, double triangularity) const {
    // Miller parameterization of a D-shaped tokamak cross-section:
    //   R(ρ,θ) = R₀ + ρ·a·cos(θ + δ·sin(θ))
    //   Z(ρ,θ) = ρ·κ·a·sin(θ)
    //
    // Given a point (R,Z), find the normalized radius ρ by minimizing
    // the distance to the parametric curve over θ at each trial ρ.
    // This is more robust than analytical inversion, especially on the
    // inboard side where the D-shape creates non-trivial topology.

    double kappa = elongation;
    double delta = triangularity;

    // Special case: at the magnetic axis
    double dR = R_pos - R0;
    double dZ = Z_pos;
    if (dR * dR + dZ * dZ < 1e-20) return 0.0;

    // Step 1: Find the poloidal angle θ that best matches this (R,Z)
    // by scanning and then refining.
    // For a given θ, the point lies at:
    //   ρ = Z / (κ·a·sin(θ))   [from Z equation, when sin(θ) ≠ 0]
    //   ρ = (R - R₀) / (a·cos(θ + δ·sin(θ)))   [from R equation]
    // We find θ where both give the same ρ.

    double best_rho = 1e10;
    double best_err = 1e30;

    // Coarse scan over θ
    int nscan = 64;
    for (int k = 0; k < nscan; ++k) {
        double theta = 2.0 * M_PI * k / nscan;
        double sin_t = std::sin(theta);
        double cos_shift = std::cos(theta + delta * sin_t);

        // Compute ρ from R equation
        double rho_r = (std::abs(cos_shift) > 1e-10) ? dR / (a * cos_shift) : -1.0;
        // Compute ρ from Z equation
        double rho_z = (std::abs(sin_t) > 1e-10) ? dZ / (kappa * a * sin_t) : -1.0;

        // If both are valid, check consistency
        double rho_candidate, err;
        if (rho_r >= 0 && rho_z >= 0) {
            // Both valid: use weighted average and measure error
            rho_candidate = (rho_r + rho_z) / 2.0;
            // Error = residual distance
            double R_fit = R0 + rho_candidate * a * cos_shift;
            double Z_fit = rho_candidate * kappa * a * sin_t;
            err = (R_pos - R_fit) * (R_pos - R_fit) + (Z_pos - Z_fit) * (Z_pos - Z_fit);
        } else if (rho_r >= 0) {
            rho_candidate = rho_r;
            double R_fit = R0 + rho_candidate * a * cos_shift;
            double Z_fit = rho_candidate * kappa * a * sin_t;
            err = (R_pos - R_fit) * (R_pos - R_fit) + (Z_pos - Z_fit) * (Z_pos - Z_fit);
        } else if (rho_z >= 0) {
            rho_candidate = rho_z;
            double R_fit = R0 + rho_candidate * a * cos_shift;
            double Z_fit = rho_candidate * kappa * a * sin_t;
            err = (R_pos - R_fit) * (R_pos - R_fit) + (Z_pos - Z_fit) * (Z_pos - Z_fit);
        } else {
            continue;
        }

        if (err < best_err) {
            best_err = err;
            best_rho = rho_candidate;
        }
    }

    // Step 2: Refine with a finer local scan around the best θ
    // Find the best theta index from coarse scan
    double best_theta = 0.0;
    best_err = 1e30;
    for (int k = 0; k < nscan; ++k) {
        double theta = 2.0 * M_PI * k / nscan;
        double sin_t = std::sin(theta);
        double cos_shift = std::cos(theta + delta * sin_t);
        double rho_r = (std::abs(cos_shift) > 1e-10) ? dR / (a * cos_shift) : -1.0;
        double rho_z = (std::abs(sin_t) > 1e-10) ? dZ / (kappa * a * sin_t) : -1.0;
        double rho_c = (rho_r >= 0 && rho_z >= 0) ? (rho_r + rho_z) / 2.0 :
                        (rho_r >= 0) ? rho_r : (rho_z >= 0) ? rho_z : -1.0;
        if (rho_c < 0) continue;
        double R_fit = R0 + rho_c * a * cos_shift;
        double Z_fit = rho_c * kappa * a * sin_t;
        double err = (R_pos - R_fit) * (R_pos - R_fit) + (Z_pos - Z_fit) * (Z_pos - Z_fit);
        if (err < best_err) {
            best_err = err;
            best_theta = theta;
            best_rho = rho_c;
        }
    }

    // Fine scan around best_theta
    double dtheta = 2.0 * M_PI / nscan;
    for (int refine = 0; refine < 3; ++refine) {
        double theta_lo = best_theta - dtheta;
        double theta_hi = best_theta + dtheta;
        int nfine = 32;
        double step = (theta_hi - theta_lo) / nfine;
        for (int k = 0; k <= nfine; ++k) {
            double theta = theta_lo + k * step;
            double sin_t = std::sin(theta);
            double cos_shift = std::cos(theta + delta * sin_t);
            double rho_r = (std::abs(cos_shift) > 1e-10) ? dR / (a * cos_shift) : -1.0;
            double rho_z = (std::abs(sin_t) > 1e-10) ? dZ / (kappa * a * sin_t) : -1.0;
            double rho_c = (rho_r >= 0 && rho_z >= 0) ? (rho_r + rho_z) / 2.0 :
                            (rho_r >= 0) ? rho_r : (rho_z >= 0) ? rho_z : -1.0;
            if (rho_c < 0) continue;
            double R_fit = R0 + rho_c * a * cos_shift;
            double Z_fit = rho_c * kappa * a * sin_t;
            double err = (R_pos - R_fit) * (R_pos - R_fit) + (Z_pos - Z_fit) * (Z_pos - Z_fit);
            if (err < best_err) {
                best_err = err;
                best_theta = theta;
                best_rho = rho_c;
            }
        }
        dtheta = step;
    }

    // Verify the solution: check residual error
    // If the best fit has significant error, the point may be in a region
    // not well-represented by the Miller parameterization (e.g., above/below
    // the elongated plasma). In that case, compute ρ from distance to boundary.
    double R_fit = R0 + best_rho * a * std::cos(best_theta + delta * std::sin(best_theta));
    double Z_fit = best_rho * kappa * a * std::sin(best_theta);
    double residual = std::sqrt((R_pos - R_fit) * (R_pos - R_fit) +
                                 (Z_pos - Z_fit) * (Z_pos - Z_fit));
    double scale = std::max(a, kappa * a);

    if (residual > 0.01 * scale) {
        // Point is not well-mapped by Miller parameterization.
        // Compute effective ρ as distance from axis divided by distance
        // from axis to the nearest boundary point.
        double dist_from_axis = std::sqrt(dR * dR + dZ * dZ);

        // Find nearest boundary point
        double min_dist_to_boundary = 1e30;
        double boundary_dist_from_axis = 1.0;
        int nb = 360;
        for (int k = 0; k < nb; ++k) {
            double theta = 2.0 * M_PI * k / nb;
            double Rb = R0 + a * std::cos(theta + delta * std::sin(theta));
            double Zb = kappa * a * std::sin(theta);
            double d = std::sqrt((R_pos - Rb) * (R_pos - Rb) +
                                  (Z_pos - Zb) * (Z_pos - Zb));
            if (d < min_dist_to_boundary) {
                min_dist_to_boundary = d;
                boundary_dist_from_axis = std::sqrt((Rb - R0) * (Rb - R0) +
                                                      Zb * Zb);
            }
        }

        // If closest boundary point is closer than axis, we're outside
        if (boundary_dist_from_axis > 1e-10) {
            best_rho = dist_from_axis / boundary_dist_from_axis;
        }
        // If point is farther from axis than the boundary in that direction,
        // it's outside (ρ > 1)
    }

    return std::max(0.0, best_rho);
}

} // namespace tokamak
