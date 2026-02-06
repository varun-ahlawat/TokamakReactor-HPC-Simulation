#pragma once
/// @file grid.h
/// @brief 2D computational grid for tokamak cross-section simulation.
///
/// Implements a structured grid in (R, Z) cylindrical coordinates
/// representing the poloidal cross-section of a tokamak.

#include <vector>
#include <cstddef>
#include <mpi.h>

namespace tokamak {

/// Domain decomposition information for MPI parallelization.
struct DomainDecomp {
    int rank = 0;
    int num_procs = 1;
    int i_start = 0;   // Local start index in R direction
    int i_end = 0;      // Local end index in R direction (exclusive)
    int j_start = 0;   // Local start index in Z direction
    int j_end = 0;      // Local end index in Z direction (exclusive)
    int local_nr = 0;   // Local number of R grid points
    int local_nz = 0;   // Local number of Z grid points
    MPI_Comm comm = MPI_COMM_WORLD;
};

/// 2D structured grid in (R, Z) cylindrical coordinates.
class Grid {
public:
    /// Construct a grid for the tokamak poloidal cross-section.
    /// @param nr Number of radial grid points
    /// @param nz Number of vertical grid points
    /// @param R_min Minimum major radius [m]
    /// @param R_max Maximum major radius [m]
    /// @param Z_min Minimum vertical position [m]
    /// @param Z_max Maximum vertical position [m]
    Grid(int nr, int nz, double R_min, double R_max, double Z_min, double Z_max);

    /// Setup domain decomposition for MPI
    void setup_decomposition(MPI_Comm comm);

    // Accessors
    int nr() const { return nr_; }
    int nz() const { return nz_; }
    double dr() const { return dr_; }
    double dz() const { return dz_; }
    double R_min() const { return R_min_; }
    double R_max() const { return R_max_; }
    double Z_min() const { return Z_min_; }
    double Z_max() const { return Z_max_; }

    /// Get R coordinate at grid index i
    double R(int i) const { return R_min_ + i * dr_; }
    /// Get Z coordinate at grid index j
    double Z(int j) const { return Z_min_ + j * dz_; }

    /// Convert (i,j) to linear index
    size_t index(int i, int j) const { return static_cast<size_t>(i) * nz_ + j; }

    /// Total number of grid points
    size_t total_points() const { return static_cast<size_t>(nr_) * nz_; }

    /// Check if point (R,Z) is inside the tokamak plasma region.
    /// Uses a Miller-parameterized D-shaped cross-section:
    ///   R(θ) = R₀ + a·cos(θ + δ·sin(θ))
    ///   Z(θ) = κ·a·sin(θ)
    /// where κ is elongation (1.0 = circular) and δ is triangularity (0.0 = symmetric).
    bool is_inside_plasma(double R, double Z, double R0, double a,
                          double elongation = 1.0,
                          double triangularity = 0.0) const;

    /// Compute normalized minor radius ρ for a point (R,Z) in shaped geometry.
    /// Returns ρ ∈ [0,1] inside plasma, >1 outside.
    /// Uses the Miller parameterization with elongation κ and triangularity δ.
    double normalized_radius(double R, double Z, double R0, double a,
                             double elongation = 1.0,
                             double triangularity = 0.0) const;

    /// Domain decomposition info
    const DomainDecomp& decomp() const { return decomp_; }
    DomainDecomp& decomp() { return decomp_; }

private:
    int nr_, nz_;
    double R_min_, R_max_, Z_min_, Z_max_;
    double dr_, dz_;
    DomainDecomp decomp_;
};

} // namespace tokamak
