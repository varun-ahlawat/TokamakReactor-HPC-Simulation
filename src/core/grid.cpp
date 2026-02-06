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
                             double elongation) const {
    double r_norm = (R_pos - R0) / a;
    double z_norm = Z_pos / (a * elongation);
    return (r_norm * r_norm + z_norm * z_norm) <= 1.0;
}

} // namespace tokamak
