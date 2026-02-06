/// @file field.cpp
#include "core/field.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <omp.h>

namespace tokamak {

// --- ScalarField ---

ScalarField::ScalarField(const Grid& grid, const std::string& name, double init_value)
    : grid_(&grid), data_(grid.total_points(), init_value), name_(name) {}

double& ScalarField::operator()(int i, int j) {
    return data_[grid_->index(i, j)];
}

const double& ScalarField::operator()(int i, int j) const {
    return data_[grid_->index(i, j)];
}

void ScalarField::fill(double value) {
    std::fill(data_.begin(), data_.end(), value);
}

void ScalarField::fill(std::function<double(double, double)> func) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < grid_->nr(); ++i) {
        for (int j = 0; j < grid_->nz(); ++j) {
            data_[grid_->index(i, j)] = func(grid_->R(i), grid_->Z(j));
        }
    }
}

double ScalarField::max_abs() const {
    double result = 0.0;
    #pragma omp parallel for reduction(max:result)
    for (size_t k = 0; k < data_.size(); ++k) {
        double val = std::abs(data_[k]);
        if (val > result) result = val;
    }
    return result;
}

double ScalarField::l2_norm() const {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (size_t k = 0; k < data_.size(); ++k) {
        sum += data_[k] * data_[k];
    }
    return std::sqrt(sum / static_cast<double>(data_.size()));
}

ScalarField& ScalarField::operator+=(const ScalarField& other) {
    #pragma omp parallel for
    for (size_t k = 0; k < data_.size(); ++k) {
        data_[k] += other.data_[k];
    }
    return *this;
}

ScalarField& ScalarField::operator*=(double scalar) {
    #pragma omp parallel for
    for (size_t k = 0; k < data_.size(); ++k) {
        data_[k] *= scalar;
    }
    return *this;
}

void ScalarField::exchange_ghosts() {
    // MPI ghost zone exchange for domain decomposition
    const auto& d = grid_->decomp();
    if (d.num_procs <= 1) return;

    int nz = grid_->nz();
    MPI_Status status;

    // Send right boundary to right neighbor, receive left boundary from left
    if (d.rank < d.num_procs - 1) {
        MPI_Send(&data_[(d.i_end - 1) * nz], nz, MPI_DOUBLE,
                 d.rank + 1, 0, d.comm);
    }
    if (d.rank > 0) {
        std::vector<double> buf(nz);
        MPI_Recv(buf.data(), nz, MPI_DOUBLE,
                 d.rank - 1, 0, d.comm, &status);
    }
}

// --- VectorField ---

VectorField::VectorField(const Grid& grid, const std::string& name)
    : r_component_(grid, name + "_R"),
      z_component_(grid, name + "_Z"),
      name_(name) {}

ScalarField VectorField::magnitude() const {
    const auto& grid = r_component_.grid();
    ScalarField mag(grid, name_ + "_mag");

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < grid.nr(); ++i) {
        for (int j = 0; j < grid.nz(); ++j) {
            double vr = r_component_(i, j);
            double vz = z_component_(i, j);
            mag(i, j) = std::sqrt(vr * vr + vz * vz);
        }
    }
    return mag;
}

void VectorField::exchange_ghosts() {
    r_component_.exchange_ghosts();
    z_component_.exchange_ghosts();
}

} // namespace tokamak
