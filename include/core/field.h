#pragma once
/// @file field.h
/// @brief 2D field data structures for plasma quantities on the grid.

#include "core/grid.h"
#include <vector>
#include <string>
#include <functional>

namespace tokamak {

/// A scalar field defined on the 2D grid (e.g., temperature, density, pressure).
class ScalarField {
public:
    ScalarField() = default;
    explicit ScalarField(const Grid& grid, const std::string& name = "unnamed",
                         double init_value = 0.0);

    /// Access element at (i, j)
    double& operator()(int i, int j);
    const double& operator()(int i, int j) const;

    /// Access raw data
    double* data() { return data_.data(); }
    const double* data() const { return data_.data(); }
    size_t size() const { return data_.size(); }

    /// Field name
    const std::string& name() const { return name_; }
    void set_name(const std::string& n) { name_ = n; }

    /// Grid reference
    const Grid& grid() const { return *grid_; }

    /// Fill with a value
    void fill(double value);

    /// Fill with a function f(R, Z)
    void fill(std::function<double(double, double)> func);

    /// Compute the maximum absolute value (useful for CFL checks)
    double max_abs() const;

    /// Compute L2 norm
    double l2_norm() const;

    /// Element-wise operations
    ScalarField& operator+=(const ScalarField& other);
    ScalarField& operator*=(double scalar);

    /// Exchange ghost zones with MPI neighbors
    void exchange_ghosts();

private:
    const Grid* grid_ = nullptr;
    std::vector<double> data_;
    std::string name_;
};

/// A vector field with (R, Z) components (e.g., velocity, magnetic field).
class VectorField {
public:
    VectorField() = default;
    VectorField(const Grid& grid, const std::string& name = "unnamed");

    /// Access R and Z components
    ScalarField& R_comp() { return r_component_; }
    ScalarField& Z_comp() { return z_component_; }
    const ScalarField& R_comp() const { return r_component_; }
    const ScalarField& Z_comp() const { return z_component_; }

    /// Compute magnitude field
    ScalarField magnitude() const;

    /// Field name
    const std::string& name() const { return name_; }

    /// Exchange ghost zones for both components
    void exchange_ghosts();

private:
    ScalarField r_component_;
    ScalarField z_component_;
    std::string name_;
};

} // namespace tokamak
