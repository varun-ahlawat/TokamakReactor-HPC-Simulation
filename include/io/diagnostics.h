#pragma once
/// @file diagnostics.h
/// @brief Diagnostic measurements for the tokamak simulation.

#include "core/grid.h"
#include "core/field.h"
#include "coupling/multi_physics_coupler.h"
#include <vector>
#include <string>
#include <fstream>

namespace tokamak {

/// Time series data point
struct DiagnosticPoint {
    double time;
    DisruptionPhase phase;
    double temperature_avg;    // Volume-averaged Te [keV]
    double temperature_max;    // Peak Te [keV]
    double plasma_current;     // Total Ip [A]
    double magnetic_energy;    // Total magnetic energy [J]
    double kinetic_energy;     // Total kinetic energy [J]
    double island_width;       // Magnetic island width [m]
    double E_field_avg;        // Average toroidal E field [V/m]
    double RE_current;         // Runaway electron current [A]
    double RE_fraction;        // RE current fraction
    double radiated_power;     // Total radiated power [W]
};

/// Diagnostic recorder for simulation data
class Diagnostics {
public:
    explicit Diagnostics(const std::string& output_dir = "output");

    /// Record a diagnostic point
    void record(const DiagnosticPoint& point);

    /// Write all diagnostics to CSV file
    void write_csv(const std::string& filename = "diagnostics.csv") const;

    /// Write current field snapshot to binary file
    void write_snapshot(const Grid& grid, const ScalarField& field,
                        int step, const std::string& prefix = "field") const;

    /// Get all recorded points
    const std::vector<DiagnosticPoint>& points() const { return points_; }

    /// Get number of recorded points
    size_t size() const { return points_.size(); }

private:
    std::string output_dir_;
    std::vector<DiagnosticPoint> points_;
};

} // namespace tokamak
