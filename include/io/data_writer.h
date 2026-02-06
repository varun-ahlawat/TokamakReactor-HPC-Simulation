#pragma once
/// @file data_writer.h
/// @brief Binary and text data output for simulation fields.

#include "core/grid.h"
#include "core/field.h"
#include <string>

namespace tokamak {

/// Data output format
enum class OutputFormat {
    BINARY,    // Raw binary (fast, compact)
    CSV,       // Comma-separated values (portable)
};

/// Write field data to files
class DataWriter {
public:
    explicit DataWriter(const std::string& output_dir = "output",
                        OutputFormat format = OutputFormat::BINARY);

    /// Write a scalar field to file
    void write_field(const ScalarField& field, int step,
                     const std::string& prefix = "") const;

    /// Write grid coordinates to file (once at start)
    void write_grid(const Grid& grid) const;

    /// Write a time series to CSV
    void write_timeseries(const std::vector<double>& times,
                          const std::vector<double>& values,
                          const std::string& name) const;

    /// Get output directory
    const std::string& output_dir() const { return output_dir_; }

private:
    std::string output_dir_;
    OutputFormat format_;

    void write_binary(const double* data, size_t n,
                      const std::string& filename) const;
    void write_csv_field(const Grid& grid, const ScalarField& field,
                         const std::string& filename) const;
};

} // namespace tokamak
