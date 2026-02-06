/// @file data_writer.cpp
#include "io/data_writer.h"
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <iostream>

namespace tokamak {

DataWriter::DataWriter(const std::string& output_dir, OutputFormat format)
    : output_dir_(output_dir), format_(format) {
    std::filesystem::create_directories(output_dir_);
}

void DataWriter::write_field(const ScalarField& field, int step,
                              const std::string& prefix) const {
    std::string name = prefix.empty() ? field.name() : prefix;
    std::string filename = output_dir_ + "/" + name + "_" +
                           std::to_string(step);

    if (format_ == OutputFormat::BINARY) {
        filename += ".dat";
        write_binary(field.data(), field.size(), filename);
    } else {
        filename += ".csv";
        write_csv_field(field.grid(), field, filename);
    }
}

void DataWriter::write_grid(const Grid& grid) const {
    std::string filename = output_dir_ + "/grid.csv";
    std::ofstream file(filename);
    if (!file.is_open()) return;

    file << "i,j,R,Z\n";
    file << std::fixed << std::setprecision(6);
    for (int i = 0; i < grid.nr(); ++i) {
        for (int j = 0; j < grid.nz(); ++j) {
            file << i << "," << j << ","
                 << grid.R(i) << "," << grid.Z(j) << "\n";
        }
    }
    file.close();
}

void DataWriter::write_timeseries(const std::vector<double>& times,
                                    const std::vector<double>& values,
                                    const std::string& name) const {
    std::string filename = output_dir_ + "/" + name + "_timeseries.csv";
    std::ofstream file(filename);
    if (!file.is_open()) return;

    file << "time," << name << "\n";
    file << std::scientific << std::setprecision(8);
    for (size_t i = 0; i < times.size() && i < values.size(); ++i) {
        file << times[i] << "," << values[i] << "\n";
    }
    file.close();
}

void DataWriter::write_binary(const double* data, size_t n,
                                const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not write to " << filename << std::endl;
        return;
    }
    file.write(reinterpret_cast<const char*>(data),
               static_cast<std::streamsize>(n * sizeof(double)));
    file.close();
}

void DataWriter::write_csv_field(const Grid& grid, const ScalarField& field,
                                  const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) return;

    file << "R,Z," << field.name() << "\n";
    file << std::scientific << std::setprecision(8);
    for (int i = 0; i < grid.nr(); ++i) {
        for (int j = 0; j < grid.nz(); ++j) {
            file << grid.R(i) << "," << grid.Z(j) << ","
                 << field(i, j) << "\n";
        }
    }
    file.close();
}

} // namespace tokamak
