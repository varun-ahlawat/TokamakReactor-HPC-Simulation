/// @file diagnostics.cpp
#include "io/diagnostics.h"
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <iostream>

namespace tokamak {

Diagnostics::Diagnostics(const std::string& output_dir)
    : output_dir_(output_dir) {
    std::filesystem::create_directories(output_dir_);
}

void Diagnostics::record(const DiagnosticPoint& point) {
    points_.push_back(point);
}

void Diagnostics::write_csv(const std::string& filename) const {
    std::string filepath = output_dir_ + "/" + filename;
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filepath << std::endl;
        return;
    }

    // Header
    file << "time_ms,phase,Te_avg_keV,Te_max_keV,Ip_A,E_mag_J,E_kin_J,"
         << "island_width_m,E_field_Vm,I_RE_A,RE_fraction,P_rad_W\n";

    // Data
    file << std::scientific << std::setprecision(6);
    for (const auto& p : points_) {
        file << p.time * 1000.0 << ","
             << phase_to_string(p.phase) << ","
             << p.temperature_avg << ","
             << p.temperature_max << ","
             << p.plasma_current << ","
             << p.magnetic_energy << ","
             << p.kinetic_energy << ","
             << p.island_width << ","
             << p.E_field_avg << ","
             << p.RE_current << ","
             << p.RE_fraction << ","
             << p.radiated_power << "\n";
    }

    file.close();
}

void Diagnostics::write_snapshot(const Grid& grid, const ScalarField& field,
                                  int step, const std::string& prefix) const {
    std::string filepath = output_dir_ + "/" + prefix + "_" +
                           std::to_string(step) + ".dat";
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) return;

    // Write header: nr, nz, R_min, R_max, Z_min, Z_max
    int nr = grid.nr(), nz = grid.nz();
    double bounds[4] = {grid.R_min(), grid.R_max(), grid.Z_min(), grid.Z_max()};
    file.write(reinterpret_cast<const char*>(&nr), sizeof(int));
    file.write(reinterpret_cast<const char*>(&nz), sizeof(int));
    file.write(reinterpret_cast<const char*>(bounds), 4 * sizeof(double));

    // Write field data
    file.write(reinterpret_cast<const char*>(field.data()),
               static_cast<std::streamsize>(field.size() * sizeof(double)));
    file.close();
}

} // namespace tokamak
