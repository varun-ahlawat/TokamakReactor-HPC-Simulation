/// @file physical_constants.cpp
#include "utils/physical_constants.h"
#include <cmath>

namespace tokamak {
namespace constants {

double coulomb_logarithm(double n_e, double T_e_keV) {
    // Coulomb logarithm for thermal plasma
    // ln Λ ≈ 15.2 - 0.5 ln(n_e/10²⁰) + ln(T_e_keV)
    double n20 = n_e / 1.0e20;
    return 15.2 - 0.5 * std::log(n20) + std::log(T_e_keV);
}

double spitzer_resistivity(double T_e_keV, double Z_eff, double ln_lambda) {
    // Spitzer resistivity: η = (Z_eff * e² * m_e^{1/2} * ln Λ) /
    //                          (1.96 * (4π ε₀)² * (k_B T_e)^{3/2})
    // Simplified formula: η ≈ 1.65e-9 * Z_eff * ln_lambda / T_keV^{3/2} [Ohm·m]
    if (T_e_keV < 1.0e-3) T_e_keV = 1.0e-3; // Floor to avoid divergence
    return 1.65e-9 * Z_eff * ln_lambda / std::pow(T_e_keV, 1.5);
}

double dreicer_field(double n_e, double T_e_keV, double ln_lambda) {
    // Dreicer field: E_D = n_e * e³ * ln Λ / (4π ε₀² * m_e * c²)
    // Simplified: E_D ≈ n_e * e * ln_lambda / (4π ε₀ * m_e * c²)
    double T_J = T_e_keV * keV_to_J;
    if (T_J < 1.0e-25) T_J = 1.0e-25;
    return n_e * e_charge * ln_lambda / (4.0 * M_PI * epsilon_0 * T_J);
}

double connor_hastie_critical_energy(double E_over_Ed) {
    // Critical energy above which electrons run away
    // E_crit ≈ m_e * c² * (1/sqrt(E/E_D) - 1) for E > E_D
    if (E_over_Ed <= 1.0) return 1.0e6; // Very high energy = no runaways
    double mc2_keV = m_electron * c_light * c_light / keV_to_J; // 511 keV
    return mc2_keV / std::sqrt(E_over_Ed - 1.0);
}

double collision_frequency(double n_e, double T_e_keV, double ln_lambda) {
    // Electron-electron collision frequency
    // ν_ee = n_e * e⁴ * ln Λ / (4π ε₀² * m_e² * v_th³)
    double T_J = T_e_keV * keV_to_J;
    if (T_J < 1.0e-25) T_J = 1.0e-25;
    double v_th = std::sqrt(2.0 * T_J / m_electron);
    return n_e * std::pow(e_charge, 4) * ln_lambda /
           (4.0 * M_PI * epsilon_0 * epsilon_0 * m_electron * m_electron *
            v_th * v_th * v_th);
}

} // namespace constants
} // namespace tokamak
