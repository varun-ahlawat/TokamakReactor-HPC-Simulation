#pragma once
/// @file physical_constants.h
/// @brief Physical constants used throughout the tokamak simulation.

namespace tokamak {
namespace constants {

// Fundamental constants (SI units)
constexpr double mu_0       = 1.2566370614e-6;  // Vacuum permeability [T·m/A]
constexpr double epsilon_0  = 8.8541878128e-12;  // Vacuum permittivity [F/m]
constexpr double c_light    = 2.99792458e8;      // Speed of light [m/s]
constexpr double e_charge   = 1.602176634e-19;   // Elementary charge [C]
constexpr double m_electron = 9.1093837015e-31;  // Electron mass [kg]
constexpr double m_proton   = 1.67262192369e-27;  // Proton mass [kg]
constexpr double m_deuteron = 3.3435837724e-27;  // Deuteron mass [kg]
constexpr double k_boltzmann= 1.380649e-23;      // Boltzmann constant [J/K]
constexpr double eV_to_J    = 1.602176634e-19;   // eV to Joules conversion
constexpr double keV_to_J   = 1.602176634e-16;   // keV to Joules conversion

// Tokamak-specific defaults (ITER-like parameters)
constexpr double default_major_radius  = 6.2;    // Major radius R₀ [m]
constexpr double default_minor_radius  = 2.0;    // Minor radius a [m]
constexpr double default_B_toroidal    = 5.3;    // Toroidal field on axis [T]
constexpr double default_plasma_current= 15.0e6; // Plasma current [A]
constexpr double default_n_e           = 1.0e20; // Electron density [m⁻³]
constexpr double default_T_e           = 10.0;   // Electron temperature [keV]
constexpr double default_T_i           = 8.0;    // Ion temperature [keV]
constexpr double default_Z_eff         = 1.7;    // Effective charge number

// Derived constants
constexpr double classical_electron_radius = 2.8179403262e-15; // [m]

/// Coulomb logarithm for typical tokamak conditions
double coulomb_logarithm(double n_e, double T_e_keV);

/// Spitzer resistivity [Ohm·m]
double spitzer_resistivity(double T_e_keV, double Z_eff, double ln_lambda);

/// Dreicer electric field E_D = n_e e³ lnΛ / (4π ε₀² kT) [V/m]
double dreicer_field(double n_e, double T_e_keV, double ln_lambda);

/// Connor-Hastie critical energy for runaway electrons [keV]
double connor_hastie_critical_energy(double E_over_Ed);

/// Collision frequency [s⁻¹]
double collision_frequency(double n_e, double T_e_keV, double ln_lambda);

} // namespace constants
} // namespace tokamak
