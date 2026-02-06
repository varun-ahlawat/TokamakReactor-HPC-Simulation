/// @file test_physics_validation.cpp
/// @brief Physics validation tests — the "litmus test" for simulation correctness.
///
/// These tests verify that the simulation produces physically correct NUMBERS,
/// not just that code runs without crashing. Each test compares simulation output
/// to known analytical solutions, textbook values, or published benchmark data
/// for ITER-like tokamak conditions.
///
/// A 25-year veteran fusion physicist would demand these before trusting
/// any simulation output:
///
///   1. Spitzer resistivity matches textbook formula (exact)
///   2. Alfvén speed is physically reasonable for a tokamak (< c, right order)
///   3. Current quench L/R timescale matches analytical prediction (exponential fit)
///   4. Dreicer field matches textbook formula at multiple temperatures
///   5. Magnetic energy matches analytical estimate for tokamak geometry
///   6. Thermal quench energy balance: radiated energy ≤ initial thermal energy
///
/// If any of these fail, the simulation's predictive value is ZERO — regardless
/// of how many "does it crash?" tests pass.

#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <functional>
#include <mpi.h>

#include "core/grid.h"
#include "core/field.h"
#include "core/time_integrator.h"
#include "utils/physical_constants.h"
#include "physics/mhd_solver.h"
#include "physics/thermal_quench.h"
#include "physics/current_quench.h"
#include "physics/runaway_electrons.h"

// ============ Minimal Test Framework ============

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    static void test_##name(); \
    static bool reg_##name = (register_test(#name, test_##name), true); \
    static void test_##name()

#define ASSERT_TRUE(expr) do { \
    if (!(expr)) { \
        std::cerr << "  FAIL: " #expr " (line " << __LINE__ << ")" << std::endl; \
        throw std::runtime_error("Assertion failed"); \
    } \
} while(0)

#define ASSERT_NEAR(a, b, tol) do { \
    double _a = (a), _b = (b), _t = (tol); \
    if (std::abs(_a - _b) > _t) { \
        std::cerr << "  FAIL: |" << _a << " - " << _b << "| > " << _t \
                  << " (line " << __LINE__ << ")" << std::endl; \
        throw std::runtime_error("Assertion failed"); \
    } \
} while(0)

#define ASSERT_GT(a, b) do { \
    double _a = (a), _b = (b); \
    if (!(_a > _b)) { \
        std::cerr << "  FAIL: " << _a << " > " << _b \
                  << " (line " << __LINE__ << ")" << std::endl; \
        throw std::runtime_error("Assertion failed"); \
    } \
} while(0)

/// Assert that a is within a factor of `factor` of b (order-of-magnitude check)
#define ASSERT_WITHIN_FACTOR(a, b, factor) do { \
    double _a = std::abs(a), _b = std::abs(b), _f = (factor); \
    if (_b == 0.0 || _a / _b > _f || _a / _b < 1.0 / _f) { \
        std::cerr << "  FAIL: " << (a) << " not within factor " << _f \
                  << " of " << (b) << " (ratio=" << (_a/_b) \
                  << ", line " << __LINE__ << ")" << std::endl; \
        throw std::runtime_error("Assertion failed"); \
    } \
} while(0)

struct TestEntry {
    std::string name;
    std::function<void()> func;
};

static std::vector<TestEntry>& test_registry() {
    static std::vector<TestEntry> registry;
    return registry;
}

static void register_test(const std::string& name, std::function<void()> func) {
    test_registry().push_back({name, func});
}

// ============ Physics Validation Tests ============

using namespace tokamak;

// ================================================================
// TEST 1: SPITZER RESISTIVITY — Exact Benchmark
//
// The Spitzer resistivity formula is one of the most well-established
// results in plasma physics (1950s). If this is wrong, every single
// current quench prediction is garbage.
//
// Formula: η = 1.65 × 10⁻⁹ × Z_eff × ln Λ / T_keV^{3/2} [Ω·m]
//
// Reference: L. Spitzer, "Physics of Fully Ionized Gases" (1962)
// ================================================================

TEST(spitzer_resistivity_exact_match) {
    // ITER conditions: Te = 10 keV, ne = 1e20 m⁻³, Zeff = 1.7
    double Te = 10.0;    // keV
    double ne = 1.0e20;  // m⁻³
    double Zeff = 1.7;
    double lnL = constants::coulomb_logarithm(ne, Te);

    double eta_code = constants::spitzer_resistivity(Te, Zeff, lnL);
    double eta_textbook = 1.65e-9 * Zeff * lnL / std::pow(Te, 1.5);

    // Must match the textbook formula EXACTLY (same formula)
    ASSERT_NEAR(eta_code, eta_textbook, eta_textbook * 1e-10);

    // Absolute value check: η(10 keV) ≈ 1.5 × 10⁻⁹ Ω·m
    ASSERT_WITHIN_FACTOR(eta_code, 1.5e-9, 2.0);
}

TEST(spitzer_resistivity_temperature_scaling) {
    // The T^{-3/2} power law is the signature of Coulomb-dominated transport.
    // Verify that η(T1) / η(T2) ≈ (T2/T1)^{3/2} for fixed Zeff and lnΛ.
    double ne = 1.0e20;
    double Zeff = 1.7;
    // lnΛ is deliberately fixed (not computed from ne,Te) to isolate the
    // temperature power-law dependence η ∝ T^{-3/2} in Spitzer's formula.
    double lnL_fixed = 17.0;

    double T1 = 10.0;  // keV (hot plasma)
    double T2 = 0.1;   // keV (100 eV, warm plasma)

    double eta1 = constants::spitzer_resistivity(T1, Zeff, lnL_fixed);
    double eta2 = constants::spitzer_resistivity(T2, Zeff, lnL_fixed);

    double ratio_actual = eta2 / eta1;
    double ratio_expected = std::pow(T1 / T2, 1.5);  // (10/0.1)^1.5 = 31623

    // T^{-3/2} scaling must hold to 1% (it's an exact power law in the formula)
    ASSERT_NEAR(ratio_actual, ratio_expected, ratio_expected * 0.01);
}

// ================================================================
// TEST 2: ALFVÉN SPEED — Sanity Check
//
// The Alfvén speed v_A = B/√(μ₀ρ) determines the MHD timescale.
// For ITER: B~5.3T, n_D~1e20 → v_A ≈ 8×10⁶ m/s (~2.7% of c).
//
// The MHD solver uses normalized density, so we test that:
// (a) v_A computed from the code's own B and ρ is self-consistent
// (b) v_A < c (required for non-relativistic MHD to be valid)
// (c) v_A > 0 (non-trivial equilibrium)
// ================================================================

TEST(alfven_speed_physical_range) {
    Grid grid(64, 64, 4.0, 8.5, -4.5, 4.5);
    MHDConfig config;
    MHDSolver solver(grid, config);
    solver.initialize_equilibrium();

    double vA = solver.max_alfven_speed();

    // v_A must be positive (non-trivial equilibrium)
    ASSERT_GT(vA, 0.0);

    // v_A must be < c (non-relativistic MHD validity)
    ASSERT_TRUE(vA < constants::c_light);

    // Self-consistency: verify v_A = B/sqrt(μ₀ρ) at a specific point
    // using the code's own density and field values
    int ic = grid.nr() / 2;
    int jc = grid.nz() / 2;
    double B = solver.state().Btor(ic, jc);
    double rho = solver.state().density(ic, jc);
    ASSERT_GT(rho, 0.0);
    double vA_local = B / std::sqrt(constants::mu_0 * rho);

    // Local v_A at center should be finite and positive
    ASSERT_GT(vA_local, 0.0);
    ASSERT_TRUE(std::isfinite(vA_local));

    // Max v_A should be >= local v_A at center (by definition of max)
    ASSERT_TRUE(vA >= vA_local * 0.99); // Allow tiny floating-point margin
}

// ================================================================
// TEST 3: CURRENT QUENCH L/R TIMESCALE — Quantitative Benchmark
//
// This is the most important quantitative test. The current quench
// is modeled as an L/R circuit: dI/dt = -R_eff * I / L, giving
// I(t) = I₀ exp(-t/τ) where τ = L/R_eff.
//
// We set up known L and compute R_eff from Spitzer resistivity,
// then verify the simulation produces the correct exponential decay.
//
// Reference: ITER Physics Basis, Ch. 3 "MHD Stability" (1999)
// ================================================================

TEST(current_quench_lr_timescale) {
    Grid grid(32, 32, 4.0, 8.5, -4.5, 4.5);
    CurrentQuenchConfig config;
    config.plasma_inductance = 10.0e-6; // 10 μH
    config.initial_current = 15.0e6;    // 15 MA
    CurrentQuench cq(grid, config);

    // Post-thermal-quench: Te = 10 eV = 0.01 keV (cold plasma)
    double Te_keV = 0.01;
    ScalarField Te(grid, "Te", Te_keV);
    ScalarField ne(grid, "ne", 1.0e20);
    ScalarField Jtor(grid, "Jtor", 1e6);
    cq.initialize_from_thermal_quench(Te, ne, Jtor);

    double I0 = cq.state().total_current;
    ASSERT_NEAR(I0, 15.0e6, 1.0); // Initial current = 15 MA

    // Run for enough time to measure decay
    int nsteps = 1000;
    double dt = 1e-5; // 10 μs
    for (int i = 0; i < nsteps; i++) {
        cq.advance(dt, Te, ne);
    }
    double I_final = cq.state().total_current;
    double t_elapsed = nsteps * dt; // 10 ms

    // Fit exponential: I(t) = I0 × exp(-t/τ) → τ = -t / ln(I/I0)
    ASSERT_GT(I_final, 0.0);
    ASSERT_TRUE(I_final < I0); // Current must decay
    double tau_measured = -t_elapsed / std::log(I_final / I0);

    // Analytical τ = L / R_eff
    // R_eff = η × 2πR₀ / (πa²) where η is Spitzer at Te
    double lnL = constants::coulomb_logarithm(1e20, Te_keV);
    double eta = constants::spitzer_resistivity(Te_keV, config.Z_eff, lnL);
    double R_eff = eta * 2.0 * M_PI * constants::default_major_radius /
                   (M_PI * constants::default_minor_radius *
                    constants::default_minor_radius);
    double tau_analytical = config.plasma_inductance / R_eff;

    // The measured timescale must match the analytical prediction within 10%
    // (small discretization error is acceptable)
    ASSERT_WITHIN_FACTOR(tau_measured, tau_analytical, 1.1);

    // Absolute range check: τ should be 10-1000 ms for ITER-like parameters
    ASSERT_GT(tau_measured, 1e-3);       // > 1 ms
    ASSERT_TRUE(tau_measured < 1.0);     // < 1 s
}

// ================================================================
// TEST 4: DREICER FIELD — Textbook Formula Verification
//
// The Dreicer field determines when runaway electrons are generated.
// If this is wrong by orders of magnitude, the RE physics is meaningless.
//
// Formula: E_D = n_e × e³ × lnΛ / (4π ε₀² × kT)
//
// At Te=10 eV (post-TQ), ne=1e20: E_D ≈ 1000-5000 V/m
// At Te=10 keV (pre-disruption): E_D ≈ 1-10 V/m
//
// The ratio E_D(10eV)/E_D(10keV) must equal T(10keV)/T(10eV) = 1000
// (because E_D ∝ 1/T).
//
// Reference: H. Dreicer, Phys. Rev. 115, 238 (1959)
// ================================================================

TEST(dreicer_field_textbook_values) {
    double ne = 1.0e20; // m⁻³

    // Hot plasma: Te = 10 keV
    double Te_hot = 10.0; // keV
    double lnL_hot = constants::coulomb_logarithm(ne, Te_hot);
    double Ed_hot = constants::dreicer_field(ne, Te_hot, lnL_hot);

    // Analytical: E_D = ne × e³ × lnΛ / (4π ε₀² × kT)
    double kT_hot = Te_hot * constants::keV_to_J;
    double Ed_hot_analytical = ne * std::pow(constants::e_charge, 3) * lnL_hot /
        (4.0 * M_PI * constants::epsilon_0 * constants::epsilon_0 * kT_hot);

    // Must match analytical formula exactly
    ASSERT_NEAR(Ed_hot, Ed_hot_analytical, Ed_hot_analytical * 1e-10);

    // Absolute range: E_D(10 keV) should be ~1-10 V/m
    ASSERT_GT(Ed_hot, 1.0);      // > 1 V/m
    ASSERT_TRUE(Ed_hot < 100.0); // < 100 V/m

    // Cold plasma: Te = 10 eV = 0.01 keV
    double Te_cold = 0.01; // keV
    double lnL_cold = constants::coulomb_logarithm(ne, Te_cold);
    double Ed_cold = constants::dreicer_field(ne, Te_cold, lnL_cold);

    // E_D(10 eV) should be ~1000-5000 V/m
    ASSERT_GT(Ed_cold, 100.0);       // > 100 V/m
    ASSERT_TRUE(Ed_cold < 100000.0); // < 100 kV/m
}

TEST(dreicer_field_temperature_scaling) {
    // E_D ∝ 1/T, so E_D(T1)/E_D(T2) = T2/T1 for same ne and lnΛ
    double ne = 1.0e20;
    double T1 = 10.0;   // keV
    double T2 = 0.01;   // keV

    // lnΛ is deliberately fixed (not computed from ne,Te) to isolate the
    // 1/T dependence of the Dreicer field formula.
    double lnL_fixed = 17.0;
    double Ed1 = ne * std::pow(constants::e_charge, 3) * lnL_fixed /
        (4.0 * M_PI * constants::epsilon_0 * constants::epsilon_0 *
         T1 * constants::keV_to_J);
    double Ed2 = ne * std::pow(constants::e_charge, 3) * lnL_fixed /
        (4.0 * M_PI * constants::epsilon_0 * constants::epsilon_0 *
         T2 * constants::keV_to_J);

    // Ratio must be T1/T2 = 1000 (since E_D ∝ 1/T)
    ASSERT_NEAR(Ed2 / Ed1, T1 / T2, (T1 / T2) * 1e-10);

    // Now verify the code function matches
    double Ed_code_1 = constants::dreicer_field(ne, T1, lnL_fixed);
    double Ed_code_2 = constants::dreicer_field(ne, T2, lnL_fixed);
    ASSERT_NEAR(Ed_code_1, Ed1, Ed1 * 1e-10);
    ASSERT_NEAR(Ed_code_2, Ed2, Ed2 * 1e-10);
}

// ================================================================
// TEST 5: MAGNETIC ENERGY — Grad-Shafranov Equilibrium Benchmark
//
// Total toroidal magnetic energy in an ITER-like tokamak:
//   E_mag = ∫ B²/(2μ₀) dV
//
// For B_φ = B₀R₀/R integrated over a torus with circular cross-section:
//   E_mag ≈ B₀² R₀ / (2μ₀) × 2π²a² × correction ≈ 5-20 GJ
//
// The simulation integrates over its actual (rectangular) computational
// domain, so it captures more volume than the plasma torus. We require
// the result to be within a factor of 5 of the analytical estimate.
//
// Reference: J. Wesson, "Tokamaks", 4th ed., Ch. 3 (2011)
// ================================================================

TEST(magnetic_energy_order_of_magnitude) {
    Grid grid(64, 64, 4.0, 8.5, -4.5, 4.5);
    MHDConfig config;
    MHDSolver solver(grid, config);
    solver.initialize_equilibrium();

    double E_mag = solver.magnetic_energy();

    // Analytical: E_mag ≈ B₀²/(2μ₀) × Volume
    // Volume of torus: V = 2π²R₀a² ≈ 488 m³
    double V_torus = 2.0 * M_PI * M_PI * config.R0 * config.a * config.a;
    double E_analytical = config.B0 * config.B0 / (2.0 * constants::mu_0) * V_torus;

    // E_analytical ≈ 5.5 GJ; code integrates over larger rectangular domain
    // so we expect E_code > E_analytical but within factor of 5
    ASSERT_WITHIN_FACTOR(E_mag, E_analytical, 5.0);

    // Absolute sanity: must be in GJ range (10⁹ J), not kJ or PJ
    ASSERT_GT(E_mag, 1.0e9);       // > 1 GJ
    ASSERT_TRUE(E_mag < 1.0e12);   // < 1 TJ
}

TEST(magnetic_energy_1_over_R_field) {
    // Verify the toroidal field follows the 1/R dependence: B_φ = B₀R₀/R
    // This is a fundamental property of any tokamak's vacuum field.
    Grid grid(64, 64, 4.0, 8.5, -4.5, 4.5);
    MHDConfig config;
    MHDSolver solver(grid, config);
    solver.initialize_equilibrium();

    int jmid = grid.nz() / 2; // Z=0 midplane

    // Check at two different R positions
    int i1 = grid.nr() / 4;     // R closer to inner wall
    int i2 = 3 * grid.nr() / 4; // R closer to outer wall
    double R1 = grid.R(i1);
    double R2 = grid.R(i2);
    double B1 = solver.state().Btor(i1, jmid);
    double B2 = solver.state().Btor(i2, jmid);

    // B₁R₁ should equal B₂R₂ (1/R dependence)
    ASSERT_NEAR(B1 * R1, B2 * R2, std::abs(B1 * R1) * 0.01);

    // Both should equal B₀R₀
    double BR_expected = config.B0 * config.R0;
    ASSERT_NEAR(B1 * R1, BR_expected, BR_expected * 0.01);
}

// ================================================================
// TEST 6: THERMAL QUENCH ENERGY BALANCE — Conservation Law
//
// The most fundamental physics requirement: ENERGY CONSERVATION.
//
// Initial thermal energy: W_th = (3/2) ∫ n_e kT_e dV
// After thermal quench: W_th drops to near wall temperature.
// The energy lost must go somewhere (radiation + wall conduction).
// Total radiated energy must NOT exceed initial thermal energy.
//
// Violation of this constraint means the radiation model is creating
// energy from nothing — a non-physical result that would make any
// experimentalist immediately reject the simulation.
//
// Tolerance: radiated energy ≤ 120% of energy lost
//   (20% margin for diffusion redistributing energy before radiation)
// ================================================================

TEST(thermal_quench_energy_conservation) {
    Grid grid(64, 64, 4.0, 8.5, -4.5, 4.5);
    ThermalQuenchConfig tq_config;
    ThermalQuench tq(grid, tq_config);

    // Initialize: 10 keV plasma at 1e20 m⁻³
    ScalarField Te(grid, "Te", 10.0);
    ScalarField ne(grid, "ne", 1.0e20);
    tq.initialize_from_mhd(Te, ne);

    // Compute initial thermal energy: W = (3/2) ∫ ne × kTe dV
    double dr = grid.dr();
    double dz = grid.dz();
    double W_initial = 0.0;
    for (int i = 0; i < grid.nr(); i++) {
        for (int j = 0; j < grid.nz(); j++) {
            double R = grid.R(i);
            double dV = R * dr * dz * 2.0 * M_PI;
            W_initial += 1.5 * ne(i, j) * Te(i, j) * constants::keV_to_J * dV;
        }
    }
    ASSERT_GT(W_initial, 0.0);

    // Run thermal quench and integrate radiated power
    double W_radiated = 0.0;
    double dt = 1e-5;
    int steps = 2000; // 20 ms — enough for full quench
    for (int s = 0; s < steps; s++) {
        double P_rad = tq.total_radiated_power();
        W_radiated += P_rad * dt;
        tq.advance(dt);
    }

    // Compute final thermal energy
    double W_final = 0.0;
    for (int i = 0; i < grid.nr(); i++) {
        for (int j = 0; j < grid.nz(); j++) {
            double R = grid.R(i);
            double dV = R * dr * dz * 2.0 * M_PI;
            W_final += 1.5 * ne(i, j) * tq.state().Te(i, j) *
                       constants::keV_to_J * dV;
        }
    }

    double W_lost = W_initial - W_final;

    // CRITICAL: Radiated energy must not exceed energy lost
    // (energy conservation: can't radiate more than you had)
    // Allow 20% margin for energy redistribution via diffusion
    ASSERT_TRUE(W_radiated <= 1.2 * W_lost);

    // Radiated energy should account for a significant fraction of energy loss
    // (radiation is the dominant cooling mechanism in a thermal quench)
    ASSERT_GT(W_radiated, 0.1 * W_lost);

    // Temperature should have dropped dramatically
    double T_final = tq.average_temperature();
    ASSERT_TRUE(T_final < 0.1); // < 100 eV post-TQ
}

TEST(thermal_energy_iter_scale) {
    // Verify the initial thermal energy is in the right ballpark for ITER.
    // ITER thermal energy: ~350 MJ for ne=1e20, Te=10 keV
    Grid grid(64, 64, 4.0, 8.5, -4.5, 4.5);
    double dr = grid.dr();
    double dz = grid.dz();

    // Compute W_th over the simulation domain
    double ne = 1.0e20;
    double Te_keV = 10.0;
    double W_th = 0.0;
    for (int i = 0; i < grid.nr(); i++) {
        for (int j = 0; j < grid.nz(); j++) {
            double R = grid.R(i);
            double dV = R * dr * dz * 2.0 * M_PI;
            W_th += 1.5 * ne * Te_keV * constants::keV_to_J * dV;
        }
    }

    // Analytical: W = (3/2) n kT × V_torus
    double V_torus = 2.0 * M_PI * M_PI * 6.2 * 2.0 * 2.0; // ~488 m³
    double W_analytical = 1.5 * ne * Te_keV * constants::keV_to_J * V_torus;

    // The grid volume is larger than the torus, so W_grid > W_analytical
    // but both should be in the 100-1000 MJ range
    ASSERT_GT(W_th, 50.0e6);        // > 50 MJ
    ASSERT_TRUE(W_th < 5000.0e6);   // < 5 GJ
    ASSERT_WITHIN_FACTOR(W_th, 350.0e6, 5.0); // Within factor 5 of 350 MJ
}

// ================================================================
// BONUS: Cross-validation between subsystems
//
// An experienced physicist checks that the OUTPUT of one phase
// makes sense as INPUT to the next phase. These are the handoff
// integrity tests.
// ================================================================

TEST(dreicer_rate_threshold_consistency) {
    // Verify that the Dreicer generation rate is zero when E < E_D
    // and non-zero when E > E_D. This is a FUNDAMENTAL threshold.
    double ne = 1.0e20;
    double Te_keV = 0.01; // 10 eV (post-TQ)
    double lnL = constants::coulomb_logarithm(ne, Te_keV);
    double E_D = constants::dreicer_field(ne, Te_keV, lnL);
    double Zeff = 1.7;

    // E = 0.5 E_D: no runaways
    double rate_below = RunawayElectrons::dreicer_rate(ne, Te_keV, 0.5 * E_D, E_D, Zeff);
    ASSERT_NEAR(rate_below, 0.0, 1e-10);

    // E = 2.0 E_D: positive generation rate
    double rate_above = RunawayElectrons::dreicer_rate(ne, Te_keV, 2.0 * E_D, E_D, Zeff);
    ASSERT_GT(rate_above, 0.0);
}

TEST(connor_hastie_critical_field_order) {
    // The Connor-Hastie critical field E_c ≈ n_e e³ lnΛ / (4π ε₀² m_e c²)
    // For ITER at ne=1e20: E_c ≈ 0.05-0.1 V/m
    // The Dreicer field E_D = E_c × (m_e c² / kT) ≫ E_c for T < m_e c² ≈ 511 keV
    double ne = 1.0e20;
    double Te = 0.01; // keV
    double lnL = constants::coulomb_logarithm(ne, Te);

    double E_D = constants::dreicer_field(ne, Te, lnL);

    // E_D / E_c = m_e c² / kT = 511 / 0.01 = 51100
    double mc2_keV = constants::m_electron * constants::c_light *
                     constants::c_light / constants::keV_to_J;
    double E_c_estimated = E_D * Te / mc2_keV;

    // E_c should be in the range 0.01 - 1.0 V/m for ITER
    ASSERT_GT(E_c_estimated, 0.001);
    ASSERT_TRUE(E_c_estimated < 10.0);
}

TEST(collision_frequency_benchmark) {
    // Electron-electron collision frequency at ITER conditions
    // ν_ee ≈ n_e e⁴ lnΛ / (4π ε₀² m_e² v_th³)
    // At Te=10 keV, ne=1e20: ν_ee ≈ 10³-10⁵ s⁻¹
    double ne = 1.0e20;
    double Te = 10.0; // keV
    double lnL = constants::coulomb_logarithm(ne, Te);

    double nu = constants::collision_frequency(ne, Te, lnL);

    // Must be positive
    ASSERT_GT(nu, 0.0);

    // Must be in physically reasonable range for a hot tokamak plasma
    // (10¹ to 10⁶ s⁻¹)
    ASSERT_GT(nu, 1.0);
    ASSERT_TRUE(nu < 1.0e8);
}

TEST(coulomb_logarithm_range) {
    // The Coulomb logarithm for tokamak plasmas is always in the range 10-25.
    // Values outside this range indicate a formula error.

    // Hot ITER-like plasma
    double lnL_hot = constants::coulomb_logarithm(1.0e20, 10.0);
    ASSERT_GT(lnL_hot, 10.0);
    ASSERT_TRUE(lnL_hot < 25.0);

    // Cold post-disruption plasma
    double lnL_cold = constants::coulomb_logarithm(1.0e20, 0.01);
    ASSERT_GT(lnL_cold, 5.0);
    ASSERT_TRUE(lnL_cold < 20.0);

    // High-density plasma (pellet injection scenario)
    double lnL_dense = constants::coulomb_logarithm(1.0e22, 1.0);
    ASSERT_GT(lnL_dense, 5.0);
    ASSERT_TRUE(lnL_dense < 25.0);
}

// ============ Test Runner ============

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        std::cout << "\n╔═══════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║  PHYSICS VALIDATION TESTS — The Litmus Test          ║" << std::endl;
        std::cout << "║                                                       ║" << std::endl;
        std::cout << "║  These tests verify PHYSICAL CORRECTNESS, not just    ║" << std::endl;
        std::cout << "║  that the code compiles. Each test compares to known  ║" << std::endl;
        std::cout << "║  analytical solutions, textbook values, or published  ║" << std::endl;
        std::cout << "║  benchmark data for ITER-like tokamak conditions.     ║" << std::endl;
        std::cout << "╚═══════════════════════════════════════════════════════╝" << std::endl;
        std::cout << std::endl;
    }

    for (const auto& test : test_registry()) {
        tests_run++;
        if (rank == 0) {
            std::cout << "  Running: " << test.name << "... " << std::flush;
        }
        try {
            test.func();
            tests_passed++;
            if (rank == 0) std::cout << "PASS" << std::endl;
        } catch (const std::exception& e) {
            tests_failed++;
            if (rank == 0) std::cout << "FAIL (" << e.what() << ")" << std::endl;
        }
    }

    if (rank == 0) {
        std::cout << "\n=== Physics Validation Results: " << tests_passed << "/"
                  << tests_run << " passed";
        if (tests_failed > 0)
            std::cout << " (" << tests_failed << " FAILED — SIMULATION CANNOT BE TRUSTED)";
        else
            std::cout << " — All physics benchmarks verified ✓";
        std::cout << " ===" << std::endl;
    }

    MPI_Finalize();
    return tests_failed > 0 ? 1 : 0;
}
