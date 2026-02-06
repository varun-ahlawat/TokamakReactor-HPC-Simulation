/// @file test_main.cpp
/// @brief Unit tests for the Tokamak Disruption Simulation.
///
/// Minimal test framework — no external dependencies required.

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
#include "physics/disruption_mitigation.h"
#include "coupling/multi_physics_coupler.h"

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

// ============ Tests ============

using namespace tokamak;

// --- Grid Tests ---

TEST(grid_construction) {
    Grid grid(32, 32, 4.0, 8.5, -4.5, 4.5);
    ASSERT_TRUE(grid.nr() == 32);
    ASSERT_TRUE(grid.nz() == 32);
    ASSERT_GT(grid.dr(), 0.0);
    ASSERT_GT(grid.dz(), 0.0);
    ASSERT_NEAR(grid.R(0), 4.0, 1e-10);
    ASSERT_NEAR(grid.Z(0), -4.5, 1e-10);
}

TEST(grid_inside_plasma) {
    Grid grid(32, 32, 4.0, 8.5, -4.5, 4.5);
    // Point at plasma center should be inside
    ASSERT_TRUE(grid.is_inside_plasma(6.2, 0.0, 6.2, 2.0));
    // Point far outside should not be
    ASSERT_TRUE(!grid.is_inside_plasma(1.0, 0.0, 6.2, 2.0));
}

TEST(grid_total_points) {
    Grid grid(64, 32, 4.0, 8.5, -4.5, 4.5);
    ASSERT_TRUE(grid.total_points() == 64 * 32);
}

// --- Field Tests ---

TEST(scalar_field_creation) {
    Grid grid(16, 16, 4.0, 8.5, -4.5, 4.5);
    ScalarField f(grid, "test", 3.14);
    ASSERT_NEAR(f(0, 0), 3.14, 1e-10);
    ASSERT_NEAR(f(8, 8), 3.14, 1e-10);
    ASSERT_TRUE(f.size() == 16 * 16);
}

TEST(scalar_field_fill_function) {
    Grid grid(16, 16, 4.0, 8.5, -4.5, 4.5);
    ScalarField f(grid, "test");
    f.fill([](double R, double Z) { return R * R + Z * Z; });

    double R0 = grid.R(0);
    double Z0 = grid.Z(0);
    ASSERT_NEAR(f(0, 0), R0 * R0 + Z0 * Z0, 1e-10);
}

TEST(scalar_field_max_abs) {
    Grid grid(16, 16, 4.0, 8.5, -4.5, 4.5);
    ScalarField f(grid, "test", 0.0);
    f(5, 5) = -7.0;
    f(10, 10) = 3.0;
    ASSERT_NEAR(f.max_abs(), 7.0, 1e-10);
}

TEST(scalar_field_l2_norm) {
    Grid grid(4, 4, 0.0, 1.0, 0.0, 1.0);
    ScalarField f(grid, "test", 2.0);
    ASSERT_NEAR(f.l2_norm(), 2.0, 1e-10);
}

TEST(vector_field_magnitude) {
    Grid grid(16, 16, 4.0, 8.5, -4.5, 4.5);
    VectorField v(grid, "vel");
    v.R_comp().fill(3.0);
    v.Z_comp().fill(4.0);
    auto mag = v.magnitude();
    ASSERT_NEAR(mag(8, 8), 5.0, 1e-10);
}

// --- Time Integrator Tests ---

TEST(rk4_exponential_decay) {
    // Test RK4 on dy/dt = -y, y(0) = 1. Solution: y = exp(-t)
    RK4Integrator rk4;
    std::vector<double> y = {1.0};
    auto rhs = [](const std::vector<double>& u, std::vector<double>& dudt, double) {
        dudt[0] = -u[0];
    };

    double dt = 0.01;
    double t = 0.0;
    for (int i = 0; i < 100; ++i) {
        rk4.step(y, rhs, t, dt);
        t += dt;
    }
    ASSERT_NEAR(y[0], std::exp(-1.0), 1e-8);
}

TEST(rk2_exponential_decay) {
    RK2Integrator rk2;
    std::vector<double> y = {1.0};
    auto rhs = [](const std::vector<double>& u, std::vector<double>& dudt, double) {
        dudt[0] = -u[0];
    };

    double dt = 0.001;
    double t = 0.0;
    for (int i = 0; i < 1000; ++i) {
        rk2.step(y, rhs, t, dt);
        t += dt;
    }
    ASSERT_NEAR(y[0], std::exp(-1.0), 1e-4);
}

TEST(adaptive_timestepper) {
    AdaptiveTimeStepper stepper(1e-10, 1e-3, 0.5);
    double dt = stepper.compute_dt(0.01, 100.0);
    ASSERT_GT(dt, 0.0);
    ASSERT_TRUE(dt <= 1e-3);
    ASSERT_NEAR(dt, 0.5 * 0.01 / 100.0, 1e-10);
}

// --- Physical Constants Tests ---

TEST(coulomb_logarithm) {
    double ln_lambda = constants::coulomb_logarithm(1e20, 10.0);
    ASSERT_GT(ln_lambda, 10.0);
    ASSERT_TRUE(ln_lambda < 25.0);
}

TEST(spitzer_resistivity) {
    double eta = constants::spitzer_resistivity(10.0, 1.7, 17.0);
    ASSERT_GT(eta, 0.0);
    // Hot plasma should have low resistivity
    ASSERT_TRUE(eta < 1e-6);

    // Cold plasma should have higher resistivity
    double eta_cold = constants::spitzer_resistivity(0.01, 1.7, 17.0);
    ASSERT_GT(eta_cold, eta);
}

TEST(dreicer_field) {
    double Ed = constants::dreicer_field(1e20, 10.0, 17.0);
    ASSERT_GT(Ed, 0.0);
}

TEST(collision_frequency) {
    double nu = constants::collision_frequency(1e20, 10.0, 17.0);
    ASSERT_GT(nu, 0.0);
}

// --- MHD Solver Tests ---

TEST(mhd_equilibrium_initialization) {
    Grid grid(32, 32, 4.0, 8.5, -4.5, 4.5);
    MHDConfig config;
    MHDSolver solver(grid, config);
    solver.initialize_equilibrium();

    // Toroidal field should be positive and follow 1/R
    ASSERT_GT(solver.state().Btor(16, 16), 0.0);

    // Density should be positive
    ASSERT_GT(solver.state().density(16, 16), 0.0);

    // Pressure should be positive
    ASSERT_GT(solver.state().pressure(16, 16), 0.0);

    // Magnetic energy should be positive
    ASSERT_GT(solver.magnetic_energy(), 0.0);
}

TEST(mhd_perturbation) {
    Grid grid(32, 32, 4.0, 8.5, -4.5, 4.5);
    MHDSolver solver(grid);
    solver.initialize_equilibrium();

    // Record flux values before perturbation
    double psi_sum_before = 0.0;
    for (int i = 8; i < 24; ++i) {
        for (int j = 8; j < 24; ++j) {
            psi_sum_before += std::abs(solver.state().Bpol_psi(i, j));
        }
    }

    solver.apply_perturbation();

    // Check that flux values changed after perturbation
    double psi_sum_after = 0.0;
    for (int i = 8; i < 24; ++i) {
        for (int j = 8; j < 24; ++j) {
            psi_sum_after += std::abs(solver.state().Bpol_psi(i, j));
        }
    }

    ASSERT_TRUE(std::abs(psi_sum_after - psi_sum_before) > 1e-10);
}

TEST(mhd_advance) {
    Grid grid(16, 16, 4.0, 8.5, -4.5, 4.5);
    MHDSolver solver(grid);
    solver.initialize_equilibrium();
    solver.apply_perturbation();

    double E0 = solver.magnetic_energy();
    double dt_suggest = solver.advance(1e-8);

    ASSERT_GT(dt_suggest, 0.0);
    // Energy should be finite and positive after advance
    double E1 = solver.magnetic_energy();
    ASSERT_GT(E1, 0.0);
    ASSERT_TRUE(std::isfinite(E1));
}

TEST(mhd_alfven_speed) {
    Grid grid(16, 16, 4.0, 8.5, -4.5, 4.5);
    MHDSolver solver(grid);
    solver.initialize_equilibrium();

    double vA = solver.max_alfven_speed();
    ASSERT_GT(vA, 0.0);
    // Alfvén speed should be physically reasonable (< c)
    ASSERT_TRUE(vA < constants::c_light);
}

// --- Thermal Quench Tests ---

TEST(thermal_quench_initialization) {
    Grid grid(16, 16, 4.0, 8.5, -4.5, 4.5);
    ThermalQuench tq(grid);

    ScalarField Te(grid, "Te", 10.0);  // 10 keV
    ScalarField ne(grid, "ne", 1e20);  // 10^20 m^-3
    tq.initialize_from_mhd(Te, ne);

    ASSERT_NEAR(tq.state().Te(8, 8), 10.0, 1e-10);
    ASSERT_GT(tq.average_temperature(), 0.0);
}

TEST(thermal_quench_cooling) {
    Grid grid(16, 16, 4.0, 8.5, -4.5, 4.5);
    ThermalQuenchConfig tq_config;
    tq_config.impurity_fraction_initial = 0.1; // Higher impurity for faster cooling
    tq_config.stochasticity_factor = 1.0;       // Strong stochastic transport
    ThermalQuench tq(grid, tq_config);

    ScalarField Te(grid, "Te", 10.0);
    ScalarField ne(grid, "ne", 1e20);
    tq.initialize_from_mhd(Te, ne);

    double T_before = tq.average_temperature();
    for (int i = 0; i < 500; ++i) {
        tq.advance(1e-5);
    }
    double T_after = tq.average_temperature();

    // Temperature should decrease during thermal quench
    ASSERT_TRUE(T_after < T_before);
}

// --- Current Quench Tests ---

TEST(current_quench_initialization) {
    Grid grid(16, 16, 4.0, 8.5, -4.5, 4.5);
    CurrentQuench cq(grid);

    ScalarField Te(grid, "Te", 0.1);   // Post-TQ temperature
    ScalarField ne(grid, "ne", 1e20);
    ScalarField Jtor(grid, "Jtor", 1e6);
    cq.initialize_from_thermal_quench(Te, ne, Jtor);

    ASSERT_GT(cq.state().total_current, 0.0);
}

TEST(current_quench_decay) {
    Grid grid(16, 16, 4.0, 8.5, -4.5, 4.5);
    CurrentQuench cq(grid);

    ScalarField Te(grid, "Te", 0.1);
    ScalarField ne(grid, "ne", 1e20);
    ScalarField Jtor(grid, "Jtor", 1e6);
    cq.initialize_from_thermal_quench(Te, ne, Jtor);

    double I_before = cq.state().total_current;
    for (int i = 0; i < 100; ++i) {
        cq.advance(1e-5, Te, ne);
    }
    double I_after = cq.state().total_current;

    // Current should decay during quench
    ASSERT_TRUE(std::abs(I_after) < std::abs(I_before));
}

// --- Runaway Electron Tests ---

TEST(dreicer_rate_below_critical) {
    // Below critical field, no Dreicer generation
    double rate = RunawayElectrons::dreicer_rate(1e20, 0.1, 0.5, 1.0, 1.7);
    ASSERT_NEAR(rate, 0.0, 1e-10);
}

TEST(dreicer_rate_above_critical) {
    // Above critical field, positive generation
    double rate = RunawayElectrons::dreicer_rate(1e20, 0.1, 10.0, 1.0, 1.7);
    ASSERT_GT(rate, 0.0);
}

TEST(avalanche_rate_positive) {
    double rate = RunawayElectrons::avalanche_rate(1e20, 10.0, 1.0, 17.0, 1.7);
    ASSERT_GT(rate, 0.0);
}

TEST(avalanche_rate_zero_below_critical) {
    double rate = RunawayElectrons::avalanche_rate(1e20, 0.5, 1.0, 17.0, 1.7);
    ASSERT_NEAR(rate, 0.0, 1e-10);
}

// --- DMS Tests ---

TEST(dms_trigger) {
    Grid grid(16, 16, 4.0, 8.5, -4.5, 4.5);
    DisruptionMitigation dms(grid);

    ASSERT_TRUE(!dms.is_triggered());
    dms.trigger(0.005);
    ASSERT_TRUE(dms.is_triggered());
}

TEST(dms_deposition) {
    Grid grid(16, 16, 4.0, 8.5, -4.5, 4.5);
    DMSConfig config;
    config.trigger_delay = 0.0; // No delay for test
    config.assimilation_time = 0.001;
    DisruptionMitigation dms(grid, config);

    ScalarField Te(grid, "Te", 1.0);
    ScalarField ne(grid, "ne", 1e20);

    dms.trigger(0.0);
    for (int i = 0; i < 100; ++i) {
        dms.advance(1e-5, i * 1e-5, Te, ne);
    }

    // Density should have increased
    ASSERT_GT(ne.max_abs(), 1e20);
}

// --- Multi-Physics Coupler Tests ---

TEST(coupler_initialization) {
    DisruptionSimConfig config;
    config.nr = 16;
    config.nz = 16;
    config.t_max = 0.0001; // Very short
    config.diagnostic_interval = 10;

    MultiPhysicsCoupler coupler(config);
    coupler.initialize();

    ASSERT_TRUE(coupler.current_phase() == DisruptionPhase::MHD_EQUILIBRIUM);
    ASSERT_NEAR(coupler.time(), 0.0, 1e-15);
}

TEST(coupler_single_step) {
    DisruptionSimConfig config;
    config.nr = 16;
    config.nz = 16;
    config.t_max = 1.0;

    MultiPhysicsCoupler coupler(config);
    coupler.initialize();

    bool should_continue = coupler.step();
    ASSERT_TRUE(should_continue);
    ASSERT_GT(coupler.time(), 0.0);
    ASSERT_TRUE(coupler.step_count() == 1);
}

TEST(phase_to_string_test) {
    ASSERT_TRUE(std::string(phase_to_string(DisruptionPhase::MHD_EQUILIBRIUM))
                == "MHD_EQUILIBRIUM");
    ASSERT_TRUE(std::string(phase_to_string(DisruptionPhase::THERMAL_QUENCH))
                == "THERMAL_QUENCH");
    ASSERT_TRUE(std::string(phase_to_string(DisruptionPhase::RUNAWAY_PLATEAU))
                == "RUNAWAY_PLATEAU");
}

// ============ Test Runner ============

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        std::cout << "\n=== Tokamak Simulation Unit Tests ===\n" << std::endl;
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
        std::cout << "\n=== Results: " << tests_passed << "/" << tests_run
                  << " passed";
        if (tests_failed > 0)
            std::cout << " (" << tests_failed << " failed)";
        std::cout << " ===" << std::endl;
    }

    MPI_Finalize();
    return tests_failed > 0 ? 1 : 0;
}
