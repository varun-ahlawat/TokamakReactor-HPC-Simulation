#pragma once
/// @file multi_physics_coupler.h
/// @brief Multi-physics coupling framework for the disruption simulation.
///
/// This is "the computational glue" described in the README — the framework
/// that manages handoff between different physics solvers operating at
/// vastly different timescales:
///   - MHD:          ~μs timestep
///   - Thermal quench: ~ms timestep
///   - Current quench: ~ms timestep
///   - RE avalanche:   ~ns timestep (within ms window)
///
/// The coupler manages:
///   1. Phase detection and transition logic
///   2. Data transfer between physics domains
///   3. Multi-rate time-stepping
///   4. Energy/momentum conservation at interfaces

#include "core/grid.h"
#include "physics/mhd_solver.h"
#include "physics/thermal_quench.h"
#include "physics/current_quench.h"
#include "physics/runaway_electrons.h"
#include "physics/disruption_mitigation.h"
#include <memory>
#include <functional>

namespace tokamak {

/// Phases of a tokamak disruption
enum class DisruptionPhase {
    MHD_EQUILIBRIUM,    // Pre-disruption MHD evolution
    MHD_UNSTABLE,       // MHD instability growing
    THERMAL_QUENCH,     // Rapid temperature collapse
    CURRENT_QUENCH,     // Current decay with RE generation
    RUNAWAY_PLATEAU,    // RE beam plateau (if not mitigated)
    MITIGATED,          // DMS successfully suppressed disruption
    POST_DISRUPTION     // Simulation complete
};

/// Convert phase to string
const char* phase_to_string(DisruptionPhase phase);

/// Callback type for diagnostics at each step
using DiagnosticCallback = std::function<void(DisruptionPhase phase,
                                               double time, int step)>;

/// Configuration for the full disruption simulation
struct DisruptionSimConfig {
    // Grid parameters
    int nr = 128;
    int nz = 128;
    double R_min = 4.0;
    double R_max = 8.5;
    double Z_min = -4.5;
    double Z_max = 4.5;

    // Time parameters
    double t_max = 0.050;         // Maximum simulation time [s] (50 ms)
    double dt_mhd = 1.0e-7;      // MHD timestep [s]
    double dt_thermal = 1.0e-5;  // Thermal quench timestep [s]
    double dt_current = 1.0e-5;  // Current quench timestep [s]
    double dt_runaway = 1.0e-7;  // RE timestep [s]

    // Phase transition criteria
    double mhd_island_threshold = 0.1;  // Island width / minor radius for TQ trigger
    double tq_temperature_threshold = 0.1; // Temperature [keV] for CQ start
    double cq_current_threshold = 0.01;    // Current fraction for RE plateau

    // Sub-solver configurations
    MHDConfig mhd_config;
    ThermalQuenchConfig tq_config;
    CurrentQuenchConfig cq_config;
    RunawayConfig re_config;
    DMSConfig dms_config;

    // Options
    bool enable_dms = true;
    int diagnostic_interval = 100;       // Steps between diagnostics
    int output_interval = 1000;          // Steps between file outputs
};

/// Multi-physics coupler — orchestrates the entire disruption simulation
class MultiPhysicsCoupler {
public:
    explicit MultiPhysicsCoupler(const DisruptionSimConfig& config);

    /// Initialize all physics solvers
    void initialize();

    /// Run the complete disruption simulation
    void run();

    /// Run a single step of the current phase
    /// @return true if simulation should continue
    bool step();

    /// Get the current disruption phase
    DisruptionPhase current_phase() const { return phase_; }

    /// Get simulation time
    double time() const { return time_; }

    /// Get step count
    int step_count() const { return step_count_; }

    /// Register a diagnostic callback
    void set_diagnostic_callback(DiagnosticCallback cb) { diag_callback_ = cb; }

    /// Access individual solvers
    MHDSolver& mhd() { return *mhd_; }
    ThermalQuench& thermal_quench() { return *tq_; }
    CurrentQuench& current_quench() { return *cq_; }
    RunawayElectrons& runaway() { return *re_; }
    DisruptionMitigation& dms() { return *dms_; }
    const Grid& grid() const { return *grid_; }

    /// Get simulation statistics
    struct Stats {
        int total_steps = 0;
        double wall_time = 0.0;
        int mhd_steps = 0;
        int tq_steps = 0;
        int cq_steps = 0;
        int re_steps = 0;
        double peak_island_width = 0.0;
        double min_temperature = 1e10;
        double peak_E_field = 0.0;
        double peak_RE_current = 0.0;
        bool disruption_mitigated = false;
    };
    const Stats& stats() const { return stats_; }

private:
    DisruptionSimConfig config_;
    std::unique_ptr<Grid> grid_;
    std::unique_ptr<MHDSolver> mhd_;
    std::unique_ptr<ThermalQuench> tq_;
    std::unique_ptr<CurrentQuench> cq_;
    std::unique_ptr<RunawayElectrons> re_;
    std::unique_ptr<DisruptionMitigation> dms_;

    DisruptionPhase phase_ = DisruptionPhase::MHD_EQUILIBRIUM;
    double time_ = 0.0;
    int step_count_ = 0;
    Stats stats_;
    DiagnosticCallback diag_callback_;

    /// Phase transition logic
    void check_phase_transition();

    /// Transfer data between physics solvers at phase boundaries
    void transfer_mhd_to_thermal_quench();
    void transfer_thermal_quench_to_current_quench();
    void transfer_current_quench_to_runaway();

    /// Get the timestep for the current phase
    double current_dt() const;
};

} // namespace tokamak
