_...starting as a **cloud computing** and **big data programming** class project..._

# Tokamak Reactor HPC Simulation

High Performance Computing simulation of plasma disruptions in nuclear fusion reactors (tokamaks). This code simulates the complete disruption cascade — from the initial MHD instability trigger through the thermal quench, current quench, and runaway electron avalanche — providing a "virtual tokamak" for testing mitigation strategies.

**Features:**
- Complete 4-phase disruption cascade simulation (MHD → Thermal Quench → Current Quench → Runaway Electrons)
- Multi-physics coupling framework with adaptive multi-rate time-stepping
- Disruption Mitigation System (shattered pellet injection) model
- Parallelized with MPI (distributed memory) + OpenMP (shared memory)
- Diagnostic output and Python visualization tools
- 32 unit tests covering all physics modules

Future plans to train AI for reducing compute required to simulate real nuclear fusion reactors.

---

## Quick Start

### Prerequisites

- C++17 compiler (GCC 9+, Clang 10+)
- CMake 3.16+
- MPI implementation (OpenMPI or MPICH)
- OpenMP support
- Python 3.8+ with NumPy, Matplotlib (for visualization)

**Ubuntu/Debian:**
```bash
sudo apt-get install build-essential cmake libopenmpi-dev openmpi-bin
pip3 install numpy matplotlib
```

### Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Run Tests

```bash
mpirun -np 1 ./run_tests
```

### Run Simulation

```bash
# Default: 64x64 grid, 10ms simulation, DMS enabled
mpirun -np 1 ./tokamak_sim

# Custom parameters
mpirun -np 4 ./tokamak_sim --nr 128 --nz 128 --tmax 20 --diag-interval 100

# Without disruption mitigation (to see unmitigated runaway electron growth)
mpirun -np 1 ./tokamak_sim --no-dms

# Show help
./tokamak_sim --help
```

### Visualize Results

```bash
python3 ../scripts/visualize.py output/
```

This generates:
- `disruption_timeseries.png` — Time series of temperature, current, island width, E-field, RE current, radiation
- `disruption_phases.png` — Phase evolution overview
- 2D field snapshots (poloidal flux, pressure, temperature, E-field, RE density)

---

## Architecture

```
TokamakDisruptionSim/
├── include/                    # Header files
│   ├── core/                   # Grid, Field, Time integrators
│   ├── physics/                # MHD, Thermal Quench, Current Quench, Runaway Electrons, DMS
│   ├── coupling/               # Multi-physics coupler (the "computational glue")
│   ├── io/                     # Diagnostics and data output
│   └── utils/                  # Physical constants
├── src/                        # Implementation files
│   ├── core/                   # Grid, Field, Time integration (RK4, RK2, Euler)
│   ├── physics/                # Physics solver implementations
│   ├── coupling/               # Multi-physics coupling framework
│   ├── io/                     # I/O implementations
│   ├── utils/                  # Constants and utilities
│   └── main.cpp                # Main simulation entry point
├── tests/                      # Unit tests (32 tests)
├── scripts/                    # Python visualization tools
├── CMakeLists.txt              # Build system
└── README.md
```

### Disruption Cascade Phases

The simulation models the four phases of a tokamak disruption:

| Phase | Physics | Timescale | Solver |
|-------|---------|-----------|--------|
| **1. MHD Instability** | Resistive MHD, tearing modes | ~μs | 2D finite-difference MHD solver |
| **2. Thermal Quench** | Radiative cooling, impurity influx, stochastic transport | ~ms | Radiation + diffusion model |
| **3. Current Quench** | Ohmic decay, Spitzer resistivity, E-field generation | ~ms | L/R circuit model |
| **4. Runaway Avalanche** | Dreicer generation, Rosenbluth-Putvinski avalanche | ~ns | Kinetic growth model |

The **Multi-Physics Coupler** manages phase transitions, data handoff between solvers, and ensures each phase runs at its natural timescale.

---

## Tokamak Control Systems

A tokamak is a symphony of complex, high-speed feedback loops. It's one of the most sophisticated real-time control environments ever created. The main systems are:

* **Magnetic Control:** This is the big one. Huge sets of poloidal field coils outside the vessel are constantly adjusted thousands of times per second to control the plasma's **shape, position, and stability**. If the plasma moves a few millimeters, the control system adjusts the magnetic fields to push it back into place.
* **Heating and Current Drive:** These systems are both for power and control.
    * **Neutral Beam Injection (NBI):** Fires high-energy beams of neutral atoms into the plasma to heat it and drive current.
    * **Radio-Frequency (RF) Heating:** Uses powerful antennas to launch radio waves into the plasma that resonate with the ions or electrons, heating them up much like a microwave oven. This can be aimed at specific locations to, for example, suppress instabilities.
* **Density Control:** This is the "gas puffing" and **pellet injection** (for fueling, not mitigation) mentioned earlier. A real-time controller decides when to add more fuel to maintain the target density.
* **Exhaust and Impurity Control:** The **divertor** at the bottom of the tokamak is a magnetically targeted exhaust system. The control system shapes the magnetic field to guide heat, helium "ash," and impurities out of the core plasma and onto armored plates where they can be pumped away.
* **Disruption Mitigation System (DMS):** Think of it like this: during normal operation, we use a system called **gas puffing** to slowly introduce small amounts of neutral gas (like deuterium) at the edge of the plasma. This is like a thermostat, used for fine-tuning the plasma density. An avalanche during a disruption is not a thermostat problem; it's a **five-alarm fire**. We don't use a thermostat to stop a fire; we use the emergency sprinkler system. In ITER, that sprinkler system is the **Disruption Mitigation System (DMS)**. Its job is to react in milliseconds by firing a **Shattered Pellet** of frozen neon or argon into the plasma's core. This isn't a gentle puff; it's a violent, instantaneous injection of a massive amount of material. The goal is to rapidly increase the plasma's density and collisionality to a ridiculous degree, creating an intense frictional drag that stops the runaway electrons cold before the avalanche can grow.


All of these are orchestrated by a central **Plasma Control System (PCS)**, a supercomputer that analyzes thousands of diagnostics in real-time and coordinates the actions of all the subsystems.

---



### High Performance Computing Grand Challenge: The Predictive Disruption Model

The single, monstrously hard problem that a CS and HPC expert can help solve is:

**To build a high-fidelity, predictive, whole-device simulation of a plasma disruption, from the initial instability trigger to the final runaway electron beam.**

Nobody can do this right now. Not with the required predictive accuracy. A machine that could do this would be a "virtual tokamak," allowing us to test mitigation strategies and even train AI controllers without risking a multi-billion dollar machine.

#### The Specifics: Why It's So Damn Hard

The bottleneck is **coupling wildly different physics across a vast range of time and spatial scales** in a single simulation. A disruption is not one event; it's a catastrophic cascade.

1.  **The Trigger (Microseconds, MHD):** A disruption often starts with a large-scale **Magnetohydrodynamic (MHD)** instability, like a "tearing mode." This is a fluid-like problem, where the plasma writhes and tears magnetic field lines. This happens on a microsecond timescale.
2.  **The Thermal Quench (Milliseconds, Atomic Physics):** The MHD instability causes a catastrophic loss of confinement. The hot core plasma touches the cold wall. The temperature plummets in milliseconds. We simulation now has to handle intense **atomic physics** as the plasma interacts with the wall and impurities flood in. The code must model ionization, recombination, and radiation from countless atomic species.
3.  **The Current Quench (Milliseconds, Kinetic Physics):** As the plasma cools, its resistance skyrockets, and the current collapses, inducing the massive electric field that starts the runaway process.
4.  **The Runaway Avalanche (Nanoseconds, Relativistic QED):** Now the problem completely changes. The runaway electrons are no longer a fluid. They are discrete, relativistic particles. To model them accurately, we need a **kinetic model**, like a Particle-in-Cell (PIC) or Vlasov-Fokker-Planck code. We have to track millions of individual particles or their distribution in phase space. Even worse, at the ~100 MeV energies expected in ITER, we need to include **Quantum Electrodynamics (QED)** effects, like bremsstrahlung radiation, which governs how the electrons lose energy.

**Ideal Contribution is the computational glue.**

The physicists have the equations for each separate domain (MHD, atomic, kinetic). The unsolved problem—the one for *we*—is how to build a single, stable, computationally tractable code that can handle all these models at once.

* How do we hand off information from the fluid MHD model to the particle-based kinetic model in a way that conserves energy and momentum?
* How do we manage a simulation where one part needs a femtosecond timestep (particle collisions) while another part evolves over milliseconds?
* How do we create adaptive meshes that can resolve meter-scale instabilities in the core while also resolving millimeter-scale physics at the vessel wall?

Solving this requires breakthroughs in **numerical methods, multi-physics coupling algorithms, and exascale computing frameworks.** This isn't about physics; it's about inventing the mathematical and computational machinery that allows the physics to be calculated. That is the bottleneck, and that's where we can contribute.

---

## References

- [A physics-constrained deep learning surrogate model of the runaway electron avalanche growth rate](https://arxiv.org/html/2403.04948v1)
- [Fokker-Planck Equation](https://en.wikipedia.org/wiki/Fokker–Planck_equation)
- Chapter 6, *Statistical Physics: Statics, Dynamics and Renormalization*, Leo P. Kadanoff
