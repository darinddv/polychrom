"""
Basic Polymer Simulation Example
================================

This is a well-documented example showing how to create a basic polymer simulation
in polychrom. This simulation represents a simple polymer chain (like chromatin)
confined to a spherical volume (like a cell nucleus).

The simulation demonstrates:
- Setting up a Simulation object with appropriate parameters
- Creating an initial polymer conformation
- Adding essential forces for realistic polymer behavior
- Running the simulation and collecting data
- Basic data analysis

Physical system:
- 10,000 monomers (representing ~20 Mb of chromatin at 2 kb/monomer)
- Spherical confinement (nuclear boundary)
- Thermal fluctuations at room temperature
- Realistic polymer connectivity and excluded volume

This serves as a foundation for more complex simulations involving
loop extrusion, chromatin modifications, or multi-chromosome systems.
"""

import os
import sys

import openmm

import polychrom
from polychrom import forcekits, forces, simulation, starting_conformations
from polychrom.hdf5_format import HDF5Reporter

print("Starting Basic Polymer Simulation")
print("=" * 40)

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

# System size: Number of monomers in the polymer chain
# In chromatin modeling, each monomer typically represents 1-10 kb of DNA
N = 10000
print(f"System size: {N} monomers")

# ============================================================================
# DATA STORAGE SETUP
# ============================================================================

# HDF5Reporter handles saving simulation trajectories
# max_data_length: maximum number of frames to store in memory before writing to disk
# overwrite: whether to overwrite existing trajectory data
reporter = HDF5Reporter(folder="trajectory", max_data_length=5, overwrite=True)
print("Data will be saved to 'trajectory/' folder")

# ============================================================================
# SIMULATION OBJECT CREATION
# ============================================================================

# The Simulation class controls all aspects of the molecular dynamics simulation
sim = simulation.Simulation(
    platform="CUDA",                    # Use GPU acceleration (change to "CPU" if no GPU available)
    integrator="variableLangevin",       # Variable timestep Langevin integrator (adaptive)
    error_tol=0.003,                     # Integration error tolerance (smaller = more accurate)
    GPU="1",                             # GPU device index (use "0" for first GPU)
    collision_rate=0.03,                 # Langevin collision rate (controls friction/temperature)
    N=N,                                 # Number of particles
    save_decimals=2,                     # Decimal precision when saving coordinates
    PBCbox=False,                        # No periodic boundary conditions (confined system)
    reporters=[reporter],                # List of reporters for data collection
)

print(f"Simulation platform: {sim.platform}")
print(f"Integration error tolerance: {sim.error_tol}")

# ============================================================================
# INITIAL CONFORMATION
# ============================================================================

# Create initial polymer conformation using a self-avoiding random walk
# grow_cubic generates a polymer that doesn't overlap with itself
# The second parameter (100) sets the size of the cubic box for generation
polymer = starting_conformations.grow_cubic(10000, 100)
print("Created initial self-avoiding random walk conformation")

# Load the initial conformation into the simulation
# center=True places the center of mass at the origin
sim.set_data(polymer, center=True)
print("Loaded initial conformation (centered at origin)")

# ============================================================================
# FORCES: THE PHYSICS OF THE SIMULATION
# ============================================================================

print("\nAdding physical forces:")

# FORCE 1: Spherical Confinement
# Represents the nuclear boundary - keeps the polymer within a spherical volume
# density: what fraction of the nuclear volume is occupied by the polymer
# k: force constant for the confining potential
spherical_force = forces.spherical_confinement(sim, density=0.85, k=1)
sim.add_force(spherical_force)
print("  ✓ Spherical confinement (nuclear boundary)")

# FORCE 2: Polymer Chain Connectivity and Interactions
# This forcekit combines multiple essential forces for realistic polymer behavior:
#   - Bonds: harmonic springs connecting adjacent monomers
#   - Angles: bending stiffness to control chain flexibility  
#   - Non-bonded interactions: excluded volume repulsion between monomers
polymer_forces = forcekits.polymer_chains(
    sim,
    chains=[(0, None, False)],           # One chain from monomer 0 to end, not a ring
    # By default assumes one continuous polymer chain
    # For multiple chains or rings, modify this parameter
    # Example: [(0,50,True),(50,None,False)] = 50-monomer ring + chain from 50 to end
    
    # Bond force: harmonic springs between consecutive monomers
    bond_force_func=forces.harmonic_bonds,
    bond_force_kwargs={
        "bondLength": 1.0,               # Rest length of bonds (equilibrium distance)
        "bondWiggleDistance": 0.05,      # Bond fluctuation amplitude (~thermal energy scale)
    },
    
    # Angle force: controls chain stiffness and persistence length
    angle_force_func=forces.angle_force,
    angle_force_kwargs={
        "k": 1.5,                        # Angular spring constant
        # k=4 corresponds to persistence length ≈ 4 monomers
        # k=1.5 gives realistic flexibility for chromatin
        # k=8 would be very stiff (like dsDNA)
    },
    
    # Non-bonded force: excluded volume interactions
    nonbonded_force_func=forces.polynomial_repulsive,
    nonbonded_force_kwargs={
        "trunc": 3.0,                    # Cutoff distance for repulsive interactions
        # trunc=3.0: allows some chain crossing (realistic for chromatin)
        # trunc=10.0: prevents chain crossing entirely (more like synthetic polymers)
    },
    
    except_bonds=True,                   # Don't apply repulsion between bonded neighbors
)
sim.add_force(polymer_forces)
print("  ✓ Polymer connectivity (bonds, angles, excluded volume)")

print(f"\nTotal forces applied: {len(sim.force_dict)}")

# ============================================================================
# RUNNING THE SIMULATION
# ============================================================================

print("\nRunning simulation...")
print("Each block represents a data collection point")

# Run simulation in blocks
# Each block performs molecular dynamics steps and saves a trajectory frame
num_blocks = 10
steps_per_block = 100

for block in range(num_blocks):
    # Perform molecular dynamics simulation for steps_per_block time steps
    # Data is automatically saved by the reporter at the end of each block
    sim.do_block(steps_per_block)
    print(f"  Completed block {block + 1}/{num_blocks}")

print("Simulation completed!")

# ============================================================================
# SIMULATION STATISTICS AND CLEANUP
# ============================================================================

# Print basic simulation statistics
# This shows information about energy, performance, etc.
sim.print_stats()

# IMPORTANT: Always call dump_data() to finalize data writing
# This ensures all buffered data is written to disk
reporter.dump_data()
print("\nData finalization complete")

# ============================================================================
# BASIC ANALYSIS EXAMPLE
# ============================================================================

print("\nBasic Analysis:")

try:
    import polychrom.polymer_analyses as polymer_analyses
    import polychrom.hdf5_format as h5f
    
    # Load the trajectory data we just created
    trajectory_data = h5f.load_URI("trajectory")
    
    # Calculate radius of gyration for each frame
    radii_of_gyration = polymer_analyses.radius_of_gyration(trajectory_data['pos'])
    
    print(f"  Average radius of gyration: {radii_of_gyration.mean():.2f} ± {radii_of_gyration.std():.2f}")
    print(f"  Minimum Rg: {radii_of_gyration.min():.2f}")
    print(f"  Maximum Rg: {radii_of_gyration.max():.2f}")
    
    # Estimate polymer density in nucleus
    nuclear_volume = (4/3) * 3.14159 * (radii_of_gyration.mean() * 1.5)**3  # Rough estimate
    monomer_volume = N * 1.0  # Assume each monomer has unit volume
    density = monomer_volume / nuclear_volume
    print(f"  Estimated nuclear density: {density:.3f}")
    
except ImportError:
    print("  Analysis modules not available")
except Exception as e:
    print(f"  Analysis error: {e}")

# ============================================================================
# SUMMARY AND NEXT STEPS
# ============================================================================

print("\n" + "=" * 40)
print("SIMULATION SUMMARY")
print("=" * 40)

print(f"✓ Successfully simulated {N} monomers for {num_blocks} blocks")
print(f"✓ Trajectory saved in 'trajectory/' folder")
print(f"✓ Data format: HDF5 (compatible with polychrom analysis tools)")

print("\nOutput files:")
print("  • trajectory/blocks.h5: Main trajectory data")
print("  • trajectory/*.json: Force parameters and metadata")

print("\nNext steps to explore:")
print("  1. Analyze contact maps: see polychrom.contactmaps")
print("  2. Calculate polymer statistics: see polychrom.polymer_analyses")
print("  3. Try loop extrusion: see examples/loopExtrusion/")
print("  4. Visualize conformations: see polychrom.pymol_show")
print("  5. Customize forces: see polychrom.forces documentation")

print("\nParameters to experiment with:")
print("  • density in spherical_confinement: controls nuclear crowding")
print("  • k in angle_force: controls chain stiffness/persistence length")
print("  • trunc in polynomial_repulsive: controls chain crossing behavior")
print("  • bondWiggleDistance: controls local chain dynamics")

print("\nHappy simulating with polychrom! 🧬")
