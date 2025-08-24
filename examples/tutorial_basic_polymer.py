"""
Comprehensive Polymer Simulation Tutorial
==========================================

This tutorial demonstrates the essential concepts of polymer simulations in polychrom,
walking through each step with detailed explanations of the physics and parameters.

This example creates a polymer chain confined to a spherical volume (representing 
a cell nucleus) and demonstrates:
1. Setting up the simulation environment
2. Creating initial polymer conformations
3. Adding physical forces (connectivity, confinement, repulsion)
4. Running the simulation and collecting data
5. Basic analysis of results

Learning objectives:
- Understand the role of each simulation component
- Learn how forces work together to create realistic polymer behavior
- Practice parameter tuning for stable simulations
- Analyze simulation results

Physical system modeled:
- A chromatin fiber (1000 monomers, ~2 Mb at 2 kb/monomer resolution)
- Confined in a spherical nucleus
- Subject to thermal fluctuations
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Import polychrom modules
import polychrom
from polychrom import forcekits, forces, simulation, starting_conformations
from polychrom.hdf5_format import HDF5Reporter
import polychrom.polymer_analyses as polymer_analyses

print("Starting Polychrom Tutorial: Confined Polymer Simulation")
print("=" * 60)

# ============================================================================
# PART 1: SIMULATION SETUP AND PARAMETERS
# ============================================================================

print("\n1. Setting up simulation parameters...")

# System size: number of monomers
# In chromatin simulations, each monomer typically represents:
# - 1-10 kb of DNA (nucleosome resolution)
# - 10-100 kb of DNA (coarse-grained chromatin)
# Here we use 1000 monomers representing ~2 Mb of chromatin
N = 1000

# Physical parameters (these create realistic polymer behavior)
collision_rate = 0.03  # Langevin friction coefficient (1/time units)
temperature = 1.0      # Temperature in simulation units (kT = 1)
bond_length = 1.0      # Rest length of polymer bonds
bond_flexibility = 0.05  # How much bonds can stretch (smaller = stiffer)
chain_stiffness = 1.5    # Angular stiffness (higher = less flexible backbone)

# Confinement parameters
nucleus_density = 0.85   # Volume fraction of nucleus occupied by polymer
confinement_strength = 1.0  # How strongly polymer is pushed back from boundary

# Simulation length
num_blocks = 20          # Number of data collection points
steps_per_block = 100    # MD steps between each data point

print(f"  • System size: {N} monomers")
print(f"  • Collision rate: {collision_rate}")
print(f"  • Bond flexibility: {bond_flexibility}")
print(f"  • Chain stiffness: {chain_stiffness}")
print(f"  • Nucleus density: {nucleus_density}")

# ============================================================================
# PART 2: DATA STORAGE SETUP
# ============================================================================

print("\n2. Setting up data storage...")

# Create reporter for saving simulation trajectory
# This saves all polymer coordinates at regular intervals
reporter = HDF5Reporter(
    folder="tutorial_trajectory",    # Folder name for output
    max_data_length=num_blocks + 5,  # Maximum number of frames to store
    overwrite=True                   # Overwrite existing data
)

print(f"  • Output folder: tutorial_trajectory")
print(f"  • Will save {num_blocks} trajectory frames")

# ============================================================================  
# PART 3: SIMULATION OBJECT CREATION
# ============================================================================

print("\n3. Creating simulation object...")

# The Simulation object controls all aspects of the molecular dynamics
sim = simulation.Simulation(
    platform="CPU",                    # Use CPU (change to "CUDA" for GPU)
    integrator="variableLangevin",      # Adaptive timestep integrator
    error_tol=0.003,                    # Integration accuracy (smaller = more accurate)
    collision_rate=collision_rate,      # Thermal friction
    N=N,                               # Number of particles
    save_decimals=2,                   # Coordinate precision when saving
    PBCbox=False,                      # No periodic boundary conditions
    reporters=[reporter],              # Where to save data
    temperature=temperature
)

print(f"  • Platform: CPU (for compatibility)")
print(f"  • Integrator: Variable Langevin (adaptive timestep)")
print(f"  • Integration error tolerance: {sim.error_tol}")

# ============================================================================
# PART 4: INITIAL POLYMER CONFORMATION
# ============================================================================

print("\n4. Creating initial polymer conformation...")

# Create initial polymer configuration
# grow_cubic creates a self-avoiding random walk in a cubic box
initial_conformation = starting_conformations.grow_cubic(
    N=N,           # Number of monomers
    boxSize=50     # Size of box for initial random walk
)

# Load the initial conformation into the simulation
# center=True places the center of mass at the origin
sim.set_data(initial_conformation, center=True)

print(f"  • Generated self-avoiding random walk")
print(f"  • Initial box size: 50 units")
print(f"  • Polymer centered at origin")

# Calculate initial polymer statistics
initial_rg = polymer_analyses.radius_of_gyration(initial_conformation[np.newaxis, :, :])
initial_end_to_end = np.linalg.norm(initial_conformation[-1] - initial_conformation[0])

print(f"  • Initial radius of gyration: {initial_rg[0]:.2f}")
print(f"  • Initial end-to-end distance: {initial_end_to_end:.2f}")

# ============================================================================
# PART 5: ADDING FORCES - THE PHYSICS OF THE SIMULATION
# ============================================================================

print("\n5. Adding physical forces...")

# FORCE 1: Spherical Confinement
# This represents the nuclear boundary, keeping chromatin inside the nucleus
print("  • Adding spherical confinement (nuclear boundary)...")

confinement_force = forces.spherical_confinement(
    sim,
    density=nucleus_density,        # How much of nucleus volume is filled
    k=confinement_strength         # Force strength at boundary
)
sim.add_force(confinement_force)

# Calculate the nucleus radius from density and polymer size
# This assumes each monomer occupies a unit volume
nucleus_radius = (3 * N / (4 * np.pi * nucleus_density)) ** (1/3)
print(f"    - Nucleus radius: {nucleus_radius:.2f} units")
print(f"    - Volume fraction filled: {nucleus_density}")

# FORCE 2: Polymer Chain Connectivity and Interactions
# This forcekit adds three essential components:
# 1. Bonds: keep adjacent monomers connected
# 2. Angles: provide chain stiffness
# 3. Non-bonded interactions: prevent overlap and provide realistic packing
print("  • Adding polymer connectivity forces...")

polymer_forcekit = forcekits.polymer_chains(
    sim,
    chains=[(0, None, False)],       # One chain from monomer 0 to end, not a ring
    
    # Bond force: harmonic springs between adjacent monomers
    bond_force_func=forces.harmonic_bonds,
    bond_force_kwargs={
        "bondLength": bond_length,           # Rest length
        "bondWiggleDistance": bond_flexibility  # Fluctuation amplitude
    },
    
    # Angle force: provides persistence length and chain stiffness
    angle_force_func=forces.angle_force,
    angle_force_kwargs={
        "k": chain_stiffness,  # Angular spring constant
    },
    
    # Non-bonded force: repulsion between non-adjacent monomers
    nonbonded_force_func=forces.polynomial_repulsive,
    nonbonded_force_kwargs={
        "trunc": 3.0,         # Cutoff distance for repulsion
        "radiusMult": 1.0     # Effective monomer radius
    },
    
    except_bonds=True,  # Don't apply repulsion to bonded neighbors
)
sim.add_force(polymer_forcekit)

print(f"    - Bond rest length: {bond_length}")
print(f"    - Bond flexibility: {bond_flexibility}")
print(f"    - Angular stiffness: {chain_stiffness}")
print(f"    - Repulsion cutoff: 3.0 units")

print(f"\nTotal forces added: {len(sim.force_dict)}")

# ============================================================================
# PART 6: RUNNING THE SIMULATION
# ============================================================================

print("\n6. Running simulation...")
print("  • This may take a few minutes...")

# Store trajectory data for analysis
radii_of_gyration = []
end_to_end_distances = []

# Run simulation in blocks
for block in range(num_blocks):
    # Run molecular dynamics for steps_per_block time steps
    sim.do_block(steps_per_block)
    
    # Get current polymer configuration
    current_pos = sim.get_data()[0]
    
    # Calculate polymer statistics
    current_rg = polymer_analyses.radius_of_gyration(current_pos[np.newaxis, :, :])
    current_e2e = np.linalg.norm(current_pos[-1] - current_pos[0])
    
    radii_of_gyration.append(current_rg[0])
    end_to_end_distances.append(current_e2e)
    
    # Progress update
    if (block + 1) % 5 == 0:
        print(f"    - Completed block {block + 1}/{num_blocks}, Rg = {current_rg[0]:.2f}")

# Print final simulation statistics
print("\n  • Simulation completed!")
sim.print_stats()

# Finalize data saving
reporter.dump_data()
print("  • Trajectory data saved to 'tutorial_trajectory' folder")

# ============================================================================
# PART 7: ANALYSIS AND VISUALIZATION
# ============================================================================

print("\n7. Analyzing results...")

# Convert lists to numpy arrays for analysis
radii_of_gyration = np.array(radii_of_gyration)
end_to_end_distances = np.array(end_to_end_distances)

# Calculate equilibrium statistics (skip first few blocks for equilibration)
equilibration_blocks = 5
equilibrium_rg = radii_of_gyration[equilibration_blocks:]
equilibrium_e2e = end_to_end_distances[equilibration_blocks:]

print(f"  • Average radius of gyration: {np.mean(equilibrium_rg):.2f} ± {np.std(equilibrium_rg):.2f}")
print(f"  • Average end-to-end distance: {np.mean(equilibrium_e2e):.2f} ± {np.std(equilibrium_e2e):.2f}")
print(f"  • Nucleus radius: {nucleus_radius:.2f}")
print(f"  • Polymer fits comfortably: {'Yes' if np.mean(equilibrium_rg) < nucleus_radius * 0.8 else 'No'}")

# Theoretical predictions for ideal chain
# For a freely jointed chain: <R_g^2> = N * b^2 / 6, <R_ee^2> = N * b^2
theoretical_rg = np.sqrt(N * bond_length**2 / 6)
theoretical_e2e = np.sqrt(N * bond_length**2)

print(f"\n  • Theoretical ideal chain Rg: {theoretical_rg:.2f}")
print(f"  • Theoretical ideal chain end-to-end: {theoretical_e2e:.2f}")
print(f"  • Confinement effect on Rg: {np.mean(equilibrium_rg)/theoretical_rg:.2f}x")

# ============================================================================
# PART 8: CREATING PLOTS
# ============================================================================

print("\n8. Creating analysis plots...")

try:
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Radius of gyration over time
    ax1.plot(radii_of_gyration, 'b-', linewidth=2, label='Simulation')
    ax1.axhline(y=theoretical_rg, color='r', linestyle='--', label='Ideal chain')
    ax1.axhline(y=nucleus_radius, color='k', linestyle=':', label='Nucleus radius')
    ax1.set_xlabel('Block')
    ax1.set_ylabel('Radius of Gyration')
    ax1.set_title('Polymer Size Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: End-to-end distance over time
    ax2.plot(end_to_end_distances, 'g-', linewidth=2, label='Simulation')
    ax2.axhline(y=theoretical_e2e, color='r', linestyle='--', label='Ideal chain')
    ax2.set_xlabel('Block')
    ax2.set_ylabel('End-to-End Distance')
    ax2.set_title('Chain Extension Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Histogram of Rg values (equilibrium distribution)
    ax3.hist(equilibrium_rg, bins=15, alpha=0.7, color='blue', density=True)
    ax3.axvline(x=np.mean(equilibrium_rg), color='red', linestyle='-', linewidth=2, label=f'Mean: {np.mean(equilibrium_rg):.2f}')
    ax3.axvline(x=theoretical_rg, color='orange', linestyle='--', linewidth=2, label=f'Ideal: {theoretical_rg:.2f}')
    ax3.set_xlabel('Radius of Gyration')
    ax3.set_ylabel('Probability Density')
    ax3.set_title('Rg Distribution (Equilibrium)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: 2D projection of final polymer conformation
    final_pos = sim.get_data()[0]
    ax4.plot(final_pos[:, 0], final_pos[:, 1], 'o-', linewidth=1, markersize=2, alpha=0.7)
    ax4.plot(final_pos[0, 0], final_pos[0, 1], 'go', markersize=8, label='Start')
    ax4.plot(final_pos[-1, 0], final_pos[-1, 1], 'ro', markersize=8, label='End')
    
    # Draw nucleus boundary
    circle = plt.Circle((0, 0), nucleus_radius, fill=False, linestyle='--', color='black', linewidth=2)
    ax4.add_patch(circle)
    ax4.set_xlim(-nucleus_radius*1.2, nucleus_radius*1.2)
    ax4.set_ylim(-nucleus_radius*1.2, nucleus_radius*1.2)
    ax4.set_xlabel('X coordinate')
    ax4.set_ylabel('Y coordinate')
    ax4.set_title('Final Polymer Conformation (2D projection)')
    ax4.legend()
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tutorial_analysis.png', dpi=150, bbox_inches='tight')
    print("  • Analysis plots saved as 'tutorial_analysis.png'")
    
except ImportError:
    print("  • Matplotlib not available, skipping plots")
except Exception as e:
    print(f"  • Error creating plots: {e}")

# ============================================================================
# PART 9: SUMMARY AND NEXT STEPS
# ============================================================================

print("\n" + "=" * 60)
print("TUTORIAL SUMMARY")
print("=" * 60)

print(f"✓ Successfully simulated a {N}-monomer polymer chain")
print(f"✓ Polymer equilibrated within nucleus (Rg = {np.mean(equilibrium_rg):.2f})")
print(f"✓ Confinement compressed polymer by {(1 - np.mean(equilibrium_rg)/theoretical_rg)*100:.1f}%")
print(f"✓ Saved {num_blocks} trajectory frames for further analysis")

print("\nKey files created:")
print("  • tutorial_trajectory/: Complete simulation trajectory")
print("  • tutorial_analysis.png: Analysis plots (if matplotlib available)")

print("\nNext steps to explore:")
print("  1. Try different confinement densities (0.1 to 2.0)")
print("  2. Vary chain stiffness to model different chromatin states")
print("  3. Add loop extrusion forces (see examples/loopExtrusion/)")
print("  4. Generate contact maps (see polychrom.contactmaps)")
print("  5. Run longer simulations for better statistics")

print("\nParameters to experiment with:")
print("  • nucleus_density: Controls nuclear crowding")
print("  • chain_stiffness: Affects persistence length")
print("  • bond_flexibility: Controls local chain dynamics")
print("  • num_blocks: Longer simulations for equilibrium statistics")

print("\nHappy simulating with polychrom! 🧬")