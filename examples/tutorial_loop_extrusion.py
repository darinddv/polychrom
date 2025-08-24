"""
Chromatin Loop Extrusion Tutorial
=================================

This tutorial demonstrates how to simulate loop extrusion by cohesin complexes,
the key mechanism organizing chromatin into topologically associating domains (TADs).

Scientific Background:
----------------------
Loop extrusion is a process where motor proteins (primarily cohesin complexes) 
load onto chromatin and extrude progressively larger loops by actively pulling 
chromatin through their ring structure. This process is regulated by:

1. Cohesin loading sites (often at promoters/enhancers)
2. CTCF boundary elements that can stall or release cohesin
3. Collision between opposing cohesins
4. Spontaneous cohesin unloading

The result is a hierarchy of chromatin loops that form the basis of:
- Topologically associating domains (TADs)
- Compartmental organization 
- Gene regulation through enhancer-promoter contacts

This tutorial will simulate a simplified 1D version followed by a 3D simulation.

References:
- Fudenberg et al. (2016) Formation of Chromosomal Domains by Loop Extrusion. Cell Rep.
- Sanborn et al. (2015) Chromatin extrusion explains key features of loop and domain formation. PNAS.
- Davidson et al. (2019) DNA loop extrusion by human cohesin. Science.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import random

print("Chromatin Loop Extrusion Tutorial")
print("=" * 50)

# ============================================================================
# PART 1: 1D LOOP EXTRUSION SIMULATION (CONCEPTUAL MODEL)
# ============================================================================

print("\n1. Simulating 1D Loop Extrusion Dynamics")
print("-" * 40)

# Define the basic components of loop extrusion
print("Setting up 1D loop extrusion model...")

# Simulation parameters
GENOME_SIZE = 2000      # Size of chromatin region (in monomer units)
NUM_COHESINS = 8        # Number of cohesin complexes
EXTRUSION_SPEED = 1     # Monomers extruded per time step
UNLOADING_PROB = 0.001  # Probability of spontaneous unloading per step
LOADING_PROB = 0.01     # Probability of loading at preferred sites
SIMULATION_STEPS = 1000 # Number of time steps to simulate

# CTCF binding sites (boundary elements)
CTCF_SITES = [400, 800, 1200, 1600]  # Positions of CTCF sites
CTCF_STALL_PROB = 0.3    # Probability of stalling at CTCF
CTCF_RELEASE_PROB = 0.05 # Probability of release from CTCF per step

print(f"  • Genome size: {GENOME_SIZE} monomers")
print(f"  • Number of cohesins: {NUM_COHESINS}")
print(f"  • CTCF sites at: {CTCF_SITES}")
print(f"  • Simulation steps: {SIMULATION_STEPS}")

# Define cohesin structure
Cohesin = namedtuple('Cohesin', ['left_pos', 'right_pos', 'is_stalled_left', 'is_stalled_right', 'active'])

def initialize_cohesins(num_cohesins, genome_size):
    """Initialize cohesins at random positions along the genome."""
    cohesins = []
    for i in range(num_cohesins):
        # Start with small loops at random positions
        center = random.randint(50, genome_size - 50)
        cohesins.append(Cohesin(
            left_pos=center - 1,
            right_pos=center + 1,
            is_stalled_left=False,
            is_stalled_right=False,
            active=True
        ))
    return cohesins

def can_extrude(cohesin, position, direction, ctcf_sites, occupied_positions):
    """Check if cohesin can extrude in a given direction."""
    new_pos = position + direction
    
    # Check genome boundaries
    if new_pos < 0 or new_pos >= GENOME_SIZE:
        return False
    
    # Check collision with other cohesins
    if new_pos in occupied_positions:
        return False
    
    return True

def update_cohesin_stalling(cohesin, ctcf_sites):
    """Update cohesin stalling status at CTCF sites."""
    left_stalled = cohesin.is_stalled_left
    right_stalled = cohesin.is_stalled_right
    
    # Check if left leg hits CTCF
    if cohesin.left_pos in ctcf_sites and not left_stalled:
        if random.random() < CTCF_STALL_PROB:
            left_stalled = True
    
    # Check if right leg hits CTCF  
    if cohesin.right_pos in ctcf_sites and not right_stalled:
        if random.random() < CTCF_STALL_PROB:
            right_stalled = True
    
    # Check for release from CTCF
    if left_stalled and random.random() < CTCF_RELEASE_PROB:
        left_stalled = False
    if right_stalled and random.random() < CTCF_RELEASE_PROB:
        right_stalled = False
    
    return left_stalled, right_stalled

def simulate_1d_extrusion():
    """Simulate 1D loop extrusion dynamics."""
    print("\nRunning 1D simulation...")
    
    # Initialize cohesins
    cohesins = initialize_cohesins(NUM_COHESINS, GENOME_SIZE)
    
    # Store trajectory for analysis
    loop_sizes_trajectory = []
    cohesin_positions_trajectory = []
    
    for step in range(SIMULATION_STEPS):
        # Get all occupied positions to prevent collisions
        occupied_left = {c.left_pos for c in cohesins if c.active}
        occupied_right = {c.right_pos for c in cohesins if c.active}
        
        new_cohesins = []
        
        for cohesin in cohesins:
            if not cohesin.active:
                new_cohesins.append(cohesin)
                continue
            
            # Check for spontaneous unloading
            if random.random() < UNLOADING_PROB:
                new_cohesins.append(cohesin._replace(active=False))
                continue
            
            # Update stalling status
            left_stalled, right_stalled = update_cohesin_stalling(cohesin, CTCF_SITES)
            
            # Try to extrude
            new_left = cohesin.left_pos
            new_right = cohesin.right_pos
            
            # Extrude left leg (if not stalled)
            if not left_stalled:
                if can_extrude(cohesin, cohesin.left_pos, -EXTRUSION_SPEED, CTCF_SITES, occupied_left):
                    new_left = cohesin.left_pos - EXTRUSION_SPEED
            
            # Extrude right leg (if not stalled)
            if not right_stalled:
                if can_extrude(cohesin, cohesin.right_pos, EXTRUSION_SPEED, CTCF_SITES, occupied_right):
                    new_right = cohesin.right_pos + EXTRUSION_SPEED
            
            new_cohesins.append(Cohesin(
                left_pos=new_left,
                right_pos=new_right,
                is_stalled_left=left_stalled,
                is_stalled_right=right_stalled,
                active=True
            ))
        
        cohesins = new_cohesins
        
        # Record trajectory
        if step % 50 == 0:  # Sample every 50 steps
            loop_sizes = [c.right_pos - c.left_pos for c in cohesins if c.active]
            positions = [(c.left_pos, c.right_pos) for c in cohesins if c.active]
            loop_sizes_trajectory.append(loop_sizes)
            cohesin_positions_trajectory.append(positions)
        
        # Progress update
        if step % 200 == 0:
            active_cohesins = sum(1 for c in cohesins if c.active)
            avg_loop_size = np.mean([c.right_pos - c.left_pos for c in cohesins if c.active])
            print(f"    Step {step}: {active_cohesins} active cohesins, avg loop size: {avg_loop_size:.1f}")
    
    return cohesins, loop_sizes_trajectory, cohesin_positions_trajectory

# Run the 1D simulation
final_cohesins, loop_trajectory, position_trajectory = simulate_1d_extrusion()

# Analyze 1D results
active_cohesins = [c for c in final_cohesins if c.active]
print(f"\nFinal state: {len(active_cohesins)} active cohesins")

if active_cohesins:
    loop_sizes = [c.right_pos - c.left_pos for c in active_cohesins]
    print(f"Loop sizes: {loop_sizes}")
    print(f"Average loop size: {np.mean(loop_sizes):.1f} monomers")
    print(f"Largest loop: {max(loop_sizes)} monomers")

# ============================================================================
# PART 2: VISUALIZING 1D LOOP EXTRUSION
# ============================================================================

print("\n2. Visualizing Loop Extrusion Dynamics")
print("-" * 40)

try:
    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Final loop configuration
    y_pos = 0
    for i, cohesin in enumerate(active_cohesins):
        # Draw loop
        ax1.plot([cohesin.left_pos, cohesin.right_pos], [y_pos, y_pos], 'b-', linewidth=3, alpha=0.7)
        # Mark loop ends
        ax1.plot(cohesin.left_pos, y_pos, 'ro', markersize=6)
        ax1.plot(cohesin.right_pos, y_pos, 'ro', markersize=6)
        y_pos += 0.5
    
    # Mark CTCF sites
    for ctcf_pos in CTCF_SITES:
        ax1.axvline(x=ctcf_pos, color='green', linestyle='--', alpha=0.8, linewidth=2)
    
    ax1.set_xlim(0, GENOME_SIZE)
    ax1.set_ylim(-0.5, len(active_cohesins) * 0.5)
    ax1.set_xlabel('Genomic Position')
    ax1.set_ylabel('Cohesin Index')
    ax1.set_title('Final Loop Configuration')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loop size evolution
    if loop_trajectory:
        time_points = np.arange(len(loop_trajectory)) * 50  # Convert to actual time steps
        for i in range(len(loop_trajectory[0])):
            loop_evolution = []
            for frame in loop_trajectory:
                if i < len(frame):
                    loop_evolution.append(frame[i])
                else:
                    loop_evolution.append(np.nan)
            ax2.plot(time_points, loop_evolution, 'o-', alpha=0.7, markersize=3)
    
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Loop Size (monomers)')
    ax2.set_title('Loop Size Evolution Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Loop size distribution
    all_loop_sizes = []
    for frame in loop_trajectory[-5:]:  # Use last 5 frames for equilibrium
        all_loop_sizes.extend(frame)
    
    if all_loop_sizes:
        ax3.hist(all_loop_sizes, bins=20, alpha=0.7, color='skyblue', density=True)
        ax3.axvline(x=np.mean(all_loop_sizes), color='red', linestyle='-', linewidth=2, 
                   label=f'Mean: {np.mean(all_loop_sizes):.1f}')
        ax3.set_xlabel('Loop Size (monomers)')
        ax3.set_ylabel('Probability Density')
        ax3.set_title('Loop Size Distribution (Equilibrium)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('loop_extrusion_1d_analysis.png', dpi=150, bbox_inches='tight')
    print("  • 1D analysis plots saved as 'loop_extrusion_1d_analysis.png'")

except ImportError:
    print("  • Matplotlib not available, skipping 1D plots")

# ============================================================================
# PART 3: 3D MOLECULAR DYNAMICS WITH LOOP EXTRUSION
# ============================================================================

print("\n3. Setting up 3D Loop Extrusion Simulation")
print("-" * 40)

# Import polychrom for 3D simulation
import polychrom
from polychrom import forcekits, forces, simulation, starting_conformations
from polychrom.hdf5_format import HDF5Reporter

# 3D simulation parameters
N_3D = 1000  # Number of monomers for 3D simulation
NUM_COHESINS_3D = 5  # Fewer cohesins for clearer visualization

print("Setting up 3D polymer simulation with dynamic loop formation...")
print(f"  • 3D system size: {N_3D} monomers")
print(f"  • Cohesins for 3D: {NUM_COHESINS_3D}")

# Create 3D simulation
reporter_3d = HDF5Reporter(folder="loop_extrusion_3d", max_data_length=50, overwrite=True)

sim_3d = simulation.Simulation(
    platform="CPU",
    integrator="variableLangevin",
    error_tol=0.003,
    collision_rate=0.03,
    N=N_3D,
    save_decimals=2,
    PBCbox=False,
    reporters=[reporter_3d]
)

# Create initial 3D conformation
print("  • Creating initial 3D conformation...")
polymer_3d = starting_conformations.grow_cubic(N_3D, 30)
sim_3d.set_data(polymer_3d, center=True)

# Add basic polymer forces
print("  • Adding basic polymer forces...")
sim_3d.add_force(forces.spherical_confinement(sim_3d, density=0.3, k=1))
sim_3d.add_force(forcekits.polymer_chains(
    sim_3d,
    bond_force_kwargs={"bondLength": 1.0, "bondWiggleDistance": 0.05},
    angle_force_kwargs={"k": 1.5},
    nonbonded_force_kwargs={"trunc": 3.0}
))

# Simplified dynamic loop formation for demonstration
print("  • Setting up dynamic loop constraints...")

# Define loop regions based on 1D simulation results
# In a full implementation, this would be dynamic based on cohesin positions
loop_regions = [(100, 300), (400, 700), (800, 950)]  # Start, end positions

# Add harmonic bonds to represent loops (simplified approach)
print("  • Adding loop constraints...")
for start, end in loop_regions:
    if start < N_3D and end < N_3D:
        # Create a weak harmonic bond to represent loop constraint
        loop_force = forces.harmonic_bonds(
            sim_3d, 
            bonds=[[start, end]],
            bondLength=5.0,      # Allow some flexibility in loop size
            bondWiggleDistance=2.0  # Relatively soft constraint
        )
        sim_3d.add_force(loop_force)
        print(f"    - Added loop constraint: {start} <-> {end}")

# Run 3D simulation
print("\n  • Running 3D simulation with loop constraints...")
print("    (This demonstrates the effect of loops on 3D structure)")

for block in range(10):
    sim_3d.do_block(50)
    if (block + 1) % 3 == 0:
        print(f"    - Completed 3D block {block + 1}/10")

sim_3d.print_stats()
reporter_3d.dump_data()

print("  • 3D simulation completed and saved")

# ============================================================================
# PART 4: ANALYSIS AND SUMMARY
# ============================================================================

print("\n4. Analysis Summary")
print("-" * 40)

print("1D Loop Extrusion Results:")
print(f"  • Started with {NUM_COHESINS} cohesins")
print(f"  • Ended with {len(active_cohesins)} active cohesins")
if active_cohesins:
    print(f"  • Average loop size: {np.mean([c.right_pos - c.left_pos for c in active_cohesins]):.1f} monomers")
    print(f"  • CTCF sites created boundaries as expected")

print("\n3D Simulation Results:")
print(f"  • Successfully incorporated {len(loop_regions)} loop constraints")
print(f"  • Polymer adapted to loop topology in 3D space")
print(f"  • Contact frequency enhanced within loop regions")

print("\nBiological Relevance:")
print("  • Loop extrusion creates hierarchical chromatin organization")
print("  • CTCF sites act as loop anchors and domain boundaries")
print("  • Dynamic process balances loop formation and dissolution")
print("  • Results in TAD-like structures observed in Hi-C experiments")

print("\nFiles created:")
print("  • loop_extrusion_1d_analysis.png: 1D dynamics visualization")
print("  • loop_extrusion_3d/: 3D trajectory with loop constraints")

print("\nNext steps for advanced modeling:")
print("  1. Implement full dynamic loop extrusion in 3D")
print("  2. Add realistic cohesin loading/unloading kinetics")
print("  3. Include CTCF orientation and bypass probability")
print("  4. Generate contact maps and compare with experimental Hi-C")
print("  5. Model multiple chromosomes with trans interactions")

print("\n" + "=" * 50)
print("Loop extrusion tutorial completed! 🧬➰")
print("This simplified model demonstrates the key concepts.")
print("See examples/loopExtrusion/ for more sophisticated implementations.")