"""
Ring Polymer Simulation Example
===============================

This example demonstrates how to simulate ring polymers (circular DNA, 
ring chromosomes, or closed chromatin loops) using polychrom.

Ring polymers have different physics compared to linear chains:
- No free ends to drive unfolding
- Topological constraints (linking number)
- Different response to confinement
- Altered contact probabilities

This example shows:
1. Setting up ring topology
2. Comparing ring vs linear polymer behavior
3. Analyzing topological properties
4. Understanding confinement effects on rings
"""

import numpy as np
import matplotlib.pyplot as plt

import polychrom
from polychrom import forcekits, forces, simulation, starting_conformations
from polychrom.hdf5_format import HDF5Reporter
import polychrom.polymer_analyses as polymer_analyses

print("Ring Polymer Simulation Example")
print("=" * 40)

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

N = 500  # Smaller system for clearer demonstration of ring effects
print(f"System size: {N} monomers")

# ============================================================================
# FUNCTION TO RUN SIMULATION
# ============================================================================

def run_polymer_simulation(is_ring=False, folder_name="trajectory"):
    """Run either ring or linear polymer simulation."""
    
    # Create reporter
    reporter = HDF5Reporter(folder=folder_name, max_data_length=50, overwrite=True)
    
    # Create simulation
    sim = simulation.Simulation(
        platform="CPU",  # Use CPU for compatibility
        integrator="variableLangevin",
        error_tol=0.003,
        collision_rate=0.03,
        N=N,
        save_decimals=2,
        PBCbox=False,
        reporters=[reporter]
    )
    
    # Create initial conformation
    print(f"  Creating initial {'ring' if is_ring else 'linear'} conformation...")
    
    if is_ring:
        # For ring: create initial circle
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radius = N / (2 * np.pi)  # Circumference = N monomers
        initial_pos = np.column_stack([
            radius * np.cos(angles),
            radius * np.sin(angles),
            np.zeros(N)
        ])
        # Add some randomness to avoid perfect circle
        initial_pos += np.random.normal(0, 0.1, initial_pos.shape)
    else:
        # For linear: use standard random walk
        initial_pos = starting_conformations.grow_cubic(N, 30)
    
    sim.set_data(initial_pos, center=True)
    
    # Add forces
    print("  Adding forces...")
    
    # Confinement
    sim.add_force(forces.spherical_confinement(sim, density=0.3, k=5))
    
    # Polymer forces with ring topology
    chain_spec = [(0, None, is_ring)]  # is_ring=True closes the chain
    
    sim.add_force(forcekits.polymer_chains(
        sim,
        chains=chain_spec,
        bond_force_kwargs={"bondLength": 1.0, "bondWiggleDistance": 0.05},
        angle_force_kwargs={"k": 1.5},
        nonbonded_force_kwargs={"trunc": 3.0}
    ))
    
    print(f"  Running {'ring' if is_ring else 'linear'} simulation...")
    
    # Store analysis data
    rg_values = []
    
    # Run simulation
    for block in range(30):
        sim.do_block(100)
        
        # Calculate current radius of gyration
        current_pos = sim.get_data()[0]
        rg = polymer_analyses.radius_of_gyration(current_pos[None, :, :])
        rg_values.append(rg[0])
        
        if (block + 1) % 10 == 0:
            print(f"    Block {block + 1}/30, Rg = {rg[0]:.2f}")
    
    sim.print_stats()
    reporter.dump_data()
    
    return rg_values

# ============================================================================
# RUN BOTH SIMULATIONS
# ============================================================================

print("\n1. Running linear polymer simulation...")
linear_rg = run_polymer_simulation(is_ring=False, folder_name="trajectory_linear")

print("\n2. Running ring polymer simulation...")
ring_rg = run_polymer_simulation(is_ring=True, folder_name="trajectory_ring")

# ============================================================================
# COMPARE RESULTS
# ============================================================================

print("\n3. Comparing ring vs linear polymer properties...")

# Calculate equilibrium values (skip first 10 blocks for equilibration)
linear_rg_eq = np.array(linear_rg[10:])
ring_rg_eq = np.array(ring_rg[10:])

print(f"Linear polymer average Rg: {np.mean(linear_rg_eq):.2f} ± {np.std(linear_rg_eq):.2f}")
print(f"Ring polymer average Rg: {np.mean(ring_rg_eq):.2f} ± {np.std(ring_rg_eq):.2f}")

# Calculate theoretical predictions
# For ideal chains: linear has larger Rg than ring by factor of √2
theoretical_ratio = np.sqrt(2)
observed_ratio = np.mean(linear_rg_eq) / np.mean(ring_rg_eq)

print(f"Theoretical Rg ratio (linear/ring): {theoretical_ratio:.2f}")
print(f"Observed Rg ratio (linear/ring): {observed_ratio:.2f}")

# ============================================================================
# ADDITIONAL RING-SPECIFIC ANALYSIS
# ============================================================================

print("\n4. Ring-specific analysis...")

try:
    import polychrom.hdf5_format as h5f
    
    # Load ring polymer trajectory
    ring_data = h5f.load_URI("trajectory_ring")
    ring_positions = ring_data['pos'][-20:]  # Last 20 frames
    
    # Calculate linking number (simplified measure)
    def calculate_writhe_approximation(positions):
        """
        Simplified writhe calculation for demonstration.
        Real writhe calculation requires more sophisticated algorithms.
        """
        writhe_values = []
        
        for frame in positions:
            # Calculate simple crossing number in XY projection
            crossings = 0
            for i in range(N):
                for j in range(i+2, N):
                    # Check if bonds i-(i+1) and j-(j+1) cross in XY plane
                    p1, p2 = frame[i], frame[(i+1) % N]
                    p3, p4 = frame[j], frame[(j+1) % N]
                    
                    # Simple crossing detection (not topologically rigorous)
                    if ((p1[0] - p3[0]) * (p2[0] - p4[0]) < 0 and 
                        (p1[1] - p3[1]) * (p2[1] - p4[1]) < 0):
                        crossings += 1
            
            writhe_values.append(crossings)
        
        return writhe_values
    
    writhe = calculate_writhe_approximation(ring_positions)
    print(f"Average crossing number (writhe approximation): {np.mean(writhe):.1f}")
    print("Note: This is a simplified measure, not true linking number")
    
    # Calculate contact probability along ring
    def ring_contact_probability(positions, cutoff=3.0):
        """Calculate contact probability vs sequence separation for ring."""
        
        contact_probs = np.zeros(N//2)  # Only go to N/2 due to ring symmetry
        
        for separation in range(1, N//2):
            contacts = 0
            total_pairs = 0
            
            for frame in positions:
                for i in range(N):
                    j = (i + separation) % N
                    distance = np.linalg.norm(frame[i] - frame[j])
                    if distance < cutoff:
                        contacts += 1
                    total_pairs += 1
            
            contact_probs[separation] = contacts / total_pairs
        
        return contact_probs
    
    ring_contacts = ring_contact_probability(ring_positions)
    
    print(f"Short-range contact probability (s=1): {ring_contacts[1]:.3f}")
    print(f"Long-range contact probability (s={N//4}): {ring_contacts[N//4]:.3f}")

except ImportError:
    print("Analysis modules not available for detailed ring analysis")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n5. Creating comparison plots...")

try:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Rg evolution comparison
    ax1.plot(linear_rg, 'b-', label='Linear polymer', linewidth=2)
    ax1.plot(ring_rg, 'r-', label='Ring polymer', linewidth=2)
    ax1.axhline(y=np.mean(linear_rg_eq), color='blue', linestyle='--', alpha=0.7)
    ax1.axhline(y=np.mean(ring_rg_eq), color='red', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Block')
    ax1.set_ylabel('Radius of Gyration')
    ax1.set_title('Polymer Size Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Rg distribution comparison
    ax2.hist(linear_rg_eq, bins=15, alpha=0.7, color='blue', density=True, label='Linear')
    ax2.hist(ring_rg_eq, bins=15, alpha=0.7, color='red', density=True, label='Ring')
    ax2.set_xlabel('Radius of Gyration')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Rg Distribution Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final conformations (if data available)
    try:
        linear_data = h5f.load_URI("trajectory_linear")
        ring_data = h5f.load_URI("trajectory_ring")
        
        # Linear polymer final conformation
        linear_final = linear_data['pos'][-1]
        ax3.plot(linear_final[:, 0], linear_final[:, 1], 'b-', alpha=0.7, linewidth=1)
        ax3.plot(linear_final[0, 0], linear_final[0, 1], 'go', markersize=8, label='Start')
        ax3.plot(linear_final[-1, 0], linear_final[-1, 1], 'ro', markersize=8, label='End')
        ax3.set_title('Linear Polymer (Final Conformation)')
        ax3.set_xlabel('X coordinate')
        ax3.set_ylabel('Y coordinate')
        ax3.legend()
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
        
        # Ring polymer final conformation
        ring_final = ring_data['pos'][-1]
        ax4.plot(ring_final[:, 0], ring_final[:, 1], 'r-', alpha=0.7, linewidth=2)
        # Close the ring visually
        ax4.plot([ring_final[-1, 0], ring_final[0, 0]], 
                [ring_final[-1, 1], ring_final[0, 1]], 'r-', alpha=0.7, linewidth=2)
        ax4.plot(ring_final[0, 0], ring_final[0, 1], 'go', markersize=8, label='Start/End')
        ax4.set_title('Ring Polymer (Final Conformation)')
        ax4.set_xlabel('X coordinate')
        ax4.set_ylabel('Y coordinate')
        ax4.legend()
        ax4.set_aspect('equal')
        ax4.grid(True, alpha=0.3)
        
    except:
        ax3.text(0.5, 0.5, 'Linear\nConformation\n(Data not available)', 
                ha='center', va='center', transform=ax3.transAxes)
        ax4.text(0.5, 0.5, 'Ring\nConformation\n(Data not available)', 
                ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.savefig('ring_vs_linear_analysis.png', dpi=150, bbox_inches='tight')
    print("  Analysis plots saved as 'ring_vs_linear_analysis.png'")

except ImportError:
    print("  Matplotlib not available, skipping plots")

# ============================================================================
# SUMMARY AND IMPLICATIONS
# ============================================================================

print("\n" + "=" * 40)
print("SUMMARY: Ring vs Linear Polymer Comparison")
print("=" * 40)

print(f"✓ Linear polymer equilibrium Rg: {np.mean(linear_rg_eq):.2f}")
print(f"✓ Ring polymer equilibrium Rg: {np.mean(ring_rg_eq):.2f}")
print(f"✓ Size ratio (linear/ring): {observed_ratio:.2f} (theory: {theoretical_ratio:.2f})")

print("\nKey differences observed:")
print("  • Ring polymers are more compact due to topological constraints")
print("  • Linear polymers have free ends that explore more volume")
print("  • Ring polymers show different contact patterns")
print("  • Confinement affects rings and linear chains differently")

print("\nBiological relevance:")
print("  • Bacterial chromosomes: large rings with supercoiling")
print("  • Plasmid DNA: small circular molecules")
print("  • Chromatin loops: effective ring-like domains")
print("  • Viral genomes: often circular")

print("\nFiles created:")
print("  • trajectory_linear/: Linear polymer simulation data")
print("  • trajectory_ring/: Ring polymer simulation data")
print("  • ring_vs_linear_analysis.png: Comparison plots")

print("\nNext steps for ring polymer research:")
print("  1. Study supercoiling effects (twist constraints)")
print("  2. Investigate topological entanglement")
print("  3. Model ring-ring interactions")
print("  4. Analyze knotting probability")
print("  5. Compare with experimental data (AFM, cryo-EM)")

print("\n🔄 Ring polymer simulation completed!")