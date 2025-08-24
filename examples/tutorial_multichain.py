"""
Multi-Chain Polymer Simulation Example
======================================

This example demonstrates how to simulate multiple polymer chains in the same
system, representing scenarios like:
- Multiple chromosomes in a nucleus
- Separate DNA molecules
- Chromatin domains with breaks
- Polymer mixtures

This tutorial covers:
1. Setting up multiple chain topology
2. Inter-chain interactions
3. Chain segregation analysis
4. Territorial organization
5. Comparison with single-chain behavior
"""

import numpy as np
import matplotlib.pyplot as plt

import polychrom
from polychrom import forcekits, forces, simulation, starting_conformations
from polychrom.hdf5_format import HDF5Reporter
import polychrom.polymer_analyses as polymer_analyses

print("Multi-Chain Polymer Simulation Example")
print("=" * 45)

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

# System design: 3 chains of different sizes
chain_sizes = [200, 300, 250]  # Different chain lengths
total_N = sum(chain_sizes)
num_chains = len(chain_sizes)

print(f"System design: {num_chains} chains")
print(f"Chain sizes: {chain_sizes}")
print(f"Total monomers: {total_N}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_chain_topology(chain_sizes):
    """Create chain topology specification for forcekits.polymer_chains."""
    chains = []
    start = 0
    
    for size in chain_sizes:
        end = start + size
        chains.append((start, end, False))  # (start, end, is_ring)
        start = end
    
    return chains

def create_initial_multichain_conformation(chain_sizes, separation=15):
    """Create initial conformation with separated chains."""
    positions = []
    
    for i, size in enumerate(chain_sizes):
        # Create individual chain as random walk
        chain_pos = starting_conformations.grow_cubic(size, size//3)
        
        # Offset each chain to avoid initial overlap
        offset = np.array([i * separation, 0, 0])
        chain_pos += offset
        
        positions.append(chain_pos)
    
    return np.vstack(positions)

def analyze_chain_separation(positions, chain_sizes):
    """Analyze spatial separation between chains."""
    # Calculate center of mass for each chain
    com_positions = []
    start = 0
    
    for size in chain_sizes:
        end = start + size
        chain_com = np.mean(positions[start:end], axis=0)
        com_positions.append(chain_com)
        start = end
    
    # Calculate pairwise distances between chain centers
    distances = []
    for i in range(len(com_positions)):
        for j in range(i+1, len(com_positions)):
            dist = np.linalg.norm(com_positions[i] - com_positions[j])
            distances.append(dist)
    
    return np.array(distances), com_positions

def calculate_inter_chain_contacts(positions, chain_sizes, cutoff=3.0):
    """Calculate contact frequencies between different chains."""
    contact_matrix = np.zeros((num_chains, num_chains))
    
    # Define chain boundaries
    boundaries = [0]
    for size in chain_sizes:
        boundaries.append(boundaries[-1] + size)
    
    # Count contacts between all chain pairs
    for i in range(num_chains):
        for j in range(i, num_chains):
            contacts = 0
            total_pairs = 0
            
            start_i, end_i = boundaries[i], boundaries[i+1]
            start_j, end_j = boundaries[j], boundaries[j+1]
            
            for mono_i in range(start_i, end_i):
                for mono_j in range(start_j, end_j):
                    if i == j and abs(mono_i - mono_j) < 2:
                        continue  # Skip adjacent monomers in same chain
                    
                    distance = np.linalg.norm(positions[mono_i] - positions[mono_j])
                    if distance < cutoff:
                        contacts += 1
                    total_pairs += 1
            
            contact_freq = contacts / total_pairs if total_pairs > 0 else 0
            contact_matrix[i, j] = contact_freq
            contact_matrix[j, i] = contact_freq  # Symmetric
    
    return contact_matrix

# ============================================================================
# RUN SIMULATION
# ============================================================================

print("\n1. Setting up multi-chain simulation...")

# Create reporter
reporter = HDF5Reporter(folder="trajectory_multichain", max_data_length=50, overwrite=True)

# Create simulation
sim = simulation.Simulation(
    platform="CPU",  # Use CPU for compatibility
    integrator="variableLangevin",
    error_tol=0.003,
    collision_rate=0.03,
    N=total_N,
    save_decimals=2,
    PBCbox=False,
    reporters=[reporter]
)

# Create initial multi-chain conformation
print("  Creating initial multi-chain conformation...")
initial_pos = create_initial_multichain_conformation(chain_sizes)
sim.set_data(initial_pos, center=True)

# Set up chain topology
chains_topology = create_chain_topology(chain_sizes)
print(f"  Chain topology: {chains_topology}")

# Add forces
print("  Adding forces...")

# Confinement (shared by all chains)
sim.add_force(forces.spherical_confinement(sim, density=0.2, k=5))

# Multi-chain polymer forces
sim.add_force(forcekits.polymer_chains(
    sim,
    chains=chains_topology,  # Multiple chains specified here
    bond_force_kwargs={"bondLength": 1.0, "bondWiggleDistance": 0.05},
    angle_force_kwargs={"k": 1.5},
    nonbonded_force_kwargs={"trunc": 3.0}  # Inter-chain repulsion
))

print("  Running multi-chain simulation...")

# Store analysis data
chain_distances = []
chain_rg_values = []
inter_chain_contact_matrices = []

# Run simulation
for block in range(40):
    sim.do_block(100)
    
    # Get current positions
    current_pos = sim.get_data()[0]
    
    # Analyze chain separation
    distances, com_positions = analyze_chain_separation(current_pos, chain_sizes)
    chain_distances.append(distances)
    
    # Calculate radius of gyration for each chain
    rg_per_chain = []
    start = 0
    for size in chain_sizes:
        end = start + size
        chain_pos = current_pos[start:end]
        rg = polymer_analyses.radius_of_gyration(chain_pos[None, :, :])
        rg_per_chain.append(rg[0])
        start = end
    chain_rg_values.append(rg_per_chain)
    
    # Calculate inter-chain contacts (every 5 blocks to save time)
    if block % 5 == 0:
        contact_matrix = calculate_inter_chain_contacts(current_pos, chain_sizes)
        inter_chain_contact_matrices.append(contact_matrix)
    
    if (block + 1) % 10 == 0:
        avg_distance = np.mean(distances)
        print(f"    Block {block + 1}/40, avg chain separation: {avg_distance:.2f}")

sim.print_stats()
reporter.dump_data()

# ============================================================================
# ANALYSIS
# ============================================================================

print("\n2. Analyzing multi-chain behavior...")

# Convert to numpy arrays
chain_distances = np.array(chain_distances)
chain_rg_values = np.array(chain_rg_values)
inter_chain_contact_matrices = np.array(inter_chain_contact_matrices)

# Skip equilibration for analysis
eq_start = 15
eq_distances = chain_distances[eq_start:]
eq_rg_values = chain_rg_values[eq_start:]
eq_contact_matrices = inter_chain_contact_matrices[eq_start//5:]

print("Chain separation analysis:")
for i, (size1, size2) in enumerate([(chain_sizes[0], chain_sizes[1]),
                                    (chain_sizes[0], chain_sizes[2]),
                                    (chain_sizes[1], chain_sizes[2])]):
    avg_dist = np.mean(eq_distances[:, i])
    std_dist = np.std(eq_distances[:, i])
    print(f"  Chain pair {i+1} (sizes {size1}, {size2}): {avg_dist:.2f} ± {std_dist:.2f}")

print("\nIndividual chain properties:")
for i, size in enumerate(chain_sizes):
    avg_rg = np.mean(eq_rg_values[:, i])
    std_rg = np.std(eq_rg_values[:, i])
    theoretical_rg = np.sqrt(size * 1.0**2 / 6)  # Ideal chain
    print(f"  Chain {i+1} (N={size}): Rg = {avg_rg:.2f} ± {std_rg:.2f} (ideal: {theoretical_rg:.2f})")

# Inter-chain contact analysis
print("\nInter-chain contact frequencies:")
avg_contact_matrix = np.mean(eq_contact_matrices, axis=0)
for i in range(num_chains):
    for j in range(i+1, num_chains):
        contact_freq = avg_contact_matrix[i, j]
        print(f"  Chain {i+1} - Chain {j+1}: {contact_freq:.4f}")

# ============================================================================
# TERRITORIAL ORGANIZATION ANALYSIS
# ============================================================================

print("\n3. Analyzing territorial organization...")

try:
    import polychrom.hdf5_format as h5f
    
    # Load trajectory data
    data = h5f.load_URI("trajectory_multichain")
    trajectory_positions = data['pos'][-20:]  # Last 20 frames
    
    # Calculate chain territories
    def calculate_chain_territories(positions_list, chain_sizes):
        """Calculate volume occupied by each chain."""
        territories = []
        
        for positions in positions_list:
            chain_volumes = []
            start = 0
            
            for size in chain_sizes:
                end = start + size
                chain_pos = positions[start:end]
                
                # Approximate volume as convex hull volume (simplified)
                # Here we use bounding box volume as approximation
                bbox_volume = np.prod(np.ptp(chain_pos, axis=0))
                chain_volumes.append(bbox_volume)
                start = end
            
            territories.append(chain_volumes)
        
        return np.array(territories)
    
    territories = calculate_chain_territories(trajectory_positions, chain_sizes)
    avg_territories = np.mean(territories, axis=0)
    
    print("Chain territorial volumes (approximate):")
    for i, (size, volume) in enumerate(zip(chain_sizes, avg_territories)):
        print(f"  Chain {i+1} (N={size}): Volume = {volume:.2f}")
    
    # Check for territorial segregation
    total_volume = np.sum(avg_territories)
    print(f"Total occupied volume: {total_volume:.2f}")
    
    # Calculate territory overlap (simplified measure)
    def calculate_territory_overlap(positions_list, chain_sizes, grid_resolution=20):
        """Calculate spatial overlap between chain territories using grid."""
        overlap_scores = []
        
        for positions in positions_list:
            # Create 3D grid
            min_coords = np.min(positions, axis=0)
            max_coords = np.max(positions, axis=0)
            
            grid_size = max_coords - min_coords
            grid_spacing = grid_size / grid_resolution
            
            # Assign each monomer to grid cells
            grid_occupancy = np.zeros((grid_resolution, grid_resolution, grid_resolution, num_chains))
            
            start = 0
            for chain_idx, size in enumerate(chain_sizes):
                end = start + size
                chain_pos = positions[start:end]
                
                # Convert positions to grid indices
                grid_indices = ((chain_pos - min_coords) / grid_spacing).astype(int)
                grid_indices = np.clip(grid_indices, 0, grid_resolution - 1)
                
                for pos_idx in grid_indices:
                    grid_occupancy[pos_idx[0], pos_idx[1], pos_idx[2], chain_idx] = 1
                
                start = end
            
            # Calculate overlap
            overlap = 0
            total_cells = 0
            for i in range(grid_resolution):
                for j in range(grid_resolution):
                    for k in range(grid_resolution):
                        occupied_chains = np.sum(grid_occupancy[i, j, k, :])
                        if occupied_chains > 1:
                            overlap += 1
                        if occupied_chains > 0:
                            total_cells += 1
            
            overlap_fraction = overlap / total_cells if total_cells > 0 else 0
            overlap_scores.append(overlap_fraction)
        
        return np.array(overlap_scores)
    
    overlap_scores = calculate_territory_overlap(trajectory_positions, chain_sizes)
    avg_overlap = np.mean(overlap_scores)
    
    print(f"Average territorial overlap: {avg_overlap:.3f}")
    if avg_overlap < 0.1:
        print("  → Strong territorial segregation")
    elif avg_overlap < 0.3:
        print("  → Moderate territorial segregation")
    else:
        print("  → Weak territorial segregation")

except ImportError:
    print("  Advanced analysis modules not available")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n4. Creating analysis plots...")

try:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Chain separation over time
    time_blocks = np.arange(len(chain_distances))
    pair_labels = [f"Chain {i+1}-{j+1}" for i in range(num_chains) for j in range(i+1, num_chains)]
    
    for i, label in enumerate(pair_labels):
        ax1.plot(time_blocks, chain_distances[:, i], label=label, linewidth=2)
    
    ax1.set_xlabel('Block')
    ax1.set_ylabel('Center-to-Center Distance')
    ax1.set_title('Chain Separation Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Individual chain Rg evolution
    for i in range(num_chains):
        ax2.plot(time_blocks, chain_rg_values[:, i], label=f'Chain {i+1} (N={chain_sizes[i]})', linewidth=2)
    
    ax2.set_xlabel('Block')
    ax2.set_ylabel('Radius of Gyration')
    ax2.set_title('Individual Chain Compaction')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Inter-chain contact matrix (heatmap)
    if len(eq_contact_matrices) > 0:
        im = ax3.imshow(avg_contact_matrix, cmap='Blues', vmin=0, vmax=np.max(avg_contact_matrix))
        ax3.set_title('Average Inter-Chain Contact Frequencies')
        ax3.set_xlabel('Chain Index')
        ax3.set_ylabel('Chain Index')
        
        # Add text annotations
        for i in range(num_chains):
            for j in range(num_chains):
                text = ax3.text(j, i, f'{avg_contact_matrix[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=10)
        
        # Set ticks
        ax3.set_xticks(range(num_chains))
        ax3.set_yticks(range(num_chains))
        ax3.set_xticklabels([f'Chain {i+1}' for i in range(num_chains)])
        ax3.set_yticklabels([f'Chain {i+1}' for i in range(num_chains)])
        
        plt.colorbar(im, ax=ax3, label='Contact Frequency')
    
    # Plot 4: Final conformation (3D projection to 2D)
    try:
        final_pos = trajectory_positions[-1]
        colors = ['red', 'blue', 'green', 'orange', 'purple'][:num_chains]
        
        start = 0
        for i, (size, color) in enumerate(zip(chain_sizes, colors)):
            end = start + size
            chain_pos = final_pos[start:end]
            
            ax4.plot(chain_pos[:, 0], chain_pos[:, 1], 'o-', color=color, 
                    alpha=0.7, markersize=3, linewidth=1, label=f'Chain {i+1}')
            
            # Mark chain start
            ax4.plot(chain_pos[0, 0], chain_pos[0, 1], 'o', color=color, 
                    markersize=8, markeredgecolor='black', markeredgewidth=2)
            
            start = end
        
        ax4.set_xlabel('X coordinate')
        ax4.set_ylabel('Y coordinate')
        ax4.set_title('Final Multi-Chain Conformation (XY projection)')
        ax4.legend()
        ax4.set_aspect('equal')
        ax4.grid(True, alpha=0.3)
        
    except:
        ax4.text(0.5, 0.5, 'Final Conformation\n(Data not available)', 
                ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.savefig('multichain_analysis.png', dpi=150, bbox_inches='tight')
    print("  Analysis plots saved as 'multichain_analysis.png'")

except ImportError:
    print("  Matplotlib not available, skipping plots")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 45)
print("SUMMARY: Multi-Chain Polymer Simulation")
print("=" * 45)

print(f"✓ Successfully simulated {num_chains} chains with sizes {chain_sizes}")
print(f"✓ Total system size: {total_N} monomers")

print(f"\nKey findings:")
print(f"  • Average chain separation: {np.mean(eq_distances):.2f}")
print(f"  • Inter-chain contact frequency: {np.mean(avg_contact_matrix[np.triu_indices(num_chains, k=1)]):.4f}")
print(f"  • Individual chain compaction varies with chain length")

print(f"\nBiological relevance:")
print(f"  • Multiple chromosomes in nucleus")
print(f"  • Chromosome territories in interphase")
print(f"  • Plasmid segregation in bacteria")
print(f"  • Mitochondrial DNA organization")

print(f"\nFiles created:")
print(f"  • trajectory_multichain/: Multi-chain simulation data")
print(f"  • multichain_analysis.png: Analysis plots")

print(f"\nNext steps for multi-chain research:")
print(f"  1. Study chromosome territorial organization")
print(f"  2. Model inter-chromosomal translocations") 
print(f"  3. Investigate chain length effects on segregation")
print(f"  4. Add specific inter-chain interactions")
print(f"  5. Compare with Hi-C trans contact data")

print(f"\n🧬 Multi-chain simulation completed!")