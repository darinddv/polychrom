Best Practices Guide
====================

This guide provides recommendations for setting up, running, and analyzing polychrom simulations effectively.

Simulation Design
-----------------

**Choose appropriate system size**

For different research questions:

* **Parameter testing**: N = 100-1000 monomers
* **Local dynamics**: N = 1000-5000 monomers  
* **Domain organization**: N = 5000-20000 monomers
* **Chromosome-scale**: N = 20000-100000 monomers

**Select realistic parameters**

Typical parameter ranges for chromatin:

.. code-block:: python

    # Bond parameters
    bondLength = 1.0          # 1-2 nm (nucleosome scale)
    bondWiggleDistance = 0.05 # Moderate flexibility
    
    # Angle parameters  
    k = 1.5                   # Realistic persistence length
    
    # Confinement
    density = 0.3             # Typical nuclear density
    k_confinement = 5.0       # Moderate boundary strength
    
    # Non-bonded
    trunc = 3.0               # Balance accuracy vs performance

**Plan simulation length**

Typical equilibration times:

* **Bond relaxation**: ~10 blocks
* **Local structure**: ~50 blocks
* **Global equilibration**: ~200-500 blocks
* **Loop extrusion steady state**: ~1000 blocks

Rule of thumb: Simulate for at least 10 × longest relaxation time.

Initial Conformations
---------------------

**Use appropriate starting structures**

Good practices:

.. code-block:: python

    # Self-avoiding random walk (recommended)
    polymer = starting_conformations.grow_cubic(N, box_size=50)
    sim.set_data(polymer, center=True, random_offset=1e-5)
    
    # Pre-equilibrated structure (even better)
    previous_data = h5f.load_URI("equilibrated_trajectory")
    sim.set_data(previous_data['pos'][-1], center=True)

**Avoid problematic starts**

Common mistakes:

.. code-block:: python

    # Bad: all particles at origin
    polymer = np.zeros((N, 3))
    
    # Bad: regular grid (too ordered)
    polymer = np.mgrid[0:10, 0:10, 0:10].reshape(3, -1).T
    
    # Bad: linear chain (unrealistic for chromatin)
    polymer = np.column_stack([np.arange(N), np.zeros(N), np.zeros(N)])

**Test initial stability**

Before long simulations:

.. code-block:: python

    # Run short test
    for i in range(5):
        sim.do_block(10)
        current_pos = sim.get_data()[0]
        rg = polymer_analyses.radius_of_gyration(current_pos[None, :, :])
        print(f"Block {i}, Rg = {rg[0]:.2f}")
    
    # Check for explosions or collapse
    assert 5 < rg[0] < 50, "Unrealistic polymer size detected"

Force Selection and Tuning
---------------------------

**Start with standard forcekits**

Use ``polymer_chains`` as foundation:

.. code-block:: python

    # Standard setup (works for most cases)
    sim.add_force(forcekits.polymer_chains(sim))
    
    # Add confinement
    sim.add_force(forces.spherical_confinement(sim, density=0.3))

**Tune forces systematically**

Priority order for parameter adjustment:

1. **Confinement density** (biggest impact on overall structure)
2. **Non-bonded cutoff** (affects chain crossing and performance)
3. **Angle stiffness** (controls persistence length)
4. **Bond flexibility** (fine-tune local dynamics)

**Validate force combinations**

Check for conflicts:

.. code-block:: python

    # Test forces individually first
    sim_test = simulation.Simulation(platform="CPU", N=100, reporters=[])
    sim_test.set_data(test_polymer, center=True)
    
    # Add one force at a time
    sim_test.add_force(forces.spherical_confinement(sim_test, density=0.3))
    sim_test.do_block(10)  # Should not explode
    
    sim_test.add_force(forcekits.polymer_chains(sim_test))
    sim_test.do_block(10)  # Check combined effect

Performance Optimization
------------------------

**Use GPU acceleration**

Always prefer CUDA when available:

.. code-block:: python

    # Check available platforms
    platforms = [openmm.Platform.getPlatform(i).getName() 
                 for i in range(openmm.Platform.getNumPlatforms())]
    print("Available platforms:", platforms)
    
    # Use CUDA if available
    platform = "CUDA" if "CUDA" in platforms else "CPU"
    sim = simulation.Simulation(platform=platform, ...)

**Optimize data handling**

Efficient trajectory management:

.. code-block:: python

    # Good: reasonable buffer size
    reporter = HDF5Reporter(folder="trajectory", max_data_length=100)
    
    # Good: batch simulation and save
    for batch in range(10):
        for block in range(20):
            sim.do_block(100)
        reporter.dump_data()  # Save every 20 blocks
    
    # Good: monitor file sizes
    import os
    size_mb = os.path.getsize("trajectory/blocks.h5") / 1024**2
    print(f"Trajectory size: {size_mb:.1f} MB")

**Balance accuracy vs speed**

Performance hierarchy (fastest to slowest):

1. Larger timesteps (higher ``error_tol``)
2. Smaller non-bonded cutoff (``trunc``)
3. Fewer particles
4. CPU instead of GPU (for small systems)
5. Lower precision

.. code-block:: python

    # Fast but less accurate
    sim = simulation.Simulation(
        error_tol=0.01,         # Larger timesteps
        precision="single",     # Lower precision
        save_decimals=1         # Less coordinate precision
    )
    
    # Add forces with performance considerations
    sim.add_force(forcekits.polymer_chains(
        sim,
        nonbonded_force_kwargs={"trunc": 2.5}  # Smaller cutoff
    ))

Data Management
---------------

**Organize simulation data**

Directory structure:

.. code-block:: bash

    project/
    ├── simulations/
    │   ├── test_runs/          # Parameter testing
    │   ├── equilibration/      # Long equilibration runs
    │   ├── production/         # Final data collection
    │   └── analysis/           # Analysis notebooks
    ├── scripts/
    │   ├── setup_simulation.py
    │   ├── run_production.py
    │   └── analyze_results.py
    └── results/
        ├── figures/
        ├── contact_maps/
        └── statistics/

**Save metadata**

Document simulation parameters:

.. code-block:: python

    import json
    
    # Save parameters with trajectory
    params = {
        "N": N,
        "density": 0.3,
        "bond_flexibility": 0.05,
        "simulation_length": num_blocks,
        "platform": "CUDA",
        "date": str(datetime.now())
    }
    
    with open("trajectory/parameters.json", "w") as f:
        json.dump(params, f, indent=2)

**Version control simulation scripts**

Use git to track simulation code:

.. code-block:: bash

    git init
    git add simulation_script.py
    git commit -m "Initial simulation setup"
    
    # Before major parameter changes
    git add -A
    git commit -m "Changed confinement density to 0.5"

Analysis Best Practices
-----------------------

**Allow for equilibration**

Skip initial frames in analysis:

.. code-block:: python

    data = h5f.load_URI("trajectory")
    
    # Skip first 20% of trajectory
    equilibrated_start = len(data['pos']) // 5
    analysis_data = data['pos'][equilibrated_start:]
    
    # Analyze equilibrated portion
    rg = polymer_analyses.radius_of_gyration(analysis_data)

**Use adequate sampling**

Statistics require sufficient data:

.. code-block:: python

    # Good: many frames for contact maps
    contactmap = polychrom.contactmaps.monomerResolutionContactMap(
        data['pos'][-100:],  # Last 100 frames
        cutoff=3.0
    )
    
    # Good: bootstrap error estimates
    rg_samples = []
    for i in range(100):
        # Sample random subset of frames
        indices = np.random.choice(len(analysis_data), size=50)
        rg_sample = polymer_analyses.radius_of_gyration(analysis_data[indices])
        rg_samples.append(np.mean(rg_sample))
    
    rg_mean = np.mean(rg_samples)
    rg_error = np.std(rg_samples)

**Validate results**

Check against expectations:

.. code-block:: python

    # Compare with theoretical predictions
    theoretical_rg = np.sqrt(N * bond_length**2 / 6)  # Ideal chain
    observed_rg = np.mean(rg)
    
    print(f"Theoretical Rg: {theoretical_rg:.2f}")
    print(f"Observed Rg: {observed_rg:.2f}")
    print(f"Ratio: {observed_rg/theoretical_rg:.2f}")
    
    # Typical ratios for confined polymers: 0.3-0.8

Loop Extrusion Best Practices
------------------------------

**Design realistic kinetics**

Use experimentally motivated parameters:

.. code-block:: python

    # Typical cohesin parameters
    extrusion_speed = 1.0      # monomers per time step
    loading_probability = 0.01  # per step per loading site
    unloading_probability = 0.001  # per step per cohesin
    CTCF_stall_probability = 0.3   # at CTCF sites

**Balance 1D and 3D simulations**

Efficient workflow:

1. **1D simulation**: Fast exploration of kinetic parameters
2. **Parameter optimization**: Find steady-state regime
3. **3D simulation**: Full spatial dynamics with optimized parameters
4. **Analysis**: Contact maps, loop statistics

**Validate loop extrusion**

Check key properties:

.. code-block:: python

    # Monitor loop size distribution
    loop_sizes = get_loop_sizes_from_trajectory(data)
    
    # Check for expected power-law decay
    plt.hist(loop_sizes, bins=50, density=True)
    plt.yscale('log')
    plt.xlabel('Loop size')
    plt.ylabel('Probability density')
    
    # Verify CTCF boundary effect
    # Should see loop accumulation at CTCF sites

Reproducibility
---------------

**Set random seeds**

Ensure reproducible results:

.. code-block:: python

    import numpy as np
    import random
    
    # Set all random seeds
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    
    # OpenMM random seed
    sim.integrator.setRandomNumberSeed(seed)

**Document environment**

Save computational environment:

.. code-block:: python

    import sys
    import polychrom
    import openmm
    
    print(f"Python: {sys.version}")
    print(f"Polychrom: {polychrom.__version__}")
    print(f"OpenMM: {openmm.__version__}")
    print(f"Platform: {sim.platform.getName()}")

**Create analysis pipelines**

Write reusable analysis scripts:

.. code-block:: python

    def standard_polymer_analysis(trajectory_folder):
        """Standard analysis pipeline for polymer simulations."""
        
        # Load data
        data = h5f.load_URI(trajectory_folder)
        
        # Skip equilibration
        equil_data = data['pos'][len(data['pos'])//5:]
        
        # Calculate properties
        results = {
            'rg': polymer_analyses.radius_of_gyration(equil_data),
            'end_to_end': polymer_analyses.end_to_end_distance(equil_data),
            'contact_map': polychrom.contactmaps.monomerResolutionContactMap(
                equil_data[-50:], cutoff=3.0
            )
        }
        
        return results

Common Pitfalls to Avoid
------------------------

**Don't skip equilibration**

Always allow sufficient time for equilibration before collecting statistics.

**Don't use unrealistic parameters**

Stay within physically reasonable ranges unless you have specific reasons.

**Don't ignore performance**

Monitor simulation speed and optimize bottlenecks early.

**Don't forget error analysis**

Always estimate uncertainties in your results.

**Don't neglect validation**

Compare results with known limits and experimental data when possible.

**Don't hardcode parameters**

Use configuration files or command-line arguments for parameter sweeps.

**Don't ignore file sizes**

Trajectory files can become very large - plan storage accordingly.

Summary Checklist
-----------------

Before starting production simulations:

□ Tested parameters with small system
□ Verified equilibration time requirements  
□ Optimized performance for your hardware
□ Set up organized directory structure
□ Documented all parameters and settings
□ Planned analysis strategy
□ Estimated storage requirements
□ Set random seeds for reproducibility

During simulations:

□ Monitor progress and key quantities
□ Save intermediate states
□ Check for signs of instability
□ Back up important data

After simulations:

□ Allow adequate equilibration time in analysis
□ Estimate uncertainties
□ Validate against theoretical expectations
□ Document and archive results
□ Version control analysis scripts