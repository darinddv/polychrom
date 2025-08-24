Troubleshooting Guide
====================

This guide helps you diagnose and fix common issues when running polychrom simulations.

Installation Issues
-------------------

**OpenMM import errors**

Error: ``ImportError: No module named 'openmm'``

For OpenMM 8+ (recommended)::

    conda install -c conda-forge openmm

For compatibility with older systems::

    conda install -c omnia openmm

**GLIBCXX version errors**

Error: ``version GLIBCXX_3.4.30 not found``

Solution::

    conda install -c conda-forge libstdcxx-ng=12

**CUDA not found**

Error: ``CUDA platform is not available`` or simulation runs on CPU instead of GPU

Solutions:

1. Check CUDA installation::

    nvidia-smi  # Should show GPU status
    
2. Install CUDA-compatible OpenMM.

For OpenMM 8+ (recommended)::

    conda install -c conda-forge openmm

For older systems::

    conda install -c omnia openmm cudatoolkit

3. Force CPU platform if GPU unavailable::

    sim = simulation.Simulation(platform="CPU", ...)

Simulation Startup Issues
-------------------------

**Particle explosion**

Symptoms: Coordinates become very large (>1000), NaN errors, simulation crashes

Common causes and solutions:

1. **Poor initial conformation**::

    # Bad: overlapping particles
    data = np.zeros((1000, 3))  # All particles at origin
    
    # Good: self-avoiding initial structure
    data = starting_conformations.grow_cubic(1000, 50)
    sim.set_data(data, center=True, random_offset=1e-5)

2. **Forces too strong**::

    # Bad: very stiff bonds
    forcekits.polymer_chains(sim, bond_force_kwargs={"bondWiggleDistance": 0.001})
    
    # Good: reasonable bond flexibility
    forcekits.polymer_chains(sim, bond_force_kwargs={"bondWiggleDistance": 0.05})

3. **Integration error tolerance too large**::

    # Bad: inaccurate integration
    sim = simulation.Simulation(error_tol=0.1, ...)
    
    # Good: tighter tolerance
    sim = simulation.Simulation(error_tol=0.003, ...)

4. **Incompatible force combinations**::

    # Bad: strong confinement + large polymer
    forces.spherical_confinement(sim, density=2.0, k=100)
    
    # Good: reasonable density and force strength
    forces.spherical_confinement(sim, density=0.3, k=5)

**Integration failures**

Error: ``IntegrationFailError`` or very slow simulation

Solutions:

1. Reduce error tolerance::

    sim = simulation.Simulation(error_tol=0.001, ...)  # More accurate

2. Increase collision rate::

    sim = simulation.Simulation(collision_rate=0.1, ...)  # More damping

3. Use CPU platform for debugging::

    sim = simulation.Simulation(platform="CPU", ...)

Performance Issues
------------------

**Simulation too slow**

Check these factors in order of impact:

1. **Use GPU acceleration**::

    sim = simulation.Simulation(platform="CUDA", ...)  # Much faster than CPU

2. **Optimize non-bonded cutoff**::

    # Slow: large cutoff distance
    forcekits.polymer_chains(sim, nonbonded_force_kwargs={"trunc": 10.0})
    
    # Fast: smaller cutoff (allows some chain crossing)
    forcekits.polymer_chains(sim, nonbonded_force_kwargs={"trunc": 3.0})

3. **Reduce integration accuracy**::

    sim = simulation.Simulation(error_tol=0.01, ...)  # Less accurate but faster

4. **Use efficient data saving**::

    # Slow: save every step
    for i in range(1000):
        sim.do_block(1)
    
    # Fast: save every 100 steps
    for i in range(10):
        sim.do_block(100)

**Memory issues**

Error: ``Out of memory`` or system becomes unresponsive

Solutions:

1. Reduce trajectory storage::

    reporter = HDF5Reporter(folder="trajectory", max_data_length=50)  # Smaller buffer

2. Clear trajectory data more frequently::

    for i in range(20):
        sim.do_block(100)
        if i % 5 == 0:
            reporter.dump_data()  # Write to disk periodically

3. Use data decimation::

    sim = simulation.Simulation(save_decimals=1, ...)  # Less precision, smaller files

4. Reduce system size for testing::

    N = 1000  # Smaller system for parameter testing
    N = 10000  # Full system after validation

Physics and Results Issues
--------------------------

**Unrealistic polymer behavior**

**Polymer too compact**

Symptoms: Radius of gyration much smaller than expected

Solutions:

1. Reduce confinement density::

    forces.spherical_confinement(sim, density=0.1, k=5)  # Larger nucleus

2. Reduce non-bonded repulsion::

    forcekits.polymer_chains(sim, nonbonded_force_kwargs={"trunc": 2.0})

3. Increase chain flexibility::

    forcekits.polymer_chains(sim, angle_force_kwargs={"k": 0.5})

**Polymer too extended**

Symptoms: Radius of gyration much larger than expected

Solutions:

1. Increase confinement::

    forces.spherical_confinement(sim, density=1.0, k=10)  # Stronger confinement

2. Increase non-bonded repulsion::

    forcekits.polymer_chains(sim, nonbonded_force_kwargs={"trunc": 5.0})

3. Add attractive interactions (advanced)::

    forces.selective_SSW(sim, ...)  # Specialized attractive forces

**Chain crossing issues**

Problem: Unrealistic chain crossing behavior

Solutions:

1. Increase non-bonded cutoff::

    forcekits.polymer_chains(sim, nonbonded_force_kwargs={"trunc": 10.0})

2. Strengthen repulsion::

    forcekits.polymer_chains(sim, nonbonded_force_kwargs={"radiusMult": 1.5})

3. Use smaller timesteps::

    sim = simulation.Simulation(error_tol=0.001, ...)

**Poor equilibration**

Symptoms: Properties change continuously, no steady state

Solutions:

1. Run longer simulations::

    for i in range(100):  # More blocks
        sim.do_block(100)

2. Check equilibration::

    import polychrom.polymer_analyses as polymer_analyses
    rg = polymer_analyses.radius_of_gyration(data['pos'])
    plt.plot(rg)  # Should stabilize

3. Start from better initial condition::

    # Use equilibrated structure from previous simulation
    data = h5f.load_URI("previous_trajectory")
    sim.set_data(data['pos'][-1], center=True)

Data Analysis Issues
--------------------

**Loading trajectory data**

Error: ``KeyError`` or missing data files

Solutions:

1. Ensure proper data finalization::

    reporter.dump_data()  # Always call at end of simulation

2. Check file paths::

    import os
    print(os.listdir("trajectory"))  # Verify files exist

3. Use correct loading function::

    import polychrom.hdf5_format as h5f
    data = h5f.load_URI("trajectory")  # Correct format
    
    # Not: data = h5py.File("trajectory/blocks.h5")  # Low-level access

**Analysis errors**

Common analysis problems and fixes:

1. **Wrong coordinate dimensions**::

    # Error: coordinates not shaped correctly
    rg = polymer_analyses.radius_of_gyration(data['pos'][0])  # Single frame
    
    # Fix: ensure correct shape
    rg = polymer_analyses.radius_of_gyration(data['pos'])  # All frames

2. **Missing trajectory frames**::

    # Check data shape
    print(data['pos'].shape)  # Should be (n_frames, n_particles, 3)
    
    # If too few frames, run longer simulation

3. **Contact map issues**::

    # Ensure adequate sampling
    contactmap = polychrom.contactmaps.monomerResolutionContactMap(
        data['pos'][-50:],  # Use last 50 frames
        cutoff=3.0
    )

Best Practices
--------------

**Development workflow**

1. **Start small**::

    N = 100  # Test with small system first
    num_blocks = 5  # Short simulation for parameter testing

2. **Test forces individually**::

    # Test confinement alone
    sim.add_force(forces.spherical_confinement(sim, density=0.3))
    
    # Then add polymer forces
    sim.add_force(forcekits.polymer_chains(sim))

3. **Monitor key quantities**::

    # Track radius of gyration
    rg = polymer_analyses.radius_of_gyration(sim.get_data()[0][None, :, :])
    print(f"Current Rg: {rg[0]:.2f}")

4. **Save intermediate states**::

    # Save state every 10 blocks for restart capability
    if block % 10 == 0:
        np.save(f"state_block_{block}.npy", sim.get_data()[0])

**Parameter exploration**

1. **Systematic testing**::

    densities = [0.1, 0.3, 0.5, 1.0]
    for density in densities:
        # Run simulation with each density
        # Compare results

2. **Monitor convergence**::

    # Check if simulation has equilibrated
    rg_values = []
    for block in range(100):
        sim.do_block(100)
        rg = polymer_analyses.radius_of_gyration(sim.get_data()[0][None, :, :])
        rg_values.append(rg[0])
        
        # Check if last 20 values are stable
        if len(rg_values) > 20:
            recent_std = np.std(rg_values[-20:])
            if recent_std < 0.1:
                print("Equilibrated!")
                break

**Error prevention**

1. **Validate inputs**::

    assert len(bonds) > 0, "No bonds provided"
    assert np.all(np.array(bonds) < N), "Bond indices out of range"

2. **Use try-except blocks**::

    try:
        sim.do_block(100)
    except Exception as e:
        print(f"Simulation failed: {e}")
        # Save current state for debugging
        np.save("debug_state.npy", sim.get_data()[0])

3. **Regular data backups**::

    # Dump data every few blocks
    if block % 5 == 0:
        reporter.dump_data()

Getting Help
------------

If you're still having issues:

1. **Check the examples**: Look at working code in ``examples/`` directory
2. **Read the documentation**: Each function has detailed docstrings
3. **Test with minimal examples**: Strip down to simplest working case
4. **Check OpenMM documentation**: For low-level force and platform issues
5. **Contact the developers**: Report bugs or ask questions on GitHub

Debugging Checklist
-------------------

When encountering problems, work through this checklist:

□ OpenMM properly installed and CUDA working (if using GPU)
□ Starting conformation is reasonable (no overlaps, proper size)
□ Force parameters are in reasonable ranges
□ Integration tolerance appropriate for your forces
□ System size appropriate for your computer's memory
□ Simulation long enough for equilibration
□ Data properly saved with ``reporter.dump_data()``
□ Analysis functions used correctly with proper data shapes

Remember: Most simulation problems are caused by unrealistic parameters or poor starting conditions rather than bugs in the code!