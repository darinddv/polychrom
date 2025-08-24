Quickstart Guide
================

Welcome to polychrom! This guide will walk you through the basics of setting up and running polymer simulations, with a focus on chromatin loop extrusion modeling.

What is polychrom?
------------------

Polychrom is a Python library for running molecular dynamics simulations of polymer chains, specifically designed for modeling chromatin and chromosome dynamics. It's built on top of OpenMM, a high-performance molecular simulation toolkit that can leverage GPU acceleration.

Key features:

* **Mechanistic modeling**: Build biophysically realistic models of chromatin behavior
* **Loop extrusion simulation**: Specialized tools for modeling cohesin-mediated loop extrusion
* **GPU acceleration**: Fast simulations using OpenMM's GPU capabilities  
* **Flexible force system**: Easy-to-use and customizable force definitions
* **Analysis tools**: Built-in functions for analyzing simulation results

Installation
------------

First, install the dependencies. OpenMM is required and best installed via conda::

    conda install -c omnia openmm

Then install polychrom's Python dependencies::

    pip install -r requirements.txt

For CUDA support (recommended for performance), ensure you have compatible CUDA drivers installed.

Basic Concepts
--------------

Before diving into simulations, let's understand the key concepts:

**Monomers**: Individual units that make up the polymer chain (representing nucleosomes or larger chromatin segments)

**Forces**: Physical interactions that govern monomer behavior:
  
  * *Connectivity forces*: Keep the polymer chain connected (bonds, angles)
  * *Confinement forces*: Keep the polymer in a bounded space (nucleus)
  * *Repulsion forces*: Prevent monomers from overlapping
  * *Loop extrusion forces*: Model the action of cohesin complexes

**Integrator**: Numerical method for evolving the system over time

**Reporter**: System for saving simulation data to files

Your First Simulation
---------------------

Let's create a simple polymer simulation step by step.

Step 1: Import Required Modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    import polychrom
    from polychrom import forcekits, forces, simulation, starting_conformations
    from polychrom.hdf5_format import HDF5Reporter

Step 2: Set Up Basic Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Number of monomers in our polymer chain
    N = 1000
    
    # Create a reporter to save our simulation data
    reporter = HDF5Reporter(folder="trajectory", max_data_length=100, overwrite=True)

Step 3: Create a Simulation Object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    sim = simulation.Simulation(
        platform="CUDA",           # Use GPU acceleration (or "CPU" if no GPU)
        integrator="variableLangevin",  # Adaptive timestep integrator
        error_tol=0.003,           # Integration error tolerance
        GPU="0",                   # GPU index (if multiple GPUs)
        collision_rate=0.03,       # Langevin collision rate (friction)
        N=N,                       # Number of particles
        save_decimals=2,           # Precision for saving coordinates
        PBCbox=False,              # No periodic boundary conditions
        reporters=[reporter],      # Where to save data
    )

Step 4: Set Initial Polymer Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Create an initial random walk conformation
    polymer = starting_conformations.grow_cubic(N, 50)
    
    # Load the starting conformation into the simulation
    sim.set_data(polymer, center=True)

Step 5: Add Forces
~~~~~~~~~~~~~~~~~~

Now we define the physical forces acting on our polymer:

.. code-block:: python

    # Confine the polymer to a sphere (like a cell nucleus)
    sim.add_force(forces.spherical_confinement(sim, density=0.85, k=1))
    
    # Add polymer connectivity and repulsion
    sim.add_force(
        forcekits.polymer_chains(
            sim,
            chains=[(0, None, False)],  # One chain from monomer 0 to end
            
            # Harmonic bonds keep adjacent monomers connected
            bond_force_func=forces.harmonic_bonds,
            bond_force_kwargs={
                "bondLength": 1.0,         # Rest length of bonds
                "bondWiggleDistance": 0.05, # Flexibility of bonds
            },
            
            # Angle force provides chain stiffness
            angle_force_func=forces.angle_force,
            angle_force_kwargs={
                "k": 1.5,  # Stiffness parameter (higher = stiffer)
            },
            
            # Repulsive force prevents chain overlap
            nonbonded_force_func=forces.polynomial_repulsive,
            nonbonded_force_kwargs={
                "trunc": 3.0,  # Cutoff distance for repulsion
            },
            
            except_bonds=True,  # Don't apply repulsion to bonded neighbors
        )
    )

Step 6: Run the Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Run 10 blocks of 100 time steps each
    for i in range(10):
        sim.do_block(100)  # Simulate 100 steps and save data
        print(f"Completed block {i+1}/10")
    
    # Print basic statistics
    sim.print_stats()
    
    # Finalize data saving
    reporter.dump_data()

Understanding the Output
------------------------

Your simulation will create a "trajectory" folder containing:

* ``blocks.h5``: Main trajectory data with polymer coordinates over time
* ``forcekit_polymer_chains.json``: Information about bonds and angles  
* ``spherical_confinement.json``: Confinement force parameters

You can load and analyze the data:

.. code-block:: python

    import polychrom.polymer_analyses as polymer_analyses
    import polychrom.hdf5_format as h5f
    
    # Load trajectory data
    data = h5f.load_URI("trajectory")
    
    # Calculate radius of gyration over time
    rg = polymer_analyses.radius_of_gyration(data['pos'])
    print(f"Average radius of gyration: {np.mean(rg):.2f}")

Loop Extrusion Simulation
-------------------------

One of polychrom's key applications is simulating loop extrusion by cohesin complexes. Here's a simple example:

.. code-block:: python

    from polychrom.forces import harmonic_bonds
    from polychrom.lib.extrusion import ExtrustionContext
    
    # Set up simulation (similar to basic example)
    N = 4000
    sim = simulation.Simulation(platform="CUDA", N=N, ...)
    
    # Add basic polymer forces
    sim.add_force(forcekits.polymer_chains(sim, ...))
    sim.add_force(forces.spherical_confinement(sim, ...))
    
    # Create loop extrusion system
    context = ExtrustionContext(
        sim=sim,
        cohesins=10,  # Number of cohesin complexes
        CTCF_sites=[100, 500, 1000, 1500],  # Boundary elements
    )
    
    # Run simulation with dynamic loop formation
    for i in range(100):
        context.extrude_step()  # Move cohesins and form loops
        sim.do_block(50)        # Run MD simulation
        context.update_bonds()  # Update loop constraints

This creates a simulation where cohesin complexes extrude loops that are anchored at CTCF binding sites, mimicking the process that organizes chromatin into topologically associating domains (TADs).

Next Steps
----------

Now that you understand the basics, explore these topics:

1. **Custom Forces**: Learn to create your own force definitions in :doc:`polychrom.forces`
2. **Analysis Tools**: Discover analysis functions in :doc:`polychrom.polymer_analyses`
3. **Advanced Examples**: Check out detailed examples in the ``examples/`` directory
4. **Loop Extrusion**: Dive deeper into chromatin modeling with loop extrusion examples

Tips for Success
----------------

* Start with small systems (N < 1000) to test parameters quickly
* Use GPU acceleration when possible for faster simulations
* Monitor the radius of gyration to ensure your polymer doesn't collapse or expand unrealistically  
* Adjust collision rate and timestep if the simulation becomes unstable
* Save data frequently to avoid losing progress from long simulations

Common Issues
-------------

**Simulation explodes (coordinates become very large)**:
  - Reduce timestep or increase collision rate
  - Check that forces are properly balanced
  - Ensure proper initial configuration

**Simulation is too slow**:
  - Use GPU platform instead of CPU
  - Reduce integration error tolerance
  - Optimize force cutoff distances

**Memory issues**:
  - Reduce ``max_data_length`` in reporter
  - Use fewer particles or shorter simulations
  - Clear trajectory data more frequently

Getting Help
------------

* Check the API documentation for detailed parameter descriptions
* Look at examples in the ``examples/`` directory for working code
* The polychrom paper describes the scientific background and validation

Happy simulating!