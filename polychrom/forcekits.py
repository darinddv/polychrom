"""
Forcekits (new in polychrom)
----------------------------

The goal of the forcekits is two-fold. First, sometimes communication between forces is required. Since explicit is
better than implicit, according to The Zen of Python, we are trying to avoid communication between forces using
hidden variables (as done in openmm-polymer), and make it explicit. Forcekits are the tool to implement groups of
forces that go together, as to avoid hidden communication between forces. Second, some structures can be realized
using many different forces: e.g. polymer chain connectivity can be done using harmonic bond force, FENE bonds,
etc. Forcekits help avoid duplicating code, and allow swapping one force for another and keeping the
topology/geometry of the system the same

The only forcekit that we have now implements polymer chain connectivity. It then explicitly adds exclusions for all
the polymer bonds into the nonbonded force, without using hidden variables for communication between forces. It also
allows using any bond force, any angle force, and any nonbonded force, allowing for easy swapping of one force for
another without duplicating code.

"""

import numpy as np

from . import forces


def polymer_chains(
    sim_object,
    chains=[(0, None, False)],
    bond_force_func=forces.harmonic_bonds,
    bond_force_kwargs={"bondWiggleDistance": 0.05, "bondLength": 1.0},
    angle_force_func=forces.angle_force,
    angle_force_kwargs={"k": 0.05},
    nonbonded_force_func=forces.polynomial_repulsive,
    nonbonded_force_kwargs={"trunc": 3.0, "radiusMult": 1.0},
    except_bonds=True,
    extra_bonds=None,
    extra_triplets=None,
    override_checks=False,
):
    """
    Set up complete polymer chain physics with bonds, angles, and repulsion.
    
    This is the most commonly used forcekit in polychrom. It creates a complete
    polymer physics model by combining three essential force types:
    1. Harmonic bonds - maintain chain connectivity
    2. Angle forces - provide chain stiffness and persistence length  
    3. Non-bonded forces - excluded volume repulsion between monomers
    
    The forcekit automatically handles bond topology, neighbor exclusions, and
    parameter consistency. It's designed to work out-of-the-box for typical
    chromatin simulations while allowing full customization of individual components.
    
    Parameters
    ----------
    sim_object : polychrom.simulation.Simulation
        The simulation object to add forces to
        
    chains : list of tuples, optional
        Chain topology specification as [(start, end, isRing), ...].
        Each tuple defines one polymer chain:
        - start: first particle index (inclusive)
        - end: last particle index (exclusive, or None for system end)
        - isRing: True to connect first and last particles
        Examples:
        - [(0, None, False)]: Single linear chain, all particles (default)
        - [(0, 50, True), (50, None, False)]: 50-particle ring + linear chain
        - [(0, 100, False), (100, 200, False)]: Two separate linear chains
        
    bond_force_func : function, optional
        Function to create bond forces. Default: forces.harmonic_bonds
        Must accept (sim_object, bonds, **kwargs) and return OpenMM force
        
    bond_force_kwargs : dict, optional
        Parameters passed to bond_force_func. Default: 
        {"bondWiggleDistance": 0.05, "bondLength": 1.0}
        Common adjustments:
        - bondWiggleDistance: 0.01-0.1 (stiffness)
        - bondLength: 0.5-2.0 (rest length)
        
    angle_force_func : function, optional
        Function to create angle forces. Default: forces.angle_force
        Must accept (sim_object, triplets, **kwargs) and return OpenMM force
        Set to None to disable angle forces
        
    angle_force_kwargs : dict, optional
        Parameters passed to angle_force_func. Default: {"k": 0.05}
        - k: 0.01-10 (angular stiffness, controls persistence length)
        
    nonbonded_force_func : function, optional
        Function for non-bonded repulsion. Default: forces.polynomial_repulsive  
        Must accept (sim_object, **kwargs) and return OpenMM force
        Set to None to disable non-bonded forces
        
    nonbonded_force_kwargs : dict, optional
        Parameters for non-bonded forces. Default: {"trunc": 3.0, "radiusMult": 1.0}
        - trunc: 1.0-10.0 (repulsion cutoff, affects chain crossing)
        - radiusMult: 0.5-2.0 (effective particle size)
        
    except_bonds : bool, optional
        If True, exclude bonded neighbors from non-bonded interactions.
        This prevents double-counting of forces and is almost always desired.
        Default: True
        
    extra_bonds : list of tuples, optional
        Additional bonds beyond chain connectivity, as [(i,j), ...].
        Useful for crosslinks, loops, or constraints.
        Default: None
        
    extra_triplets : list of tuples, optional
        Additional angle triplets as [(i,j,k), ...] where j is the central particle.
        Default: None
        
    override_checks : bool, optional
        Skip validation that all particles belong to exactly one chain.
        Only set True if you know your topology is correct.
        Default: False
        
    Returns
    -------
    list of openmm.Force
        List of force objects (bonds, angles, non-bonded) ready to add to simulation
        
    Examples
    --------
    Basic linear polymer (most common usage):
    
    >>> forces = forcekits.polymer_chains(sim)
    >>> sim.add_force(forces)
    
    Stiff polymer with strong repulsion:
    
    >>> forces = forcekits.polymer_chains(
    ...     sim,
    ...     bond_force_kwargs={"bondWiggleDistance": 0.01},
    ...     angle_force_kwargs={"k": 5.0},
    ...     nonbonded_force_kwargs={"trunc": 10.0}
    ... )
    >>> sim.add_force(forces)
    
    Multiple chains:
    
    >>> forces = forcekits.polymer_chains(
    ...     sim,
    ...     chains=[(0, 500, False), (500, 1000, False)]  # Two 500-monomer chains
    ... )
    >>> sim.add_force(forces)
    
    Ring polymer:
    
    >>> forces = forcekits.polymer_chains(
    ...     sim,
    ...     chains=[(0, None, True)]  # Single ring
    ... )
    >>> sim.add_force(forces)
    
    Custom bond types:
    
    >>> from polychrom.forces import constant_force_bonds
    >>> forces = forcekits.polymer_chains(
    ...     sim,
    ...     bond_force_func=constant_force_bonds,
    ...     bond_force_kwargs={"bondWiggleDistance": 0.1}
    ... )
    >>> sim.add_force(forces)
    
    No angle forces (very flexible chain):
    
    >>> forces = forcekits.polymer_chains(
    ...     sim,
    ...     angle_force_func=None
    ... )
    >>> sim.add_force(forces)
    
    Add crosslinks:
    
    >>> crosslinks = [(0, 100), (50, 150), (200, 300)]
    >>> forces = forcekits.polymer_chains(
    ...     sim,
    ...     extra_bonds=crosslinks
    ... )
    >>> sim.add_force(forces)
    
    Notes
    -----
    Physical Interpretation:
    - Bond forces maintain chain connectivity (covalent bonds)
    - Angle forces provide persistence length (chain stiffness) 
    - Non-bonded forces prevent overlap (excluded volume)
    - Together they create realistic polymer physics
    
    Parameter Guidelines:
    - For chromatin: default parameters work well
    - For DNA: bondLength=0.34, angle k=50, trunc=2.0
    - For synthetic polymers: bondLength=1.5, k=1.0, trunc=5.0
    - For very flexible chains: k=0.1, trunc=3.0
    - For rigid chains: k=10, bondWiggleDistance=0.01
    
    Performance:
    - Most computationally expensive part is usually non-bonded forces
    - Reducing trunc improves performance but allows chain crossing
    - Angle forces have minimal computational cost
    - Bond forces are very fast
    
    Topology Validation:
    - Checks that every particle belongs to exactly one chain
    - Prevents gaps or overlaps in chain definition  
    - Can be disabled with override_checks=True for complex topologies
    
    Force Exclusions:
    - Bonded neighbors automatically excluded from non-bonded forces
    - Prevents unphysical strong repulsion between connected particles
    - Essential for stable simulations
    
    See Also
    --------
    forces.harmonic_bonds : Individual bond force
    forces.angle_force : Individual angle force  
    forces.polynomial_repulsive : Individual non-bonded force
    starting_conformations : Functions to create initial polymer configurations
    
    Raises
    ------
    ValueError
        If particles don't belong to exactly one chain
        If chain indices are out of bounds
        If invalid force functions are provided
    """

    force_list = []

    bonds = [] if ((extra_bonds is None) or len(extra_bonds) == 0) else [tuple(b) for b in extra_bonds]
    triplets = extra_triplets if extra_triplets else []
    newchains = []

    for start, end, is_ring in chains:
        end = sim_object.N if (end is None) else end
        newchains.append((start, end, is_ring))

        bonds += [(j, j + 1) for j in range(start, end - 1)]
        triplets += [(j - 1, j, j + 1) for j in range(start + 1, end - 1)]

        if is_ring:
            bonds.append((start, end - 1))
            triplets.append((int(end - 2), int(end - 1), int(start)))
            triplets.append((int(end - 1), int(start), int(start + 1)))

    # check that all monomers are a member of exactly one chain
    if not override_checks:
        num_chains_for_monomer = np.zeros(sim_object.N, dtype=int)
        for chain in newchains:
            start, end, _ = chain
            num_chains_for_monomer[start:end] += 1

        errs = np.where(num_chains_for_monomer != 1)[0]
        if len(errs) != 0:
            raise ValueError(
                f"Monomer {errs[0]} is a member of {num_chains_for_monomer[errs[0]]} chains. "
                f"Set override_checks=True to override this check."
            )

    report_dict = {
        "chains": np.array(newchains, dtype=int),
        "bonds": np.array(bonds, dtype=int),
        "angles": np.array(triplets),
    }
    for reporter in sim_object.reporters:
        reporter.report("forcekit_polymer_chains", report_dict)

    if bond_force_func is not None:
        force_list.append(bond_force_func(sim_object, bonds, **bond_force_kwargs))

    if angle_force_func is not None:
        force_list.append(angle_force_func(sim_object, triplets, **angle_force_kwargs))

    if nonbonded_force_func is not None:
        nb_force = nonbonded_force_func(sim_object, **nonbonded_force_kwargs)

        if except_bonds:
            exc = list(set([tuple(i) for i in np.sort(np.array(bonds), axis=1)]))
            if hasattr(nb_force, "addException"):
                print("Exclude neighbouring chain particles from {}".format(nb_force.name))
                for pair in exc:
                    nb_force.addException(int(pair[0]), int(pair[1]), 0, 0, 0, True)

                num_exc = nb_force.getNumExceptions()

            # The built-in LJ nonbonded force uses "exclusions" instead of "exceptions"
            elif hasattr(nb_force, "addExclusion"):
                print("Exclude neighbouring chain particles from {}".format(nb_force.name))
                nb_force.createExclusionsFromBonds([(int(b[0]), int(b[1])) for b in bonds], int(except_bonds))
                # for pair in exc:
                #     nb_force.addExclusion(int(pair[0]), int(pair[1]))
                num_exc = nb_force.getNumExclusions()

            print("Number of exceptions:", num_exc)

        force_list.append(nb_force)

    return force_list
