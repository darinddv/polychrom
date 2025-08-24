.. polychrom documentation master file, created by
   sphinx-quickstart on Sat Jan  4 15:17:36 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the documentation for the polychrom package
======================================================

Polychrom is a powerful Python package for setting up, performing and analyzing polymer simulations of chromosomes and chromatin. 
The simulation engine is built on top of OpenMM, a high-performance GPU-accelerated molecular dynamics framework, 
while the analysis tools are developed by the Mirnylab to specifically address chromatin organization questions.

**Key Features:**

* High-performance GPU-accelerated molecular dynamics simulations
* Specialized tools for chromatin loop extrusion modeling  
* Comprehensive force system for polymer physics
* Built-in analysis tools for contact maps, polymer statistics, and more
* Flexible architecture for custom force development

Getting Started
---------------

New to polychrom? Start here:

* **New users**: Begin with the :doc:`quickstart` guide for a hands-on introduction
* **Installation help**: See the installation section below
* **Examples**: Check out the ``examples/`` directory in the repository
* **API reference**: Browse the detailed module documentation below

Installation
------------

**Step 1: Install OpenMM**

Polychrom requires OpenMM, which is best installed through conda::

    conda install -c omnia openmm

See the `OpenMM installation guide <http://docs.openmm.org/latest/userguide/application.html#installing-openmm>`_ for more details. 
Adding ``-c conda-forge`` as mentioned in their guide is optional in our experience.

**Step 2: Install Python dependencies**

Install polychrom's Python dependencies::

    pip install -r requirements.txt

**Step 3: GPU Support (Recommended)**

For optimal performance, ensure you have CUDA support:

* Install compatible CUDA drivers for your GPU
* OpenMM will automatically detect and use CUDA if available
* CPU-only simulations are possible but much slower

**Common Installation Issues**

If you encounter the error ``version GLIBCXX_3.4.30 not found`` when importing openmm::

    conda install -c conda-forge libstdcxx-ng=12

Package Structure
-----------------

Polychrom is designed as a flexible API where each simulation is set up as a Python script. The main components are:

**Core Simulation:**
  * :py:mod:`polychrom.simulation` - Main simulation class and control
  * :py:mod:`polychrom.forces` - Individual force definitions  
  * :py:mod:`polychrom.forcekits` - Combined force systems for common setups

**Data Handling:**
  * :py:mod:`polychrom.hdf5_format` - Trajectory storage and loading
  * :py:mod:`polychrom.polymerutils` - Individual conformation utilities
  * :py:mod:`polychrom.starting_conformations` - Initial polymer configurations

**Analysis:**
  * :py:mod:`polychrom.polymer_analyses` - Radius of gyration, end-to-end distance, etc.
  * :py:mod:`polychrom.contactmaps` - Hi-C-like contact map generation

**Utilities:**
  * :py:mod:`polychrom.param_units` - Physical parameter calculations
  * :py:mod:`polychrom.pymol_show` - Visualization helpers

Documentation
-------------

.. toctree::
   :maxdepth: 2   
   
   quickstart
   best_practices
   troubleshooting
   polychrom.simulation
   polychrom.forces
   polychrom.forcekits
   polychrom.polymerutils
   polychrom.hdf5_format
   polychrom.polymer_analyses
   polychrom.contactmaps
   polychrom.starting_conformations
   polychrom.param_units
   


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
