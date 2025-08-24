# polychrom

[![DOI](https://zenodo.org/badge/178608195.svg)](https://zenodo.org/badge/latestdoi/178608195)

## Open2C polymer simulation library

Polychrom is a powerful Python library for simulating polymer dynamics, specifically designed for modeling chromatin and chromosome organization. Built on OpenMM's high-performance GPU-accelerated molecular dynamics engine, polychrom enables researchers to build mechanistic models of chromatin behavior and compare results directly with experimental data from Hi-C, microscopy, and other sources.

### 🧬 What makes polychrom special?

**Mechanistic modeling focus**: Unlike data-driven approaches, polychrom helps you build biophysically realistic models that simulate actual biological processes like loop extrusion, chromatin compaction, and nuclear organization.

**Loop extrusion specialization**: State-of-the-art tools for modeling cohesin-mediated loop extrusion, the key process organizing chromatin into topologically associating domains (TADs).

**High performance**: GPU acceleration through OpenMM provides fast simulations suitable for genome-scale modeling.

**Scientific rigor**: Extensively validated against experimental data and used in numerous published studies.

> **Philosophy**: Polychrom is designed to build mechanistic models that test biological hypotheses, not to fit models to Hi-C data alone. For the distinction between mechanistic and data-driven modeling, see [this review](https://pubmed.ncbi.nlm.nih.gov/26364723/).

### 🚀 Quick Start

**Try it immediately**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/darinddv/polychrom/blob/master/tutorials/01_basic_polymer_simulation.ipynb) **← Start here for interactive learning!**

```python
import polychrom
from polychrom import forcekits, forces, simulation, starting_conformations
from polychrom.hdf5_format import HDF5Reporter

# Create a simple polymer simulation
N = 1000  # Number of monomers
reporter = HDF5Reporter(folder="trajectory", overwrite=True)

sim = simulation.Simulation(
    platform="CUDA", N=N, reporters=[reporter]
)

# Set initial conformation
polymer = starting_conformations.grow_cubic(N, 50)
sim.set_data(polymer, center=True)

# Add forces
sim.add_force(forces.spherical_confinement(sim, density=0.85, k=1))
sim.add_force(forcekits.polymer_chains(sim))

# Run simulation  
for i in range(10):
    sim.do_block(100)
sim.print_stats()
reporter.dump_data()
```

### 📚 Documentation and Examples

**Comprehensive documentation**: https://polychrom.readthedocs.io/en/latest/

**Getting started**: Check out our [quickstart guide](https://polychrom.readthedocs.io/en/latest/quickstart.html) for a step-by-step tutorial.

**Interactive Google Colab Tutorials** 🚀:
- **Basic polymer simulation**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/darinddv/polychrom/blob/master/tutorials/01_basic_polymer_simulation.ipynb) - Learn polychrom fundamentals with hands-on simulation
- **Loop extrusion modeling**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/darinddv/polychrom/blob/master/tutorials/02_loop_extrusion_simulation.ipynb) - Model cohesin-mediated chromatin organization  
- **Contact map analysis**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/darinddv/polychrom/blob/master/tutorials/03_contact_map_analysis.ipynb) - Analyze Hi-C-like contact maps from simulations

**Key examples**:
- **Basic polymer simulation**: [`examples/example/example.py`](examples/example/example.py) - Simple polymer in spherical confinement
- **Loop extrusion**: [`examples/loopExtrusion/`](examples/loopExtrusion/) - Complete loop extrusion simulation pipeline
- **Storage formats**: [`examples/storage_formats/`](examples/storage_formats/) - Data handling and analysis workflows

### 🛠️ Installation

**Requirements**:
- Python 3.7+
- OpenMM (GPU acceleration recommended)
- Standard scientific Python stack (numpy, scipy, h5py, pandas)

**Install OpenMM first** (required dependency):

For OpenMM 8+ (recommended):
```bash
conda install -c conda-forge openmm
```

For compatibility with older systems:
```bash
conda install -c omnia openmm
```

**Install polychrom dependencies**:
```bash
pip install -r requirements.txt
```

**For best performance**: Ensure CUDA support for GPU acceleration.

**Common installation issue**: If you see `version GLIBCXX_3.4.30 not found`:
```bash
conda install -c conda-forge libstdcxx-ng=12
```

### 🔬 Scientific Applications

Polychrom has been used to study:
- **Chromatin loop extrusion** and TAD formation
- **Nuclear organization** and chromosome positioning  
- **Chromatin compaction** mechanisms
- **Hi-C contact map** prediction and interpretation
- **Polymer physics** of confined chromosomes

### 📊 Key Features

**Simulation Engine**:
- GPU-accelerated molecular dynamics via OpenMM
- Specialized integrators for polymer simulations
- Flexible force system with pre-built and custom forces
- Efficient trajectory storage and analysis

**Loop Extrusion Modeling**:
- Dynamic cohesin loading, translocation, and unloading
- CTCF boundary element interactions
- Realistic loop extrusion kinetics
- Multi-chromosome simulations

**Analysis Tools**:
- Contact map generation (Hi-C-like)
- Polymer statistics (Rg, end-to-end distance, P(s))
- Visualization and data export utilities
- Integration with standard analysis pipelines

### 🔄 Transitioning from openmm-polymer

Polychrom replaces the older openmm-polymer library with:
- **New storage format**: Described in [`examples/storage_formats/hdf5_reporter.ipynb`](examples/storage_formats/hdf5_reporter.ipynb)
- **Legacy compatibility**: Use [`examples/storage_formats/legacy_reporter.ipynb`](examples/storage_formats/legacy_reporter.ipynb)
- **Backwards compatibility**: `polychrom.polymerutils.load` works with both old and new formats

### 🤝 Contributing and Support

**Getting help**:
- Check the [documentation](https://polychrom.readthedocs.io/en/latest/)
- Browse [examples](examples/) for working code
- Read the polychrom paper for scientific background

**Contributing**:
- Report bugs and request features via GitHub issues
- Contribute examples and documentation improvements
- Follow the existing code style and testing practices

### 📖 Citation

If you use polychrom in your research, please cite:
- The polychrom paper: [DOI to be added]
- This repository: [![DOI](https://zenodo.org/badge/178608195.svg)](https://zenodo.org/badge/latestdoi/178608195)

### 📝 License

[Add license information]

### 🏗️ Architecture Overview

```
polychrom/
├── simulation.py      # Main simulation control
├── forces.py          # Individual force definitions
├── forcekits.py       # Combined force systems  
├── hdf5_format.py     # Trajectory storage
├── polymer_analyses.py # Analysis functions
├── contactmaps.py     # Contact map generation
└── starting_conformations.py # Initial configs
```

**Next steps**: 
- 🚀 **New to polychrom?** Start with our [interactive Google Colab tutorials](tutorials/) - no installation required!
- 📖 **Comprehensive guide**: Check the [quickstart guide](https://polychrom.readthedocs.io/en/latest/quickstart.html) 
- 🧬 **Advanced examples**: Explore [loop extrusion examples](examples/loopExtrusion/) for specialized capabilities
