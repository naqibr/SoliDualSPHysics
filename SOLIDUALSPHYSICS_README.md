# SoliDualSPHysics

## An Extension of DualSPHysics to Solid Mechanics

**SoliDualSPHysics** extends the capabilities of [DualSPHysics](http://dual.sphysics.org) to solid mechanics simulations, enabling the modeling of deformable structures with hyperelasticity, plasticity, and fracture.

---

## Key Features

- **Hyperelastic Material Models**: Neo-Hookean and Mooney-Rivlin constitutive models
- **J2 Plasticity**: Von Mises plasticity with isotropic hardening
- **Phase-Field Fracture**: Advanced fracture modeling with crack propagation
- **Flexible Boundary Conditions**: User-defined expressions for time and space-dependent forcing
- **Seamless Integration**: Built on top of DualSPHysics framework

---

## Authors and Contributors

### SoliDualSPHysics Development

**Dr. Naqib Rahimi** (Principal Developer)  
PhD Thesis: "Computational Mechanics of Extreme Events: Advanced Multi-physics Modeling and Simulations with Smoothed Particle Hydrodynamics, Isogeometric Analysis, and Phase Field"

**Dr. Georgios Moutsanidis** (PhD Advisor and Co-Developer)

### DualSPHysics Core Team

The SoliDualSPHysics extensions build upon the excellent foundation provided by the DualSPHysics team. See the [DualSPHysics developers page](https://dual.sphysics.org/developers) for the full list of contributors.

---

## Related Publications

### SoliDualSPHysics

Rahimi, N., & Moutsanidis, G. (2024). "SoliDualSPHysics: An extension of DualSPHysics to solid mechanics with hyperelasticity, plasticity, and fracture." *[In preparation]*

### DualSPHysics

Domínguez, J.M., et al. (2022). DualSPHysics: from fluid dynamics to multiphysics problems. *Computational Particle Mechanics*, 9(5), 867-895.  
[https://doi.org/10.1007/s40571-021-00404-2](https://doi.org/10.1007/s40571-021-00404-2)

---

## License

SoliDualSPHysics is distributed under the same licenses as DualSPHysics:

- **GNU Lesser General Public License (LGPL) v2.1+** for most components
- **GNU General Public License (GPL) v3** for certain solid mechanics modules

See the LICENSE files in the source distribution for complete license information.

---

## Getting Started

### Prerequisites

- C++ compiler with C++11 support or later
- CUDA Toolkit (for GPU execution)
- CMake 3.10 or later
- Dependencies from DualSPHysics (see [DualSPHysics documentation](http://dual.sphysics.org))

### Building from Source

```bash
# Clone the repository
git clone [your-repository-url]
cd SoliDualSPHysics

# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
make -j4
```

### Running Simulations

SoliDualSPHysics uses XML configuration files similar to DualSPHysics, with additional sections for deformable structures:

```bash
./DualSPHysics5 -cpu case_config.xml
```

See the `SolidExamples/` directory for sample cases demonstrating:
- Hyperelastic beam bending
- Plate with J2 plasticity
- Fracture propagation with phase-field method

---

## Key Components

### Core Classes

- **`JSphDeformStruc`**: Main manager for deformable structure simulations
- **`JSphDeformStrucBody`**: Individual deformable body with material properties
- **`JUserExpression`**: Expression parser for boundary conditions

### Features

#### Material Models

1. **Hyperelasticity**
   - Neo-Hookean
   - Saint Venant Kirchoff

2. **Plasticity**
   - J2 (Von Mises) hyperelastic-plasticity
   - Isotropic hardening

3. **Fracture**
   - Phase-field model for brittle fracture

#### Boundary Conditions

- Velocity boundary conditions (Dirichlet)
- Force boundary conditions (Neumann)
- Phase-field boundary conditions
- User-defined expressions supporting:
  - Time-dependent functions
  - Space-dependent functions
  - Mathematical expressions (sin, cos, pow, if, etc.)

---

## Documentation

### API Documentation

Generate Doxygen documentation:

```bash
cd doc
doxygen Doxyfile
```
---

## Examples

The `SolidExamples/` directory includes:

1. **Cantilever Beam**: Hyperelastic beam under gravity
2. **Compression Test**: J2 plasticity validation
3. **Three-Point Bending**: Phase-field fracture demonstration
4. **Impact Test**: Dynamic loading with contact

---

## Citation

If you use SoliDualSPHysics in your research, please cite:

```bibtex
@article{rahimi2024solidualsphysics,
  title={SoliDualSPHysics: An extension of DualSPHysics to solid mechanics with hyperelasticity, plasticity, and fracture},
  author={Rahimi, Mohammad Naqib and Moutsanidis, Georgios},
  year={2026},
}

@article{dominguez2022dualsphysics,
  title={DualSPHysics: from fluid dynamics to multiphysics problems},
  author={Dom{\'i}nguez, Jos{\'e} M and Fourtakas, Georgios and Altomare, Corrado and Canelas, Ricardo B and Tafuni, Angelo and Garc{\'i}a-Feal, Orlando and Mart{\'i}nez-Est{\'e}vez, Iv{\'a}n and Mokos, Athanasios and Vacondio, Renato and Crespo, Alejandro JC and others},
  journal={Computational Particle Mechanics},
  volume={9},
  number={5},
  pages={867--895},
  year={2022},
  publisher={Springer}
}

@phdthesis{Rahimi2025phdthesis,
  author  = {Rahimi, M. Naqib},
  title   = {Computational Mechanics of Extreme Events: Advanced Multi-physics Modeling and Simulations with Smoothed Particle Hydrodynamics, Isogeometric Analysis, and Phase Field},
  school  = {Stony Brook University},
  year    = {2025},
  address = {Stony Brook, NY, United States},
  type    = {Phd Dissertation},
}

```

---

## Support and Contact

- **Issues**: Please report bugs and feature requests via GitHub Issues
- **Discussions**: Use GitHub Discussions for questions and community support
- **Email**: [Your contact information]

---

## Acknowledgments

This work was developed as part of Dr. Naqib Rahimi's PhD thesis under the supervision of Dr. Georgios Moutsanidis.

We gratefully acknowledge:
- The DualSPHysics development team for providing the excellent SPH framework

---

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for guidelines.

---

## Changelog

### Version 1.0 (2024)
- Initial release of SoliDualSPHysics
- Hyperelastic materials (Neo-Hookean, Saint Venant Kirchoff)
- J2 plasticity with hardening
- Phase-field fracture
- User expression parser for boundary conditions

---
