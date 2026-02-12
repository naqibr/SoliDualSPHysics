# SoliDualSPHysics Contributors

## Core Development Team

### Dr. Naqib Rahimi
**Role**: Principal Developer and Creator  
**Affiliation**: R&D Engineer, Senior Engineer, Synopsys Inc.  
**Contact**: Naqib.rahimy123@gmail.com  
**ORCID**: https://orcid.org/0000-0002-2156-4441

### Dr. Georgios Moutsanidis
**Role**: PhD Advisor and Co-Developer  
**Affiliation**: Professor, Rutgers University  
**Contact**: George.Moutsanidis@rutgers.edu  
**ORCID**: https://orcid.org/0000-0001-9198-2641

---

## Related PhD Thesis

**Title**: "Computational Mechanics of Extreme Events: Advanced Multi-physics Modeling and Simulations with Smoothed Particle Hydrodynamics, Isogeometric Analysis, and Phase Field"

**Author**: Dr. Naqib Rahimi  
**Advisor**: Dr. Georgios Moutsanidis  
**Year**: 2025

---

## SoliDualSPHysics Components

### Core Classes and Modules

#### `JSphDeformStruc` Module
- **Files**: 
  - `JSphDeformStruc.h`
  - `JSphDeformStruc.cpp`
- **Description**: Main framework for deformable structure simulations

#### `JUserExpression` Module
- **Files**:
  - `JSphCpu_ExpressionParser.h`
  - `JSphCpu_ExpressionParser.cpp`
- **Description**: Expression parser for user-defined boundary conditions

#### Material Models
- **Neo-Hookean Hyperelasticity**
- **Mooney-Rivlin Hyperelasticity**
- **J2 Plasticity**
- **Phase-Field Fracture**

#### Boundary Conditions System
- **Velocity BC**
- **Force BC**
- **Phase-Field BC**
- **Expression Evaluation**

---

## DualSPHysics Foundation

SoliDualSPHysics is built upon the DualSPHysics framework. We gratefully acknowledge the DualSPHysics team:

### DualSPHysics Core Developers
- Dr. José M. Domínguez Alonso
- Dr. Alejandro Crespo
- Prof. Moncho Gómez Gesteira
- Prof. Benedict Rogers
- Dr. Georgios Fourtakas
- Prof. Peter Stansby
- Dr. Renato Vacondio
- Dr. Corrado Altomare
- Dr. Angelo Tafuni
- Dr. Orlando García Feal
- Iván Martínez Estévez
- Dr. Joseph O'Connor
- Dr. Aaron English

For the complete list of DualSPHysics contributors, see:  
https://dual.sphysics.org/developers

---

## Code Contributions Breakdown

### SoliDualSPHysics Specific Code (~25,000+ lines)

| Component | LOC (approx) |
|-----------|---------------|--------------|
| JSphDeformStruc Core | ~3,000 |
| Material Models  | ~5,000 |
| Phase-Field Fracture  | ~4,000 |
| Boundary Conditions  | ~3,000 |
| JUserExpression  | ~2,000 |
| Integration & Solvers  | ~6,000 |
| Utilities & I/O  | ~2,000 |

### DualSPHysics Base Framework
- SPH core algorithms
- Neighbor search
- Cell division
- GPU acceleration
- Post-processing tools

---

## Testing and Validation

**Test Cases Developed**: Dr. Naqib Rahimi
- Hyperelastic beam benchmarks
- Plasticity validation cases
- Fracture propagation tests
- Contact mechanics examples

**Validation Studies**: Dr. Naqib Rahimi, Dr. Georgios Moutsanidis

---

## Documentation

**Technical Documentation**
**User Guide**
**Code Comments**
**Example Cases**

---

## Publications

### Primary Publication (In Preparation)

**Title**: "SoliDualSPHysics: An extension of DualSPHysics to solid mechanics with hyperelasticity, plasticity, and fracture"

**Authors**: Naqib Rahimi, Georgios Moutsanidis

**Status**: In preparation (2026)

**Key Contributions**:
- Novel SPH formulation for solid mechanics
- Integration of phase-field fracture with SPH
- Validation against analytical and experimental results

---

## How to Contribute

We welcome contributions from the community! Areas where contributions are particularly valuable:

1. **Additional Material Models**
   - Anisotropic materials
   - Composite materials
   - Temperature-dependent properties

2. **Advanced Features**
   - Multi-body contact algorithms
   - Thermal coupling
   - Fluid-structure interaction enhancements

3. **Performance Optimization**
   - GPU acceleration for new features
   - Multi-GPU support
   - Memory optimization

4. **Testing and Validation**
   - Additional benchmark cases
   - Experimental validation
   - Bug reports and fixes

5. **Documentation**
   - Tutorial videos
   - User examples
   - Theory documentation

### Contribution Process

1. Fork the repository
2. Create a feature branch
3. Implement your changes with appropriate tests
4. Update documentation
5. Submit a pull request

For major contributions, please contact the development team first to discuss the proposed changes.

---

## Contact Information

Dr. Naqib Rahimi  
**Email**: Naqib.rahimi123@gmail.com

Dr. Georgios Moutsanidis  
**Email**: Georgios.moutsanidis@rutgers.edu

---

## License Information

### SoliDualSPHysics Components
Copyright (c) 2024 Dr. Naqib Rahimi and Dr. Georgios Moutsanidis

Licensed under GNU General Public License v3.0 (GPL-3.0) or GNU Lesser General Public License v2.1+ (LGPL-2.1+) depending on the component. See individual source files for specific license information.

### DualSPHysics Base
Copyright (c) 2023 by the DualSPHysics developers  
Licensed under GNU Lesser General Public License v2.1+ (LGPL-2.1+)

---

*This document reflects contributions as of 2026. For the most up-to-date information, please visit the project repository.*
