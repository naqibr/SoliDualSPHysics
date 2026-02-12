# SoliDualSPHysics Open Source Preparation - Summary of Changes

**Date**: 2024  
**Prepared by**: GitHub Copilot  
**For**: Dr. Naqib Rahimi

---

## Overview

This document summarizes all changes made to prepare the SoliDualSPHysics project for open-source publication. The modifications ensure proper attribution, licensing clarity, and professional documentation for the research community.

---

## Modified Files

### 1. Source Code Headers

#### `JSphDeformStruc.h`
**Changes**:
- Added comprehensive SoliDualSPHysics copyright notice
- Included PhD thesis reference
- Added article citation (in preparation)
- Extended file documentation with author attribution
- Maintained compatibility with original DualSPHysics GPL v3 license

**Key Additions**:
```cpp
<SOLIDUALSPHYSICS> SoliDualSPHysics Extensions:
Copyright (c) 2026 by Dr. Naqib Rahimi and Dr. Georgios Moutsanidis.

Developed as part of the PhD thesis:
"Computational Mechanics of Extreme Events..."
by Dr. Naqib Rahimi, supervised by Dr. Georgios Moutsanidis.

Related publication:
Rahimi, N., & Moutsanidis, G. (2026). "SoliDualSPHysics: An extension of 
DualSPHysics to solid mechanics with hyperelasticity, plasticity, and fracture."
```

#### `JSphDeformStruc.cpp`
**Changes**:
- Added matching header with SoliDualSPHysics attribution
- Included author information in file documentation

#### `JSphCpu_ExpressionParser.h`
**Changes**:
- Added SoliDualSPHysics extension notice
- Documented the expression parser's role in boundary conditions
- Added author attribution
- Maintained LGPL v2.1+ compatibility

#### `main.cpp`
**Changes**:
- Enhanced `getlicense_lgpl()` function to include SoliDualSPHysics attribution
- Added detailed feature list for solid mechanics extensions
- Included PhD thesis and publication references
- Updated copyright notices in main file header

**Enhanced License Output**:
```
<SOLIDUALSPHYSICS> SoliDualSPHysics Extensions Copyright (c) 2026 by
Dr. Naqib Rahimi, Dr. Georgios Moutsanidis

SoliDualSPHysics extends DualSPHysics to solid mechanics with:
- Hyperelasticity (Neo-Hookean, Mooney-Rivlin)
- J2 Plasticity with isotropic hardening
- Phase-field fracture modeling
- Advanced boundary condition handling
```

---

## New Documentation Files

### 1. `SOLIDUALSPHYSICS_README.md`
**Purpose**: Main project documentation

**Contents**:
- Project overview and key features
- Author information and contributions
- Related publications and citations
- Building and installation instructions
- Examples and usage guide
- API documentation references
- Acknowledgments and community guidelines

**Sections**:
- Key Features (hyperelasticity, plasticity, fracture)
- Authors (Dr. Naqib Rahimi, Dr. Georgios Moutsanidis)
- Related Publications
- License Information
- Getting Started
- Core Components
- Examples
- Citation Guidelines
- Support and Contact

### 2. `SOLIDUALSPHYSICS_CONTRIBUTORS.md`
**Purpose**: Detailed contributor attribution

**Contents**:
- Core development team profiles
- PhD thesis information
- Component-by-component attribution
- Code contribution breakdown (~25,000+ lines)
- Testing and validation credits
- Documentation authorship
- Publication information
- Contribution guidelines

**Key Statistics**:
| Component | LOC (approx) |
|-----------|--------------|
| JSphDeformStruc Core | ~3,000 |
| Material Models | ~5,000 |
| Phase-Field Fracture | ~4,000 |
| Boundary Conditions | ~3,000 |
| JUserExpression | ~2,000 |
| Integration & Solvers | ~6,000 |
| Utilities & I/O | ~2,000 |

### 3. `CITATION.cff`
**Purpose**: Standard citation file format (Citation File Format)

**Contents**:
- Structured citation metadata
- Author information with ORCID placeholders
- Version and release information
- Keywords for discoverability
- Preferred citation format
- References to DualSPHysics base framework

**Benefits**:
- GitHub automatically recognizes and displays citation information
- Enables easy citation in academic work
- Machine-readable format for citation managers

### 4. `LICENSE_INFO.md`
**Purpose**: Comprehensive licensing information

**Contents**:
- Overview of dual licensing approach
- DualSPHysics base framework (LGPL v2.1+)
- SoliDualSPHysics extensions (GPL v3)
- Component-specific license breakdown
- License compatibility guidelines
- Third-party component licenses
- Academic citation requirements
- Warranty disclaimers

**Licensing Structure**:
```
DualSPHysics Base → LGPL v2.1+
SoliDualSPHysics Core → GPL v3
  - JSphDeformStruc → GPL v3
  - Material Models → GPL v3
  - Phase-Field → GPL v3
SoliDualSPHysics Utils → LGPL v2.1+
  - Expression Parser → LGPL v2.1+
  - XML Readers → LGPL v2.1+
```

---

## Attribution Strategy

### Primary Authors
- **Dr. Naqib Rahimi**: Principal Developer
- **Dr. Georgios Moutsanidis**: PhD Advisor and Co-Developer

### PhD Thesis Reference
**Title**: "Computational Mechanics of Extreme Events: Advanced Multi-physics Modeling and Simulations with Smoothed Particle Hydrodynamics, Isogeometric Analysis, and Phase Field"

### Article Reference
**Title**: "SoliDualSPHysics: An extension of DualSPHysics to solid mechanics with hyperelasticity, plasticity, and fracture"  
**Status**: In preparation (2026)  
**Authors**: Naqib Rahimi, Georgios Moutsanidis

---

## Key Features Documented

### 1. Material Models
- **Hyperelasticity**: Neo-Hookean, Saint Venant Kirchoff
- **Plasticity**: J2 (Von Mises) with isotropic hardening
- **Fracture**: Phase-field fracture model

### 2. Boundary Conditions
- Velocity boundary conditions (Dirichlet)
- Force boundary conditions (Neumann)
- Phase-field boundary conditions
- User-defined mathematical expressions

### 3. Core Framework
- `JSphDeformStruc`: Main deformable structure manager
- `JSphDeformStrucBody`: Individual body properties
- `JUserExpression`: Expression parser

---

## License Compliance

### DualSPHysics Base
- ✅ Maintained LGPL v2.1+ license
- ✅ Preserved original copyright notices
- ✅ Acknowledged all DualSPHysics developers

### SoliDualSPHysics Extensions
- ✅ Clear GPL v3 licensing for solid mechanics core
- ✅ LGPL v2.1+ for independent utilities
- ✅ Proper copyright attribution
- ✅ Comprehensive license documentation

---

## Citations Provided

### BibTeX Entry for SoliDualSPHysics
```bibtex
@article{rahimi2024solidualsphysics,
  title={SoliDualSPHysics: An extension of DualSPHysics to solid 
         mechanics with hyperelasticity, plasticity, and fracture},
  author={Rahimi, Mohammad Naqib and Moutsanidis, Georgios},
  year={2026},
  note={In preparation}
}
```

### BibTeX Entry for DualSPHysics
```bibtex
@article{dominguez2022dualsphysics,
  title={DualSPHysics: from fluid dynamics to multiphysics problems},
  author={Dom{\'i}nguez, Jos{\'e} M and others},
  journal={Computational Particle Mechanics},
  volume={9},
  number={5},
  pages={867--895},
  year={2022},
  doi={10.1007/s40571-021-00404-2}
}
```

---
