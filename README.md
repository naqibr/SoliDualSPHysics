<div align="center">

# SoliDualSPHysics

**An Open-Source Extension of DualSPHysics for Solid Mechanics**

[Key Contributions](#key-contributions) •
[Features](#features) •
[Repository Structure](#repository-structure) •
[Getting Started](#getting-started) •
[Examples](#examples) •
[Citation](#citation) •
[License](#license)

</div>

---

**SoliDualSPHysics** extends [DualSPHysics](https://dual.sphysics.org) to **solid mechanics** simulations using Smoothed Particle Hydrodynamics (SPH). It introduces hyperelastic material models, finite-strain J2 plasticity with isotropic hardening, phase-field fracture modeling, and a versatile mathematical expression parser for user-defined boundary conditions, all with CPU parallelization and full GPU acceleration via CUDA.

> **Based on:** DualSPHysics v5.2.269  
> ([Domínguez et al., 2022](https://doi.org/10.1007/s40571-021-00404-2))

---

## Key Contributions

SoliDualSPHysics brings two major contributions to the DualSPHysics ecosystem:

### 1. Solid Mechanics Framework (`JSphDeformStruc`)

A complete deformable structure simulation framework within the SPH paradigm, including:

- **Hyperelastic constitutive models**: Saint Venant–Kirchhoff and Neo-Hookean formulations for large-deformation analysis.
- **Finite-strain J2 (Von Mises) plasticity**: with isotropic hardening, enabling the simulation of metallic and ductile materials under extreme loading.
- **Phase-field fracture modeling**: a diffuse-crack approach for brittle fracture, dynamic crack branching, and crack propagation without explicit crack tracking.
- **Deformable body management**: each body carries its own material properties (Young's modulus, Poisson's ratio, yield stress, hardening modulus, critical energy release rate, etc.) and boundary conditions.
- **Measuring planes**: for extracting data at user-specified locations in the domain.
- **GPU/CPU-accelerated solid mechanics kernels**: all solid mechanics operations (deformation gradients, stress updates, fracture field evolution, contact detection) are parallelized on both CPU using OpenMP and GPU via CUDA.

### 2. Expression Parser (`JUserExpression`)

A flexible, self-contained mathematical expression parser that enables user-defined boundary conditions directly in XML configuration files (no recompilation):

- **Time-dependent** expressions using the variable `t`.
- **Space-dependent** expressions using initial coordinates (`x0`, `y0`, `z0`), current coordinates (`x`, `y`, `z`), and displacements (`ux`, `uy`, `uz`).
- **Rich function library** — `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `cot`, `coth`, `sqrt`, `log`, `ln`, `pow`, `abs`, and conditional `if(condition, true_val, false_val)`.
- **Logical and comparison operators** — `<`, `>`, `<=`, `>=`, `==`, `!=`, `and`, `or`.
- **Local variables** — define reusable constants within expressions.
- **Compiled evaluation** — expressions are tokenized, parsed to Reverse Polish Notation (RPN), and compiled to an efficient operation sequence for both CPU and GPU execution.
- **`skip` sentinel** — selectively leaves particles unconstrained (free).

---

## Features

| Feature | Description |
|---|---|
| **Hyperelasticity** | Saint Venant–Kirchhoff (SVK), Neo-Hookean models |
| **Finite-strain plasticity** | J2 (Von Mises) with isotropic hardening |
| **Phase-Field Fracture** | Brittle fracture with crack branching and propagation |
| **Expression Parser** | User-defined math expressions for boundary conditions |
| **GPU Acceleration** | Full CUDA support for all solid mechanics computations |
| **CPU Execution** | Complete CPU-only mode (no GPU required) |
| **Boundary Conditions** | Velocity (Dirichlet), force (Neumann), and phase-field BCs |
| **Post-Processing** | VTK output (ParaView compatible), CSV export, measuring planes |
| **9 Benchmark Examples** | Validated against analytical and experimental references |

---

## Repository Structure

```text
SoliDualSPHysics/
├── src/
│   ├── source/                         # C++ and CUDA source code
│   │   ├── JSphDeformStruc.cpp/h        # Core solid mechanics framework
│   │   ├── JSphCpu_ExpressionParser.cpp/h  # Expression parser (CPU)
│   │   ├── JSphGpu_ExpressionParser.cu/h   # Expression parser (GPU)
│   │   ├── JSphGpu_DefStruc_ker.cu/h       # GPU kernels for solid mechanics
│   │   ├── CMakeLists.txt               # CMake build configuration
│   │   ├── Makefile                     # Makefile build configuration
│   │   └── ...                          # DualSPHysics core source files
│   ├── lib/                             # Pre-compiled static libraries
│   │   ├── linux_gcc/                   # Linux (GCC) libraries
│   │   ├── vs2022/                      # Visual Studio 2022 libraries
│   │   └── vs2026/                      # Visual Studio 2026 libraries
│   └── VS/                              # Visual Studio project files & documentation
├── SolidExamples/                       # 9 benchmark simulation cases
│   ├── 1_Free_Oscillation_of_a_Cantilever_Beam/
│   ├── 2_Free_Oscillation_of_a_Cantilever_Plate/
│   ├── 3_Large_Deformation_of_a_3D_Cantilever_Beam/
│   ├── 4_Twisting_3D_Column/
│   ├── 5_Dynamic_Crack_Branching/
│   ├── 6_Kalthoff_Winkler_Experiment/
│   ├── 7_Four_Point_Bending/
│   ├── 8_Flyer_Plates_Impact/
│   └── 9_3D_Taylor_Bar_Impact/
├── DSPartVTK/                           # Post-processing tool for VTK output
├── bin/                                 # Pre-compiled binaries
│   ├── linux/
│   └── windows/
└── doc/                                 # Documentation and guides
    ├── guides/
    ├── help/
    └── xml_format/
```

---

## Getting Started

### Prerequisites

| Requirement | Details |
|---|---|
| **C++ Compiler** | GCC (Linux) or MSVC 2022+ (Windows) with C++11 support |
| **CUDA Toolkit** | Version 7.5 or later (v11+ recommended) — *required for GPU execution* |
| **CMake** | Version 3.0 or later |
| **OpenMP** | Recommended for CPU parallel execution |

### Building from Source

#### Linux (Make — recommended)

```bash
cd src/source
# Build GPU+CPU version (requires CUDA)
make -j"$(nproc)"
# Executable placed in ../../bin/linux/
# Output: SoliDualSPHysics_linux64
```

Build options can be configured at the top of the `Makefile`:

```makefile
USE_DEBUG        ?= NO      # Debug build
USE_FAST_MATH    ?= YES     # Fast math optimizations
COMPILE_CHRONO   ?= YES     # Chrono Engine coupling
COMPILE_MOORDYN  ?= YES     # MoorDyn+ coupling
```

#### Linux (CMake)

```bash
cd src/source
mkdir -p build && cd build
cmake ..
make -j"$(nproc)"
# CPU-only executable: DualSPHysics5.2_FlexStrucCPU_linux64
# GPU executable:      DualSPHysics5.2_FlexStruc_linux64
```

#### Windows (Visual Studio)

1. Open the Visual Studio solution in `src/VS/`.
2. Select **Release** configuration and **x64** platform.
3. Build the solution.

Alternatively, with CMake:

```bat
cd src\source
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

### Verifying the Build

```bash
# Linux
./bin/linux/SoliDualSPHysics_linux64 -h

# Windows
bin\windows\SoliDualSPHysicsCPU_win64.exe -h
```

---

## Running Simulations

SoliDualSPHysics simulations follow a two-step workflow:

### Step 1: Generate the Case

```bash
./bin/linux/GenCase_linux64 \
  SolidExamples/1_Free_Oscillation_of_a_Cantilever_Beam/Solid_Case_Def \
  output/Solid_Case \
  -save:all
```

### Step 2: Run the Solver

```bash
# CPU execution
./bin/linux/SoliDualSPHysics_linux64 -cpu output/Solid_Case output -dirdataout bindata -svres

# GPU execution
./bin/linux/SoliDualSPHysics_linux64 output/Solid_Case output -dirdataout bindata -svres
```

### Step 3: Post-Processing

Convert binary output to VTK format for visualization in [ParaView](https://www.paraview.org/):

```bash
./DSPartVTK/bin/DSPartVTK_linux64 \
  -dirin output/bindata \
  -filexml output/Solid_Case.xml \
  -savevtk output/particles/DefStrucBody_%mk%
```
---

## XML Configuration

Simulations are defined in XML configuration files.

### Defining Material Properties

```xml
<special>
  <deformstrucs>
    <deformstrucbody mkbound="1">
      <!-- Material properties -->
      <density value="1000.0" comment="Mass density (kg/m³)" />
      <u_mu value="0.715e6" comment="Shear modulus (Pa)" />
      <u_bulk value="3.25e6" comment="Bulk modulus (Pa)" />

      <!-- Constitutive model: 1=SVK, 2=Neo-Hookean -->
      <constitmodel value="1" comment="Constitutive model" />

      <!-- Artificial viscosity -->
      <artvisc factor1="0.015" factor2="0.01" />

      <!-- Map factor for refined particle discretization -->
      <mapfac value="4" />

      <!-- Velocity boundary conditions (reference expression IDs) -->
      <bcvel ze="1" xe="2" ye="2" />
    </deformstrucbody>
  </deformstrucs>
</special>
```

### Enabling Fracture

```xml
<deformstrucbody mkbound="1">
  <!-- ... material properties ... -->
  <fracture value="1" comment="Enable phase-field fracture" />
  <gc value="3.0" comment="Critical energy release rate (J/m²)" />
  <pflim value="0.9" comment="Phase-field limit for crack surface" />
</deformstrucbody>
```

### Enabling Plasticity

```xml
<deformstrucbody mkbound="1">
  <!-- ... material properties ... -->
  <constitmodel value="3" comment="J2 plasticity" />
  <yieldstress value="300e6" comment="Initial yield stress (Pa)" />
  <hardening value="100e6" comment="Isotropic hardening modulus (Pa)" />
</deformstrucbody>
```

### Defining Mathematical Expressions for Boundary Conditions

```xml
<special>
  <mathexpressions>
    <userexpression id="1" comment="Initial velocity profile">
      <locals value="L0=0.2; kw=9.375; cs=57.0" />
      <expression value="if(x0<=0.0, 0.0,
        if(t<=0.0,
          0.01 * cs * ((cos(kw*L0)+cosh(kw*L0))*(cosh(kw*x0)-cos(kw*x0))
          + (sin(kw*L0)-sinh(kw*L0))*(sinh(kw*x0)-sin(kw*x0)))
          / ((cos(kw*L0)+cosh(kw*L0))*(cosh(kw*L0)-cos(kw*L0))
          + (sin(kw*L0)-sinh(kw*L0))*(sinh(kw*L0)-sin(kw*L0))),
        skip))" />
    </userexpression>

    <userexpression id="2" comment="Clamped boundary">
      <expression value="if(x0<=0.0, 0.0, skip)" />
    </userexpression>
  </mathexpressions>
</special>
```

### Expression Parser Reference

| Category | Available Items |
|---|---|
| **Spatial Variables** | `x0`, `y0`, `z0` (initial), `x`, `y`, `z` (current), `ux`, `uy`, `uz` (displacement) |
| **Temporal Variables** | `t` (current time), `dt` (time step) |
| **Other Variables** | `dx` (particle spacing) |
| **Math Functions** | `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `cot`, `coth`, `sqrt`, `log`, `ln`, `pow`, `abs` |
| **Conditional** | `if(condition, true_value, false_value)` |
| **Comparison** | `<`, `>`, `<=`, `>=`, `==`, `!=` |
| **Logical** | `and`, `or` |
| **Operators** | `+`, `-`, `*`, `/`, `^` |
| **Special** | `skip` — leaves the particle unconstrained |
| **Local Variables** | Defined in `<locals>` as semicolon-separated `name=value` pairs |

---

## Examples

The `SolidExamples/` directory contains 9 validated benchmark cases:

| # | Example | Physics | Description |
|---:|---|---|---|
| 1 | **Free Oscillation of a Cantilever Beam** | Hyperelasticity (SVK) | 2D beam vibration with analytical mode shape initialization |
| 2 | **Free Oscillation of a Cantilever Plate** | Hyperelasticity | 3D plate vibration benchmark |
| 3 | **Large Deformation of a 3D Cantilever Beam** | Hyperelasticity | Nonlinear large-displacement bending |
| 4 | **Twisting 3D Column** | Hyperelasticity | Large-deformation torsion benchmark |
| 5 | **Dynamic Crack Branching** | Fracture | Phase-field fracture with crack branching |
| 6 | **Kalthoff–Winkler Experiment** | Fracture | Impact-induced fracture validation |
| 7 | **Four-Point Bending** | Fracture | Quasi-static fracture propagation |
| 8 | **Flyer Plates Impact** | Plasticity | J2 plasticity with impact and mushrooming |
| 9 | **3D Taylor Bar Impact** | Plasticity | J2 plasticity under high-strain-rate impact |

Each example directory contains:

- `Solid_Case_Def.xml` — simulation definition file  
- `Run_Solid_Case_win` — Windows batch script for CPU/GPU execution  
- `Run_Solid_Case_linux` — Linux bash script for CPU/GPU execution  

### Running an Example with Scripts (Windows + Linux)

Each example folder includes scripts to run the case end-to-end. Both scripts are **interactive** and will prompt you for:
- **CPU vs GPU**
- What to do if the output folder already exists (delete / post-process / abort)

#### Windows: Interactive CPU/GPU Run (`.bat`)

Script name: `Run_Solid_Case_win.bat`

What it does:
1. Prompts for execution mode: **[1] CPU** or **[2] GPU**
2. Checks if the output folder exists (prompts to delete / post-process / abort)
3. Runs `GenCase_win64.exe`
4. Runs solver:
   - CPU: `SoliDualSPHysicsCPU_win64.exe`
   - GPU: `SoliDualSPHysics_win64.exe` with `-gpu`
5. Converts binary output to VTK using `DSPartVTK_win64.exe`

From inside an example directory (e.g., `SolidExamples\1_Free_Oscillation_of_a_Cantilever_Beam\`), run:

```bat
Run_Solid_Case_win.bat
```

---

## Core Architecture

### Key Classes

| Class | File | Description |
|---|---|---|
| `JSphDeformStruc` | `JSphDeformStruc.cpp/h` | Main manager for all deformable structures in a simulation |
| `JSphDeformStrucBody` | `JSphDeformStruc.cpp/h` | Individual deformable body with its own material properties, BCs, and state |
| `JUserExpression` | `JSphCpu_ExpressionParser.cpp/h` | Tokenizes, parses to RPN, and evaluates math expressions |
| `JUserExpressionList` | `JSphCpu_ExpressionParser.cpp/h` | Manages all user expressions loaded from XML |
| `JUserExpressionGPU` | `JSphGpu_ExpressionParser.cu/h` | GPU-compiled expression evaluator |

### Computation Pipeline

```text
XML Config → GenCase → Particle Generation → Neighbor Search → Kernel Evaluation → Solver Loop:
  ├── Deformation Gradient Computation
  ├── Stress Update (SVK / Neo-Hookean / J2 Plasticity)
  ├── Phase-Field Fracture Evolution (if enabled)
  ├── Boundary Condition Application (Expression Parser)
  └── Acceleration & Velocity Update (Verlet / Symplectic)
→ Output (VTK / CSV)
```

---

## Citation

If you use SoliDualSPHysics in your research, please cite both the SoliDualSPHysics extensions and the underlying DualSPHysics framework:

```bibtex
@article{rahimi2026solidualsphysics,
  title   = {SoliDualSPHysics: An extension of DualSPHysics for solid mechanics
             with hyperelasticity, plasticity, and fracture},
  author  = {Rahimi, Mohammad Naqib and Moutsanidis, George},
  year    = {2026},
  doi     = {10.48550/arXiv.2602.15149}
}

@phdthesis{rahimi2025phdthesis,
  author  = {Rahimi, M. Naqib},
  title   = {Computational Mechanics of Extreme Events: Advanced Multi-physics
             Modeling and Simulations with Smoothed Particle Hydrodynamics,
             Isogeometric Analysis, and Phase Field},
  school  = {Stony Brook University},
  year    = {2025},
  address = {Stony Brook, NY, United States},
  type    = {PhD Dissertation}
}

@article{dominguez2022dualsphysics,
  title     = {DualSPHysics: from fluid dynamics to multiphysics problems},
  author    = {Dom{\'\i}nguez, Jos{\'e} M and Fourtakas, Georgios and
               Altomare, Corrado and Canelas, Ricardo B and Tafuni, Angelo and
               Garc{\'\i}a-Feal, Orlando and Mart{\'\i}nez-Est{\'e}vez, Iv{\'a}n and
               Mokos, Athanasios and Vacondio, Renato and
               Crespo, Alejandro JC and others},
  journal   = {Computational Particle Mechanics},
  volume    = {9},
  number    = {5},
  pages     = {867--895},
  year      = {2022},
  publisher = {Springer},
  doi       = {10.1007/s40571-021-00404-2}
}
```

---

## License

SoliDualSPHysics is licensed under the GNU Lesser General Public License v2.1 or later (LGPL-2.1-or-later).

SoliDualSPHysics extends DualSPHysics, which is also licensed under LGPL-2.1-or-later. All new source files introduced in this repository are distributed under the same license for compatibility.

See the `LICENSE` file for the full license text.

---

## Contributing

We welcome contributions from the community! Areas where contributions are particularly valuable:

- **Additional material models** — anisotropic materials, composites, temperature-dependent properties
- **Advanced features** — multi-body contact, thermal coupling, fluid-structure interaction
- **Performance** — multi-GPU support, memory optimization
- **Testing & validation** — new benchmark cases, experimental validation
- **Documentation** — tutorials, user examples, theory documentation

### How to Contribute

1. **Fork** the repository.
2. **Create a feature branch** (`git checkout -b feature/my-feature`).
3. **Implement your changes** with clear comments and documentation.
4. **Test** your changes against the provided examples.
5. **Submit a pull request** with a description of your changes.

For major contributions, please open an issue first to discuss the proposed changes.

---

## Authors

### SoliDualSPHysics Development

**Dr. Naqib Rahimi** — *Principal Developer and Creator*  
- R&D Engineer, Synopsys Inc.  
- ORCID: [0000-0002-2156-4441](https://orcid.org/0000-0002-2156-4441)  
- Email: Naqib.rahimy123@gmail.com  

**Dr. George Moutsanidis** — *PhD Advisor and Co-Developer*  
- Professor, Rutgers University  
- ORCID: [0000-0001-9198-2641](https://orcid.org/0000-0001-9198-2641)  
- Email: George.Moutsanidis@rutgers.edu  

### DualSPHysics Core Team

SoliDualSPHysics is built upon the DualSPHysics framework. We gratefully acknowledge the [DualSPHysics development team](https://dual.sphysics.org/developers).

---

## Acknowledgments

This work was developed as part of Dr. Naqib Rahimi's PhD thesis under the supervision of Dr. George Moutsanidis at Stony Brook University, with the support of National Science Foundation grant #2545336. We also gratefully acknowledge the DualSPHysics development team for providing the SPH framework upon which SoliDualSPHysics is built.

---

## Support

- **Issues**: Report bugs and request features via [GitHub Issues](../../issues)  
- **Discussions**: Ask questions and engage with the community via [GitHub Discussions](../../discussions)  
- **Email**: Naqib.rahimy123@gmail.com  
