# License Information for SoliDualSPHysics

## Overview

SoliDualSPHysics consists of two main components:

1. **DualSPHysics Base Framework** - Licensed under GNU LGPL v2.1+
2. **SoliDualSPHysics Extensions** - Licensed under GNU GPL v3 or GNU LGPL v2.1+ (depending on the component)

---

## DualSPHysics Base Framework

**Copyright**: (c) 2023 by Dr José M. Dominguez et al.  
**License**: GNU Lesser General Public License v2.1 or later (LGPL-2.1-or-later)  
**Website**: http://dual.sphysics.org  
**Developers**: https://dual.sphysics.org/developers

### LGPL License Summary

The DualSPHysics base framework is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation; either version 2.1 of the License, or (at your option) any later version.

DualSPHysics is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with DualSPHysics. If not, see <http://www.gnu.org/licenses/>.

---

## SoliDualSPHysics Extensions

**Copyright**: (c) 2024 by Dr. Naqib Rahimi and Dr. Georgios Moutsanidis  
**License**: GNU General Public License v3 or later (GPL-3.0-or-later) for solid mechanics core;  
           GNU Lesser General Public License v2.1+ (LGPL-2.1-or-later) for certain components

### GPL License Summary

The SoliDualSPHysics solid mechanics extensions are free software: you can redistribute them and/or modify them under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

These extensions are distributed in the hope that they will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with SoliDualSPHysics. If not, see <http://www.gnu.org/licenses/>.

---

## Component-Specific Licenses

### GPL v3 Components

The following solid mechanics components are licensed under GPL v3:

- `JSphDeformStruc.h` / `JSphDeformStruc.cpp` - Deformable structure framework
- Material model implementations (hyperelasticity, plasticity, fracture)
- Phase-field fracture algorithms
- Solid mechanics solvers and integrators

**Reason**: These components implement novel algorithms for solid mechanics that are tightly integrated with the GPL-compatible portions of the codebase.

### LGPL v2.1+ Components

The following components maintain LGPL v2.1+ compatibility:

- `JSphCpu_ExpressionParser.h` / `JSphCpu_ExpressionParser.cpp` - Expression parser
- Utility functions and data structures that don't directly implement solid mechanics algorithms
- XML configuration readers for deformable structures
- Post-processing utilities

**Reason**: These utilities can be independently used and maintain compatibility with the DualSPHysics LGPL licensing.

---

## License Compatibility

### Using SoliDualSPHysics

- **For end users**: You can freely use SoliDualSPHysics for research, commercial, or personal purposes under the terms of the GPL v3 license.

- **For developers**: 
  - If you modify GPL v3 components, your modifications must also be GPL v3
  - If you modify LGPL v2.1+ components, you can choose LGPL v2.1+ or GPL v3
  - If you create new solid mechanics algorithms, we recommend GPL v3 to maintain consistency

- **For commercial applications**:
  - The GPL v3 license requires that derivative works be open-sourced under GPL v3
  - If you need a different licensing arrangement, please contact the authors

### Linking and Distribution

When distributing SoliDualSPHysics or derivative works:

1. **Source code**: Must be made available under GPL v3
2. **Attribution**: Must retain all copyright notices and author attributions
3. **License files**: Must include copies of GPL v3 and LGPL v2.1 licenses
4. **Modifications**: Must be clearly indicated with date and author

---

## Third-Party Components

SoliDualSPHysics may include or link to third-party libraries. Each has its own license:

| Component | License | Usage |
|-----------|---------|-------|
| DualSPHysics | LGPL v2.1+ | Core framework |
| CUDA Toolkit | NVIDIA EULA | GPU acceleration |
| VTK | BSD-3-Clause | Visualization |
| [Other libs] | [License] | [Purpose] |

Please refer to the respective component documentation for detailed license information.

---

## Academic Use and Citation

While the code is open-source, we request that academic users properly cite SoliDualSPHysics in publications:

```bibtex
@article{rahimi2024solidualsphysics,
  title={SoliDualSPHysics: An extension of DualSPHysics to solid mechanics with hyperelasticity, plasticity, and fracture},
  author={Rahimi, Naqib and Moutsanidis, Georgios},
  year={2026},
  note={In preparation}
}
```

And the underlying DualSPHysics framework:

```bibtex
@article{dominguez2022dualsphysics,
  title={DualSPHysics: from fluid dynamics to multiphysics problems},
  author={Dom{\'i}nguez, Jos{\'e} M and others},
  journal={Computational Particle Mechanics},
  volume={9},
  number={5},
  pages={867--895},
  year={2022}
}
```

---

## Patent and Intellectual Property

No patents are currently filed for the algorithms implemented in SoliDualSPHysics. The code is freely available for research and commercial use under the terms of the GPL v3 license.

---

## Warranty Disclaimer

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## License Texts

Full license texts can be found in:

- `LICENSE_GPL3.txt` - GNU General Public License v3.0
- `LICENSE_LGPL2.1.txt` - GNU Lesser General Public License v2.1
- `LICENSE_DualSPHysics.txt` - Original DualSPHysics license

Or online at:
- GPL v3: https://www.gnu.org/licenses/gpl-3.0.html
- LGPL v2.1: https://www.gnu.org/licenses/lgpl-2.1.html

---

## Contact for Licensing Questions

For questions about licensing or to discuss alternative licensing arrangements:

**Dr. Naqib Rahimi**  
Email: Naqib.rahimy123@gmail.com

**Dr. Georgios Moutsanidis**  
Email: Georgios.moutsanidis@rutgers.edu

---
