# EC-FEM: Elastic Curve-based Finite Element Method

> Companion code for the manuscript:  
> **Kilinc, M. & Al-Anbagi, M. (2024). EC-FEM: An exact finite element formulation for non-prismatic frame members.** *[Journal Name]*, doi:[your-doi]

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.xxxx%2Fxxxxx-orange)](https://doi.org/10.xxxx/xxxxx)

---

## Overview

EC-FEM derives element stiffness and mass matrices directly from the governing differential equation of the elastic curve, so the resulting shape functions satisfy equilibrium **exactly** for any continuous variation of the section properties EI(z) and EA(z).

**Key results from the paper:**

| Method | Elements needed | Tip displacement error |
|--------|----------------|----------------------|
| Stepwise (prismatic) | N = 50 | 0.016% |
| Stepwise (prismatic) | N = 1 | 44.8% |
| **EC-FEM (exact)** | **N = 1** | **< 0.001%** |

A single EC-FEM element replaces hundreds of prismatic sub-elements without loss of accuracy.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/ec-fem.git
cd ec-fem

# Install the core library
pip install .

# Install with dependencies for the example scripts
pip install ".[examples]"
```

Or install directly from GitHub:

```bash
pip install "git+https://github.com/<your-username>/ec-fem.git"
```

---

## Quickstart

```python
import sympy as sp
from ecfem import VariableFrameElement, SystemAssembler

# Define a linearly tapered beam: h(z) from 0.40 m to 0.20 m over 5 m
z = sp.symbols('z')
b, h0, h1, L = 0.2, 0.4, 0.2, 5.0
h_z = h0 + (h1 - h0) / L * z

elem = VariableFrameElement(
    E     = 200e9,
    I_expr = b * h_z**3 / 12,
    A_expr = b * h_z,
    L     = L,
    rho   = 7850,
)

K_local, M_local = elem.get_local_matrices()
print("Stiffness matrix (6×6):")
print(K_local)
```

---

## Reproducing the Paper Results

Each example script corresponds to a specific section of the manuscript:

| Script | Paper section | Description |
|--------|--------------|-------------|
| `examples/01_cantilever_validation.py` | Tables 1–2, Figure 5 | Convergence study — tapered cantilever beam |
| `examples/02_portal_frame.py`          | Table 4               | Free-vibration analysis — gable portal frame |
| `examples/03_multistory_frame.py`      | Tables 5–6            | Efficiency benchmark — 5-story 3-bay frame |

Run any example from the repository root:

```bash
python examples/01_cantilever_validation.py
```

---

## Repository Structure

```
ec-fem/
├── ecfem/                      # Core library
│   ├── __init__.py
│   ├── elements.py             # VariableFrameElement  (EC-FEM exact)
│   ├── stepwise.py             # PrismaticFrameElement (reference method)
│   └── assembly.py             # SystemAssembler + solvers
├── examples/
│   ├── 01_cantilever_validation.py
│   ├── 02_portal_frame.py
│   └── 03_multistory_frame.py
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

---

## API Reference

### `VariableFrameElement`

Exact 6×6 stiffness and mass matrices for a non-prismatic 2D frame element.

```python
VariableFrameElement(E, I_expr, A_expr, L, rho)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `E` | `float` | Young's modulus [Pa] |
| `I_expr` | `sympy.Expr` | Second moment of area I(z) [m⁴] |
| `A_expr` | `sympy.Expr` | Cross-sectional area A(z) [m²] |
| `L` | `float` | Element length [m] |
| `rho` | `float` | Mass density [kg/m³] |

**Methods:** `get_local_matrices()`, `get_global_matrices(phi)`

### `SystemAssembler`

Sparse global matrix assembly and solvers.

```python
assembler = SystemAssembler(num_nodes)
assembler.add_element(K_elem, M_elem, node_i, node_j)
K_free, M_free, free_dofs = assembler.get_reduced_system(fixed_dofs)
U = assembler.solve_static(K_free, force_vector)
freqs, modes = assembler.solve_dynamic(K_free, M_free, num_modes=3)
```

---

## Citation

If you use EC-FEM in your research, please cite:

```bibtex
@article{kilinc2024ecfem,
  title   = {EC-FEM: An exact finite element formulation for non-prismatic frame members},
  author  = {Kilinc, Muslum and Al-Anbagi, Mustafa},
  journal = {[Journal Name]},
  year    = {2024},
  doi     = {10.xxxx/xxxxx}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
