"""
EC-FEM: Elastic Curve-based Finite Element Method
==================================================

An open-source Python framework for the exact static and dynamic analysis
of non-prismatic (tapered) frame members.

The core idea is to derive element stiffness and mass matrices directly from
the governing differential equation of the elastic curve, so that the
resulting shape functions satisfy equilibrium *exactly* for any continuous
variation of EI(z) and EA(z).  A single EC-FEM element can replace hundreds
of prismatic sub-elements without loss of accuracy.

Companion code for the manuscript:
    Kilinc, M. & Al-Anbagi, M. (2024). EC-FEM: ...

Basic usage
-----------
>>> import sympy as sp
>>> from ecfem import VariableFrameElement, SystemAssembler
>>>
>>> z = sp.symbols('z')
>>> elem = VariableFrameElement(
...     E=200e9,
...     I_expr=(0.2 * (0.4 - 0.04*z)**3) / 12,
...     A_expr=0.2 * (0.4 - 0.04*z),
...     L=5.0,
...     rho=7850,
... )
>>> K, M = elem.get_local_matrices()
"""

from .elements import VariableFrameElement
from .stepwise import PrismaticFrameElement, generate_stepwise_mesh
from .assembly import SystemAssembler

__all__ = [
    "VariableFrameElement",
    "PrismaticFrameElement",
    "generate_stepwise_mesh",
    "SystemAssembler",
]

__version__ = "1.0.0"
__author__ = "Muslum Kilinc, Mustafa Al-Anbagi"
