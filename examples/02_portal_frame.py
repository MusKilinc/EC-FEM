"""
EC-FEM Benchmark 02: Tapered Portal (Gable) Frame
==================================================

Reproduces the free-vibration analysis of the tapered gable frame presented
in Section 5.1 of the manuscript (Table 4).

Geometry
--------
The frame has a total span of L = 6 m and two vertical columns of height
H_col = 3 m.  The apex is located at H_apex = 4 m above ground level
(i.e., 1 m above the eaves), giving a gently pitched roof.  All sections
are rectangular with constant width b = 0.2 m; the depth varies linearly
along each member.

Node coordinates (X, Y) in metres::

    0: (0, 0)   1: (0, 3)   2: (3, 4)   3: (6, 3)   4: (6, 0)

Rafter geometry::

    Left rafter  (node 1 -> node 2): dx=3, dy=1, L=sqrt(10) ~ 3.162 m
    Right rafter (node 2 -> node 3): dx=3, dy=-1

Run
---
    python examples/02_portal_frame.py
"""

import numpy as np
import sympy as sp

from ecfem import VariableFrameElement, SystemAssembler

# =============================================================================
# 1. Material and section parameters
# =============================================================================
E   = 200e9   # Pa
rho = 7850    # kg/m^3
b   = 0.2     # m

z = sp.symbols('z')

# Section depths (paper Section 5.1.1)
h_base = 0.25   # m - column base / rafter apex
h_knee = 0.50   # m - column top / rafter depth at eave
h_apex = 0.25   # m - rafter depth at apex

# Derived geometry
# Apex is at Y = H_apex = 4 m from ground, eaves at Y = H_col = 3 m
# => rise above eaves = 1 m, horizontal half-span = 3 m
L_col     = 3.0
L_raft    = float(np.sqrt(3.0**2 + 1.0**2))  # sqrt(10) ~ 3.1623 m
phi_left  =  np.arctan2(1, 3)                 # ~18.43 deg
phi_right =  np.arctan2(-1, 3)

# =============================================================================
# 2. Assemble (5 nodes x 3 DOF = 15 total DOF)
# =============================================================================
assembler = SystemAssembler(num_nodes=5)
print("Building exact EC-FEM model for the portal frame...")

# Element 0: Left column (node 0 -> node 1, vertical)
h_col = h_base + ((h_knee - h_base) / L_col) * z
K0, M0 = VariableFrameElement(
    E, b*h_col**3/12, b*h_col, L_col, rho
).get_global_matrices(phi=np.pi/2)
assembler.add_element(K0, M0, 0, 1)

# Element 1: Left rafter (node 1 -> node 2)
h_rl = h_knee + ((h_apex - h_knee) / L_raft) * z
K1, M1 = VariableFrameElement(
    E, b*h_rl**3/12, b*h_rl, L_raft, rho
).get_global_matrices(phi=phi_left)
assembler.add_element(K1, M1, 1, 2)

# Element 2: Right rafter (node 2 -> node 3)
h_rr = h_apex + ((h_knee - h_apex) / L_raft) * z
K2, M2 = VariableFrameElement(
    E, b*h_rr**3/12, b*h_rr, L_raft, rho
).get_global_matrices(phi=phi_right)
assembler.add_element(K2, M2, 2, 3)

# Element 3: Right column (node 4 -> node 3, vertical)
# Local z runs bottom-to-top (same taper direction as left column)
K3, M3 = VariableFrameElement(
    E, b*h_col**3/12, b*h_col, L_col, rho
).get_global_matrices(phi=np.pi/2)
assembler.add_element(K3, M3, 4, 3)

# =============================================================================
# 3. Boundary conditions and solution
# =============================================================================
# Nodes 0 and 4 fully fixed
fixed_dofs = [0, 1, 2, 12, 13, 14]
K_free, M_free, active_dofs = assembler.get_reduced_system(fixed_dofs)
freqs, _ = assembler.solve_dynamic(K_free, M_free, num_modes=3)

# =============================================================================
# 4. Output
# =============================================================================
f_ref = 18.215  # Hz - stepwise N=100 reference (paper Section 5.1.3)
print("\n=== Table 4: Natural Frequencies of the Portal Frame ===")
print(f"  Active DOFs (free)  : {len(active_dofs)}")
print(f"  Mode 1  : {freqs[0]:.4f} Hz  "
      f"(reference N=100: {f_ref:.3f} Hz, "
      f"error: {abs(freqs[0]-f_ref)/f_ref*100:.2f}%)")
print(f"  Mode 2  : {freqs[1]:.4f} Hz")
print(f"  Mode 3  : {freqs[2]:.4f} Hz")
print()
print(f"Geometry: apex at Y=4.0 m, rafter L={L_raft:.4f} m, "
      f"phi={np.degrees(phi_left):.2f} deg.")
print("Single element per member captures near-exact dynamic behaviour.")
