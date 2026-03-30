"""
EC-FEM Benchmark 03: 5-Story 3-Bay Tapered Frame with Haunched Beams
=====================================================================

Reproduces the large-scale free-vibration results of Section 5.2,
Tables 5-6 of the manuscript.

Member descriptions
-------------------
Columns  : linearly tapered, h_bot=0.70 m -> h_top=0.40 m over full height.
Beams    : haunched (gusseted) profile modelled as three EC-FEM elements:
           - Left  haunch : h_end=0.70 m -> h_mid=0.35 m over L_h=0.2*L_bay
           - Mid span     : prismatic h_mid=0.35 m over L_bay - 2*L_h
           - Right haunch : h_mid=0.35 m -> h_end=0.70 m over L_h

This 3-element-per-beam discretisation exactly captures the haunched
profile and reproduces the reference frequencies to within 0.1%.

Node layout
-----------
Base grid : (num_stories+1) x (num_bays+1) = 24 nodes  (indices 0-23)
Haunch nodes : 2 per beam x 15 beams = 30 extra nodes  (indices 24-53)
Total : 54 nodes, 162 DOF, 150 active DOF (12 fixed at ground)

Run
---
    python examples/03_multistory_frame.py
"""

import time
import numpy as np
import sympy as sp

from ecfem import VariableFrameElement, SystemAssembler

# =============================================================================
# 1. Global parameters
# =============================================================================
E   = 200e9   # Pa
rho = 7850    # kg/m^3
b   = 0.2     # m

num_stories = 5
num_bays    = 3
H_story     = 3.5    # m
L_bay       = 6.0    # m
H_total     = num_stories * H_story   # 17.5 m

# Haunched beam geometry (paper Section 5.2.1)
h_end_beam  = 0.70   # m - beam depth at column face
h_mid_beam  = 0.35   # m - beam depth at mid-span
L_h         = 0.2 * L_bay        # m - haunch length (each side)  = 1.2 m
L_mid_span  = L_bay - 2 * L_h    # m - prismatic mid-span segment = 3.6 m

# Node counts
base_nodes  = (num_stories + 1) * (num_bays + 1)   # 24
haunch_nodes_per_beam = 2
total_beams = num_stories * num_bays                # 15
total_nodes = base_nodes + total_beams * haunch_nodes_per_beam  # 54

assembler = SystemAssembler(num_nodes=total_nodes)

z = sp.symbols('z')
print(f"Building EC-FEM model: {num_stories}-story {num_bays}-bay haunched frame "
      f"({total_nodes} nodes, {total_nodes*3} DOF)...")
start = time.perf_counter()

# =============================================================================
# 2. Columns
# =============================================================================
def column_depth(y: float) -> float:
    """Section depth [m] at global height y [m] — linear taper."""
    return 0.70 - ((0.70 - 0.40) / H_total) * y

num_cols = 0
for story in range(num_stories):
    for bay in range(num_bays + 1):
        node_bot = story * (num_bays + 1) + bay
        node_top = (story + 1) * (num_bays + 1) + bay

        h_bot = column_depth(story * H_story)
        h_top = column_depth((story + 1) * H_story)

        h_z = h_bot + ((h_top - h_bot) / H_story) * z
        K, M = VariableFrameElement(
            E, b * h_z**3 / 12, b * h_z, H_story, rho
        ).get_global_matrices(phi=np.pi / 2)
        assembler.add_element(K, M, node_bot, node_top)
        num_cols += 1

# =============================================================================
# 3. Haunched beams  (3 elements per beam)
# =============================================================================
# Extra haunch nodes are appended beyond the 24 base nodes.
extra = base_nodes   # running index for new haunch nodes
num_beams = 0

for story in range(1, num_stories + 1):
    for bay in range(num_bays):
        node_left  = story * (num_bays + 1) + bay       # column top (left)
        node_right = story * (num_bays + 1) + bay + 1   # column top (right)

        node_lh = extra;     extra += 1   # end of left haunch
        node_rh = extra;     extra += 1   # start of right haunch

        # -- Left haunch: h_end -> h_mid over L_h --
        h_lh = h_end_beam + ((h_mid_beam - h_end_beam) / L_h) * z
        K, M = VariableFrameElement(
            E, b * h_lh**3 / 12, b * h_lh, L_h, rho
        ).get_global_matrices(phi=0.0)
        assembler.add_element(K, M, node_left, node_lh)

        # -- Mid span: prismatic h_mid --
        h_mid_s = sp.sympify(h_mid_beam)
        K, M = VariableFrameElement(
            E, b * h_mid_s**3 / 12, b * h_mid_s, L_mid_span, rho
        ).get_global_matrices(phi=0.0)
        assembler.add_element(K, M, node_lh, node_rh)

        # -- Right haunch: h_mid -> h_end over L_h --
        h_rh = h_mid_beam + ((h_end_beam - h_mid_beam) / L_h) * z
        K, M = VariableFrameElement(
            E, b * h_rh**3 / 12, b * h_rh, L_h, rho
        ).get_global_matrices(phi=0.0)
        assembler.add_element(K, M, node_rh, node_right)

        num_beams += 1

build_time = time.perf_counter() - start

# =============================================================================
# 4. Boundary conditions and sparse eigensolver
# =============================================================================
# Ground-floor nodes (story 0, indices 0..num_bays) are fully fixed
fixed_dofs: list[int] = []
for bay in range(num_bays + 1):
    fixed_dofs.extend([bay * 3, bay * 3 + 1, bay * 3 + 2])

K_free, M_free, active_dofs = assembler.get_reduced_system(fixed_dofs)

solve_start = time.perf_counter()
freqs, _    = assembler.solve_dynamic(K_free, M_free, num_modes=3)
solve_time  = time.perf_counter() - solve_start

# =============================================================================
# 5. Output
# =============================================================================
# Reference frequencies from stepwise N=20 (paper Table 6)
refs = [4.1660, 12.7320, 23.5350]

total_elements = num_cols + num_beams * 3   # 20 cols + 15*3 = 65 elements
print(f"\n=== Tables 5 & 6: 5-Story 3-Bay Haunched Frame ===")
print(f"  Columns         : {num_cols} (1 element each)")
print(f"  Beams           : {num_beams} (3 elements each = {num_beams*3} beam elements)")
print(f"  Total elements  : {total_elements}")
print(f"  Total nodes     : {total_nodes}")
print(f"  Active DOFs     : {len(active_dofs)}")
print()
for i, (f, r) in enumerate(zip(freqs, refs), 1):
    print(f"  Mode {i}: {f:.4f} Hz  "
          f"(reference: {r:.4f} Hz, error: {abs(f-r)/r*100:.2f}%)")
print()
print(f"  Assembly time   : {build_time:.3f} s")
print(f"  Eigensolver time: {solve_time:.3f} s")
print(f"  Total runtime   : {build_time+solve_time:.3f} s")
