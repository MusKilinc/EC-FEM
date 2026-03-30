"""
EC-FEM Benchmark 01: Tapered Cantilever Beam Validation
========================================================

Reproduces the static tip displacement and first natural frequency convergence
results presented in the manuscript (Tables 1–2 and Figure 5).

A linearly tapered cantilever beam (h varies from 0.40 m at the free end to
0.20 m at the fixed end) is analysed under a transverse tip load P = -10 kN.
The EC-FEM result with a **single element** is compared against a fine
stepwise reference solution (N = 500 prismatic sub-elements).

Run
---
    python examples/01_cantilever_validation.py
"""

import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eigh

from ecfem import VariableFrameElement

# =============================================================================
# 1. Parameters
# =============================================================================
L_total = 5.0       # m  — total beam length
b       = 0.2       # m  — constant width
h_start = 0.4       # m  — depth at the fixed end (z = 0)
h_end   = 0.2       # m  — depth at the free end  (z = L)
E       = 200e9     # Pa — Young's modulus (structural steel)
rho     = 7850      # kg/m^3
P       = -10000    # N  — transverse tip load (downward)

z = sp.symbols('z')
slope = (h_start - h_end) / L_total
h_z = h_start - slope * z       # Linear taper: h(z)
A_z = b * h_z                   # Cross-sectional area A(z)
I_z = (b * h_z**3) / 12        # Second moment of area I(z)

# =============================================================================
# 2. Reference values (high-fidelity stepwise solution, N = 500)
# =============================================================================
REF_DISP = -3.1944  # mm — tip transverse displacement
REF_FREQ = 14.1881  # Hz — first natural frequency

# =============================================================================
# 3. EC-FEM analysis with a single element (N = 1)
# =============================================================================
print("Running EC-FEM (single element) analysis...")

element_exact = VariableFrameElement(E, I_z, A_z, L_total, rho)
K_exact_local, M_exact_local = element_exact.get_local_matrices()

# Cantilever boundary conditions: u1, v1, theta1 are fixed (z = 0 end).
# Active bending DOFs: v2 (index 4) and theta2 (index 5).
free_idx = np.ix_([4, 5], [4, 5])
K_exact_free = K_exact_local[free_idx]
M_exact_free = M_exact_local[free_idx]

# Static analysis
F_exact  = np.array([P, 0.0])
U_exact  = np.linalg.solve(K_exact_free, F_exact)
disp_exact = U_exact[0] * 1000  # convert to mm

# Free-vibration analysis
w2_exact, _ = eigh(K_exact_free, M_exact_free)
freq_exact   = float(np.sqrt(np.abs(w2_exact[0])) / (2 * np.pi))

# =============================================================================
# 4. Results table
# =============================================================================
# Stepwise convergence data reproduced from Table 1 of the manuscript
results = [
    {"Method": "Stepwise",        "N":   1, "DOFs":    4, "Disp(mm)": -4.6296, "Freq(Hz)":  9.8311},
    {"Method": "Stepwise",        "N":   2, "DOFs":    6, "Disp(mm)": -3.5510, "Freq(Hz)": 12.8414},
    {"Method": "Stepwise",        "N":   5, "DOFs":   12, "Disp(mm)": -3.2489, "Freq(Hz)": 13.9545},
    {"Method": "Stepwise",        "N":  10, "DOFs":   22, "Disp(mm)": -3.2079, "Freq(Hz)": 14.1289},
    {"Method": "Stepwise",        "N":  50, "DOFs":  102, "Disp(mm)": -3.1949, "Freq(Hz)": 14.1800},
    {"Method": "Reference",       "N": 500, "DOFs": 1002, "Disp(mm)": REF_DISP, "Freq(Hz)": REF_FREQ},
    {"Method": "EC-FEM (Exact)",  "N":   1, "DOFs":    4, "Disp(mm)": disp_exact, "Freq(Hz)": freq_exact},
]

df = pd.DataFrame(results)
df["Disp Error (%)"] = (df["Disp(mm)"] - REF_DISP).abs() / abs(REF_DISP) * 100
df["Freq Error (%)"] = (df["Freq(Hz)"] - REF_FREQ).abs() / REF_FREQ * 100
df.loc[df["Method"] == "Reference", ["Disp Error (%)", "Freq Error (%)"]] = 0.0

print("\n=== Tables 1 & 2: Convergence Results ===")
print(df.to_string(index=False, float_format="%.4f"))

# =============================================================================
# 5. Figure 5 — convergence plot
# =============================================================================
df_step  = df[df["Method"] == "Stepwise"]
df_exact_row = df[df["Method"] == "EC-FEM (Exact)"].iloc[0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Static displacement error ---
ax1.plot(df_step["N"], df_step["Disp Error (%)"], 'bo-', label='Stepwise (prismatic sub-elements)')
ax1.plot(
    df_exact_row["N"], df_exact_row["Disp Error (%)"], 'r*', markersize=14,
    label=f'EC-FEM (exact), N=1\nError: {df_exact_row["Disp Error (%)"]:.4f}%',
)
ax1.set_title("Static Tip Displacement — Error Convergence")
ax1.set_xlabel("Number of elements N")
ax1.set_ylabel("Relative error (%)")
ax1.set_xscale('log')
ax1.grid(True, which="both", ls="--", alpha=0.6)
ax1.legend()

# --- First natural frequency error ---
ax2.plot(df_step["N"], df_step["Freq Error (%)"], 'go-', label='Stepwise (prismatic sub-elements)')
ax2.plot(
    df_exact_row["N"], df_exact_row["Freq Error (%)"], 'r*', markersize=14,
    label=f'EC-FEM (exact), N=1\nError: {df_exact_row["Freq Error (%)"]:.4f}%',
)
ax2.set_title("First Natural Frequency — Error Convergence")
ax2.set_xlabel("Number of elements N")
ax2.set_ylabel("Relative error (%)")
ax2.set_xscale('log')
ax2.grid(True, which="both", ls="--", alpha=0.6)
ax2.legend()

plt.tight_layout()
output_path = "Fig5_Convergence.png"
plt.savefig(output_path, dpi=300)
print(f"\nFigure saved to '{output_path}' (300 DPI).")
