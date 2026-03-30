import numpy as np
import sympy as sp
from scipy.integrate import fixed_quad


class VariableFrameElement:
    """
    Elastic Curve-based Finite Element (EC-FEM) for 2D non-prismatic frame analysis.

    Computes exact 6x6 stiffness and consistent mass matrices for members with
    continuously varying cross-sections using a hybrid symbolic-numeric approach.

    The formulation is based on the governing differential equation of the elastic
    curve, producing shape functions that satisfy equilibrium exactly for any
    continuous variation of EI(z) and EA(z).

    Parameters
    ----------
    E : float
        Young's modulus [Pa].
    I_expr : sympy.Expr
        SymPy expression for the second moment of area I(z) [m^4].
    A_expr : sympy.Expr
        SymPy expression for the cross-sectional area A(z) [m^2].
    L : float
        Element length [m].
    rho : float
        Mass density [kg/m^3].
    """

    def __init__(self, E: float, I_expr: sp.Expr, A_expr: sp.Expr, L: float, rho: float):
        self.E = float(E)
        self.I_expr = I_expr
        self.A_expr = A_expr
        self.L = float(L)
        self.rho = float(rho)
        self.z = sp.symbols('z')

        # Flag used to select optimised code paths for prismatic sub-cases
        self.is_tapered = not (I_expr.is_constant() and A_expr.is_constant())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_local_matrices(self) -> tuple:
        """
        Construct the exact 6x6 local stiffness and consistent mass matrices
        by superposing the axial (2-DOF) and bending (4-DOF) contributions.

        Returns
        -------
        K_local : ndarray, shape (6, 6)
        M_local : ndarray, shape (6, 6)
        """
        K_bend, M_bend = self._calculate_bending_matrices()
        K_axial, M_axial = self._calculate_axial_matrices()

        K_local = np.zeros((6, 6))
        M_local = np.zeros((6, 6))

        # DOF mapping: axial -> [0, 3],  bending -> [1, 2, 4, 5]
        axial_idx = [0, 3]
        bend_idx = [1, 2, 4, 5]

        for i, r in enumerate(axial_idx):
            for j, c in enumerate(axial_idx):
                K_local[r, c] = K_axial[i, j]
                M_local[r, c] = M_axial[i, j]

        for i, r in enumerate(bend_idx):
            for j, c in enumerate(bend_idx):
                K_local[r, c] = K_bend[i, j]
                M_local[r, c] = M_bend[i, j]

        return K_local, M_local

    def get_global_matrices(self, phi: float) -> tuple:
        """
        Transform the local 6x6 matrices to the global coordinate system.

        Parameters
        ----------
        phi : float
            Inclination angle of the element measured from the global X-axis [rad].

        Returns
        -------
        K_global : ndarray, shape (6, 6)
        M_global : ndarray, shape (6, 6)
        """
        K_local, M_local = self.get_local_matrices()

        c, s = np.cos(phi), np.sin(phi)
        T = np.array([
            [ c,  s,  0,  0,  0,  0],
            [-s,  c,  0,  0,  0,  0],
            [ 0,  0,  1,  0,  0,  0],
            [ 0,  0,  0,  c,  s,  0],
            [ 0,  0,  0, -s,  c,  0],
            [ 0,  0,  0,  0,  0,  1],
        ])

        K_global = T.T @ K_local @ T
        M_global = T.T @ M_local @ T
        return K_global, M_global

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_real(expr: sp.Expr) -> float:
        """
        Evaluate a SymPy expression to a real Python float.

        Symbolic integration of tapered EI(z) routinely produces imaginary
        residuals from logarithm branch cuts (e.g. integrating z/EI(z) where
        EI(z) contains a linear taper).  These artefacts cancel out when the
        full boundary-condition system is assembled; taking only the real part
        at each evaluation point is mathematically correct and matches the
        behaviour of the original implementation.
        """
        return float(sp.re(expr.evalf()))

    def _calculate_bending_matrices(self) -> tuple:
        """
        Derive the exact 4x4 bending stiffness and consistent mass matrices.

        The exact shape functions are obtained by solving the fourth-order
        governing ODE EI(z) v'''' = 0 via successive symbolic integration,
        following the EC-FEM formulation described in the manuscript.
        """
        z = self.z
        L_val = self.L

        EI_expr = sp.nsimplify(self.E) * sp.nsimplify(self.I_expr)

        # Successive integration of 1/EI(z) yields the fundamental solutions
        Q1 = sp.integrate(z / EI_expr, z)
        Q2 = sp.integrate(1 / EI_expr, z)
        Y1 = sp.integrate(Q1, z)
        Y2 = sp.integrate(Q2, z)

        # Boundary condition matrix: enforces unit displacements/rotations at each end
        M_sys = np.array([
            [self._to_real(Y1.subs(z, 0)),     self._to_real(Y2.subs(z, 0)),     0,     1],
            [self._to_real(Q1.subs(z, 0)),     self._to_real(Q2.subs(z, 0)),     1,     0],
            [self._to_real(Y1.subs(z, L_val)), self._to_real(Y2.subs(z, L_val)), L_val, 1],
            [self._to_real(Q1.subs(z, L_val)), self._to_real(Q2.subs(z, L_val)), 1,     0],
        ], dtype=float)

        C_inv = np.linalg.inv(M_sys)

        # Exact 4x4 stiffness matrix assembled from elastic curve coefficients
        K_bend = np.zeros((4, 4))
        for i in range(4):
            c_vals = C_inv[:, i]
            K_bend[:, i] = [
                c_vals[0],
                -c_vals[1],
                -c_vals[0],
                c_vals[0] * L_val + c_vals[1],
            ]

        # --- Consistent mass matrix via Gauss quadrature (hybrid symbolic-numeric) ---
        # Shape functions are evaluated numerically from symbolic primitives.
        # Complex-valued lambdify is used to safely handle log branch cuts that
        # arise from tapered EI(z); the real part is taken after evaluation.
        A_func = sp.lambdify(z, self.A_expr, 'numpy')

        log_safe = [{'log': np.emath.log}, 'numpy']
        Y1_func = sp.lambdify(z, Y1, modules=log_safe)
        Y2_func = sp.lambdify(z, Y2, modules=log_safe)

        def get_N_matrix(z_array):
            """Evaluate all four exact shape functions at an array of z values."""
            z_c = np.asarray(z_array, dtype=complex)
            phi = np.vstack([
                Y1_func(z_c),
                Y2_func(z_c),
                z_c,
                np.ones_like(z_c),
            ])
            return np.real(C_inv.T @ phi)

        def mass_integrand(z_array, i, j):
            N = get_N_matrix(z_array)
            return self.rho * A_func(z_array) * N[i, :] * N[j, :]

        # 10-point Gauss quadrature is sufficient for polynomial-like integrands
        M_bend = np.zeros((4, 4))
        for i in range(4):
            for j in range(i, 4):
                val, _ = fixed_quad(mass_integrand, 0.0, L_val, args=(i, j), n=10)
                M_bend[i, j] = val
                M_bend[j, i] = val  # Exploit symmetry

        return K_bend, M_bend

    def _calculate_axial_matrices(self) -> tuple:
        """
        Calculate the exact 2x2 axial stiffness and consistent mass matrices.

        The exact axial stiffness is the reciprocal of the axial flexibility
        integral: k = 1 / integral(1/EA dz, 0, L).
        """
        z = self.z
        L_val = self.L

        # Exact axial flexibility via symbolic integration
        flex_expr = sp.integrate(1 / (self.E * self.A_expr), (z, 0, L_val))
        k_eq = 1.0 / self._to_real(flex_expr)

        K_axial = np.array([
            [ k_eq, -k_eq],
            [-k_eq,  k_eq],
        ])

        # Consistent axial mass (linear interpolation assumed along the axis)
        m_total = self._to_real(sp.integrate(self.rho * self.A_expr, (z, 0, L_val)))
        M_axial = np.array([
            [m_total / 3, m_total / 6],
            [m_total / 6, m_total / 3],
        ])

        return K_axial, M_axial
