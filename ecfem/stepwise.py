import numpy as np
import sympy as sp


class PrismaticFrameElement:
    """
    Standard 2D Euler-Bernoulli frame element with a constant cross-section.

    Used as the reference (benchmark) method: a non-prismatic member is
    discretised into N prismatic sub-elements, each evaluated at its midpoint
    cross-section properties.  As N → ∞ the stepwise solution converges to
    the exact result provided by :class:`~ecfem.elements.VariableFrameElement`.

    Parameters
    ----------
    E : float
        Young's modulus [Pa].
    I_val : float
        Second moment of area [m^4] (constant over the element).
    A_val : float
        Cross-sectional area [m^2] (constant over the element).
    L : float
        Element length [m].
    rho : float
        Mass density [kg/m^3].
    """

    def __init__(self, E: float, I_val: float, A_val: float, L: float, rho: float):
        self.E = float(E)
        self.I = float(I_val)
        self.A = float(A_val)
        self.L = float(L)
        self.rho = float(rho)

    def get_local_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the standard 6x6 local stiffness and consistent mass matrices
        based on classical cubic Hermite shape functions.

        Returns
        -------
        K_local : ndarray, shape (6, 6)
        M_local : ndarray, shape (6, 6)
        """
        E, I, A, L, rho = self.E, self.I, self.A, self.L, self.rho

        # --- Bending (4x4) ---
        EI_L3 = (E * I) / L**3
        K_bend = EI_L3 * np.array([
            [ 12,      6 * L,    -12,      6 * L   ],
            [  6 * L,  4 * L**2,  -6 * L,  2 * L**2],
            [-12,     -6 * L,     12,     -6 * L   ],
            [  6 * L,  2 * L**2,  -6 * L,  4 * L**2],
        ])

        mu_L = (rho * A * L) / 420.0
        M_bend = mu_L * np.array([
            [156,      22 * L,    54,     -13 * L   ],
            [ 22 * L,   4 * L**2, 13 * L,  -3 * L**2],
            [ 54,      13 * L,   156,     -22 * L   ],
            [-13 * L,  -3 * L**2, -22 * L,   4 * L**2],
        ])

        # --- Axial (2x2) ---
        EA_L = (E * A) / L
        K_axial = EA_L * np.array([[ 1, -1], [-1,  1]])

        m_axial = (rho * A * L) / 6.0
        M_axial = m_axial * np.array([[2, 1], [1, 2]])

        # --- Assemble into 6x6 ---
        K_local = np.zeros((6, 6))
        M_local = np.zeros((6, 6))

        axial_idx = [0, 3]
        bend_idx  = [1, 2, 4, 5]

        for i, r in enumerate(axial_idx):
            for j, c in enumerate(axial_idx):
                K_local[r, c] = K_axial[i, j]
                M_local[r, c] = M_axial[i, j]

        for i, r in enumerate(bend_idx):
            for j, c in enumerate(bend_idx):
                K_local[r, c] = K_bend[i, j]
                M_local[r, c] = M_bend[i, j]

        return K_local, M_local

    def get_global_matrices(self, phi: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform local matrices to the global coordinate system.

        Parameters
        ----------
        phi : float
            Element inclination angle from the global X-axis [rad].

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

        return T.T @ K_local @ T, T.T @ M_local @ T


def generate_stepwise_mesh(
    E: float,
    I_expr: sp.Expr,
    A_expr: sp.Expr,
    L_total: float,
    rho: float,
    num_elements: int,
) -> list[PrismaticFrameElement]:
    """
    Discretise a non-prismatic member into *num_elements* prismatic sub-elements.

    Cross-section properties are evaluated at the midpoint of each segment,
    which provides first-order accuracy in the stepwise approximation.

    Parameters
    ----------
    E : float
        Young's modulus [Pa].
    I_expr : sympy.Expr
        Symbolic expression for I(z) [m^4].
    A_expr : sympy.Expr
        Symbolic expression for A(z) [m^2].
    L_total : float
        Total member length [m].
    rho : float
        Mass density [kg/m^3].
    num_elements : int
        Number of prismatic sub-elements.

    Returns
    -------
    elements : list of PrismaticFrameElement
        Ready-to-use element objects, ordered from z=0 to z=L_total.
    """
    z = sp.symbols('z')
    L_seg = float(L_total) / num_elements
    elements = []

    for i in range(num_elements):
        z_mid = (i + 0.5) * L_seg
        I_val = float(I_expr.subs(z, z_mid))
        A_val = float(A_expr.subs(z, z_mid))
        elements.append(PrismaticFrameElement(E, I_val, A_val, L_seg, rho))

    return elements
