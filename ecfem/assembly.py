import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh


class SystemAssembler:
    """
    Assembles global stiffness and mass matrices for 2D frame structures
    and applies boundary conditions. Uses sparse matrices for memory efficiency.

    Parameters
    ----------
    num_nodes : int
        Total number of nodes in the structural model.
    """

    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.num_dofs = num_nodes * 3  # 3 DOFs per node: u (axial), v (transverse), theta (rotation)

        # List-of-Lists sparse format is efficient for incremental assembly
        self.K_global = lil_matrix((self.num_dofs, self.num_dofs), dtype=float)
        self.M_global = lil_matrix((self.num_dofs, self.num_dofs), dtype=float)

    def add_element(self, K_elem: np.ndarray, M_elem: np.ndarray,
                    node_i: int, node_j: int) -> None:
        """
        Scatter a 6x6 element matrix pair into the global system.

        Parameters
        ----------
        K_elem : ndarray, shape (6, 6)
            Element stiffness matrix in global coordinates.
        M_elem : ndarray, shape (6, 6)
            Element mass matrix in global coordinates.
        node_i : int
            Index of the first (start) node.
        node_j : int
            Index of the second (end) node.
        """
        dofs = [
            node_i * 3, node_i * 3 + 1, node_i * 3 + 2,
            node_j * 3, node_j * 3 + 1, node_j * 3 + 2,
        ]

        for i, r in enumerate(dofs):
            for j, c in enumerate(dofs):
                self.K_global[r, c] += K_elem[i, j]
                self.M_global[r, c] += M_elem[i, j]

    def get_reduced_system(
        self, fixed_dofs: list[int]
    ) -> tuple[csr_matrix, csr_matrix, list[int]]:
        """
        Apply boundary conditions by partitioning out the fixed DOFs.

        Parameters
        ----------
        fixed_dofs : list of int
            Indices of constrained (zero-displacement) DOFs.

        Returns
        -------
        K_free : csr_matrix
            Reduced stiffness matrix for the free DOFs.
        M_free : csr_matrix
            Reduced mass matrix for the free DOFs.
        free_dofs : list of int
            Indices of the active (unconstrained) DOFs.
        """
        free_dofs = [i for i in range(self.num_dofs) if i not in fixed_dofs]

        # Convert to CSR for efficient arithmetic and slicing
        K_csr = self.K_global.tocsr()
        M_csr = self.M_global.tocsr()

        K_free = K_csr[free_dofs, :][:, free_dofs]
        M_free = M_csr[free_dofs, :][:, free_dofs]

        return K_free, M_free, free_dofs

    def solve_static(self, K_free: csr_matrix, force_vector: np.ndarray) -> np.ndarray:
        """
        Solve the linear static system K * U = F for free-DOF displacements.

        Parameters
        ----------
        K_free : csr_matrix
            Reduced stiffness matrix.
        force_vector : ndarray
            Applied load vector (free DOFs only).

        Returns
        -------
        U_free : ndarray
            Displacement vector for the free DOFs.
        """
        from scipy.sparse.linalg import spsolve
        return spsolve(K_free, force_vector)

    def solve_dynamic(
        self, K_free: csr_matrix, M_free: csr_matrix, num_modes: int = 3
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the generalised eigenvalue problem K*phi = omega^2 * M*phi
        for the lowest natural frequencies and mode shapes.

        For small systems (< 50 free DOFs) a dense solver is used.
        For larger systems the sparse Shift-Invert Lanczos method (eigsh with
        sigma=0) is used, which efficiently targets the lowest-frequency modes.

        Parameters
        ----------
        K_free : csr_matrix
            Reduced stiffness matrix.
        M_free : csr_matrix
            Reduced consistent mass matrix.
        num_modes : int, optional
            Number of modes to extract (default: 3).

        Returns
        -------
        freqs : ndarray, shape (num_modes,)
            Natural frequencies in Hz, sorted ascending.
        modes : ndarray, shape (n_free, num_modes)
            Corresponding mode shape vectors.
        """
        if K_free.shape[0] < 50:
            # Dense LAPACK solver — exact and robust for small systems
            w2, modes = eigh(K_free.toarray(), M_free.toarray())
            freqs = np.sqrt(np.abs(w2)) / (2 * np.pi)
            return freqs[:num_modes], modes[:, :num_modes]
        else:
            # Shift-Invert mode: factor (K - 0*M) once, then invert cheaply.
            # This makes eigsh converge to the smallest eigenvalues,
            # which correspond to the lowest natural frequencies.
            w2, modes = eigsh(K_free, M=M_free, k=num_modes, sigma=0.0, which='LM')
            freqs = np.sqrt(np.abs(w2)) / (2 * np.pi)
            idx = freqs.argsort()
            return freqs[idx], modes[:, idx]
