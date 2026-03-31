import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def prepare_boundary_conditions(K, ndof, nelx, nely, bc="cantilever", k_in=1.0, k_out=0.001):
    if bc == "cantilever":
        # Cantilever beam:
        #   n_force = max(2, ceil(nelx / 100))
        #
        #     ●───────────────────────●
        #     ║                       │
        #     ║  (fixed wall)         │
        #     ║                       │
        #     ●───────────────────────●
        #     ║                  F↓ (n_force nodes)
        #
        #     Left edge: ux = uy = 0 (clamped)
        #     Bottom-right: F↓ (downward load, n_force nodes)
        #
        n_force = max(2, int(np.ceil(nelx / 100.0)))
        br_nodes = nelx * (nely + 1) + nely - np.arange(n_force)
        F = np.zeros(ndof)
        F[2 * br_nodes + 1] = -1.0 / n_force
        fixed = np.arange(0, 2 * (nely + 1) - 1, 1)
        return {
            "K": K,
            "fixed": fixed,
            "F": F,
            "is_mechanism": False,
        }

    if bc == "mbb":
        # MBB beam (half-beam with symmetry):
        #
        #     F↓ (n_force nodes)
        #     ●───●───────────────────●
        #     ║                       │
        #     ║  (symmetry plane)     │
        #     ║                       │
        #     ●───────────────────────▲  ← roller (uy = 0, 1 node)
        #
        #     Left edge: ux = 0 (symmetry)
        #     Bottom-right corner: uy = 0 (single node roller)
        #
        n_force = max(2, int(np.ceil(nelx / 100.0)))
        tl_nodes = np.arange(n_force) * (nely + 1)  # top-left along top edge
        F = np.zeros(ndof)
        F[2 * tl_nodes + 1] = -1.0 / n_force
        fixed_left_x = np.arange(0, 2 * (nely + 1), 2)
        br_node = nelx * (nely + 1) + nely  # bottom-right corner node
        fixed_br_y = np.array([2 * br_node + 1])
        fixed = np.union1d(fixed_left_x, fixed_br_y)
        return {
            "K": K,
            "fixed": fixed,
            "F": F,
            "is_mechanism": False,
        }

    if bc == "mechanism":
        # Compliant mechanism (force inverter):
        #   n_io = n_fix = max(2, ceil(nelx / 100))
        #
        #     ▽───▽───▽───▽───▽───▽───▽  ← top edge: uy = 0
        #   F→● (n_io nodes, k_in)    ● (n_io nodes, k_out) → u_out
        #     │                       │
        #     │                       │
        #     ●───────────────────────●
        #     ██ (n_fix nodes)
        #
        #     Top edge: uy = 0 (roller)
        #     Bottom-left: ux = uy = 0 (clamped, n_fix nodes)
        #     Left top node(s): F→ input (with k_in spring)
        #     Right top node(s): output (with k_out spring)
        #
        n_io = max(2, int(np.ceil(nelx / 100.0)))
        in_nodes = np.arange(0, n_io)
        out_nodes = nelx * (nely + 1) + np.arange(0, n_io)
        dof_in_x = 2 * in_nodes
        dof_out_x = 2 * out_nodes

        K[dof_in_x, dof_in_x] += k_in / n_io
        K[dof_out_x, dof_out_x] += k_out / n_io

        top_nodes = np.arange(nelx + 1) * (nely + 1)
        fixed_top_y = 2 * top_nodes + 1

        n_fix_left = max(2, int(np.ceil(nelx / 100.0)))
        bl_nodes = np.arange((nely + 1) - n_fix_left, nely + 1)
        fixed_bl = np.sort(np.concatenate([2 * bl_nodes, 2 * bl_nodes + 1]))

        fixed = np.union1d(fixed_top_y, fixed_bl)

        F1 = np.zeros(ndof)
        F1[dof_in_x] = 1.0 / n_io

        F2 = np.zeros(ndof)
        F2[dof_out_x] = -1.0 / n_io

        return {
            "K": K,
            "fixed": fixed,
            "F1": F1,
            "F2": F2,
            "dof_out_x": dof_out_x,
            "is_mechanism": True,
        }

    raise ValueError(f"Unknown bc type '{bc}'. Supported: 'cantilever', 'mbb', 'mechanism'.")

def lk():
    E = 1.0
    nu = 0.3
    k = np.array([
        1 / 2 - nu / 6,
        1 / 8 + nu / 8,
        -1 / 4 - nu / 12,
        -1 / 8 + 3 * nu / 8,
        -1 / 4 + nu / 12,
        -1 / 8 - nu / 8,
        nu / 6,
        1 / 8 - 3 * nu / 8,
    ])
    KE = E / (1 - nu**2) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
    ])
    return KE


def build_element_dof_map(nelx, nely):
    nele = nelx * nely
    edofMat = np.zeros((nele, 8), dtype=int)
    e = 0
    for elx in range(nelx):
        for ely in range(nely):
            n1 = (nely + 1) * elx + (ely + 1)
            n2 = (nely + 1) * (elx + 1) + (ely + 1)
            edofMat[e, :] = np.array([
                2 * n1 - 2,
                2 * n1 - 1,
                2 * n2 - 2,
                2 * n2 - 1,
                2 * n2,
                2 * n2 + 1,
                2 * n1,
                2 * n1 + 1,
            ], dtype=int)
            e += 1
    return edofMat


def assemble_global_stiffness(nelx, nely, x, penal, E0=1.0, Emin=1e-6, KE=None):
    if KE is None:
        KE = lk()

    ndof = 2 * (nelx + 1) * (nely + 1)
    edofMat = build_element_dof_map(nelx, nely)

    rho = np.clip(x, 0.0, 1.0)
    Ee = Emin + (rho**penal) * (E0 - Emin)
    Ee_vec = Ee.ravel(order="F")

    iK = np.kron(edofMat, np.ones((8, 1), dtype=int)).ravel()
    jK = np.kron(edofMat, np.ones((1, 8), dtype=int)).ravel()
    sK = (KE.ravel()[None, :] * Ee_vec[:, None]).ravel()
    K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
    return K, ndof


def FE(nelx, nely, x, penal, bc="cantilever", k_in=1.0, k_out=0.02, E0=1.0, Emin=1e-6):
    K, ndof = assemble_global_stiffness(nelx, nely, x, penal, E0=E0, Emin=Emin)
    bc_data = prepare_boundary_conditions(K, ndof, nelx, nely, bc=bc, k_in=k_in, k_out=k_out)

    K = bc_data["K"]
    fixed = bc_data["fixed"]
    all_dofs = np.arange(ndof)
    free = np.setdiff1d(all_dofs, fixed)

    if bc_data["is_mechanism"]:
        U = np.zeros(ndof)
        Lambda = np.zeros(ndof)
        F1 = bc_data["F1"]
        F2 = bc_data["F2"]

        U[free] = spsolve(K[free][:, free], F1[free])
        Lambda[free] = spsolve(K[free][:, free], F2[free])
        return U, Lambda, bc_data["dof_out_x"]

    U = np.zeros(ndof)
    F = bc_data["F"]
    U[free] = spsolve(K[free][:, free], F[free])
    U[fixed] = 0.0
    return U
