import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def prepare_thermal_boundary_conditions(ndof, nelx, nely, sink_width_frac=0.1, sink_value=0.0):
    if ndof != (nelx + 1) * (nely + 1):
        raise ValueError("ndof must be (nelx + 1) * (nely + 1) for thermal conduction.")

    bottom_nodes = np.arange(nelx + 1) * (nely + 1) + nely

    sink_count = max(1, int(np.ceil((nelx + 1) * sink_width_frac)))
    sink_count = min(sink_count, nelx + 1)

    center = nelx // 2
    start = max(0, center - sink_count // 2)
    end = start + sink_count
    if end > (nelx + 1):
        end = nelx + 1
        start = end - sink_count

    fixed = bottom_nodes[start:end]
    fixed_values = np.full(fixed.shape[0], float(sink_value), dtype=float)

    return {
        "fixed": fixed,
        "fixed_values": fixed_values,
    }


def lk_thermal():
    """4-node bilinear element conductivity matrix on unit square."""
    return (1.0 / 6.0) * np.array([
        [4.0, -1.0, -2.0, -1.0],
        [-1.0, 4.0, -1.0, -2.0],
        [-2.0, -1.0, 4.0, -1.0],
        [-1.0, -2.0, -1.0, 4.0],
    ])


def build_element_dof_map_thermal(nelx, nely):
    nele = nelx * nely
    edofMat = np.zeros((nele, 4), dtype=int)
    e = 0
    for elx in range(nelx):
        for ely in range(nely):
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            n3 = n2 + 1
            n4 = n1 + 1
            edofMat[e, :] = np.array([n1, n2, n3, n4], dtype=int)
            e += 1
    return edofMat


def assemble_global_conductivity(nelx, nely, x, penal, k_max=1.0, k_min=1e-6, KE=None):
    if KE is None:
        KE = lk_thermal()

    ndof = (nelx + 1) * (nely + 1)
    edofMat = build_element_dof_map_thermal(nelx, nely)

    rho = np.clip(x, 0.0, 1.0)
    kappa_e = k_min + (rho**penal) * (k_max - k_min)
    kappa_vec = kappa_e.ravel(order="F")

    iK = np.kron(edofMat, np.ones((4, 1), dtype=int)).ravel()
    jK = np.kron(edofMat, np.ones((1, 4), dtype=int)).ravel()
    sK = (KE.ravel()[None, :] * kappa_vec[:, None]).ravel()

    K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
    return K, ndof, edofMat


def assemble_uniform_heat_load(nelx, nely, edofMat, q=1.0):
    ndof = (nelx + 1) * (nely + 1)
    F = np.zeros(ndof, dtype=float)
    fe = (q / 4.0) * np.ones(4, dtype=float)
    for e in range(edofMat.shape[0]):
        F[edofMat[e]] += fe
    return F


def FE_thermal(
    nelx,
    nely,
    x,
    penal,
    q=1.0,
    sink_width_frac=0.1,
    sink_value=0.0,
    k_max=1.0,
    k_min=1e-6,
):
    K, ndof, edofMat = assemble_global_conductivity(
        nelx, nely, x, penal, k_max=k_max, k_min=k_min
    )
    F = assemble_uniform_heat_load(nelx, nely, edofMat, q=q)

    bc_data = prepare_thermal_boundary_conditions(
        ndof=ndof,
        nelx=nelx,
        nely=nely,
        sink_width_frac=sink_width_frac,
        sink_value=sink_value,
    )
    fixed = bc_data["fixed"]
    fixed_values = bc_data["fixed_values"]

    all_dofs = np.arange(ndof)
    free = np.setdiff1d(all_dofs, fixed)

    T = np.zeros(ndof, dtype=float)
    T[fixed] = fixed_values

    rhs = F[free] - K[free][:, fixed] @ T[fixed]
    T[free] = spsolve(K[free][:, free], rhs)

    return T
