r"""Dimer projection (DIPRO).

The converged monomer orbitals are projected onto the dimer Fock matrix,

    J_ab = <phi_a| F_D |phi_b>,   S_ab = <phi_a|phi_b>,   e_a = <phi_a|F_D|phi_a>

and the non-orthogonality of the two fragment orbitals is divided out:

    J_eff = (J_ab - S_ab (e_a + e_b) / 2) / (1 - S_ab^2)

J_eff is the quantity to use downstream - the raw J_ab is basis-set dependent
through that non-orthogonality, and at 3 A contact the correction typically
changes it by tens of percent.

Two numbers per frame matter for transport, not one. J_eff is the coupling, and
dE = e_a - e_b is the site energy difference across the pair. A Marcus hopping
rate needs both: J sets the prefactor, dE enters the activation energy.
"""

import numpy as np


def embed(coefficients, n_dimer, offset):
    """Place a fragment orbital into the dimer AO basis by zero-padding.

    pyscf orders AOs by atom and the dimer file lists all of A before all of B,
    so the dimer AO space is exactly the direct sum of the monomer spaces in
    that order.
    """
    vector = np.zeros(n_dimer)
    vector[offset:offset + len(coefficients)] = coefficients
    return vector


def coupling(ca, cb, F, S):
    """Effective coupling between two fragment orbitals already expressed in
    the dimer AO basis.

    Returns (J_eff, S_ab, e_a, e_b, J_raw), in Hartree except S_ab.
    """
    j_raw = float(ca @ F @ cb)
    s_ab = float(ca @ S @ cb)
    e_a = float(ca @ F @ ca)
    e_b = float(cb @ F @ cb)
    if abs(s_ab) >= 1.0:
        raise ValueError(f"fragment orbital overlap {s_ab} is not physical; "
                         f"the two monomers are probably overlapping")
    j_eff = (j_raw - s_ab * (e_a + e_b) / 2) / (1 - s_ab ** 2)
    return j_eff, s_ab, e_a, e_b, j_raw


def manifold_coupling(ca, cb, F, S):
    """Coupling between two *sets* of fragment orbitals.

    A single-orbital coupling understates transport when a monomer's frontier
    orbital is near-degenerate with the next one: the carrier can arrive in
    either state, and both channels carry flux. The fix is to stop treating the
    frontier as one orbital.

    `ca` is (n_ao, n_a) and `cb` is (n_ao, n_b), both already embedded in the
    dimer AO basis. The two sets together are symmetrically (Lowdin)
    orthogonalised, and the A-B block of the transformed Fock matrix is the
    matrix of couplings between the manifolds.

    For n_a = n_b = 1 this reduces *exactly* to `coupling()` above - the
    two-state formula is itself the closed form of this transformation - so
    turning it on changes nothing for the frames that did not need it.

    Returns (J_eff, block, e_a, e_b) in Hartree. `J_eff` is the
    transport-relevant scalar: Marcus rates go as J^2 and the total rate out of
    a state is the sum over accessible final states, so the channels add in
    quadrature, averaged over the equally populated initial states:

        J_eff^2 = (1 / n_a) * sum_a sum_b |J_ab|^2
    """
    ca, cb = np.atleast_2d(ca.T).T, np.atleast_2d(cb.T).T
    n_a = ca.shape[1]
    C = np.hstack([ca, cb])

    H = C.T @ F @ C
    overlap = C.T @ S @ C

    eigenvalues, vectors = np.linalg.eigh(overlap)
    if eigenvalues.min() <= 1e-10:
        raise ValueError(
            f"the fragment orbital set is linearly dependent (smallest "
            f"overlap eigenvalue {eigenvalues.min():.2e}); reduce "
            f"dft.max_manifold or narrow dft.degeneracy_window")
    inverse_sqrt = vectors @ np.diag(eigenvalues ** -0.5) @ vectors.T
    orthogonal = inverse_sqrt @ H @ inverse_sqrt

    block = orthogonal[:n_a, n_a:]
    j_eff = float(np.sqrt(np.mean(np.sum(block ** 2, axis=1))))
    diagonal = np.diag(orthogonal)
    return j_eff, block, float(diagonal[:n_a].mean()), float(diagonal[n_a:].mean())
