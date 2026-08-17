"""Building molecules and converging their SCFs.

pyscf has no Windows build - run stage 3 under WSL, Linux or macOS.
"""

import numpy as np

from ..utils import HARTREE_TO_EV


def build_mol(symbols, coords, basis, charge=0, spin=0):
    from pyscf import gto

    mol = gto.Mole()
    mol.atom = [(s, tuple(c)) for s, c in zip(symbols, coords)]
    mol.basis = basis
    mol.charge, mol.spin = charge, spin
    mol.unit = "Angstrom"
    mol.verbose = 0
    mol.build()
    return mol


def make_mf(mol, dft_cfg):
    """Density-fitted RKS. Density fitting is a large speedup here and pyscf
    keeps the three-index tensor on disk, so its size is not a constraint."""
    from pyscf import dft as pyscf_dft

    mf = pyscf_dft.RKS(mol).density_fit()
    mf.xc = dft_cfg.xc
    mf.conv_tol = float(dft_cfg.conv_tol)
    mf.grids.level = int(dft_cfg.grid_level)
    mf.max_cycle = int(dft_cfg.max_cycle)
    return mf


def run_scf(mol, dft_cfg, dm0=None):
    """Converge an SCF, falling back to the second-order solver if the default
    DIIS iteration stalls."""
    mf = make_mf(mol, dft_cfg)
    mf.kernel(dm0=dm0)
    if not mf.converged:
        mf = mf.newton()
        mf.kernel(mf.make_rdm1())
    if not mf.converged:
        raise RuntimeError(f"SCF did not converge for {mol.natm} atoms")
    return mf


def frontier(mf):
    """HOMO and LUMO indices, their orbital energies in eV, and the gap to the
    next orbital on each side.

    The gaps tell you whether a single-orbital coupling is adequate: acceptor
    LUMOs are often near-degenerate, and if LUMO and LUMO+1 sit within ~0.1 eV
    the 2x2 frontier block should be diagonalised instead.
    """
    homo = int(np.count_nonzero(mf.mo_occ > 0)) - 1
    lumo = homo + 1
    energies = mf.mo_energy
    return (homo, lumo,
            energies[homo] * HARTREE_TO_EV, energies[lumo] * HARTREE_TO_EV,
            (energies[homo] - energies[homo - 1]) * HARTREE_TO_EV,
            (energies[lumo + 1] - energies[lumo]) * HARTREE_TO_EV)


def manifold(mf, index, window_ev, max_orbitals):
    """Orbitals degenerate with `index` to within `window_ev`.

    Restricted to orbitals of the same occupancy, so a small HOMO-LUMO gap can
    never pull an occupied orbital into a virtual manifold. Returns `index`
    alone when nothing else is close, which is the usual case.
    """
    energies = mf.mo_energy * HARTREE_TO_EV
    occupied = mf.mo_occ > 0
    same_side = np.flatnonzero(occupied == occupied[index])
    separation = np.abs(energies[same_side] - energies[index])
    near = same_side[separation <= window_ev]
    closest = near[np.argsort(np.abs(energies[near] - energies[index]))]
    return np.sort(closest[:max(1, int(max_orbitals))])
