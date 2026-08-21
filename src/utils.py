"""Things more than one stage needs: physical constants, xyz files, and the
geometry used to describe and monitor a stacked pair.
"""

import numpy as np

# ---- Physical constants --------------------------------------------------
# One definition each, so stages 4 and 5 cannot drift apart on the value of
# kB and quietly disagree about the mobility.
Q = 1.602176634e-19             # C, elementary charge
KB = 1.380649e-23               # J/K
HBAR = 1.054571817e-34          # J s
HARTREE_TO_EV = 27.211386245988
HARTREE_TO_MEV = 27211.386245988
ANGSTROM_TO_CM = 1.0e-8

# Covalent radii, Cordero et al. (2008), in A. Extend as needed - an element
# missing from this table is a hard error rather than a silent guess.
RCOV = {
    "H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39,
    "Se": 1.20, "Te": 1.38,
}


def covalent_radii(symbols):
    """Covalent radii for a list of element symbols."""
    unknown = sorted({s for s in symbols if s not in RCOV})
    if unknown:
        raise SystemExit(
            f"no covalent radius for element(s) {', '.join(unknown)} - add "
            f"them to RCOV in src/utils.py")
    return np.array([RCOV[s] for s in symbols])


# ---- xyz files -----------------------------------------------------------
def read_xyz(path):
    """Symbols and coordinates from an xyz file.

    Parsed by hand rather than through a library because the comment line the
    stages write is full of '=' and ',' characters that some readers try to
    interpret as extended-xyz key/value pairs.
    """
    with open(path, encoding="utf-8") as f:
        n = int(f.readline().split()[0])
        comment = f.readline().rstrip("\n")
        symbols, coords = [], []
        for _ in range(n):
            parts = f.readline().split()
            symbols.append(parts[0])
            coords.append([float(x) for x in parts[1:4]])
    if len(symbols) != n:
        raise SystemExit(f"{path}: header says {n} atoms, file has "
                         f"{len(symbols)}")
    return symbols, np.asarray(coords, dtype=float), comment


def write_xyz(path, symbols, coords, comment=""):
    comment = comment.replace("\n", " ")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{len(symbols)}\n{comment}\n")
        for s, r in zip(symbols, coords):
            f.write(f"{s:2s}  {r[0]:12.6f}  {r[1]:12.6f}  {r[2]:12.6f}\n")


# ---- Geometry ------------------------------------------------------------
def kabsch(P, Q_):
    """Rotation and translation taking P onto Q_ (both N x 3).

    The det() correction forces a proper rotation; without it a near-planar
    fragment can be matched by a reflection, which would report a molecule as
    superposable on its mirror image.
    """
    pc, qc = P.mean(axis=0), Q_.mean(axis=0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q_ - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, qc - R @ pc


def descriptors(positions, n_a, heavy_a, contact_cutoff=5.0):
    """Describe a stacked pair: separation along monomer A's best-fit-plane
    normal, the two in-plane slips, the centre of mass distance, the closest
    contact, and the fraction of A within `contact_cutoff` of B.

    Overlap is the descriptor that tracks how much cofacial contact there is,
    and that is what the transfer integral follows.
    """
    positions = np.asarray(positions, dtype=float)
    a, b = positions[:n_a][heavy_a], positions[n_a:][heavy_a]
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    _, _, vt = np.linalg.svd(a - ca)
    delta = cb - ca
    d = np.linalg.norm(positions[:n_a, None, :] - positions[None, n_a:, :],
                       axis=-1)
    return (abs(delta @ vt[2]), delta @ vt[0], delta @ vt[1],
            np.linalg.norm(delta), d.min(),
            (d.min(axis=1) < contact_cutoff).mean())


def drift(positions, reference, n_a):
    """RMSD of monomer B from its starting placement after superposing monomer
    A - i.e. how far the pair has moved relative to the crystal.

    Invariant to rigid-body motion of the pair as a whole, which is exactly
    what you want when the thermostat is free to rotate and translate it.
    """
    R, t = kabsch(positions[:n_a], reference[:n_a])
    moved = positions[n_a:] @ R.T + t
    return float(np.sqrt(((moved - reference[n_a:]) ** 2).sum(axis=1).mean()))


def heavy_indices(symbols):
    return np.array([i for i, s in enumerate(symbols) if s != "H"], dtype=int)


# ---- Console -------------------------------------------------------------
def banner(text, width=70):
    print(f"\n{'=' * width}\n{text}\n{'=' * width}", flush=True)


def say(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)
