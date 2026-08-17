"""The Marcus hopping rate, and the pool of couplings it draws from.

    k = |J|^2 / hbar * sqrt(pi / (lambda kB T)) * exp(-(lambda - dE)^2 / (4 lambda kB T))

returning s^-1. dE is the site energy of the current site minus that of the
destination, so a downhill hop has dE > 0 and the exponent is the usual
(lambda + dG)^2 with dG = -dE.

**Units.** Transfer integrals are in meV throughout this pipeline: stage 3
writes J.txt in meV and stage 4 reads it directly, with no conversion layer in
between. The site energies, the disorder width and the reorganisation energy
are in eV. The rate function therefore takes mixed units by design, and its
arguments are named accordingly.
"""

import numpy as np

from ..utils import HBAR, KB, Q, say

MEV = 1.0e-3        # meV -> eV, applied inside the rate expression only


def marcus(j_mev, de_ev, reorganisation_ev, temperature):
    """Hopping rate in s^-1.

    `j_mev` is the transfer integral in meV, as it comes out of J.txt.
    `de_ev` and `reorganisation_ev` are in eV. Both `j_mev` and `de_ev` may be
    arrays.
    """
    lam = reorganisation_ev * Q
    coupling_j = j_mev * MEV * Q          # meV -> eV -> Joules
    prefactor = (np.square(coupling_j) / HBAR) * np.sqrt(
        np.pi / (lam * KB * temperature))
    exponent = -np.square(lam - de_ev * Q) / (4.0 * KB * temperature * lam)
    return prefactor * np.exp(exponent)


def load_couplings(path):
    """Read a J file written by stage 3. Values are meV and stay meV."""
    values = np.loadtxt(path, comments="#")
    values = np.atleast_1d(values).astype(float)
    if values.size == 0:
        raise SystemExit(f"{path} contains no couplings")
    values = np.abs(values)
    say(f"{path.name}: {values.size} couplings, {values.min():.1f} to "
        f"{values.max():.1f} meV, rms {np.sqrt(np.mean(values ** 2)):.1f} meV")
    return values


def sample_couplings(rng, pool, n):
    """Draw n couplings from the pool, uniformly and with replacement.

    Every entry is reachable, including the first and the last - the walk is
    resampling a small ensemble millions of times, so an off-by-one in the
    range would quietly drop a configuration from the whole calculation.
    """
    return pool[rng.integers(0, pool.size, size=n)]
