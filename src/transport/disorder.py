"""Static disorder: the site energy, the positional shift that attenuates the
coupling, and the orientational factor.

These are properties of a *site*, so a carrier that returns to a site it has
already visited must find it unchanged - that is what makes the disorder
static, and static disorder is what produces trapping. Generating fresh values
on every visit would be dynamic disorder and would wash the traps out.

Rather than filling three arrays of length N up front - at 800^3 that is 4 GB
each, and they are regenerated every trial - each value is derived on demand
from a hash of (site, trial, seed). That is reproducible, consistent within a
trial, independent between trials, and costs memory proportional to the length
of the path rather than the size of the box.
"""

import numpy as np

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)
_SITE = np.uint64(0x2545F4914F6CDD1D)
_TRIAL = np.uint64(0xC2B2AE3D27D4EB4F)
_STREAM = np.uint64(0x165667B19E3779F9)

_TWO53 = 1.0 / 9007199254740992.0    # 2**-53
_TINY = 1e-300                        # keeps log(0) out of Box-Muller


def _splitmix64(x):
    """The SplitMix64 finaliser: a good avalanche from a counter to a random
    64-bit word, which is all that is needed here."""
    z = x + _GOLDEN
    z = (z ^ (z >> np.uint64(30))) * _MIX1
    z = (z ^ (z >> np.uint64(27))) * _MIX2
    return z ^ (z >> np.uint64(31))


_MASK64 = (1 << 64) - 1


def _key(seed, trial, stream):
    """Mix the scalar part of the key in Python integers.

    numpy warns on scalar unsigned overflow even though the wraparound is
    exactly what a hash wants, so the scalars are folded here and only the
    per-site part is done in numpy.
    """
    value = (int(seed) * 0x9E3779B97F4A7C15) & _MASK64
    value ^= ((int(trial) + 1) * 0xC2B2AE3D27D4EB4F) & _MASK64
    value ^= ((int(stream) + 1) * 0x165667B19E3779F9) & _MASK64
    return np.uint64(value)


def _uniform(sites, seed, trial, stream):
    """Independent U(0, 1) per site, per trial, per stream."""
    x = sites.astype(np.uint64) * _SITE
    x = x ^ _key(seed, trial, stream)
    return (_splitmix64(x) >> np.uint64(11)).astype(np.float64) * _TWO53


def _normal(sites, seed, trial, stream):
    """Box-Muller from two uniform streams."""
    u1 = np.maximum(_uniform(sites, seed, trial, stream), _TINY)
    u2 = _uniform(sites, seed, trial, stream + 1)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


class SiteDisorder:
    """Site energies, positional shifts and orientational factors.

    Streams are numbered so the three quantities never share random numbers.
    """

    ENERGY, SHIFT, ANGLE = 0, 8, 16

    def __init__(self, seed, sigma, shift_sigma):
        self.seed = int(seed)
        self.sigma = float(sigma)
        self.shift_sigma = float(shift_sigma)

    def energy(self, sites, trial):
        """Gaussian site energy disorder, N(0, sigma), in eV."""
        sites = np.atleast_1d(np.asarray(sites, dtype=np.int64))
        return self.sigma * _normal(sites, self.seed, trial, self.ENERGY)

    def shift(self, sites, trial):
        """|N(0, s)| positional shift, which attenuates the coupling as
        exp(-shift)."""
        sites = np.atleast_1d(np.asarray(sites, dtype=np.int64))
        return np.abs(self.shift_sigma
                      * _normal(sites, self.seed, trial, self.SHIFT))

    def costheta(self, sites, trial):
        """Cosine of the angle between the molecule and the hop direction,
        drawn uniformly. The coupling picks up its square."""
        sites = np.atleast_1d(np.asarray(sites, dtype=np.int64))
        return _uniform(sites, self.seed, trial, self.ANGLE)
