"""The site lattice the carrier walks on.

Every point gets a flat index from 0 to N-1, with x fastest, then y, then z:

    index = z * (L * W) + y * L + x

The 26 neighbours of a site are the full cube shell around it, reached by
adding the corresponding offsets to the flat index and wrapping modulo N.
"""

from itertools import product

import numpy as np


class Lattice:
    def __init__(self, dims):
        self.L, self.W, self.H = (int(d) for d in dims)
        if min(self.L, self.W, self.H) < 3:
            raise SystemExit("transport.lattice must be at least 3 in every "
                             "direction for a full neighbour shell")
        self.plane = self.L * self.W
        self.N = self.plane * self.H
        self.offsets = np.array(
            [dz * self.plane + dy * self.L + dx
             for dz, dy, dx in product((-1, 0, 1), repeat=3)
             if (dx, dy, dz) != (0, 0, 0)], dtype=np.int64)

    @property
    def centre(self):
        """The site every trial starts from, at the middle of the box."""
        return int(self.L // 2 + (self.W // 2 - 1) * self.L
                   + (self.H // 2 - 1) * self.plane)

    def neighbours(self, site):
        """The 26 neighbours of a site, wrapped periodically."""
        return (np.int64(site) + self.offsets) % np.int64(self.N)

    def coordinates(self, index):
        """Flat index -> (x, y, z)."""
        index = np.asarray(index, dtype=np.int64)
        return (index % self.plane) % self.L, (index % self.plane) // self.L, \
            index // self.plane

    def separation(self, a, b):
        """Distance between two sites in lattice units.

        Note this is the plain difference of coordinates, not a minimum-image
        distance: a carrier that wraps around the box would be recorded as
        having travelled backwards. With the default 800^3 box and a few
        thousand steps from the centre, the boundary is never reached, but
        shrinking transport.lattice without shortening transport.steps would
        make that assumption false.
        """
        ax, ay, az = self.coordinates(a)
        bx, by, bz = self.coordinates(b)
        return np.sqrt((az - bz) ** 2 + (ay - by) ** 2 + (ax - bx) ** 2)

    def max_excursion(self, steps):
        """The furthest a walk of `steps` hops could possibly get, used to warn
        when the box is too small to hold the trajectories."""
        return steps
