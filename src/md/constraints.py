"""Constraints used during the dynamics."""

import numpy as np


class FlatBottomCOM:
    """Harmonic restraint on the separation of two fragments' centres of mass,
    active only beyond r0. Below r0 it is exactly zero force.

    In the crystal the pair is held in place by the surrounding lattice, which
    is absent in vacuum. This keeps the two molecules from drifting apart
    without biasing any geometry the pair actually visits, because inside the
    flat region it contributes nothing at all.
    """

    def __init__(self, idx_a, idx_b, r0, k):
        self.a = np.asarray(idx_a, dtype=int)
        self.b = np.asarray(idx_b, dtype=int)
        self.r0 = float(r0)
        self.k = float(k)

    def adjust_positions(self, atoms, new):
        pass

    def adjust_forces(self, atoms, forces):
        masses = atoms.get_masses()
        ma, mb = masses[self.a], masses[self.b]
        ca = (atoms.positions[self.a] * ma[:, None]).sum(0) / ma.sum()
        cb = (atoms.positions[self.b] * mb[:, None]).sum(0) / mb.sum()
        vector = cb - ca
        r = np.linalg.norm(vector)
        if r <= self.r0:
            return
        force = self.k * (r - self.r0) * vector / r
        forces[self.a] += force * (ma / ma.sum())[:, None]
        forces[self.b] -= force * (mb / mb.sum())[:, None]

    def get_removed_dof(self, atoms):
        return 0
