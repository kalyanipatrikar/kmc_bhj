"""Bond graph, ring perception, and automatic detection of the conjugated core.

This is what replaces the hand-written "carbons numbered 24 and above are the
alkyl chains" rule. The chain of reasoning is:

  * two atoms are bonded if they are closer than a tolerance times the sum of
    their covalent radii;
  * a carbon with four bonded neighbours is sp3, and an sp3 carbon cannot be
    part of a conjugated system;
  * deleting the sp3 carbons therefore cuts every saturated side chain away
    from the conjugated framework, and the largest connected piece that
    remains is the core.

Terminal groups that matter electronically - carbonyl oxygens, nitrile
nitrogens, ring sulfurs, fluorines - survive this because none of them is an
sp3 carbon and all of them are bonded to the framework.
"""

import numpy as np
from scipy.spatial import cKDTree

from ..utils import covalent_radii


def bond_graph(positions, symbols, tolerance=1.15):
    """Adjacency list over all atoms, bonded when
    d < tolerance * (r_cov_i + r_cov_j)."""
    positions = np.asarray(positions, dtype=float)
    radii = covalent_radii(symbols)
    # 2 * max radius * tolerance bounds any bond length in the structure
    cutoff = 2.0 * radii.max() * tolerance
    pairs = cKDTree(positions).query_pairs(cutoff, output_type="ndarray")

    adjacency = [set() for _ in symbols]
    if len(pairs) == 0:
        return adjacency
    distances = np.linalg.norm(positions[pairs[:, 0]] - positions[pairs[:, 1]],
                               axis=1)
    bonded = distances < tolerance * (radii[pairs[:, 0]] + radii[pairs[:, 1]])
    for i, j in pairs[bonded]:
        adjacency[i].add(int(j))
        adjacency[j].add(int(i))
    return adjacency


def connected_components(adjacency, subset=None):
    """Connected components as a list of sorted index arrays.

    `subset` restricts the traversal to those atoms; edges leaving it are
    ignored, which is how the core is isolated once the sp3 carbons are
    removed.
    """
    allowed = set(range(len(adjacency))) if subset is None else set(map(int, subset))
    seen, components = set(), []
    for start in sorted(allowed):
        if start in seen:
            continue
        stack, component = [start], []
        seen.add(start)
        while stack:
            atom = stack.pop()
            component.append(atom)
            for neighbour in adjacency[atom]:
                if neighbour in allowed and neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        components.append(np.array(sorted(component), dtype=int))
    return components


def ring_atoms(adjacency, subset=None):
    """Indices lying on a ring, found by repeatedly deleting terminal atoms.

    Whatever survives has no free end, so every remaining atom is on a cycle or
    on a path joining two cycles.
    """
    alive = set(range(len(adjacency))) if subset is None else set(map(int, subset))
    degree = {a: len(adjacency[a] & alive) for a in alive}
    terminal = [a for a in alive if degree[a] <= 1]
    while terminal:
        atom = terminal.pop()
        if atom not in alive:
            continue
        alive.discard(atom)
        for neighbour in adjacency[atom]:
            if neighbour in alive:
                degree[neighbour] -= 1
                if degree[neighbour] <= 1:
                    terminal.append(neighbour)
    return np.array(sorted(alive), dtype=int)


def n_rings(adjacency, subset=None):
    """Number of independent rings, from Euler's formula: edges - atoms +
    components. This is the size of the cycle basis, i.e. the number of ring
    closures, which for a fused aromatic system is the number of rings."""
    allowed = set(range(len(adjacency))) if subset is None else set(map(int, subset))
    edges = sum(len(adjacency[a] & allowed) for a in allowed) // 2
    return edges - len(allowed) + len(connected_components(adjacency, allowed))


def sp3_carbons(adjacency, symbols):
    """Carbons with four bonded neighbours - saturated, so not conjugated."""
    return np.array([i for i, s in enumerate(symbols)
                     if s == "C" and len(adjacency[i]) >= 4], dtype=int)


def detect_core(positions, symbols, tolerance=1.15, keep=(), drop=()):
    """Indices of the conjugated core: heavy atoms, no hydrogens.

    `keep` and `drop` are index collections that override the heuristic in
    either direction, and are applied after it.
    """
    adjacency = bond_graph(positions, symbols, tolerance)
    saturated = set(sp3_carbons(adjacency, symbols).tolist())

    candidates = {i for i, s in enumerate(symbols)
                  if s != "H" and i not in saturated}
    components = connected_components(adjacency, candidates)
    if not components:
        raise SystemExit("no conjugated system found: every heavy atom is "
                         "either hydrogen-saturated carbon or isolated. Set "
                         "crystal.selection.core_labels explicitly.")

    core = set(max(components, key=len).tolist())
    core |= {int(i) for i in keep}
    core -= {int(i) for i in drop}
    return np.array(sorted(core), dtype=int), adjacency


def describe_core(core, adjacency, symbols):
    """A one-line summary of what the detection produced, so a user can see at
    a glance whether it found the molecule they had in mind."""
    from collections import Counter

    composition = Counter(symbols[i] for i in core)
    formula = " ".join(f"{element}{count}"
                       for element, count in sorted(composition.items()))
    rings = n_rings(adjacency, core)
    on_ring = len(ring_atoms(adjacency, core))
    return (f"{len(core)} heavy atoms ({formula}), {rings} rings, "
            f"{on_ring} of them ring atoms")
