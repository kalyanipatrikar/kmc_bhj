"""The override layer for fragment selection.

Automatic detection in `topology.py` handles an ordinary organic crystal on its
own. This module cleans the asymmetric unit before that runs - minor disorder
components and solvent - and translates the label-based overrides in
`config.yaml` into atom indices for when the heuristic needs correcting.

Everything here is optional. With an empty `crystal.selection` block the
pipeline is fully automatic.
"""

import numpy as np

from ..utils import say
from .topology import bond_graph, connected_components


def clean_asymmetric_unit(atoms, M, selection, tolerance=1.15):
    """Drop minor disorder components and solvent from the asymmetric unit.

    Solvent is identified structurally rather than by name: it is a small
    connected component, entirely separate from the molecule. That works
    without knowing what the solvent is, which a hard-coded label list does
    not.
    """
    drop_groups = set(map(str, selection.get("drop_disorder_groups", []) or []))
    kept = [a for a in atoms if str(a["grp"]) not in drop_groups]
    n_disordered = len(atoms) - len(kept)
    if n_disordered:
        say(f"  dropped {n_disordered} atoms in minor disorder component(s) "
            f"{', '.join(sorted(drop_groups))}")

    positions = np.array([M @ a["f"] for a in kept])
    symbols = [a["sym"] for a in kept]
    adjacency = bond_graph(positions, symbols, tolerance)

    minimum = int(selection.get("min_component_size", 0) or 0)
    components = connected_components(adjacency)
    keep_mask = np.zeros(len(kept), dtype=bool)
    dropped = []
    for component in components:
        if len(component) >= minimum:
            keep_mask[component] = True
        else:
            dropped.append(component)
    if dropped:
        formulas = ", ".join(
            "".join(sorted(symbols[i] for i in component)) for component in dropped)
        say(f"  dropped {sum(len(c) for c in dropped)} atoms in "
            f"{len(dropped)} component(s) smaller than {minimum} atoms "
            f"(solvent: {formulas})")
    if not keep_mask.any():
        raise SystemExit(
            "every component of the asymmetric unit is smaller than "
            "crystal.selection.min_component_size - lower it")

    return [a for a, keep in zip(kept, keep_mask) if keep]


def resolve_labels(labels, wanted, what):
    """Turn a list of atom labels from the config into indices, complaining
    about any that do not occur in the structure - a typo in an override is
    otherwise completely silent."""
    wanted = list(wanted or [])
    if not wanted:
        return np.array([], dtype=int)
    lookup = {label: i for i, label in enumerate(labels)}
    missing = [label for label in wanted if label not in lookup]
    if missing:
        raise SystemExit(
            f"crystal.selection.{what} names atom(s) not in the structure: "
            f"{', '.join(missing)}")
    return np.array([lookup[label] for label in wanted], dtype=int)
