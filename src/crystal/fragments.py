"""From the asymmetric unit to a periodic block of complete molecules.

Four steps:

  1. expand the asymmetric unit by symmetry over a small block and group the
     result into molecules by covalent connectivity, to obtain one *complete*
     molecule;
  2. cut the conjugated core out of that molecule and cap the bonds broken
     doing so;
  3. apply every symmetry operation and cell translation to the resulting
     fragment;
  4. discard the duplicates that generates.

Step 1 exists because the asymmetric unit is very often a fraction of a
molecule - half of it when the molecule sits on an inversion centre, which is
extremely common for a planar conjugated molecule. Extracting the core before
expanding would then produce half a molecule, and the packing analysis would be
describing contacts between half-molecules.

Working from a complete molecule and transforming *that* also gives the
property stages 2 and 3 both depend on and neither could repair: every molecule
in the block is the same ordered list of atoms, so monomer A and monomer B of
any pair have identical atom ordering.
"""

from collections import Counter

import numpy as np

from ..utils import say
from .cif import parse_symop
from .selection import resolve_labels
from .topology import bond_graph, connected_components, describe_core, detect_core

# a 3x3x3 block is always enough to complete a molecule split by a cell
# boundary, and is much cheaper than the block used for the packing itself
COMPLETION_CELLS = (-1, 0, 1)


def _transform(fractional, symops, cells, M):
    """Every symmetry image of a set of fractional coordinates, as Cartesians.

    Returns one array per (operation, translation), each in the same atom order
    as the input.
    """
    operations = [parse_symop(op) if isinstance(op, str) else op
                  for op in symops]
    translations = [np.array([a, b, c], dtype=float)
                    for a in cells for b in cells for c in cells]
    return [((fractional @ R.T + t + translation) @ M.T)
            for R, t in operations for translation in translations]


def complete_molecule(atoms, M, symops, tolerance=1.15):
    """One whole molecule from the packing, with its atoms in a fixed order.

    The asymmetric unit may be a fraction of a molecule, so the molecule is
    reassembled by symmetry first and only then treated as the unit of
    structure.
    """
    Minv = np.linalg.inv(M)
    fractional = np.array([a["f"] for a in atoms])
    symbols_unit = [a["sym"] for a in atoms]
    labels_unit = [a["label"] for a in atoms]

    blocks = _transform(fractional, symops, COMPLETION_CELLS, M)
    positions = np.vstack(blocks)
    n_unit = len(atoms)
    symbols = np.array(symbols_unit * len(blocks))
    labels = np.array(labels_unit * len(blocks))

    adjacency = bond_graph(positions, list(symbols), tolerance)
    components = connected_components(adjacency)
    largest = max(len(c) for c in components)
    whole = [c for c in components if len(c) == largest]

    # the most interior one, so it cannot be a molecule clipped by the edge of
    # the block that happens to have the full atom count
    centres = np.array([positions[c].mean(axis=0) for c in whole])
    interior = whole[int(np.argmin(
        np.linalg.norm(centres - positions.mean(axis=0), axis=1)))]

    per_unit = largest / n_unit
    if abs(per_unit - round(per_unit)) > 1e-9:
        say(f"  WARNING: a complete molecule is {largest} atoms and the "
            f"asymmetric unit is {n_unit}, which is not a whole-number ratio; "
            f"the structure may be disordered in a way this cannot untangle")
    if largest != n_unit:
        say(f"  the asymmetric unit is 1/{largest // n_unit} of a molecule; "
            f"reassembled to {largest} atoms")

    # plain str, not numpy's - these end up in dict keys and xyz files
    return ([str(s) for s in symbols[interior]], positions[interior],
            [str(l) for l in labels[interior]])


def core_fragment(symbols, positions, labels, selection, cap_lengths,
                  tolerance=1.15):
    """The conjugated core of one molecule, with chain attachment points capped
    by hydrogen."""
    positions = np.asarray(positions, dtype=float)

    explicit = selection.get("core_labels")
    if explicit:
        core = resolve_labels(labels, explicit, "core_labels")
        adjacency = bond_graph(positions, symbols, tolerance)
        say(f"  core taken from crystal.selection.core_labels: "
            f"{len(core)} atoms")
    else:
        keep = resolve_labels(labels, selection.get("keep_labels"), "keep_labels")
        drop = resolve_labels(labels, selection.get("drop_labels"), "drop_labels")
        core, adjacency = detect_core(positions, symbols, tolerance, keep, drop)
        say(f"  auto-detected core: {describe_core(core, adjacency, symbols)}")

    core_set = set(int(i) for i in core)

    out_positions, out_symbols, out_labels = [], [], []
    for i in sorted(core_set):
        out_positions.append(positions[i])
        out_symbols.append(symbols[i])
        out_labels.append(labels[i])

    # hydrogens already bonded to the core come along unchanged
    for i, symbol in enumerate(symbols):
        if symbol != "H":
            continue
        if any(j in core_set for j in adjacency[i]):
            out_positions.append(positions[i])
            out_symbols.append("H")
            out_labels.append(labels[i])

    # where a chain was cut, put a hydrogen along the broken bond
    n_caps = 0
    for i in sorted(core_set):
        for j in adjacency[i]:
            if j in core_set or symbols[j] == "H":
                continue
            if symbols[i] not in cap_lengths:
                raise SystemExit(f"no capping bond length for {symbols[i]}-H; "
                                 f"add it to crystal.cap_lengths")
            direction = positions[j] - positions[i]
            direction = direction / np.linalg.norm(direction)
            out_positions.append(
                positions[i] + cap_lengths[symbols[i]] * direction)
            out_symbols.append("H")
            out_labels.append(f"Hcap{labels[i]}")
            n_caps += 1

    composition = dict(sorted(Counter(out_symbols).items()))
    say(f"  capped {n_caps} attachment point(s); fragment "
        f"{len(out_symbols)} atoms {composition}")
    return out_symbols, np.array(out_positions), out_labels


def packing(symbols, positions, symops, M, cells, tolerance=1.15):
    """Every molecule in the block, each with the same atom ordering.

    Because the input is already a complete molecule, applying the symmetry
    operations to it reproduces the whole packing directly - no connectivity
    clustering, and no risk of two neighbouring molecules being fused into one
    component. Operations that map the molecule onto itself produce duplicates,
    which are removed by centroid.
    """
    Minv = np.linalg.inv(M)
    fractional = np.array([Minv @ p for p in positions])
    copies = _transform(fractional, symops, cells, M)

    seen, unique = set(), []
    for copy in copies:
        key = tuple(np.round(copy.mean(axis=0), 2))
        if key in seen:
            continue
        seen.add(key)
        unique.append(copy)

    if not unique:
        raise SystemExit("the symmetry expansion produced no molecules")

    # a sanity check on the whole construction: a molecule that overlaps
    # another is a sign the deduplication or the symmetry parsing is wrong
    stacked = np.vstack(unique)
    adjacency = bond_graph(stacked, list(symbols) * len(unique), tolerance)
    n_atoms = len(symbols)
    for atom, neighbours in enumerate(adjacency):
        molecule = atom // n_atoms
        if any(j // n_atoms != molecule for j in neighbours):
            raise SystemExit(
                "two molecules in the expanded packing are covalently bonded "
                "to each other, which means the fragment is not a complete "
                "molecule. Set crystal.selection.core_labels explicitly.")

    return unique
