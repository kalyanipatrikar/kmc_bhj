"""The neighbour shell of one molecule, and the stacked pairs in it.

Every molecule touching the reference within the contact cutoff is described
by its closest approach, interplanar spacing, contact count, and how much of
the contact is made through the ends of the molecule rather than its middle.
Symmetry-equivalent neighbours are collapsed into one entry carrying a
multiplicity.

Motifs are then named generically - motif_01, motif_02 - ranked by contact
count. The chemistry-meaningful reading of each one is left to the descriptor
columns rather than baked into the name, so the pipeline works on a crystal
whose packing nobody has classified yet.
"""

import numpy as np

MOTIF_COLUMNS = ["motif", "closest_contact_A", "interplanar_A", "n_contacts",
                 "multiplicity", "terminal_fraction", "com_distance_A",
                 "separation_A", "slip_long_A", "slip_short_A",
                 "overlap_fraction", "stacked"]


def _terminal_mask(points, centre, long_axis, extent, fraction=1.0 / 3.0):
    """True for points lying in the outer `fraction` of the molecule along its
    long axis - a label-free stand-in for "belongs to the end group"."""
    projection = np.abs((points - centre) @ long_axis)
    return projection > (1.0 - fraction) * extent


def neighbour_shell(positions, symbols, reference, others, contact_cutoff):
    """Describe every molecule in `others` that touches `reference`.

    `reference` and each entry of `others` are index arrays into `positions`.
    """
    heavy_ref = reference[symbols[reference] != "H"]
    ref_points = positions[heavy_ref]
    ref_centre = ref_points.mean(axis=0)
    _, _, ref_axes = np.linalg.svd(ref_points - ref_centre)
    long_axis = ref_axes[0]
    extent = np.abs((ref_points - ref_centre) @ long_axis).max()

    rows = []
    for molecule in others:
        heavy = molecule[symbols[molecule] != "H"]
        distances = np.linalg.norm(
            positions[heavy_ref][:, None, :] - positions[heavy][None, :, :],
            axis=-1)
        if distances.min() > contact_cutoff:
            continue

        contact_a = heavy_ref[distances.min(axis=1) < contact_cutoff]
        contact_b = heavy[distances.min(axis=0) < contact_cutoff]

        # interplanar spacing: mean out-of-plane offset of B's contact atoms
        # from the best-fit plane through A's
        if len(contact_a) >= 8 and len(contact_b) >= 8:
            centre = positions[contact_a].mean(axis=0)
            _, _, axes = np.linalg.svd(positions[contact_a] - centre)
            interplanar = float(np.abs((positions[contact_b] - centre) @ axes[2]).mean())
        else:
            interplanar = float("nan")

        contact_points = positions[np.concatenate([contact_a, contact_b])]
        terminal = float(_terminal_mask(contact_points, ref_centre,
                                        long_axis, extent).mean())

        pair = np.concatenate([reference, molecule])
        geometry = _pair_geometry(positions[pair], len(reference),
                                  symbols[reference] != "H")

        rows.append({
            "molecule": molecule,
            "closest_contact_A": float(distances.min()),
            "interplanar_A": interplanar,
            "n_contacts": int((distances < contact_cutoff).sum()),
            "terminal_fraction": terminal,
            **geometry,
        })

    rows.sort(key=lambda row: -row["n_contacts"])
    return rows


def _pair_geometry(pair_positions, n_a, heavy_mask):
    """Separation, slips, COM distance and overlap for a pair, using the same
    definitions stage 2 monitors during the dynamics."""
    from ..utils import descriptors

    heavy = np.flatnonzero(heavy_mask)
    sep, slip_long, slip_short, com, _, overlap = descriptors(
        pair_positions, n_a, heavy)
    return {"separation_A": float(sep), "slip_long_A": float(slip_long),
            "slip_short_A": float(slip_short), "com_distance_A": float(com),
            "overlap_fraction": float(overlap)}


def collapse_equivalent(rows):
    """Collapse symmetry-equivalent neighbours, carrying a multiplicity.

    Two neighbours are equivalent when their closest contact and contact count
    agree; in a crystal that is only true of genuinely related contacts.
    """
    seen, unique = {}, []
    for row in rows:
        key = (round(row["closest_contact_A"], 2), round(row["n_contacts"], -1))
        if key in seen:
            seen[key]["multiplicity"] += 1
            continue
        row = dict(row, multiplicity=1)
        seen[key] = row
        unique.append(row)
    return unique


def name_motifs(rows, min_contacts, max_motifs):
    """Assign motif_NN names, ranked by contact count, and mark which are
    stacked pairs rather than peripheral tip contacts."""
    named, n_written = [], 0
    for row in rows:
        stacked = row["n_contacts"] >= min_contacts and n_written < max_motifs
        if stacked:
            n_written += 1
            row = dict(row, motif=f"motif_{n_written:02d}", stacked=True)
        else:
            row = dict(row, motif="", stacked=False)
        named.append(row)
    return named
