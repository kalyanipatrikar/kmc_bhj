"""Stage 1: crystal structure -> stacked dimer motifs.

Extracts the conjugated core of the molecule from the experimental structure,
rebuilds the local packing by symmetry, and writes out every stacked pair in
the neighbour shell of one molecule.

**Why the crystal rather than a built dimer.** Planar conjugated molecules with
any curvature do not stack by translating a copy along a plane normal - they
stack by nesting, offset and matched to their own shape. Constructing a pair in
the gas phase produces clashes or edge-only overlap for exactly that reason.
The crystal already contains the answer.

**The alkyl chains are dropped.** They are usually the most disordered part of
the structure, so their coordinates are unreliable, and clustering on them
fuses neighbouring molecules into one blob. They are also what you would
truncate for a coupling calculation anyway - the frontier orbitals live on the
conjugated core - so the core is kept and its attachment points are capped with
hydrogen.

    python -m src.stage1_pair_structure [config.yaml]
"""

import argparse
import csv

import numpy as np

from . import config as config_module
from .crystal.cif import cart_matrix, parse_cif
from .crystal.fragments import complete_molecule, core_fragment, packing
from .crystal.motifs import (MOTIF_COLUMNS, collapse_equivalent, name_motifs,
                             neighbour_shell)
from .crystal.selection import clean_asymmetric_unit
from .utils import banner, read_xyz, say, write_xyz


def run(cfg):
    crystal = cfg.crystal
    cfg.make_dirs("dimers", "results")

    banner(f"stage 1: {cfg.project.name} - pair structures from the crystal")
    if not cfg.cif_path.exists():
        raise SystemExit(f"{cfg.cif_path} not found - put the structure there "
                         f"and point crystal.cif at it")
    say(f"reading {cfg.cif_path}")

    # ---- Step 1: the ordered part of the asymmetric unit -----------------
    cell, symops, atoms = parse_cif(cfg.cif_path)
    M = cart_matrix(cell)
    say(f"  {len(symops)} symmetry operations, {len(atoms)} atoms in the "
        f"asymmetric unit")
    atoms = clean_asymmetric_unit(atoms, M, crystal.selection,
                                  crystal.bond_tolerance)

    # ---- Step 2: one complete molecule, then its conjugated core ----------
    whole_symbols, whole_positions, whole_labels = complete_molecule(
        atoms, M, symops, crystal.bond_tolerance)
    symbols, positions, labels = core_fragment(
        whole_symbols, whole_positions, whole_labels, crystal.selection,
        dict(crystal.cap_lengths), crystal.bond_tolerance)

    # ---- Step 3: build the packing ----------------------------------------
    cells = list(crystal.supercell)
    copies = packing(symbols, positions, symops, M, cells,
                     crystal.bond_tolerance)
    say(f"  {len(copies)} molecules of {len(symbols)} atoms in a "
        f"{len(cells)}x{len(cells)}x{len(cells)} block")

    all_positions = np.vstack(copies)
    all_symbols = np.array(list(symbols) * len(copies))
    n_atoms = len(symbols)
    index = [np.arange(i * n_atoms, (i + 1) * n_atoms) for i in range(len(copies))]

    # ---- Step 4: the neighbour shell of the most interior molecule --------
    centres = np.array([all_positions[m].mean(axis=0) for m in index])
    interior = int(np.argmin(np.linalg.norm(centres - centres.mean(axis=0),
                                            axis=1)))
    reference = index[interior]
    others = [m for i, m in enumerate(index) if i != interior]

    rows = neighbour_shell(all_positions, all_symbols, reference, others,
                           crystal.contact_cutoff)
    say(f"  {len(rows)} neighbours within {crystal.contact_cutoff} A")
    rows = name_motifs(collapse_equivalent(rows), crystal.min_contacts,
                       crystal.max_motifs)

    # ---- Step 5: write the dimers and the motif table ---------------------
    say(f"\n{'motif':<10} {'contact':>8} {'interpl':>8} {'pairs':>6} "
        f"{'mult':>5} {'term%':>6} {'COM':>7}")
    written = []
    for row in rows:
        name = row["motif"] or "-"
        say(f"{name:<10} {row['closest_contact_A']:8.2f} "
            f"{row['interplanar_A']:8.2f} {row['n_contacts']:6d} "
            f"{row['multiplicity']:5d} {row['terminal_fraction'] * 100:5.0f}% "
            f"{row['com_distance_A']:7.2f}"
            f"{'' if row['stacked'] else '   peripheral contact'}")
        if not row["stacked"]:
            continue

        pair = np.concatenate([reference, row["molecule"]])
        note = (f"{cfg.project.name} {row['motif']} dimer from "
                f"{crystal.cif}: closest contact "
                f"{row['closest_contact_A']:.3f} A, interplanar "
                f"{row['interplanar_A']:.3f} A, {row['n_contacts']} contacts, "
                f"multiplicity {row['multiplicity']}, n_A={len(reference)}")
        path = cfg.dimers_dir / f"{row['motif']}_dimer.xyz"
        write_xyz(path, all_symbols[pair], all_positions[pair], note)
        written.append(path)

    table = cfg.results_dir / "motifs.csv"
    with open(table, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MOTIF_COLUMNS,
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in MOTIF_COLUMNS})

    say(f"\nwrote {len(written)} dimer(s) to {cfg.dimers_dir}/ and the motif "
        f"table to {table}")
    verify(written)
    if written:
        say("\nready for stage 2: python -m src.stage2_generate_configs")
    else:
        say("\nno stacked pair passed crystal.min_contacts "
            f"({crystal.min_contacts}) - lower it, or widen "
            f"crystal.contact_cutoff")
    return written


def verify(paths):
    """Stage 2 splits each dimer down the middle and requires the two halves to
    be the same molecule in the same atom order. Anything wrong with that shows
    up here rather than an hour into an MD run."""
    for path in sorted(paths):
        symbols, positions, _ = read_xyz(path)
        n_a = len(symbols) // 2
        if len(symbols) % 2 or symbols[:n_a] != symbols[n_a:]:
            raise SystemExit(
                f"{path}: the two halves are not the same molecule in the "
                f"same order - stage 2 would refuse this file")
        distances = np.linalg.norm(
            positions[:n_a, None, :] - positions[None, n_a:, :], axis=-1)
        say(f"  {path.name}: {len(symbols)} atoms ({n_a} per monomer), "
            f"closest contact {distances.min():.2f} A, overlap "
            f"{(distances.min(axis=1) < 5.0).mean() * 100:.0f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", nargs="?", default=None,
                        help="path to config.yaml")
    args = parser.parse_args()
    run(config_module.load(args.config))


if __name__ == "__main__":
    main()
