"""Stage 3: energies and charge transfer integrals.

For a sample of the configurations of each motif, this computes the electronic
coupling between the frontier orbitals of the two monomers by dimer projection,
together with the site energies that go with it. See `src/dft/dipro.py` for the
formulation.

Output, into `out_folder/results/`:

    transfer_integrals_<motif>.csv   every quantity, one row per frame
    J_<motif>.txt                    LUMO-LUMO couplings in meV, per motif
    J.txt                            the same, pooled across motifs, which is
                                     what stage 4 reads

Where a monomer's frontier orbital is near-degenerate with the next one, the
coupling is computed between the two frontier *manifolds* rather than between
two single orbitals - see `src/dft/dipro.py`. That costs no extra SCF and
reduces exactly to the single-orbital result when nothing is degenerate.

Results are appended per frame and frames already in a motif's CSV are skipped,
so an interrupted run loses at most the frame in progress. Delete the CSVs to
start over, and do delete them if you change dft.seed, dft.n_sample or any of
the method settings.

Needs pyscf, which is Linux/macOS only - under Windows run this from WSL.

    python -m src.stage3_transfer_integrals [config.yaml] [--motif motif_01]
"""

import argparse
import csv
import os

import numpy as np

from . import config as config_module
from .dft.driver import (J_COLUMN, append_row, completed, process,
                         select_frames, write_j_files)
from .utils import banner, say


def run(cfg, only=None):
    cfg.make_dirs("results")
    motifs = only or cfg.motifs()

    from pyscf import lib
    n_threads = cfg.dft.n_threads or os.cpu_count()
    lib.num_threads(n_threads)

    banner(f"stage 3: {cfg.project.name} - transfer integrals")
    say(f"basis {cfg.dft.basis}, functional {cfg.dft.xc}, mode "
        f"{cfg.dft.mode}, {n_threads} threads")
    say(f"{cfg.dft.n_sample} frames total, "
        f"{'stratified across' if cfg.dft.stratify else 'pooled over'} "
        f"{len(motifs)} motif(s)\n")

    selection = select_frames(cfg, motifs)

    couplings = {}
    for motif in motifs:
        output = cfg.results_dir / f"transfer_integrals_{motif}.csv"
        banner(motif)
        done = completed(output)
        if done:
            say(f"{len(done)} frames already done, skipping those")
        for tag in selection[motif]:
            if tag in done:
                continue
            append_row(output, process(cfg, motif, tag))
        couplings[motif] = _read_j(output)

    banner("summary")
    for motif in motifs:
        summarise(cfg, motif)
    say("")
    write_j_files(cfg, motifs, couplings)
    say("\nready for stage 4: python -m src.stage4_random_walk")
    return couplings


def _read_j(path):
    if not path.exists():
        return np.array([])
    with open(path, newline="", encoding="utf-8") as handle:
        return np.array([float(row[J_COLUMN])
                         for row in csv.DictReader(handle)])


def summarise(cfg, motif):
    path = cfg.results_dir / f"transfer_integrals_{motif}.csv"
    rows = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    say(f"\n{motif}  ({len(rows)} frames)")
    for name in ("lumo", "homo"):
        j = np.abs(rows[f"J_{name}_manifold_meV"])
        # rms is the transport-relevant average: Marcus rates go as J^2, so
        # averaging J itself over a disordered ensemble understates it
        say(f"  |J| {name.upper():4s} mean {j.mean():6.1f}  sd {j.std():5.1f}"
            f"  rms {np.sqrt((j ** 2).mean()):6.1f} meV  "
            f"(min {j.min():.1f}, max {j.max():.1f})")
    de = rows["dE_lumo_eV"]
    say(f"  site energy difference, LUMO: mean {de.mean():+.3f} eV, "
        f"sd {de.std():.3f} eV")

    gaps = np.minimum(rows["lumo_gap_A_eV"], rows["lumo_gap_B_eV"])
    sizes = np.maximum(rows["n_lumo_A"], rows["n_lumo_B"])
    degenerate = int((sizes > 1).sum())
    say(f"  smallest LUMO/LUMO+1 gap: {gaps.min():.3f} eV")
    if degenerate:
        single = np.abs(rows["J_lumo_meV"])
        multi = np.abs(rows["J_lumo_manifold_meV"])
        say(f"  {degenerate} of {len(rows)} frames had near-degenerate "
            f"frontier orbitals and were treated as a manifold "
            f"(up to {int(sizes.max())} orbitals per monomer); rms |J| "
            f"{np.sqrt((single ** 2).mean()):.1f} -> "
            f"{np.sqrt((multi ** 2).mean()):.1f} meV")
    say(f"  mean wall time per frame: {rows['seconds'].mean() / 60:.1f} min")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", nargs="?", default=None,
                        help="path to config.yaml")
    parser.add_argument("--motif", action="append", dest="motifs",
                        help="run only this motif; repeatable")
    args = parser.parse_args()
    run(config_module.load(args.config), only=args.motifs)


if __name__ == "__main__":
    main()
