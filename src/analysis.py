"""A read-only summary of whatever the pipeline has produced so far.

Useful for picking up a run you left days ago, and for checking what a stage
actually wrote before spending compute on the next one.

    python -m src.analysis [config.yaml]
"""

import argparse
import csv

import numpy as np

from . import config as config_module
from .dft.driver import J_COLUMN
from .utils import banner, say


def _table(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run(cfg):
    banner(f"{cfg.project.name}: pipeline status")

    # ---- Stage 1 ---------------------------------------------------------
    motifs_csv = cfg.results_dir / "motifs.csv"
    rows = _table(motifs_csv)
    stacked = [r for r in rows if r.get("stacked") in ("True", "true", "1")]
    if not rows:
        say("stage 1  not run")
    else:
        say(f"stage 1  {len(rows)} neighbours, {len(stacked)} stacked motif(s)")
        for row in stacked:
            say(f"           {row['motif']}: contact "
                f"{float(row['closest_contact_A']):.2f} A, interplanar "
                f"{float(row['interplanar_A']):.2f} A, "
                f"{row['n_contacts']} contacts, multiplicity "
                f"{row['multiplicity']}, terminal "
                f"{float(row['terminal_fraction']) * 100:.0f}%")

    # ---- Stage 2 ---------------------------------------------------------
    names = [r["motif"] for r in stacked] or []
    if not names:
        say("stage 2  not run")
    for motif in names:
        manifest = cfg.results_dir / f"manifest_{motif}.csv"
        frames = _table(manifest)
        if not frames:
            say(f"stage 2  {motif}: not run")
            continue
        drift = np.array([float(f["drift_A"]) for f in frames])
        overlap = np.array([float(f["overlap_fraction"]) for f in frames])
        say(f"stage 2  {motif}: {len(frames)}/{cfg.md.n_configs} frames, "
            f"drift {drift.mean():.2f} +/- {drift.std():.2f} A (max "
            f"{drift.max():.2f}), overlap {overlap.mean() * 100:.0f}%")

    # ---- Stage 3 ---------------------------------------------------------
    for motif in names:
        path = cfg.results_dir / f"transfer_integrals_{motif}.csv"
        frames = _table(path)
        if not frames:
            say(f"stage 3  {motif}: not run")
            continue
        j = np.abs(np.array([float(f[J_COLUMN]) for f in frames]))
        seconds = np.array([float(f["seconds"]) for f in frames])
        degenerate = sum(1 for f in frames
                         if max(int(f["n_lumo_A"]), int(f["n_lumo_B"])) > 1)
        say(f"stage 3  {motif}: {len(frames)} frames, |J| LUMO rms "
            f"{np.sqrt((j ** 2).mean()):.1f} meV (mean {j.mean():.1f}, sd "
            f"{j.std():.1f}), {seconds.mean() / 60:.1f} min/frame"
            f"{f', {degenerate} with a frontier manifold' if degenerate else ''}")

    pooled = cfg.results_dir / cfg.transport.j_file
    if pooled.exists():
        values = np.atleast_1d(np.loadtxt(pooled, comments="#"))
        say(f"stage 3  pooled {pooled.name}: {values.size} couplings, rms "
            f"{np.sqrt(np.mean(values ** 2)):.1f} meV")

    # ---- Stage 4 ---------------------------------------------------------
    x_path = cfg.trajectories_dir / "x.npy"
    if x_path.exists():
        sites = np.load(x_path, mmap_mode="r")
        say(f"stage 4  {sites.shape[0]} trials of {sites.shape[1]} steps")
    else:
        say("stage 4  not run")

    # ---- Stage 5 ---------------------------------------------------------
    mobility = _table(cfg.results_dir / "mobility.csv")
    if mobility:
        row = mobility[0]
        say(f"stage 5  mobility {float(row['mobility_cm2_per_Vs']):.4e} "
            f"cm^2/(V s), diffusion "
            f"{float(row['diffusion_cm2_per_s']):.4e} cm^2/s")
    else:
        say("stage 5  not run")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", nargs="?", default=None)
    args = parser.parse_args()
    run(config_module.load(args.config))


if __name__ == "__main__":
    main()
