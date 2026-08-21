"""Stage 2: sample thermal configurations of each dimer motif.

Every motif written by stage 1 is run here, `md.n_configs` configurations each.

Per motif, into `out_folder/configs/<motif>/`:

    config_NNN_dimer.xyz    both molecules, for the dimer SCF
    config_NNN_A.xyz        monomer A alone, in the dimer's frame
    config_NNN_B.xyz        monomer B alone, in the dimer's frame

plus `results/manifest_<motif>.csv`, written incrementally. Everything is
motif-tagged, so two motifs cannot overwrite each other.

Each motif checkpoints itself: if a run is interrupted, run it again and it
resumes from the last completed frame. Delete
`out_folder/checkpoints/checkpoint_<motif>.npz` to force a fresh run.

Needs torch, ase>=3.28 and mace-torch. A GPU is close to essential - tens of ps
of MACE dynamics per motif on a CPU is impractically slow.

    python -m src.stage2_generate_configs [config.yaml] [--motif motif_01]
"""

import argparse

from . import config as config_module
from .md.calculators import device
from .md.sampling import report, run_motif
from .utils import banner, say


def run(cfg, only=None):
    cfg.make_dirs("configs", "checkpoints", "logs", "results")
    motifs = cfg.motifs()
    if only:
        unknown = [m for m in only if m not in motifs]
        if unknown:
            raise SystemExit(f"unknown motif(s) {', '.join(unknown)}; stage 1 "
                             f"wrote {', '.join(motifs)}")
        motifs = list(only)

    banner(f"stage 2: {cfg.project.name} - thermal sampling")
    dev = device()
    say(f"device {dev}, motifs {', '.join(motifs)}")
    if dev == "cpu":
        say("  WARNING: no GPU visible - this will be impractically slow")

    results = {}
    for motif in motifs:
        results[motif] = run_motif(cfg, motif)

    banner("summary")
    for motif, (manifest, crystal) in results.items():
        report(cfg, manifest, crystal, motif)
    say("\nready for stage 3: python -m src.stage3_transfer_integrals")
    return results


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
