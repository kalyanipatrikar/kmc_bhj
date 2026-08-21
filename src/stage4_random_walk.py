"""Stage 4: kinetic Monte Carlo random walk on a lattice of hopping sites.

A charge carrier moves through a grid of scattered conformation pairs. At each
step a coupling is drawn from the ensemble stage 3 produced, representing a hop
across a molecular pair in that configuration, and the Marcus rate to each of
the 26 neighbouring sites decides where the carrier goes and how long it takes.

Output, into `out_folder/trajectories/`:

    x.npy       site index after each step, shape (trials, steps)
    time.npy    elapsed time after each step, shape (trials, steps)

Both are read by stage 5.

    python -m src.stage4_random_walk [config.yaml]
"""

import argparse

import numpy as np

from . import config as config_module
from .transport.kmc import walk
from .transport.lattice import Lattice
from .transport.rates import load_couplings
from .utils import banner, say


def run(cfg):
    cfg.make_dirs("trajectories")
    transport = cfg.transport

    banner(f"stage 4: {cfg.project.name} - kinetic Monte Carlo")
    j_path = cfg.results_dir / transport.j_file
    if not j_path.exists():
        raise SystemExit(f"{j_path} not found - run stage 3 first")
    pool = load_couplings(j_path)

    lattice = Lattice(transport.lattice)
    say(f"lattice {lattice.L}x{lattice.W}x{lattice.H} = {lattice.N:,} sites, "
        f"starting at {lattice.centre}")
    say(f"{transport.trials} trials of {transport.steps} steps at "
        f"{transport.temperature:.0f} K")
    say(f"site energy disorder sigma {transport.sigma} eV, reorganisation "
        f"energy {transport.reorganisation_energy} eV")

    sites, times = walk(cfg, pool)

    np.save(cfg.trajectories_dir / "x.npy", sites)
    np.save(cfg.trajectories_dir / "time.npy", times)
    say(f"\nwrote x.npy and time.npy to {cfg.trajectories_dir}/")

    final = times[:, -1]
    displacement = lattice.separation(lattice.centre, sites[:, -1])
    say(f"  after {transport.steps} steps: total time "
        f"{final.mean():.3e} +/- {final.std():.1e} s, displacement "
        f"{displacement.mean():.1f} +/- {displacement.std():.1f} sites")
    say("\nready for stage 5: python -m src.stage5_mobility")
    return sites, times


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", nargs="?", default=None,
                        help="path to config.yaml")
    args = parser.parse_args()
    run(config_module.load(args.config))


if __name__ == "__main__":
    main()
