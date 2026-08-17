"""Run the whole pipeline, stage 1 through stage 5.

Every stage is resumable, so re-running this after an interruption picks up
roughly where it stopped rather than starting over.

    python scripts/run_all.py [config.yaml] [--from 3] [--to 4]

Note the wall times are very uneven: stage 3 is hours to days of DFT, stage 2
is tens of minutes on a GPU, and stages 1, 4 and 5 are minutes. Running the
whole thing unattended is reasonable; running it interactively is not.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as config_module              # noqa: E402
from src import (stage1_pair_structure, stage2_generate_configs,  # noqa: E402
                 stage3_transfer_integrals, stage4_random_walk,
                 stage5_mobility)
from src.utils import banner, say                    # noqa: E402

STAGES = [
    ("1  pair structures", stage1_pair_structure.run),
    ("2  thermal sampling", stage2_generate_configs.run),
    ("3  transfer integrals", stage3_transfer_integrals.run),
    ("4  random walk", stage4_random_walk.run),
    ("5  mobility", stage5_mobility.run),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", nargs="?", default=None)
    parser.add_argument("--from", dest="first", type=int, default=1,
                        choices=range(1, 6))
    parser.add_argument("--to", dest="last", type=int, default=5,
                        choices=range(1, 6))
    args = parser.parse_args()
    if args.first > args.last:
        raise SystemExit("--from must not be greater than --to")

    cfg = config_module.load(args.config)
    for number, (name, run) in enumerate(STAGES, start=1):
        if not args.first <= number <= args.last:
            continue
        run(cfg)

    banner("pipeline complete")
    say("summary: python -m src.analysis")


if __name__ == "__main__":
    main()
