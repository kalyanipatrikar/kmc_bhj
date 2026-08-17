"""The kinetic Monte Carlo walk.

A carrier starts at the centre of the box. At each step the rate to all 26
neighbours is computed from a coupling drawn at random from the stage 3
ensemble, attenuated by the destination site's positional shift and
orientational factor, and the destination is chosen with probability
proportional to its rate.

Stochasticity comes from resampling a fixed ensemble of DFT couplings rather
than from new DFT: the frames are independent draws from the same thermal
distribution either way, so redrawing the ensemble per trial would cost orders
of magnitude more for no extra statistical information. What sharpens the
result is the number of distinct configurations in the ensemble, which is
dft.n_sample.
"""

import time as wallclock

import numpy as np

from ..utils import say
from .disorder import SiteDisorder
from .lattice import Lattice
from .rates import marcus, sample_couplings


def walk(cfg, pool, progress_every=25):
    """Run every trial. Returns (sites, times), both (trials, steps).

    NOTE ON `times`: each entry is the running total elapsed time, not the
    dwell time of that step, and stage 5 applies a further cumulative sum to
    it. That is the behaviour of the original script and it is preserved here
    deliberately, pending review - see the note in
    `src/transport/mobility.py`.
    """
    transport = cfg.transport
    lattice = Lattice(transport.lattice)
    disorder = SiteDisorder(transport.seed, transport.sigma,
                            transport.shift_sigma)
    rng = np.random.default_rng(transport.seed)

    trials, steps = int(transport.trials), int(transport.steps)
    temperature = float(transport.temperature)
    reorganisation = float(transport.reorganisation_energy)

    if lattice.max_excursion(steps) > min(lattice.L, lattice.W, lattice.H) // 2:
        say(f"  WARNING: {steps} steps could reach the edge of a "
            f"{lattice.L}x{lattice.W}x{lattice.H} box, where the periodic wrap "
            f"would corrupt the displacement - enlarge transport.lattice")

    sites = np.empty((trials, steps), dtype=np.int64)
    times = np.empty((trials, steps), dtype=np.float64)

    started = wallclock.time()
    for trial in range(trials):
        position = lattice.centre
        elapsed = 0.0
        for step in range(steps):
            neighbours = lattice.neighbours(position)

            coupling = (sample_couplings(rng, pool, neighbours.size)
                        * np.exp(-disorder.shift(neighbours, trial))
                        * disorder.costheta(neighbours, trial) ** 2)

            here = disorder.energy(position, trial)[0]
            there = disorder.energy(neighbours, trial)
            rate = marcus(coupling, here - there, reorganisation, temperature)

            total = rate.sum()
            if total <= 0.0 or not np.isfinite(total):
                raise SystemExit(
                    f"trial {trial}, step {step}: every hop rate is zero or "
                    f"non-finite. Check the couplings in "
                    f"{cfg.transport.j_file} and "
                    f"transport.reorganisation_energy.")

            cumulative = np.cumsum(rate / total)
            draw = rng.random()
            cumulative[cumulative < draw] = 2.0    # exclude the bins below the
            chosen = int(np.argmin(cumulative))    # draw; the first remaining
                                                   # bin is the selected hop
            position = int(neighbours[chosen])
            elapsed += 1.0 / rate[chosen]
            sites[trial, step] = position
            times[trial, step] = elapsed

        if progress_every and (trial + 1) % progress_every == 0:
            rate_per_trial = (wallclock.time() - started) / (trial + 1)
            say(f"  trial {trial + 1}/{trials}  "
                f"({rate_per_trial:.2f} s/trial, "
                f"{rate_per_trial * (trials - trial - 1) / 60:.1f} min left)")

    return sites, times
