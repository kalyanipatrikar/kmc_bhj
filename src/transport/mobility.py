"""Mobility from the trajectories.

The trajectories are resampled onto a uniform time grid, the mean square
displacement is averaged over trials in each time bin, and the diffusion
coefficient is the slope of MSD against time in the diffusive regime. The
Einstein relation then gives the mobility:

    D = MSD / (2 d t),      mu = q D / (kB T)

Only bins from `mobility.fit_from` onwards enter the fit, because the walk is
not diffusive until it has taken enough hops to forget where it started.
"""

import numpy as np

from ..utils import KB, Q, say
from .lattice import Lattice

# ---------------------------------------------------------------------------
# NOTE, carried over from the original script and left in place deliberately:
# stage 4 stores the *running total* elapsed time, and the cumulative sum below
# is then applied to it a second time. If the intent was for stage 4 to store
# per-step dwell times, this cumsum is correct and stage 4 needs changing; if
# stage 4 is right, this cumsum should go. Either way one of the two is
# double-counting, and the mobility scales with that choice. Flagged for review
# rather than silently changed.
# ---------------------------------------------------------------------------


def mean_square_displacement(sites, times, lattice, intersite_a, n_bins,
                             time_bin):
    """MSD in cm^2 and mean time in s, per time bin."""
    cumulative = np.cumsum(times, axis=1)      # see the note above
    trials = sites.shape[0]
    start = lattice.centre
    index = np.arange(trials)

    msd = np.full(n_bins, np.nan)
    mean_time = np.full(n_bins, np.nan)
    for i in range(n_bins):
        first = (cumulative > i * time_bin).argmax(axis=1)
        reached = cumulative[index, first]
        arrived = sites[index, first]

        # trials that have not reached this bin, or have already run past it,
        # do not contribute
        inside = (reached > i * time_bin) & (reached <= (i + 1) * time_bin)
        if not inside.any():
            continue
        displacement = (lattice.separation(start, arrived[inside])
                        * intersite_a * 1.0e-8)      # A -> cm
        msd[i] = np.mean(displacement ** 2)
        mean_time[i] = np.mean(reached[inside])
    return mean_time, msd


def fit_mobility(mean_time, msd, fit_from, temperature, dimensionality):
    """Slope of MSD against time, then the Einstein relation."""
    usable = np.isfinite(mean_time) & np.isfinite(msd)
    usable[:fit_from] = False
    if usable.sum() < 2:
        raise SystemExit(
            f"only {usable.sum()} usable time bin(s) after mobility.fit_from="
            f"{fit_from}; lower it, or raise mobility.time_bin so the walk "
            f"reaches the later bins")

    slope, intercept = np.polyfit(mean_time[usable], msd[usable], 1)
    diffusion = slope / (2.0 * dimensionality)
    return {
        "slope_cm2_per_s": float(slope),
        "intercept_cm2": float(intercept),
        "diffusion_cm2_per_s": float(diffusion),
        "mobility_cm2_per_Vs": float(diffusion * Q / (KB * temperature)),
        "n_bins_fitted": int(usable.sum()),
    }


def analyse(cfg, sites, times):
    lattice = Lattice(cfg.transport.lattice)
    intersite = intersite_distance(cfg)
    mean_time, msd = mean_square_displacement(
        sites, times, lattice, intersite, int(cfg.mobility.n_bins),
        float(cfg.mobility.time_bin))
    result = fit_mobility(mean_time, msd, int(cfg.mobility.fit_from),
                          float(cfg.transport.temperature),
                          int(cfg.mobility.dimensionality))
    result["intersite_distance_A"] = intersite
    return mean_time, msd, result


def intersite_distance(cfg):
    """Spacing between hopping sites, in A.

    Taken from the config when set; otherwise the mean centre-of-mass distance
    of the stacked motifs found in stage 1, which is the physically meaningful
    spacing for a stack of molecules.
    """
    value = cfg.mobility.intersite_distance
    if value is not None:
        return float(value)

    import csv
    table = cfg.results_dir / "motifs.csv"
    if not table.exists():
        raise SystemExit(
            "mobility.intersite_distance is null and motifs.csv is not "
            "available to take it from - set it explicitly")
    with open(table, newline="", encoding="utf-8") as handle:
        distances = [float(row["com_distance_A"]) for row in csv.DictReader(handle)
                     if row.get("stacked", "").strip() in ("True", "true", "1")]
    if not distances:
        raise SystemExit("no stacked motif in motifs.csv to take an intersite "
                         "distance from - set mobility.intersite_distance")
    value = float(np.mean(distances))
    say(f"  intersite distance {value:.2f} A, from the stacked motifs in "
        f"motifs.csv")
    return value
