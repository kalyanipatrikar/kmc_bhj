"""Stage 5, against a random walk whose diffusion coefficient is known.

An unbiased walk on the 26-neighbour lattice with a constant dwell time has

    <r^2> = n * <step^2> * d^2,    t = n * tau

so D = <step^2> d^2 / (2 * 3 * tau) exactly, and the mobility follows from the
Einstein relation. If the binning, the unit conversion or the fit is wrong,
this test says so.

NOTE: the `times` array built here holds per-step dwell times, which is what
`mean_square_displacement` assumes when it takes a cumulative sum. Stage 4
currently stores running totals instead - the discrepancy flagged in
`src/transport/mobility.py`. This test pins down the module's own contract; it
is deliberately not a test of the two stages agreeing.
"""

import numpy as np
import pytest

from src.config import Section
from src.transport.lattice import Lattice
from src.transport.mobility import (fit_mobility, mean_square_displacement)
from src.utils import KB, Q

TAU = 1.0e-12          # s per hop
INTERSITE = 3.27       # A
TRIALS, STEPS = 300, 400


def synthetic_walk(lattice, rng):
    """Unbiased hops among the 26 neighbours, constant dwell time."""
    sites = np.empty((TRIALS, STEPS), dtype=np.int64)
    for trial in range(TRIALS):
        position = lattice.centre
        for step in range(STEPS):
            neighbours = lattice.neighbours(position)
            position = int(neighbours[rng.integers(0, 26)])
            sites[trial, step] = position
    return sites, np.full((TRIALS, STEPS), TAU)


def expected_diffusion(lattice):
    """<step^2> over the 26 offsets: 6 face, 12 edge, 8 corner neighbours."""
    x, y, z = lattice.coordinates(lattice.centre + lattice.offsets)
    x0, y0, z0 = lattice.coordinates(lattice.centre)
    step_squared = ((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2).mean()
    assert step_squared == pytest.approx(54.0 / 26.0)

    d_cm = INTERSITE * 1.0e-8
    return step_squared * d_cm ** 2 / (6.0 * TAU)


def test_msd_of_a_free_walk_recovers_the_analytic_diffusion_constant():
    lattice = Lattice((1000, 1000, 1000))
    rng = np.random.default_rng(0)
    sites, times = synthetic_walk(lattice, rng)

    mean_time, msd = mean_square_displacement(
        sites, times, lattice, INTERSITE, n_bins=40, time_bin=1.0e-11)
    result = fit_mobility(mean_time, msd, fit_from=5, temperature=300.0,
                          dimensionality=3)

    assert result["diffusion_cm2_per_s"] == \
        pytest.approx(expected_diffusion(lattice), rel=0.10)


def test_mobility_follows_the_einstein_relation():
    lattice = Lattice((1000, 1000, 1000))
    rng = np.random.default_rng(1)
    sites, times = synthetic_walk(lattice, rng)

    mean_time, msd = mean_square_displacement(
        sites, times, lattice, INTERSITE, n_bins=40, time_bin=1.0e-11)
    result = fit_mobility(mean_time, msd, fit_from=5, temperature=300.0,
                          dimensionality=3)

    assert result["mobility_cm2_per_Vs"] == pytest.approx(
        result["diffusion_cm2_per_s"] * Q / (KB * 300.0), rel=1e-9)


def test_msd_grows_linearly_with_time():
    """The signature of diffusion, and the thing the fit assumes."""
    lattice = Lattice((1000, 1000, 1000))
    rng = np.random.default_rng(2)
    sites, times = synthetic_walk(lattice, rng)

    mean_time, msd = mean_square_displacement(
        sites, times, lattice, INTERSITE, n_bins=40, time_bin=1.0e-11)
    usable = np.isfinite(msd) & np.isfinite(mean_time)

    ratio = msd[usable][5:] / mean_time[usable][5:]
    assert ratio.std() / ratio.mean() < 0.1


def test_too_few_usable_bins_is_a_clear_error():
    mean_time = np.full(20, np.nan)
    msd = np.full(20, np.nan)
    mean_time[:3], msd[:3] = [1e-11, 2e-11, 3e-11], [1e-14, 2e-14, 3e-14]

    with pytest.raises(SystemExit):
        fit_mobility(mean_time, msd, fit_from=10, temperature=300.0,
                     dimensionality=3)


def test_config_section_attribute_access():
    section = Section({"a": 1, "b": {"c": 2}})
    assert section.a == 1
    assert section.b.c == 2
    with pytest.raises(AttributeError):
        _ = section.missing
