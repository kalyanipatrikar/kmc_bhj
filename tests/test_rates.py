"""The Marcus rate and the coupling pool it draws from."""

import numpy as np
import pytest

from src.transport.rates import load_couplings, marcus, sample_couplings

LAMBDA = 0.10        # eV
T = 300.0
J = 10.0             # meV - the unit couplings are carried in throughout


def test_rate_is_maximal_at_activationless_transfer():
    """The exponent is -(lambda - dE)^2 / 4 lambda kT, so the rate peaks when
    the driving force matches the reorganisation energy."""
    peak = marcus(J, LAMBDA, LAMBDA, T)
    assert peak > marcus(J, 0.0, LAMBDA, T)
    assert peak > marcus(J, 2 * LAMBDA, LAMBDA, T)
    # and it is symmetric about that point
    assert marcus(J, LAMBDA - 0.03, LAMBDA, T) == \
        pytest.approx(marcus(J, LAMBDA + 0.03, LAMBDA, T))


def test_rate_goes_as_the_square_of_the_coupling():
    single = marcus(J, 0.0, LAMBDA, T)
    double = marcus(2 * J, 0.0, LAMBDA, T)
    assert double / single == pytest.approx(4.0)


def test_uphill_hops_are_slower_than_downhill_ones():
    downhill = marcus(J, 0.05, LAMBDA, T)     # dE > 0: destination lower
    uphill = marcus(J, -0.05, LAMBDA, T)
    assert downhill > uphill


def test_rate_is_a_plausible_hopping_frequency():
    """10 meV coupling, 100 meV reorganisation energy, no energy offset: an
    organic semiconductor hops on the order of 10^12 - 10^13 per second.

    This is the guard on the mixed-unit signature - the coupling arrives in
    meV and the energies in eV, and confusing the two would move this by six
    orders of magnitude without changing anything else that is observable.
    """
    rate = marcus(J, 0.0, LAMBDA, T)
    assert 1e11 < rate < 1e14


def test_arrays_are_handled_elementwise():
    j = np.array([10.0, 20.0])
    de = np.array([0.0, 0.0])
    rates = marcus(j, de, LAMBDA, T)
    assert rates.shape == (2,)
    assert rates[1] / rates[0] == pytest.approx(4.0)


# ---- The pool -------------------------------------------------------------
def test_sampling_reaches_every_entry_including_the_ends():
    """The original selection drew from randint(1, len(pool)), which could
    never return the first or the last configuration - with a ten-entry
    ensemble that silently discards a fifth of the DFT."""
    pool = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rng = np.random.default_rng(0)
    drawn = set(sample_couplings(rng, pool, 5000).tolist())
    assert drawn == set(pool.tolist())


def test_sampling_returns_the_requested_count():
    pool = np.array([1.0, 2.0, 3.0])
    rng = np.random.default_rng(0)
    assert sample_couplings(rng, pool, 26).shape == (26,)


def test_couplings_are_read_as_written(tmp_path):
    """J.txt is meV and stays meV - no conversion on the way in. The sign is
    dropped because it follows an arbitrary orbital phase convention."""
    path = tmp_path / "J.txt"
    path.write_text("# a comment\n10.0\n-20.0\n", encoding="utf-8")

    assert load_couplings(path) == pytest.approx([10.0, 20.0])


def test_a_single_value_file_still_loads(tmp_path):
    path = tmp_path / "J.txt"
    path.write_text("# header\n7.5\n", encoding="utf-8")
    assert load_couplings(path).shape == (1,)


def test_an_empty_file_is_an_error(tmp_path):
    path = tmp_path / "J.txt"
    path.write_text("# nothing here\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_couplings(path)
