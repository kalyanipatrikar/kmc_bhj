"""The site lattice: index arithmetic, the neighbour shell, and displacement."""

from itertools import product

import numpy as np
import pytest

from src.transport.disorder import SiteDisorder
from src.transport.lattice import Lattice


@pytest.fixture
def lattice():
    return Lattice((10, 8, 6))


def test_there_are_twenty_six_distinct_neighbours(lattice):
    site = lattice.centre
    neighbours = lattice.neighbours(site)

    assert neighbours.size == 26
    assert len(set(neighbours.tolist())) == 26
    assert site not in neighbours.tolist()


def test_neighbour_offsets_are_the_full_cube_shell(lattice):
    """The neighbour list is built from flat index arithmetic, so this is the
    check that the arithmetic actually corresponds to the cube around a site."""
    site = lattice.centre
    x0, y0, z0 = (int(c) for c in lattice.coordinates(site))
    x, y, z = lattice.coordinates(lattice.neighbours(site))

    found = {(int(a - x0), int(b - y0), int(c - z0))
             for a, b, c in zip(x, y, z)}
    expected = set(product((-1, 0, 1), repeat=3)) - {(0, 0, 0)}
    assert found == expected


def test_coordinates_round_trip(lattice):
    for index in (0, 1, lattice.L, lattice.plane, lattice.centre,
                  lattice.N - 1):
        x, y, z = (int(c) for c in lattice.coordinates(index))
        assert z * lattice.plane + y * lattice.L + x == index


def test_separation_counts_lattice_steps(lattice):
    site = lattice.centre
    assert lattice.separation(site, site) == pytest.approx(0.0)
    assert lattice.separation(site, site + 1) == pytest.approx(1.0)
    assert lattice.separation(site, site + lattice.L) == pytest.approx(1.0)
    assert lattice.separation(site, site + lattice.plane) == pytest.approx(1.0)
    assert lattice.separation(site, site + lattice.plane + lattice.L + 1) == \
        pytest.approx(np.sqrt(3.0))


def test_neighbours_wrap_rather_than_run_off_the_end(lattice):
    """Sites at the very edge of the box must still return 26 valid indices."""
    for site in (0, lattice.N - 1):
        neighbours = lattice.neighbours(site)
        assert neighbours.min() >= 0
        assert neighbours.max() < lattice.N


def test_too_small_a_box_is_rejected():
    with pytest.raises(SystemExit):
        Lattice((2, 10, 10))


# ---- Site disorder --------------------------------------------------------
def test_a_site_keeps_its_properties_within_a_trial():
    """Static disorder: revisiting a site must find it unchanged, otherwise
    there is nothing for a carrier to be trapped by."""
    disorder = SiteDisorder(seed=0, sigma=0.1, shift_sigma=0.1)
    sites = np.array([5, 12345, 999999])

    first = disorder.energy(sites, trial=3)
    second = disorder.energy(sites, trial=3)
    assert first == pytest.approx(second)
    assert disorder.shift(sites, 3) == pytest.approx(disorder.shift(sites, 3))
    assert disorder.costheta(sites, 3) == \
        pytest.approx(disorder.costheta(sites, 3))


def test_each_trial_is_a_new_disorder_realisation():
    disorder = SiteDisorder(seed=0, sigma=0.1, shift_sigma=0.1)
    sites = np.arange(1000)
    assert not np.allclose(disorder.energy(sites, 0), disorder.energy(sites, 1))


def test_site_energies_have_the_requested_distribution():
    sigma = 0.13
    disorder = SiteDisorder(seed=7, sigma=sigma, shift_sigma=0.1)
    energies = disorder.energy(np.arange(200_000), trial=0)

    assert energies.mean() == pytest.approx(0.0, abs=0.005)
    assert energies.std() == pytest.approx(sigma, rel=0.02)


def test_shift_is_positive_and_costheta_is_a_cosine():
    disorder = SiteDisorder(seed=1, sigma=0.1, shift_sigma=0.1)
    sites = np.arange(50_000)

    shift = disorder.shift(sites, 0)
    assert (shift >= 0).all()
    # |N(0, s)| has mean s * sqrt(2/pi)
    assert shift.mean() == pytest.approx(0.1 * np.sqrt(2 / np.pi), rel=0.03)

    costheta = disorder.costheta(sites, 0)
    assert costheta.min() >= 0.0 and costheta.max() <= 1.0
    assert costheta.mean() == pytest.approx(0.5, abs=0.01)
