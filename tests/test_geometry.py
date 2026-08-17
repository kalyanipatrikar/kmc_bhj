"""Superposition, pair descriptors, and drift - pure algebra, so exact answers
are available."""

import numpy as np
import pytest

from src.utils import descriptors, drift, kabsch


def rotation(axis, degrees):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    angle = np.radians(degrees)
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def test_kabsch_recovers_a_known_rigid_motion():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(20, 3))
    R_true = rotation([0.3, 1.0, -0.5], 37.0)
    t_true = np.array([1.5, -2.0, 0.75])
    Q = P @ R_true.T + t_true

    R, t = kabsch(P, Q)
    assert R == pytest.approx(R_true, abs=1e-9)
    assert t == pytest.approx(t_true, abs=1e-9)
    assert (P @ R.T + t) == pytest.approx(Q, abs=1e-9)


def test_kabsch_never_returns_a_reflection():
    """Without the determinant correction a mirrored set superposes perfectly,
    which would report a molecule as identical to its mirror image."""
    rng = np.random.default_rng(1)
    P = rng.normal(size=(20, 3))
    mirrored = P * np.array([1.0, 1.0, -1.0])

    R, _ = kabsch(P, mirrored)
    assert np.linalg.det(R) == pytest.approx(1.0)


def stacked_pair(separation, slip_long, slip_short):
    """Two identical planar sheets, offset by a known amount.

    The sheet is wider in x than in y, so the best-fit axes come out ordered
    long, short, normal.
    """
    x, y = np.meshgrid(np.linspace(-4.0, 4.0, 9), np.linspace(-1.0, 1.0, 3))
    sheet = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1)
    offset = np.array([slip_long, slip_short, separation])
    return np.vstack([sheet, sheet + offset]), len(sheet)


def test_descriptors_recover_a_known_offset():
    positions, n_a = stacked_pair(3.4, 1.2, 0.5)
    heavy = np.arange(n_a)
    sep, slip_l, slip_s, com, contact, overlap = descriptors(
        positions, n_a, heavy)

    assert sep == pytest.approx(3.4)
    assert abs(slip_l) == pytest.approx(1.2)
    assert abs(slip_s) == pytest.approx(0.5)
    assert com == pytest.approx(np.linalg.norm([1.2, 0.5, 3.4]))
    assert contact == pytest.approx(np.linalg.norm([1.2, 0.5, 3.4]) - 0.0,
                                    abs=1.5)   # closest pair is roughly the
                                               # offset, up to the grid spacing
    assert 0.0 < overlap <= 1.0


def test_descriptors_overlap_falls_when_the_sheets_separate():
    close, n_a = stacked_pair(3.4, 0.0, 0.0)
    far, _ = stacked_pair(12.0, 0.0, 0.0)
    heavy = np.arange(n_a)

    assert descriptors(close, n_a, heavy)[5] > descriptors(far, n_a, heavy)[5]
    assert descriptors(far, n_a, heavy)[5] == pytest.approx(0.0)


def test_drift_is_zero_for_an_unmoved_pair():
    positions, n_a = stacked_pair(3.4, 1.0, 0.0)
    assert drift(positions, positions.copy(), n_a) == pytest.approx(0.0,
                                                                    abs=1e-9)


def test_drift_ignores_rigid_body_motion_of_the_whole_pair():
    """The thermostat is free to rotate and translate the pair; that must not
    register as the pair having come apart."""
    positions, n_a = stacked_pair(3.4, 1.0, 0.0)
    R = rotation([0.2, -1.0, 0.4], 55.0)
    moved = positions @ R.T + np.array([10.0, -3.0, 2.0])

    assert drift(moved, positions, n_a) == pytest.approx(0.0, abs=1e-9)


def test_drift_measures_movement_of_b_relative_to_a():
    positions, n_a = stacked_pair(3.4, 0.0, 0.0)
    shifted = positions.copy()
    shifted[n_a:] += np.array([0.7, 0.0, 0.0])

    assert drift(shifted, positions, n_a) == pytest.approx(0.7, abs=1e-6)
