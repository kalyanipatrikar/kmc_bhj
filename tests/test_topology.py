"""Ring perception and automatic core detection.

This is the module that replaced a hand-written "carbons numbered 24 and above
are the chains" rule, so it is the one that most needs to be pinned down: if it
picks the wrong core the whole pipeline still runs and produces plausible
numbers for the wrong molecule.
"""

import numpy as np
import pytest

from src.crystal.fragments import core_fragment
from src.crystal.topology import (bond_graph, connected_components, detect_core,
                                  n_rings, ring_atoms, sp3_carbons)

BOND = 1.4


def hexagon(centre=(0.0, 0.0)):
    angles = np.radians(np.arange(6) * 60.0)
    return np.stack([centre[0] + BOND * np.cos(angles),
                     centre[1] + BOND * np.sin(angles),
                     np.zeros(6)], axis=1)


def benzene():
    """Six carbons and six hydrogens."""
    carbons = hexagon()
    hydrogens = carbons * (1.0 + 1.09 / BOND)
    return ["C"] * 6 + ["H"] * 6, np.vstack([carbons, hydrogens])


def naphthalene():
    """Two hexagons fused across one edge: rotating the ring by 180 degrees
    about the midpoint of the shared edge generates the second ring."""
    ring = hexagon()
    midpoint = (ring[0] + ring[5]) / 2.0
    fused = 2.0 * midpoint - ring          # v -> 2m - v
    new = fused[[1, 2, 3, 4]]              # v0 and v5 are the shared edge
    return ["C"] * 10, np.vstack([ring, new])


def test_benzene_is_one_ring():
    symbols, positions = benzene()
    adjacency = bond_graph(positions, symbols)
    carbons = list(range(6))

    assert all(len(adjacency[i] & set(carbons)) == 2 for i in carbons)
    assert n_rings(adjacency, carbons) == 1
    assert sorted(ring_atoms(adjacency, carbons)) == carbons


def test_naphthalene_is_two_fused_rings():
    symbols, positions = naphthalene()
    adjacency = bond_graph(positions, symbols)

    assert len(connected_components(adjacency)) == 1
    assert n_rings(adjacency) == 2
    assert len(ring_atoms(adjacency)) == 10


def test_two_separate_rings_are_two_components():
    symbols_a, positions_a = benzene()
    symbols_b, positions_b = benzene()
    positions_b = positions_b + np.array([20.0, 0.0, 0.0])
    symbols = symbols_a + symbols_b
    positions = np.vstack([positions_a, positions_b])

    adjacency = bond_graph(positions, symbols)
    assert len(connected_components(adjacency)) == 2
    assert n_rings(adjacency) == 2


def with_alkyl_chain():
    """Naphthalene with a -CH3 hung off ring atom 2."""
    symbols, positions = naphthalene()
    anchor = positions[2]
    direction = anchor / np.linalg.norm(anchor)
    chain_c = anchor + 1.52 * direction

    # a proper tetrahedral methyl, pointing away from the ring - hydrogens
    # placed carelessly end up within bonding distance of the anchor carbon,
    # which would make it look four-coordinate and therefore sp3
    e1 = np.cross(direction, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(direction, e1)
    tilt = np.radians(180.0 - 109.5)
    hydrogens = []
    for phi in np.radians([0.0, 120.0, 240.0]):
        unit = (np.cos(tilt) * direction
                + np.sin(tilt) * (np.cos(phi) * e1 + np.sin(phi) * e2))
        hydrogens.append(chain_c + 1.09 * unit)

    symbols = symbols + ["C"] + ["H"] * 3
    positions = np.vstack([positions, chain_c[None, :], np.array(hydrogens)])
    return symbols, positions


def test_sp3_carbon_is_detected_and_excluded_from_the_core():
    symbols, positions = with_alkyl_chain()
    adjacency = bond_graph(positions, symbols)

    saturated = sp3_carbons(adjacency, symbols)
    assert saturated.tolist() == [10]       # the chain carbon, nothing else

    core, _ = detect_core(positions, symbols)
    assert core.tolist() == list(range(10))  # the ten ring carbons


def test_capping_replaces_the_chain_with_a_hydrogen():
    symbols, positions = with_alkyl_chain()
    labels = [f"{s}{i}" for i, s in enumerate(symbols)]
    selection = {"core_labels": None, "keep_labels": [], "drop_labels": []}

    out_symbols, out_positions, out_labels = core_fragment(
        symbols, positions, labels, selection, {"C": 1.09, "N": 1.01})

    # ten ring carbons, no ring hydrogens in this construction, one cap
    assert out_symbols.count("C") == 10
    assert out_symbols.count("H") == 1
    assert out_labels[-1].startswith("Hcap")

    anchor = out_positions[2]
    cap = out_positions[-1]
    assert np.linalg.norm(cap - anchor) == pytest.approx(1.09)
    # the cap must lie along the bond that was broken, not just anywhere
    broken = positions[10] - positions[2]
    along = (cap - anchor) @ broken / (np.linalg.norm(broken) * 1.09)
    assert along == pytest.approx(1.0)


def test_explicit_overrides_win_over_the_heuristic():
    symbols, positions = with_alkyl_chain()
    core, _ = detect_core(positions, symbols, keep=[10], drop=[0])
    assert 10 in core and 0 not in core
