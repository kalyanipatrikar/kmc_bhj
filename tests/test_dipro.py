"""The dimer projection formula, checked without any SCF.

A hand-built two-orbital problem makes every quantity in
J_eff = (J_ab - S_ab (e_a + e_b)/2) / (1 - S_ab^2) exact.
"""

import numpy as np
import pytest

from src.dft.dipro import coupling, embed, manifold_coupling


def two_level(j=0.01, s=0.05, ea=-0.20, eb=-0.18):
    F = np.array([[ea, j], [j, eb]])
    S = np.array([[1.0, s], [s, 1.0]])
    ca = np.array([1.0, 0.0])
    cb = np.array([0.0, 1.0])
    return ca, cb, F, S


def test_coupling_matches_the_closed_form():
    j, s, ea, eb = 0.01, 0.05, -0.20, -0.18
    ca, cb, F, S = two_level(j, s, ea, eb)

    j_eff, s_ab, e_a, e_b, j_raw = coupling(ca, cb, F, S)

    assert j_raw == pytest.approx(j)
    assert s_ab == pytest.approx(s)
    assert e_a == pytest.approx(ea)
    assert e_b == pytest.approx(eb)
    assert j_eff == pytest.approx((j - s * (ea + eb) / 2) / (1 - s ** 2))


def test_orthogonal_fragments_need_no_correction():
    ca, cb, F, S = two_level(s=0.0)
    j_eff, s_ab, _, _, j_raw = coupling(ca, cb, F, S)
    assert s_ab == pytest.approx(0.0)
    assert j_eff == pytest.approx(j_raw)


def test_the_correction_is_not_negligible_at_realistic_overlap():
    """If this ever came out tiny, the correction would not be worth having -
    it is here because at ~3 A contact it changes J by tens of percent."""
    ca, cb, F, S = two_level(j=0.01, s=0.05, ea=-0.20, eb=-0.18)
    j_eff, _, _, _, j_raw = coupling(ca, cb, F, S)
    assert abs(j_eff - j_raw) / abs(j_raw) > 0.1


def test_swapping_the_fragments_preserves_the_magnitude():
    ca, cb, F, S = two_level()
    forward = coupling(ca, cb, F, S)[0]
    backward = coupling(cb, ca, F, S)[0]
    assert abs(forward) == pytest.approx(abs(backward))


def test_flipping_an_orbital_phase_flips_only_the_sign():
    """The sign of J follows the phase convention of two independent SCFs and
    is therefore arbitrary; stage 3 writes it out anyway because every
    downstream use squares it."""
    ca, cb, F, S = two_level()
    reference = coupling(ca, cb, F, S)[0]
    flipped = coupling(ca, -cb, F, S)[0]
    assert flipped == pytest.approx(-reference)


def test_unphysical_overlap_is_rejected():
    ca, cb, F, S = two_level(s=1.0)
    with pytest.raises(ValueError):
        coupling(ca, cb, F, S)


def test_embedding_places_fragments_side_by_side():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0])
    ca, cb = embed(a, 5, 0), embed(b, 5, 3)

    assert ca.tolist() == [1.0, 2.0, 3.0, 0.0, 0.0]
    assert cb.tolist() == [0.0, 0.0, 0.0, 4.0, 5.0]
    # disjoint support is what makes the block-diagonal Fock construction valid
    assert float(ca @ cb) == pytest.approx(0.0)


# ---- Near-degenerate frontier orbitals ------------------------------------
def test_manifold_of_one_reproduces_the_two_state_formula_exactly():
    """The single-orbital formula is itself the closed form of the Lowdin
    transformation, so switching the manifold treatment on must leave every
    non-degenerate frame bit-for-bit unchanged."""
    for j, s, ea, eb in [(0.01, 0.05, -0.20, -0.18),
                         (0.02, 0.20, -0.25, -0.15),
                         (-0.005, 0.001, -0.30, -0.30)]:
        ca, cb, F, S = two_level(j, s, ea, eb)
        expected = coupling(ca, cb, F, S)[0]
        _, block, _, _ = manifold_coupling(ca[:, None], cb[:, None], F, S)
        assert float(block[0, 0]) == pytest.approx(expected, abs=1e-14)


def degenerate_pair(j11, j12, j21, j22, gap=0.0):
    """Two orbitals on each fragment, separated by `gap`, with a known
    coupling block between them and no inter-fragment overlap."""
    F = np.zeros((4, 4))
    F[0, 0], F[1, 1] = -0.20, -0.20 + gap
    F[2, 2], F[3, 3] = -0.20, -0.20 + gap
    block = np.array([[j11, j12], [j21, j22]])
    F[:2, 2:] = block
    F[2:, :2] = block.T
    return np.eye(4), F, block


def test_channels_add_in_quadrature():
    """Marcus rates go as J^2 and the rate out of a state is the sum over
    accessible final states, so the effective coupling of a 2x2 manifold is
    the root mean square over initial states of the sum over final ones."""
    S, F, block = degenerate_pair(0.01, 0.004, -0.003, 0.008)
    ca, cb = np.eye(4)[:, :2], np.eye(4)[:, 2:]

    j_eff, returned, _, _ = manifold_coupling(ca, cb, F, S)

    assert returned == pytest.approx(block)
    assert j_eff == pytest.approx(
        np.sqrt(np.mean(np.sum(block ** 2, axis=1))))


def test_a_degenerate_manifold_carries_more_than_its_strongest_channel():
    """The point of the whole treatment: a second open channel adds flux that
    a single-orbital coupling cannot see."""
    S, F, block = degenerate_pair(0.01, 0.01, 0.01, 0.01)
    ca, cb = np.eye(4)[:, :2], np.eye(4)[:, 2:]

    j_eff, _, _, _ = manifold_coupling(ca, cb, F, S)
    single = abs(block[0, 0])

    assert j_eff > single
    # four equal channels, two reachable from each initial state
    assert j_eff == pytest.approx(single * np.sqrt(2.0))


def test_manifold_site_energies_come_from_the_orthogonalised_diagonal():
    S, F, _ = degenerate_pair(0.01, 0.0, 0.0, 0.01, gap=0.05)
    ca, cb = np.eye(4)[:, :2], np.eye(4)[:, 2:]

    _, _, e_a, e_b = manifold_coupling(ca, cb, F, S)
    # identical fragments here, so the manifold-averaged energies must agree
    assert e_a == pytest.approx(e_b)
    assert e_a == pytest.approx(-0.20 + 0.05 / 2)


def test_linearly_dependent_manifolds_are_rejected():
    """Widening the window until two fragment orbitals become linearly
    dependent in the dimer basis has to fail loudly, not return a number."""
    ca = np.eye(4)[:, :2]
    cb = ca.copy()                      # the same orbitals on both fragments
    with pytest.raises(ValueError):
        manifold_coupling(ca, cb, np.eye(4), np.eye(4) * 0.0 + np.ones((4, 4)))


def test_manifold_selection_respects_the_window_and_the_cap():
    from src.dft.scf import manifold
    from src.utils import HARTREE_TO_EV

    class FakeMF:
        # HOMO at index 2; LUMO at 3 with LUMO+1 only 0.02 eV above it
        mo_energy = np.array([-0.40, -0.30, -0.25, -0.10,
                              -0.10 + 0.02 / HARTREE_TO_EV,
                              -0.10 + 0.30 / HARTREE_TO_EV])
        mo_occ = np.array([2.0, 2.0, 2.0, 0.0, 0.0, 0.0])

    mf = FakeMF()
    assert manifold(mf, 3, 0.10, 4).tolist() == [3, 4]
    assert manifold(mf, 3, 0.00, 4).tolist() == [3]     # window off
    assert manifold(mf, 3, 0.40, 4).tolist() == [3, 4, 5]
    assert manifold(mf, 3, 0.40, 2).tolist() == [3, 4]  # cap applies
    # an occupied orbital can never be pulled into a virtual manifold
    assert 2 not in manifold(mf, 3, 5.0, 4).tolist()


def test_every_declared_column_must_be_populated(tmp_path):
    """The manifold work added ten columns; a row that quietly omits one would
    be written as a blank field rather than rejected."""
    from src.dft.driver import COLUMNS, J_COLUMN, append_row, completed

    assert J_COLUMN in COLUMNS
    path = tmp_path / "transfer_integrals_motif_01.csv"

    with pytest.raises(SystemExit):
        append_row(path, {"config": "config_000"})

    append_row(path, {column: "0" for column in COLUMNS} | {"config": "c0"})
    assert completed(path) == {"c0"}


@pytest.mark.slow
def test_end_to_end_against_a_hydrogen_dimer():
    """A minimal-basis H2 dimer through the whole stage-3 path.

    Cheap, but it exercises the one assumption that unit tests cannot reach:
    that pyscf orders the dimer AOs as monomer A followed by monomer B.
    """
    pyscf = pytest.importorskip("pyscf")
    from src.config import Section
    from src.dft.scf import build_mol, make_mf, run_scf

    cfg = Section({"basis": "sto-3g", "xc": "b3lyp", "conv_tol": 1e-9,
                   "grid_level": 1, "max_cycle": 100})
    sym_a, pos_a = ["H", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    sym_b, pos_b = ["H", "H"], pos_a + np.array([0.0, 3.0, 0.0])

    mol_a = build_mol(sym_a, pos_a, cfg.basis)
    mol_b = build_mol(sym_b, pos_b, cfg.basis)
    mol_d = build_mol(sym_a + sym_b, np.vstack([pos_a, pos_b]), cfg.basis)
    assert mol_a.nao + mol_b.nao == mol_d.nao

    mf_a, mf_b = run_scf(mol_a, cfg), run_scf(mol_b, cfg)
    dm = np.zeros((mol_d.nao, mol_d.nao))
    dm[:mol_a.nao, :mol_a.nao] = mf_a.make_rdm1()
    dm[mol_a.nao:, mol_a.nao:] = mf_b.make_rdm1()

    F = make_mf(mol_d, cfg).get_fock(dm=dm)
    S = mol_d.intor("int1e_ovlp")

    homo = int(np.count_nonzero(mf_a.mo_occ > 0)) - 1
    ca = embed(mf_a.mo_coeff[:, homo], mol_d.nao, 0)
    cb = embed(mf_b.mo_coeff[:, homo], mol_d.nao, mol_a.nao)
    j_eff, s_ab, e_a, e_b, _ = coupling(ca, cb, F, S)

    # identical fragments 3 A apart: equal site energies, small positive
    # overlap, and a coupling of a sensible size rather than zero or a Hartree
    assert e_a == pytest.approx(e_b, abs=1e-6)
    assert 0.0 < abs(s_ab) < 0.1
    assert 1e-6 < abs(j_eff) < 0.1
