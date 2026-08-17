"""The CIF reader, against a small structure whose answers are known by hand."""

import numpy as np
import pytest

from src.crystal.cif import apply_op, cart_matrix, num, parse_cif, parse_symop

TRICLINIC = """
data_test
_cell_length_a     10.0000(5)
_cell_length_b      8.0000(4)
_cell_length_c      6.0000(3)
_cell_angle_alpha  90.000
_cell_angle_beta   100.000(4)
_cell_angle_gamma  90.000

loop_
_space_group_symop_operation_xyz
'x, y, z'
'-x, y+1/2, -z'

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_disorder_group
C1 C 0.1000(2) 0.2000(3) 0.3000(4) .
C2 C 0.4000 0.5000 0.6000 2
O1 O 0.7000 0.8000 0.9000 .
"""

NO_DISORDER_COLUMN = """
data_test
_cell_length_a 5.0
_cell_length_b 5.0
_cell_length_c 5.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0

loop_
_symmetry_equiv_pos_as_xyz
1 'x, y, z'

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Cl1 Cl 0.0 0.0 0.0
"""


def write(tmp_path, text, name="test.cif"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_num_strips_standard_uncertainty():
    assert num("1.234(5)") == pytest.approx(1.234)
    assert num("-0.5") == pytest.approx(-0.5)
    assert num("10") == pytest.approx(10.0)


def test_parse_cif_reads_cell_symops_and_atoms(tmp_path):
    cell, symops, atoms = parse_cif(write(tmp_path, TRICLINIC))

    assert cell["_cell_length_a"] == pytest.approx(10.0)
    assert cell["_cell_angle_beta"] == pytest.approx(100.0)
    assert symops == ["x, y, z", "-x, y+1/2, -z"]

    assert [a["label"] for a in atoms] == ["C1", "C2", "O1"]
    assert [a["sym"] for a in atoms] == ["C", "C", "O"]
    assert [a["grp"] for a in atoms] == [".", "2", "."]
    assert atoms[0]["f"] == pytest.approx([0.1, 0.2, 0.3])


def test_parse_cif_without_a_disorder_column(tmp_path):
    """The column is optional; indexing it unconditionally would crash here."""
    _, symops, atoms = parse_cif(write(tmp_path, NO_DISORDER_COLUMN))
    assert symops == ["x, y, z"]
    assert atoms[0]["sym"] == "Cl"      # two-letter element, not 'C'
    assert atoms[0]["grp"] == "."


def test_cart_matrix_reproduces_the_cell():
    cell = {"_cell_length_a": 10.0, "_cell_length_b": 8.0, "_cell_length_c": 6.0,
            "_cell_angle_alpha": 90.0, "_cell_angle_beta": 100.0,
            "_cell_angle_gamma": 90.0}
    M = cart_matrix(cell)
    a, b, c = M[:, 0], M[:, 1], M[:, 2]

    assert np.linalg.norm(a) == pytest.approx(10.0)
    assert np.linalg.norm(b) == pytest.approx(8.0)
    assert np.linalg.norm(c) == pytest.approx(6.0)

    def angle(u, v):
        return np.degrees(np.arccos(u @ v / (np.linalg.norm(u)
                                             * np.linalg.norm(v))))

    assert angle(b, c) == pytest.approx(90.0)
    assert angle(a, c) == pytest.approx(100.0)
    assert angle(a, b) == pytest.approx(90.0)

    # the determinant is the cell volume, V = abc sqrt(...) - here abc sin(beta)
    assert np.linalg.det(M) == pytest.approx(
        10.0 * 8.0 * 6.0 * np.sin(np.radians(100.0)))


def test_symop_parsing():
    R, t = parse_symop("-x, y+1/2, -z")
    assert R == pytest.approx(np.diag([-1.0, 1.0, -1.0]))
    assert t == pytest.approx([0.0, 0.5, 0.0])

    assert apply_op("-x, y+1/2, -z", [0.1, 0.2, 0.3]) == \
        pytest.approx([-0.1, 0.7, -0.3])
    # a general position mixing axes, as monoclinic and trigonal CIFs contain
    assert apply_op("x-y, x, z+2/3", [0.5, 0.2, 0.1]) == \
        pytest.approx([0.3, 0.5, 0.1 + 2 / 3])


def test_symop_rejects_malformed_operations():
    with pytest.raises(ValueError):
        parse_symop("x, y")
