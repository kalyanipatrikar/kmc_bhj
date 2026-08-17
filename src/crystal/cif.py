"""A small CIF reader: cell parameters, symmetry operations, and the atom site
loop.

Written out rather than taken from a library so the only dependencies are
numpy and scipy - CIF loops are simple enough to walk directly. Symmetry
operations are parsed into a matrix and a vector rather than passed to eval(),
so a malformed or hostile CIF cannot execute anything.
"""

import re

import numpy as np

CELL_KEYS = ("_cell_length_a", "_cell_length_b", "_cell_length_c",
             "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma")


def num(token):
    """A CIF number, dropping the parenthesised standard uncertainty."""
    return float(re.sub(r"\(\d+\)", "", token))


# ---- Symmetry operations -------------------------------------------------
_TERM = re.compile(r"[+-]?[^+-]+")


def _parse_component(text):
    """One component of a symmetry operation, e.g. '-x+1/2', as (row, shift)."""
    row, shift = np.zeros(3), 0.0
    text = text.replace(" ", "").lower()
    if not text:
        raise ValueError("empty symmetry component")
    for match in _TERM.finditer(text):
        term = match.group()
        sign = -1.0 if term.startswith("-") else 1.0
        term = term.lstrip("+-")
        axis = None
        for candidate in "xyz":
            if candidate in term:
                axis, term = candidate, term.replace(candidate, "")
                break
        term = term.strip("*")
        if term:
            if "/" in term:
                numer, denom = term.split("/")
                coefficient = float(numer or 1.0) / float(denom)
            else:
                coefficient = float(term)
        else:
            coefficient = 1.0
        if axis is None:
            shift += sign * coefficient
        else:
            row["xyz".index(axis)] += sign * coefficient
    return row, shift


def parse_symop(op):
    """'-x, y+1/2, -z' -> (3x3 rotation, 3-vector translation)."""
    parts = op.split(",")
    if len(parts) != 3:
        raise ValueError(f"symmetry operation {op!r} does not have three "
                         f"components")
    rows, shifts = zip(*(_parse_component(p) for p in parts))
    return np.array(rows), np.array(shifts)


def apply_op(op, fractional):
    """Apply one symmetry operation to a fractional coordinate."""
    R, t = parse_symop(op) if isinstance(op, str) else op
    return R @ np.asarray(fractional, dtype=float) + t


# ---- Cell ----------------------------------------------------------------
def cart_matrix(cell):
    """Fractional to Cartesian, for a general triclinic cell."""
    a, b, c = (cell["_cell_length_a"], cell["_cell_length_b"],
               cell["_cell_length_c"])
    al, be, ga = (np.radians(cell["_cell_angle_alpha"]),
                  np.radians(cell["_cell_angle_beta"]),
                  np.radians(cell["_cell_angle_gamma"]))
    v = np.sqrt(1 - np.cos(al) ** 2 - np.cos(be) ** 2 - np.cos(ga) ** 2
                + 2 * np.cos(al) * np.cos(be) * np.cos(ga))
    return np.array([
        [a, b * np.cos(ga), c * np.cos(be)],
        [0.0, b * np.sin(ga),
         c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)],
        [0.0, 0.0, c * v / np.sin(ga)],
    ])


# ---- The file ------------------------------------------------------------
def parse_cif(path):
    """Return (cell, symops, atoms).

    `atoms` is a list of dicts with 'label', 'sym', 'f' (fractional position)
    and 'grp' (disorder group, or '.' when the CIF has no such column).
    """
    with open(path, encoding="utf-8", errors="replace") as handle:
        lines = handle.read().splitlines()

    cell, symops, atoms, i = {}, [], [], 0
    while i < len(lines):
        line = lines[i].strip()
        for key in CELL_KEYS:
            if line.startswith(key + " ") or line.startswith(key + "\t"):
                cell[key] = num(line.split()[1])

        if line == "loop_":
            j, names = i + 1, []
            while j < len(lines) and lines[j].strip().startswith("_"):
                names.append(lines[j].strip().split()[0])
                j += 1
            body = []
            while j < len(lines):
                entry = lines[j].strip()
                if not entry or entry.startswith(("_", "#")) or entry == "loop_":
                    break
                body.append(entry)
                j += 1

            if "_space_group_symop_operation_xyz" in names or \
                    "_symmetry_equiv_pos_as_xyz" in names:
                symops += [_symop_token(b) for b in body]

            if "_atom_site_label" in names and "_atom_site_fract_x" in names:
                atoms += _atom_rows(names, body)

            i = j
            continue
        i += 1

    missing = [k for k in CELL_KEYS if k not in cell]
    if missing:
        raise SystemExit(f"{path}: missing cell parameter(s) "
                         f"{', '.join(missing)}")
    if not symops:
        symops = ["x, y, z"]        # P1, or a CIF that omits the identity loop
    if not atoms:
        raise SystemExit(f"{path}: no atom site loop found")
    return cell, symops, atoms


def _symop_token(row):
    """A symop loop row is either "'-x, y, z'" or "1 '-x, y, z'"."""
    row = row.strip()
    quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", row)
    if quoted:
        return (quoted[0][0] or quoted[0][1]).strip()
    parts = row.split(maxsplit=1)
    return (parts[1] if len(parts) == 2 and parts[0].isdigit() else row).strip()


def _atom_rows(names, body):
    """Rows of the atom site loop, tolerating an absent disorder group column
    and skipping rows that do not have one token per column."""
    index = {name: position for position, name in enumerate(names)}
    if "_atom_site_type_symbol" not in index:
        raise SystemExit("the atom site loop has no _atom_site_type_symbol "
                         "column, so element identities are unknown")
    group_col = index.get("_atom_site_disorder_group")

    rows = []
    for entry in body:
        tokens = entry.split()
        if len(tokens) < len(names):
            continue
        rows.append({
            "label": tokens[index["_atom_site_label"]],
            "sym": _element(tokens[index["_atom_site_type_symbol"]]),
            "f": np.array([num(tokens[index["_atom_site_fract_x"]]),
                           num(tokens[index["_atom_site_fract_y"]]),
                           num(tokens[index["_atom_site_fract_z"]])]),
            "grp": tokens[group_col] if group_col is not None else ".",
        })
    return rows


def _element(token):
    """'C', 'Cl', 'O1-' -> the element symbol."""
    match = re.match(r"([A-Za-z]{1,2})", token)
    if not match:
        raise SystemExit(f"cannot read an element symbol from {token!r}")
    symbol = match.group(1)
    return symbol[0].upper() + symbol[1:].lower()
