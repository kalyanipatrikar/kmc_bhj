"""Per-frame driver for stage 3, frame selection, and the J files.

LUMO-LUMO is the coupling that matters for an electron-transporting material;
HOMO-HOMO comes out of the same SCFs at no extra cost and is recorded
alongside it.
"""

import csv
import math
import time

import numpy as np

from ..utils import HARTREE_TO_EV, HARTREE_TO_MEV, read_xyz, say
from .dipro import coupling, embed, manifold_coupling
from .scf import build_mol, frontier, make_mf, manifold, run_scf

COLUMNS = ["config", "motif",
           "E_A_eV", "E_B_eV",
           "J_lumo_meV", "S_lumo", "e_a_lumo_eV", "e_b_lumo_eV",
           "dE_lumo_eV", "J_raw_lumo_meV",
           "J_lumo_manifold_meV", "J_lumo_manifold_max_meV",
           "dE_lumo_manifold_eV", "n_lumo_A", "n_lumo_B",
           "J_homo_meV", "S_homo", "e_a_homo_eV", "e_b_homo_eV",
           "dE_homo_eV", "J_raw_homo_meV",
           "J_homo_manifold_meV", "J_homo_manifold_max_meV",
           "dE_homo_manifold_eV", "n_homo_A", "n_homo_B",
           "homo_A_eV", "lumo_A_eV", "homo_B_eV", "lumo_B_eV",
           "lumo_gap_A_eV", "lumo_gap_B_eV", "homo_gap_A_eV", "homo_gap_B_eV",
           "seconds"]

# which column stage 4's J files are built from
J_COLUMN = "J_lumo_manifold_meV"


# ---- Frame selection ------------------------------------------------------
def select_frames(cfg, motifs):
    """Which configurations go to DFT.

    With `dft.stratify` the sample is split evenly across motifs, so each one
    is represented whatever the draw happens to do. A uniform draw over the
    pooled configurations can easily return 8 frames of one motif and 2 of the
    other, which then supports no comparison between them at the same cost.
    """
    n_sample, rng = int(cfg.dft.n_sample), np.random.default_rng(cfg.dft.seed)
    available = {}
    for motif in motifs:
        directory = cfg.config_dir(motif)
        if not directory.is_dir():
            raise SystemExit(f"{directory} not found - run stage 2 first")
        tags = sorted(p.name[:-len("_dimer.xyz")]
                      for p in directory.glob("*_dimer.xyz"))
        if not tags:
            raise SystemExit(f"no configurations in {directory}")
        available[motif] = tags

    selection = {}
    if cfg.dft.stratify:
        base, extra = divmod(n_sample, len(motifs))
        for i, motif in enumerate(motifs):
            want = base + (1 if i < extra else 0)
            tags = available[motif]
            selection[motif] = sorted(rng.choice(
                tags, size=min(want, len(tags)), replace=False).tolist())
    else:
        pool = [(motif, tag) for motif in motifs for tag in available[motif]]
        picked = rng.choice(len(pool), size=min(n_sample, len(pool)),
                            replace=False)
        selection = {motif: [] for motif in motifs}
        for index in sorted(picked):
            motif, tag = pool[index]
            selection[motif].append(tag)

    for motif in motifs:
        say(f"{motif}: {len(selection[motif])} of {len(available[motif])} "
            f"frames")
        if selection[motif]:
            say("  " + ", ".join(selection[motif]))
    return selection


# ---- One frame ------------------------------------------------------------
def process(cfg, motif, tag):
    """Two monomer SCFs, the dimer Fock matrix, then the projection.

    The direct-sum structure of the dimer AO basis is checked rather than
    trusted, because a silent violation would produce plausible-looking
    couplings that are wrong.
    """
    started = time.time()
    directory = cfg.config_dir(motif)
    sym_d, pos_d, _ = read_xyz(directory / f"{tag}_dimer.xyz")
    sym_a, pos_a, _ = read_xyz(directory / f"{tag}_A.xyz")
    sym_b, pos_b, _ = read_xyz(directory / f"{tag}_B.xyz")

    if sym_d != sym_a + sym_b:
        raise SystemExit(f"{motif}/{tag}: dimer atom order is not A then B")
    if not np.allclose(pos_d, np.vstack([pos_a, pos_b]), atol=1e-6):
        raise SystemExit(f"{motif}/{tag}: monomer coordinates differ from the "
                         f"dimer")

    basis = cfg.dft.basis
    mol_d = build_mol(sym_d, pos_d, basis)
    mol_a = build_mol(sym_a, pos_a, basis)
    mol_b = build_mol(sym_b, pos_b, basis)
    na, nb = mol_a.nao, mol_b.nao
    if na + nb != mol_d.nao:
        raise SystemExit(f"{motif}/{tag}: dimer basis is not the direct sum of "
                         f"the monomer bases ({na} + {nb} != {mol_d.nao})")

    mf_a, mf_b = run_scf(mol_a, cfg.dft), run_scf(mol_b, cfg.dft)

    # the superposed monomer density is both the starting guess for the dimer
    # SCF and, in superposition mode, the density the Fock matrix is built
    # from, so it is assembled either way
    dm_super = np.zeros((mol_d.nao, mol_d.nao))
    dm_super[:na, :na] = mf_a.make_rdm1()
    dm_super[na:, na:] = mf_b.make_rdm1()

    if cfg.dft.mode == "dimer_scf":
        F = run_scf(mol_d, cfg.dft, dm0=dm_super).get_fock()
    else:
        F = make_mf(mol_d, cfg.dft).get_fock(dm=dm_super)
    S = mol_d.intor("int1e_ovlp")

    homo_a, lumo_a, eh_a, el_a, hgap_a, lgap_a = frontier(mf_a)
    homo_b, lumo_b, eh_b, el_b, hgap_b, lgap_b = frontier(mf_b)

    window = float(cfg.dft.degeneracy_window)
    max_manifold = int(cfg.dft.max_manifold)

    out = {}
    for name, ia, ib in (("lumo", lumo_a, lumo_b), ("homo", homo_a, homo_b)):
        ca = embed(mf_a.mo_coeff[:, ia], mol_d.nao, 0)
        cb = embed(mf_b.mo_coeff[:, ib], mol_d.nao, na)
        j_eff, s_ab, e_a, e_b, j_raw = coupling(ca, cb, F, S)

        # the same projection over every orbital degenerate with the frontier
        # one. Costs no SCF - these orbitals are already converged - and
        # reduces exactly to the single-orbital result when the manifolds are
        # one orbital each, which is the usual case.
        set_a = manifold(mf_a, ia, window, max_manifold)
        set_b = manifold(mf_b, ib, window, max_manifold)
        block_a = np.zeros((mol_d.nao, len(set_a)))
        block_b = np.zeros((mol_d.nao, len(set_b)))
        block_a[:na] = mf_a.mo_coeff[:, set_a]
        block_b[na:] = mf_b.mo_coeff[:, set_b]
        j_manifold, block, em_a, em_b = manifold_coupling(
            block_a, block_b, F, S)
        # keep the sign when there is one orbital on each side, so the column
        # stays continuous with J_<name>_meV; a manifold result has no sign
        signed = float(block[0, 0]) if block.shape == (1, 1) else j_manifold

        out[name] = {"J": j_eff * HARTREE_TO_MEV, "S": s_ab,
                     "e_a": e_a * HARTREE_TO_EV, "e_b": e_b * HARTREE_TO_EV,
                     "dE": (e_a - e_b) * HARTREE_TO_EV,
                     "J_raw": j_raw * HARTREE_TO_MEV,
                     "J_manifold": signed * HARTREE_TO_MEV,
                     "J_manifold_max": np.abs(block).max() * HARTREE_TO_MEV,
                     "dE_manifold": (em_a - em_b) * HARTREE_TO_EV,
                     "n_a": len(set_a), "n_b": len(set_b)}

    elapsed = time.time() - started
    row = {"config": tag, "motif": motif,
           "E_A_eV": f"{mf_a.e_tot * HARTREE_TO_EV:.6f}",
           "E_B_eV": f"{mf_b.e_tot * HARTREE_TO_EV:.6f}"}
    for name in ("lumo", "homo"):
        o = out[name]
        row.update({
            f"J_{name}_meV": f"{o['J']:.4f}", f"S_{name}": f"{o['S']:.6e}",
            f"e_a_{name}_eV": f"{o['e_a']:.6f}",
            f"e_b_{name}_eV": f"{o['e_b']:.6f}",
            f"dE_{name}_eV": f"{o['dE']:.6f}",
            f"J_raw_{name}_meV": f"{o['J_raw']:.4f}",
            f"J_{name}_manifold_meV": f"{o['J_manifold']:.4f}",
            f"J_{name}_manifold_max_meV": f"{o['J_manifold_max']:.4f}",
            f"dE_{name}_manifold_eV": f"{o['dE_manifold']:.6f}",
            f"n_{name}_A": o["n_a"], f"n_{name}_B": o["n_b"]})
    row.update({"homo_A_eV": f"{eh_a:.6f}", "lumo_A_eV": f"{el_a:.6f}",
                "homo_B_eV": f"{eh_b:.6f}", "lumo_B_eV": f"{el_b:.6f}",
                "lumo_gap_A_eV": f"{lgap_a:.4f}",
                "lumo_gap_B_eV": f"{lgap_b:.4f}",
                "homo_gap_A_eV": f"{hgap_a:.4f}",
                "homo_gap_B_eV": f"{hgap_b:.4f}",
                "seconds": f"{elapsed:.1f}"})

    lumo_manifold = (out["lumo"]["n_a"], out["lumo"]["n_b"])
    note = (f"  [LUMO manifold {lumo_manifold[0]}x{lumo_manifold[1]}]"
            if lumo_manifold != (1, 1) else "")
    say(f"  {motif}/{tag}  J(LUMO) {out['lumo']['J_manifold']:8.2f} meV  "
        f"dE(LUMO) {out['lumo']['dE']:+7.3f} eV  "
        f"J(HOMO) {out['homo']['J_manifold']:8.2f} meV  "
        f"{elapsed / 60:.1f} min{note}")
    return row


def append_row(path, row):
    # DictWriter fills a missing key with an empty field and says nothing, so
    # a column added to COLUMNS but never populated would show up as a blank
    # rather than an error
    missing = [column for column in COLUMNS if column not in row]
    if missing:
        raise SystemExit(f"internal error: no value for column(s) "
                         f"{', '.join(missing)}")
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def completed(path):
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["config"] for row in csv.DictReader(handle)
                if row.get("config")}


# ---- The J files ----------------------------------------------------------
def multiplicities(cfg, motifs):
    """Symmetry multiplicity of each motif, from the stage 1 motif table."""
    table = cfg.results_dir / "motifs.csv"
    weights = {motif: 1 for motif in motifs}
    if not table.exists():
        say("  motifs.csv not found - weighting every motif equally")
        return weights
    with open(table, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("motif") in weights:
                weights[row["motif"]] = max(1, int(float(row["multiplicity"])))
    return weights


def write_j_files(cfg, motifs, couplings):
    """One J_<motif>.txt per motif, plus the pooled J.txt that stage 4 reads.

    Pooling by multiplicity matters: if one contact occurs four times per
    molecule and another twice, a carrier meets the first twice as often, and
    an equally weighted pool would misrepresent the material. The weights are
    reduced by their greatest common divisor so the pooled file stays small.
    """
    header = (f"# LUMO-LUMO effective transfer integrals, meV\n"
              f"# {cfg.project.name}, {cfg.dft.xc}/{cfg.dft.basis}, mode "
              f"{cfg.dft.mode}\n")

    for motif, values in couplings.items():
        path = cfg.results_dir / f"J_{motif}.txt"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(header)
            handle.write(f"# motif {motif}, {len(values)} frames\n")
            for value in values:
                handle.write(f"{value:.4f}\n")
        say(f"{path.name}: {len(values)} values, rms "
            f"{np.sqrt(np.mean(np.square(values))):.1f} meV")

    weights = multiplicities(cfg, motifs)
    if cfg.dft.pool_weighting == "uniform":
        weights = {motif: 1 for motif in motifs}
    present = [m for m in motifs if len(couplings.get(m, []))]
    divisor = math.gcd(*[weights[m] for m in present]) if present else 1

    pooled, description = [], []
    for motif in present:
        repeat = max(1, weights[motif] // max(1, divisor))
        pooled += list(couplings[motif]) * repeat
        description.append(f"{motif} x{repeat}")

    path = cfg.results_dir / cfg.transport.j_file
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(header)
        handle.write(f"# pooled over motifs by {cfg.dft.pool_weighting}: "
                     f"{', '.join(description)}\n")
        for value in pooled:
            handle.write(f"{value:.4f}\n")
    say(f"{path.name}: {len(pooled)} values pooled as "
        f"{', '.join(description)}, rms "
        f"{np.sqrt(np.mean(np.square(pooled))):.1f} meV")
    return path
