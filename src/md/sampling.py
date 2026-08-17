"""One motif, start to finish: hydrogen relaxation, equilibration, production.

The input is a dimer taken from the experimental crystal structure, so the
packing motif is already correct. The job here is only to sample thermal
fluctuations around it - the dynamic disorder that makes transfer integrals a
distribution rather than a single number.

Three things follow from starting at the experimental geometry:

* **No annealing.** Heating the pair to reorganise it out of a poor constructed
  placement would destroy the very packing we went to the crystal to obtain.

* **Hydrogens are relaxed first, heavy atoms fixed.** X-ray hydrogen positions
  are placed geometrically and ride on their parent atoms, far too short for a
  quantum calculation - a crystal geometry typically starts at several eV/A of
  force almost entirely because of it. Relaxing them with the heavy skeleton
  fixed repairs that without moving the packing.

* **The heavy skeleton is NOT relaxed.** In the crystal this pair is held in
  place by the surrounding lattice, which is absent here; relaxing to the
  gas-phase minimum would slide the molecules to an arrangement that is not the
  experimental one. The trajectory samples around the crystal geometry, and the
  drift column reports how far it wanders so you can see whether that
  assumption held.
"""

import os

import numpy as np

from ..utils import banner, descriptors, drift, heavy_indices, say, write_xyz
from . import checkpoint as ckpt
from .calculators import calculator, device
from .constraints import FlatBottomCOM

MANIFEST_COLUMNS = ["config", "energy_eV", "temperature_K", "separation_A",
                    "slip_long_A", "slip_short_A", "com_distance_A",
                    "closest_contact_A", "overlap_fraction", "drift_A"]


def run_motif(cfg, motif):
    """Sample `md.n_configs` configurations of one motif. Resumable."""
    from ase import Atoms, units
    from ase.constraints import FixAtoms, FixCom
    from ase.io import read
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import (Stationary, ZeroRotation,
                                             thermalize_momenta)
    from ase.optimize import FIRE

    md = cfg.md
    input_xyz = cfg.dimers_dir / f"{motif}_dimer.xyz"
    output_dir = cfg.config_dir(motif)
    manifest = cfg.results_dir / f"manifest_{motif}.csv"
    checkpoint = cfg.checkpoints_dir / f"checkpoint_{motif}.npz"
    md_log = cfg.logs_dir / f"md_{motif}.log"
    relax_log = cfg.logs_dir / f"relax_h_{motif}.log"
    output_dir.mkdir(parents=True, exist_ok=True)

    banner(motif)
    if not input_xyz.exists():
        raise SystemExit(f"{input_xyz} not found - run stage 1 first")

    # ---- Step 1: load the crystal dimer ----------------------------------
    atoms = read(str(input_xyz), format="xyz")
    n_atoms = len(atoms)
    n_a = n_atoms // 2
    symbols = atoms.get_chemical_symbols()
    if symbols[:n_a] != symbols[n_a:]:
        raise SystemExit(f"{input_xyz}: the two halves are not the same "
                         f"molecule")
    heavy_a = heavy_indices(symbols[:n_a])
    idx_a, idx_b = np.arange(n_a), np.arange(n_a, n_atoms)

    atoms.calc = calculator(md.model, "float64")
    crystal = descriptors(atoms.get_positions(), n_a, heavy_a)
    say(f"{n_atoms} atoms ({n_a} per monomer); crystal geometry: contact "
        f"{crystal[4]:.2f} A, COM {crystal[3]:.2f} A, overlap "
        f"{crystal[5] * 100:.0f}%")

    spacing = int(round(md.sample_spacing_fs / md.timestep))
    equil_steps = int(round(md.equilibration_fs / md.timestep))
    equil_chunk = int(round(md.equil_chunk_fs / md.timestep))

    # ---- Step 2: resume, or relax the hydrogens ---------------------------
    resumed = False
    if checkpoint.exists():
        state = ckpt.load(checkpoint, motif)
        atoms.set_positions(state["positions"])
        atoms.set_momenta(state["momenta"])
        reference = state["reference"]
        equil_done, n_frames = state["equil_done"], state["n_frames"]
        resumed = True
        say(f"resuming: {equil_done}/{equil_steps} equilibration steps, "
            f"{n_frames}/{md.n_configs} frames saved")
        if n_frames >= md.n_configs:
            say("already complete")
            return manifest, crystal
    else:
        atoms.set_constraint(FixAtoms(
            indices=[i for i, s in enumerate(symbols) if s != "H"]))
        say(f"start fmax {np.abs(atoms.get_forces()).max():.2f} eV/A "
            f"(X-ray riding hydrogens)")
        FIRE(atoms, logfile=str(relax_log)).run(fmax=md.relax_fmax,
                                                steps=md.relax_steps)
        say(f"after H relaxation: fmax "
            f"{np.abs(atoms.get_forces()).max():.3f} eV/A")
        atoms.set_constraint()
        reference = atoms.get_positions().copy()
        equil_done, n_frames = 0, 0
        with open(manifest, "w", encoding="utf-8") as handle:
            handle.write(",".join(MANIFEST_COLUMNS) + "\n")

    r_flat = crystal[3] + md.com_margin
    atoms.calc = calculator(md.model, "float32")

    # ---- Step 3: thermostat ----------------------------------------------
    # FixCom holds the overall centre of mass: Langevin's own fixcm=True does
    # the same but is deprecated in ASE 3.28+ for not strictly sampling NVT.
    if not resumed:
        thermalize_momenta(atoms, temperature_K=md.temperature)
        Stationary(atoms)
        ZeroRotation(atoms)
    atoms.set_constraint([FixCom(),
                          FlatBottomCOM(idx_a, idx_b, r_flat, md.k_restraint)])

    dyn = Langevin(atoms, timestep=md.timestep * units.fs,
                   temperature_K=md.temperature,
                   friction=md.friction / units.fs, fixcm=False,
                   logfile=str(md_log), loginterval=200)

    # ---- Step 4: equilibrate ----------------------------------------------
    if equil_done < equil_steps:
        say(f"equilibrating {md.equilibration_fs / 1000:.0f} ps at "
            f"{md.temperature:.0f} K (COM restraint beyond {r_flat:.1f} A)...")
        while equil_done < equil_steps:
            chunk = min(equil_chunk, equil_steps - equil_done)
            dyn.run(chunk)
            equil_done += chunk
            ckpt.save(checkpoint, atoms, reference, equil_done, n_frames, motif)
        positions = atoms.get_positions()
        d = descriptors(positions, n_a, heavy_a)
        say(f"  equilibrated: contact {d[4]:.2f} A, overlap "
            f"{d[5] * 100:.0f}%, drift {drift(positions, reference, n_a):.2f} A")

    # ---- Step 5: production -----------------------------------------------
    # an explicit loop rather than dyn.attach, so a frame and its checkpoint
    # are written together and a resumed run picks up at exactly the right
    # frame
    say(f"production {spacing * md.n_configs * md.timestep / 1000:.0f} ps, one "
        f"frame every {md.sample_spacing_fs / 1000:.1f} ps:")
    while n_frames < md.n_configs:
        dyn.run(spacing)
        positions = atoms.get_positions()
        energy = atoms.get_potential_energy()
        temperature = atoms.get_temperature()
        sep, slip_l, slip_s, com, contact, overlap = descriptors(
            positions, n_a, heavy_a)
        moved = drift(positions, reference, n_a)

        tag = f"config_{n_frames:03d}"
        elapsed = (equil_steps + (n_frames + 1) * spacing) * md.timestep / 1000
        note = (f"{tag}: {motif}, t={elapsed:.2f} ps, E={energy:.4f} eV, "
                f"T={temperature:.1f} K, sep={sep:.3f} A, com={com:.3f} A, "
                f"contact={contact:.3f} A, overlap={overlap:.3f}, "
                f"drift={moved:.3f} A, n_A={n_a}")

        write_xyz(output_dir / f"{tag}_dimer.xyz", symbols, positions, note)
        write_xyz(output_dir / f"{tag}_A.xyz", symbols[:n_a], positions[:n_a],
                  note + " [fragment A]")
        write_xyz(output_dir / f"{tag}_B.xyz", symbols[n_a:], positions[n_a:],
                  note + " [fragment B]")

        with open(manifest, "a", encoding="utf-8") as handle:
            handle.write(f"{tag},{energy:.6f},{temperature:.2f},{sep:.4f},"
                         f"{slip_l:.4f},{slip_s:.4f},{com:.4f},{contact:.4f},"
                         f"{overlap:.4f},{moved:.4f}\n")
        n_frames += 1
        ckpt.save(checkpoint, atoms, reference, equil_steps, n_frames, motif)
        say(f"  {tag}  E {energy:11.3f}  T {temperature:5.1f}  contact "
            f"{contact:4.2f}  overlap {overlap * 100:3.0f}%  drift {moved:4.2f}")

    return manifest, crystal


def report(cfg, manifest, crystal, motif):
    """Summary and the sanity checks that say whether the sampling still
    represents the experimental packing."""
    rows = np.atleast_1d(np.genfromtxt(manifest, delimiter=",", names=True))
    say(f"\n{motif}: {len(rows)} configurations in {cfg.config_dir(motif)}")
    say(f"  contact   {rows['closest_contact_A'].mean():.2f} +/- "
        f"{rows['closest_contact_A'].std():.2f} A (crystal {crystal[4]:.2f})")
    say(f"  overlap   {rows['overlap_fraction'].mean() * 100:.0f} +/- "
        f"{rows['overlap_fraction'].std() * 100:.0f}% "
        f"(crystal {crystal[5] * 100:.0f}%)")
    say(f"  drift     {rows['drift_A'].mean():.2f} +/- "
        f"{rows['drift_A'].std():.2f} A (max {rows['drift_A'].max():.2f})")

    if rows["drift_A"].max() > 2.0:
        say("  WARNING: the pair has moved well away from the crystal "
            "packing; these frames no longer sample the experimental motif")
    if rows["overlap_fraction"].mean() < 0.7 * crystal[5]:
        say("  WARNING: overlap has dropped substantially below the crystal "
            "value - the molecules are sliding apart in vacuum")
    if rows["com_distance_A"].max() > crystal[3] + cfg.md.com_margin:
        say("  WARNING: COM distance passed the restraint - the pair is being "
            "held together rather than staying together")
    return rows
