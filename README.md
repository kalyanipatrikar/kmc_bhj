# Mobility of Small Molecule Semiconductors with MLIP Derived Structures

Charge carrier mobility of a molecular semiconductor, computed from nothing but
its crystal structure. The methods for computing mobility and the entire algorithm has been described in 
https://doi.org/10.1021/acs.jctc.4c01029. The paper made use of Gaussian 16 for calculating chargr transfer integrals, here a method using only python based packages is presented. If you use this for your work please cite "K. Patrikar, K. Patadia, R. Khatua, A. Mondal
J. Chem. Theory Comput, 20, 22, 10120–10131, 2024".

The pipeline goes: packing information → thermally sampled dimer geometries
from a machine-learned interatomic potential → transfer integrals from DFT →
kinetic Monte Carlo hopping → mobility.

```
   data/*.cif
       |  stage 1   symmetry expansion, core extraction, neighbour shell
       v
   crystal_dimers/motif_NN_dimer.xyz
       |  stage 2   MACE molecular dynamics at 300 K
       v
   configs/motif_NN/config_NNN_{dimer,A,B}.xyz
       |  stage 3   DFT + dimer projection
       v
   results/J.txt          (an ensemble of transfer integrals, meV)
       |  stage 4   kinetic Monte Carlo random walk
       v
   trajectories/{x,time}.npy
       |  stage 5   mean square displacement, Einstein relation
       v
   results/mobility.csv, msd.png
```

Nothing in the code is specific to a particular molecule. The conjugated core
is detected from the bond graph rather than from atom labels, and the packing
motifs are discovered and ranked rather than named in advance, so pointing
`config.yaml` at a different CIF is the whole of what it takes to run a
different material.

---

## Install

```bash
conda create -n mlip python=3.12
conda activate mlip
pip install -r requirements.txt
```

Two things `requirements.txt` cannot express:

* **GPU torch.** Stage 2 is tens of picoseconds of MACE dynamics per motif and
  is impractically slow on a CPU. Install the CUDA build first, matching the
  `cu###` suffix to the driver on the machine:
  `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124`
* **Platform.** `pyscf` has no Windows build, so stage 3 needs WSL, Linux or
  macOS. Stages 1, 2, 4 and 5 run anywhere.

## Run

Put a `.cif` in `data/`, point `crystal.cif` at it, then:

```bash
python -m src.stage1_pair_structure
python -m src.stage2_generate_configs
python -m src.stage3_transfer_integrals
python -m src.stage4_random_walk
python -m src.stage5_mobility
```

or `python scripts/run_all.py`, optionally with `--from 3 --to 4`.

`python -m src.analysis` prints what every stage has produced so far, which is
the quickest way to pick up a run you left days ago.

Every stage is resumable. Stages 2 and 3 checkpoint per frame, so an
interrupted run continues from the last completed one rather than starting
over.

### Wall times

Very uneven, and worth knowing before you start. For a molecule of ~75 atoms
per monomer:

| stage | cost |
|---|---|
| 1 pair structures | seconds |
| 2 thermal sampling | ~30–60 min per motif on a GPU |
| 3 transfer integrals | **20–45 min per frame** — hours to days |
| 4 random walk | minutes |
| 5 mobility | seconds |

Stage 3 dominates everything. `dft.n_sample` is the dial: 10 frames is a first
pass, 40 is a well-sampled coupling distribution.

---

## Configuration

`config.yaml` is the only file you normally edit, and a run is fully described
by it plus the CIF it points at. The settings that change results most:

| key | what it controls |
|---|---|
| `crystal.contact_cutoff`, `crystal.min_contacts` | what counts as a stacked pair rather than a tip touch |
| `crystal.selection` | overrides for core detection; empty means fully automatic |
| `md.n_configs` | configurations sampled per motif |
| `dft.n_sample`, `dft.stratify` | how many frames reach DFT, and whether motifs are sampled evenly |
| `dft.pool_weighting` | how per-motif couplings are mixed into the pooled `J.txt` |
| `transport.sigma`, `transport.reorganisation_energy` | the energetic disorder and the Marcus barrier |
| `mobility.fit_from` | where the diffusive regime is taken to start |

### How motifs are chosen

Stage 1 enumerates every neighbour of one molecule, collapses the
symmetry-equivalent ones into a multiplicity, ranks them by contact count and
names them `motif_01`, `motif_02`, … Chemistry-meaningful reading is left to
the descriptor columns in `results/motifs.csv` — closest contact, interplanar
spacing, and `terminal_fraction`, the share of the contact made through the
ends of the molecule rather than its middle. A contact with a high terminal
fraction is a J-type end-group stack; a low one is an H-type backbone stack.

### How couplings reach the random walk

Stage 3 writes `J_<motif>.txt` per motif and a pooled `J.txt`. Pooling is
weighted by symmetry multiplicity by default: if one contact occurs four times
per molecule and another twice, a carrier meets the first twice as often, and
an equally weighted pool would misrepresent the material.

Stage 4 then resamples that pool — 400 trials × 6000 steps × 26 neighbours is
about 6×10⁷ draws from an ensemble of a few tens of values. Redrawing fresh DFT
per trial would cost hundreds of times more for no extra statistical
information, because the frames are independent draws from the same thermal
distribution either way. What sharpens the result is the *number of distinct
configurations* in the ensemble, i.e. `dft.n_sample`.

### Frontier manifolds

A single-orbital coupling understates transport when a monomer's frontier
orbital is near-degenerate with the next one — common in acceptors, where LUMO
and LUMO+1 can sit within a few tens of meV. The carrier can arrive in either
state, and both channels carry flux.

Stage 3 handles this by not treating the frontier as one orbital. Every orbital
within `dft.degeneracy_window` of it (same occupancy, at most
`dft.max_manifold`) joins a manifold; the two manifolds are symmetrically
(Löwdin) orthogonalised together, and the A–B block of the transformed Fock
matrix is the matrix of couplings between them. Since Marcus rates go as *J*²
and the rate out of a state sums over accessible final states, the channels add
in quadrature:

*J*<sub>eff</sub>² = (1/*n*<sub>A</sub>) Σ<sub>*a*</sub> Σ<sub>*b*</sub> |*J*<sub>*ab*</sub>|²

Two things make this cheap and safe. It needs **no extra SCF** — the orbitals
are already converged, so the added cost is one small eigendecomposition per
frame. And for one orbital on each side it reduces *exactly* to the
single-orbital formula, which is itself the closed form of the same
transformation — `test_manifold_of_one_reproduces_the_two_state_formula_exactly`
pins that to 10⁻¹⁴. Setting `degeneracy_window: 0` recovers the single-orbital
treatment everywhere.

`transfer_integrals_<motif>.csv` keeps both: `J_lumo_meV` is the single-orbital
value and `J_lumo_manifold_meV` is the manifold one, alongside `n_lumo_A` /
`n_lumo_B` recording how many orbitals were used. The manifold column is what
feeds `J.txt`, and the stage 3 summary reports how many frames needed more than
one orbital and what it did to the rms coupling.

### Units

Transfer integrals are in **meV** throughout. Stage 3 writes `J.txt` in meV and
stage 4 reads it directly — there is no conversion layer and no unit option to
set. Site energies, the disorder width `transport.sigma` and the
reorganisation energy are in **eV**, so `marcus()` in
`src/transport/rates.py` takes the coupling in meV and the energies in eV by
design, and its argument names say so.

---

## Worked example: AQx-2

`data/2161927.cif` is the shipped example — AQx-2, CCDC 2161927, from
*Nat. Commun.* **14**, 5079 (2023). Stage 1 runs on it with no configuration at
all and recovers the published packing:

```
  the asymmetric unit is 1/4 of a molecule; reassembled to 380 atoms
  auto-detected core: 64 heavy atoms (C46 F4 N8 O2 S4), 12 rings
  capped 4 attachment point(s); fragment 76 atoms

motif       contact  interpl  pairs  mult  term%     COM
motif_01       3.43     3.51    210     2    35%   13.42
motif_02       3.29     3.54    123     2    55%   19.23
```

against the paper's H-aggregation between backbones at 3.424 Å and
J-aggregation between end groups at 3.341 Å, four overlaps per molecule. The
alkyl chains are dropped and their attachment points capped: they are severely
disordered in this structure, and the frontier orbitals live on the conjugated
core in any case.

---

## Tests

```bash
python -m pytest              # 51 tests, a few seconds
python -m pytest -m slow      # adds one end-to-end pyscf check
```

They cover the deterministic layer: the CIF reader and symmetry parsing, ring
perception and core detection, superposition and pair descriptors, the dimer
projection algebra, lattice index arithmetic and site disorder, the Marcus
rate, and the mobility fit against a random walk whose diffusion coefficient is
known analytically. MACE and the SCFs themselves are not unit tested — they are
too expensive and too stochastic — and are instead guarded from the inside by
the drift, overlap and orbital-degeneracy warnings the stages print.

---

**Near-degenerate frontier orbitals** are handled, but the manifold width is a
choice. `dft.degeneracy_window` (0.10 eV) and `dft.max_manifold` (4) decide how
many orbitals join the frontier one; widening the window too far eventually
makes the fragment orbital set linearly dependent in the dimer basis, which fails loudly rather than silently.
---
Next update will include an alternate way to get starting structures of dimers, so that packing information (.cif) is not necessary.

---

## Layout

```
config.yaml            the only file you normally edit
data/                  input structures
out_folder/            everything generated (gitignored)
src/
  config.py            config loading, validation, output paths
  utils.py             constants, xyz I/O, superposition, pair descriptors
  crystal/             stage 1: CIF, topology, fragments, motifs
  md/                  stage 2: constraints, calculators, checkpoints, sampling
  dft/                 stage 3: SCF, dimer projection, driver
  transport/           stages 4-5: lattice, disorder, rates, kMC, mobility
  stage[1-5]_*.py      the command line entry points
  analysis.py          read-only status summary
scripts/run_all.py     the whole pipeline
tests/                 pytest suite
```
