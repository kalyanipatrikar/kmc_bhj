"""Loading and validating config.yaml, and resolving every path the stages
write to.

One Config object is threaded through all five stages, so a run is fully
described by the config file plus the CIF it points at.
"""

import os
from pathlib import Path

import yaml

DEFAULT_CONFIG = "config.yaml"


class Section(dict):
    """A dict whose keys are also attributes, so config.md.timestep reads the
    way the stage code talks about it."""

    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError:
            raise AttributeError(
                f"'{key}' is missing from this section of the config file; "
                f"available keys are {sorted(self)}") from None
        return Section(value) if isinstance(value, dict) else value


class Config(Section):
    """The whole config file, plus the output directory layout."""

    @property
    def root(self):
        return Path(self["_root"])

    @property
    def data_dir(self):
        return self.root / self.project.data_dir

    @property
    def out_dir(self):
        return self.root / self.project.out_dir

    # ---- the output layout, in one place ---------------------------------
    @property
    def dimers_dir(self):
        return self.out_dir / "crystal_dimers"

    @property
    def configs_dir(self):
        return self.out_dir / "configs"

    @property
    def checkpoints_dir(self):
        return self.out_dir / "checkpoints"

    @property
    def logs_dir(self):
        return self.out_dir / "logs"

    @property
    def trajectories_dir(self):
        return self.out_dir / "trajectories"

    @property
    def results_dir(self):
        return self.out_dir / "results"

    @property
    def cif_path(self):
        return self.data_dir / self.crystal.cif

    def config_dir(self, motif):
        return self.configs_dir / motif

    def motifs(self):
        """Motif names written by stage 1, in rank order. Stages 2 and 3 both
        discover their work this way rather than from a list in the config, so
        the two can never disagree about which motifs exist."""
        if not self.dimers_dir.is_dir():
            raise SystemExit(f"{self.dimers_dir} not found - run stage 1 first")
        names = sorted(p.name[:-len("_dimer.xyz")]
                       for p in self.dimers_dir.glob("*_dimer.xyz"))
        if not names:
            raise SystemExit(f"no dimers in {self.dimers_dir} - stage 1 found "
                             f"no stacked pair, or has not been run")
        return names

    def make_dirs(self, *names):
        """Create the named output directories, e.g. make_dirs('results')."""
        for name in names:
            path = getattr(self, f"{name}_dir")
            path.mkdir(parents=True, exist_ok=True)
        return self


REQUIRED = ("project", "crystal", "md", "dft", "transport", "mobility")


def load(path=None):
    """Read the config file. `path` defaults to config.yaml beside the repo."""
    if path is None:
        path = os.environ.get("MLIP_CONFIG", DEFAULT_CONFIG)
    path = Path(path).resolve()
    if not path.exists():
        raise SystemExit(
            f"config file {path} not found - copy config.yaml from the "
            f"repository root, or set MLIP_CONFIG to point at yours")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise SystemExit(f"{path} did not parse as a mapping")

    missing = [s for s in REQUIRED if s not in raw]
    if missing:
        raise SystemExit(f"{path} is missing the section(s): "
                         f"{', '.join(missing)}")

    raw["_root"] = str(path.parent)
    cfg = Config(raw)
    _validate(cfg, path)
    return cfg


def _validate(cfg, path):
    """Catch the mistakes that would otherwise surface hours into a run."""
    if cfg.dft.mode not in ("superposition", "dimer_scf"):
        raise SystemExit(f"{path}: dft.mode must be 'superposition' or "
                         f"'dimer_scf', not {cfg.dft.mode!r}")
    if cfg.dft.pool_weighting not in ("multiplicity", "uniform"):
        raise SystemExit(f"{path}: dft.pool_weighting must be 'multiplicity' "
                         f"or 'uniform', not {cfg.dft.pool_weighting!r}")
    if cfg.dft.degeneracy_window < 0:
        raise SystemExit(f"{path}: dft.degeneracy_window must not be negative")
    if cfg.dft.max_manifold < 1:
        raise SystemExit(f"{path}: dft.max_manifold must be at least 1")
    if len(cfg.transport.lattice) != 3:
        raise SystemExit(f"{path}: transport.lattice must be [L, W, H]")
    if cfg.mobility.fit_from >= cfg.mobility.n_bins:
        raise SystemExit(f"{path}: mobility.fit_from ({cfg.mobility.fit_from}) "
                         f"leaves no bins to fit against mobility.n_bins "
                         f"({cfg.mobility.n_bins})")
    spacing = cfg.md.sample_spacing_fs / cfg.md.timestep
    if abs(spacing - round(spacing)) > 1e-9:
        raise SystemExit(f"{path}: md.sample_spacing_fs is not a whole number "
                         f"of timesteps")
