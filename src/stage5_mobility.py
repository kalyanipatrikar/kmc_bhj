"""Stage 5: mobility from the trajectories.

Averages the trajectories onto a uniform time grid, fits the mean square
displacement against time in the diffusive regime, and converts the diffusion
coefficient to a mobility with the Einstein relation.

Output, into `out_folder/results/`:

    mobility.csv    the fitted numbers and the settings behind them
    msd.png         mean square displacement against time, with the fit

    python -m src.stage5_mobility [config.yaml] [--show]
"""

import argparse
import csv

import numpy as np

from . import config as config_module
from .transport.mobility import analyse
from .utils import banner, say


def run(cfg, show=False):
    cfg.make_dirs("results")
    banner(f"stage 5: {cfg.project.name} - mobility")

    x_path = cfg.trajectories_dir / "x.npy"
    t_path = cfg.trajectories_dir / "time.npy"
    if not (x_path.exists() and t_path.exists()):
        raise SystemExit(f"{x_path} or {t_path} not found - run stage 4 first")
    sites, times = np.load(x_path), np.load(t_path)
    say(f"{sites.shape[0]} trials of {sites.shape[1]} steps")

    mean_time, msd, result = analyse(cfg, sites, times)

    say(f"\n  intersite distance   {result['intersite_distance_A']:.2f} A")
    say(f"  bins fitted          {result['n_bins_fitted']} of "
        f"{cfg.mobility.n_bins}, from bin {cfg.mobility.fit_from}")
    say(f"  diffusion constant   {result['diffusion_cm2_per_s']:.4e} cm^2/s")
    say(f"  mobility             {result['mobility_cm2_per_Vs']:.4e} "
        f"cm^2/(V s)")

    output = cfg.results_dir / "mobility.csv"
    row = dict(result,
               temperature_K=cfg.transport.temperature,
               sigma_eV=cfg.transport.sigma,
               reorganisation_eV=cfg.transport.reorganisation_energy,
               trials=int(sites.shape[0]), steps=int(sites.shape[1]))
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    say(f"\nwrote {output}")

    plot(cfg, mean_time, msd, result, show)
    return result


def plot(cfg, mean_time, msd, result, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")     # so the pipeline runs headless
    import matplotlib.pyplot as plt

    usable = np.isfinite(mean_time) & np.isfinite(msd)
    figure, axes = plt.subplots(figsize=(5.5, 4.0), constrained_layout=True)
    axes.plot(mean_time[usable], msd[usable], "o", ms=4, label="simulation")

    fitted = usable.copy()
    fitted[:int(cfg.mobility.fit_from)] = False
    line = (mean_time[fitted] * result["slope_cm2_per_s"]
            + result["intercept_cm2"])
    axes.plot(mean_time[fitted], line, "-",
              label=f"fit: mu = {result['mobility_cm2_per_Vs']:.2e} "
                    f"cm$^2$/(V s)")

    axes.set_xlabel("time (s)")
    axes.set_ylabel(r"mean square displacement (cm$^2$)")
    axes.set_title(f"{cfg.project.name} charge carrier diffusion")
    axes.legend(frameon=False, fontsize=8)

    path = cfg.results_dir / "msd.png"
    figure.savefig(path, dpi=200)
    say(f"wrote {path}")
    if show:
        plt.show()
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", nargs="?", default=None,
                        help="path to config.yaml")
    parser.add_argument("--show", action="store_true",
                        help="open the plot window as well as saving it")
    args = parser.parse_args()
    run(config_module.load(args.config), show=args.show)


if __name__ == "__main__":
    main()
