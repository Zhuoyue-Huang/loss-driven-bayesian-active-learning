"""Compare squared-loss and LINEX regression acquisition on real datasets."""

from __future__ import annotations

import argparse

from acquisition.base import run_active_learning_with_eval_reg_std_update
from acquisition.model import GPRegressorWrapper
from acquisition.strategies import RegressionPosteriorLinex, RegressionVarianceReduction
from acquisition.utils.progress import trange
from acquisition.utils.regression import suggest_initial_lengthscale_noise_and_mean
from acquisition.visualisation import create_loss_comparison_plots_reg
from data.problems import UCIRegressionProblem

DATASETS = {
    "yacht": {"dataset": "yacht_hydrodynamics", "p_init": 5, "p_targ": 60, "prefer": ["openml"]},
    "estate": {"dataset": 477, "p_init": 5, "p_targ": 80, "prefer": ["ucimlrepo"]},
    "slump": {"dataset": "slump", "p_init": 5, "p_targ": 20, "version": 2, "prefer": ["openml"]},
    "liver_disorders": {"dataset": "liver-disorders", "p_init": 5, "p_targ": 70, "prefer": ["openml"]},
}
def build_gp_model(problem, seed: int) -> GPRegressorWrapper:
    ls, sigma_n, init_mean, sigma_lin, sigma_f, kernel, info = suggest_initial_lengthscale_noise_and_mean(
        problem,
        family="m32",
        add_linear=True,
        add_rq=False,
    )
    return GPRegressorWrapper(
        ls=ls,
        nu=info.get("nu", 1.5) or 1.5,
        true_std=sigma_n,
        mean_init=init_mean,
        sigma_lin=sigma_lin,
        sigma_f=sigma_f,
        kernel=kernel,
        random_state=seed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="estate")
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--p-init", type=int, help="Override the number of initial labeled points.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed; each run adds the run index.")
    parser.add_argument("--info-update-every", type=int, default=3)
    parser.add_argument("--scale-target", action="store_true")
    parser.add_argument(
        "--target-scale-std-only",
        action="store_true",
        help="Scale y only by the initial labeled-set std, without centering.",
    )
    parser.add_argument(
        "--target-scale-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="Fit a MinMax target scaler on the initial labels and map them into [LOW, HIGH].",
    )
    parser.add_argument(
        "--stat-type",
        choices=["mean", "median"],
        default="mean",
        help="Aggregate plot statistic across runs; 'median' uses a median line with quantile band.",
    )
    parser.add_argument("--transform", choices=["none", "log10"], default="log10")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_scale_modes = int(bool(args.scale_target)) + int(bool(args.target_scale_std_only)) + int(args.target_scale_range is not None)
    if target_scale_modes > 1:
        raise ValueError(
            "Use at most one of --scale-target, --target-scale-std-only, or --target-scale-range LOW HIGH."
        )
    dataset_config = dict(DATASETS[args.dataset])
    if args.p_init is not None:
        dataset_config["p_init"] = args.p_init
    transform = None if args.transform == "none" else args.transform
    target_scale_range = tuple(args.target_scale_range) if args.target_scale_range is not None else None

    for run_idx in trange(args.n_runs, desc="Runs"):
        run_seed = args.seed + run_idx
        problem = UCIRegressionProblem(
            **dataset_config,
            seed=run_seed,
            scale_target=args.scale_target,
            target_scale_std_only=args.target_scale_std_only,
            target_scale_range=target_scale_range,
        )

        variance_strategy = RegressionVarianceReduction(build_gp_model(problem, run_seed))
        run_active_learning_with_eval_reg_std_update(
            problem,
            variance_strategy,
            args.n_steps,
            args.n_samples,
            viz=False,
            seed=run_seed,
            info_update_every=args.info_update_every,
        )

        linex_strategy = RegressionPosteriorLinex(build_gp_model(problem, run_seed))
        run_active_learning_with_eval_reg_std_update(
            problem,
            linex_strategy,
            args.n_steps,
            args.n_samples,
            viz=False,
            seed=run_seed,
            info_update_every=args.info_update_every,
        )

    create_loss_comparison_plots_reg(
        problem=problem,
        strategies=[variance_strategy, linex_strategy],
        ref_indices=(0, 1),
        stat_type=args.stat_type,
        filename="reg_loss_compare.svg",
        transform=transform,
    )
    secondary_transform = None if transform == "log10" else "log10"
    secondary_filename = "reg_loss_compare_raw.svg" if transform == "log10" else "reg_loss_compare_log10.svg"
    create_loss_comparison_plots_reg(
        problem=problem,
        strategies=[variance_strategy, linex_strategy],
        ref_indices=(0, 1),
        stat_type=args.stat_type,
        filename=secondary_filename,
        transform=secondary_transform,
    )


if __name__ == "__main__":
    main()
