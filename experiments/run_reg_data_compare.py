"""Compare regression acquisition strategies on real datasets."""

from __future__ import annotations

import argparse

import numpy as np

from acquisition.base import run_active_learning_with_eval_reg_std_update
from acquisition.model import GPRegressorWrapper
from acquisition.strategies import Random, RegressionVarianceReduction, WeightedRegressionVarianceReduction
from acquisition.utils import create_named_weight_fn
from acquisition.utils.progress import trange
from acquisition.utils.regression import suggest_initial_lengthscale_noise_and_mean
from acquisition.visualisation import (
    create_metric_comparison_plots_reg,
    create_reg1d_acquisition_plots_with_visdata,
)
from data.problems import UCIRegressionProblem

DATASETS = {
    "yacht": {"dataset": "yacht_hydrodynamics", "p_init": 5, "p_targ": 60, "prefer": ["openml"]},
    "estate": {"dataset": 477, "p_init": 5, "p_targ": 80, "prefer": ["ucimlrepo"]},
    "slump": {"dataset": "slump", "p_init": 5, "p_targ": 20, "version": 2, "prefer": ["openml"]},
    "liver_disorders": {"dataset": "liver-disorders", "p_init": 5, "p_targ": 70, "prefer": ["openml"]},
}


def build_weight_fn(name: str):
    weight_builders = {
        "exp": lambda: create_named_weight_fn(lambda z: np.exp(z), "exp", "Exponential"),
        "exp_neg": lambda: create_named_weight_fn(lambda z: np.exp(-z), "exp_neg", "Exponential Decay"),
        "squared": lambda: create_named_weight_fn(lambda z: z**2 / 100, "squared", "Squared"),
        "max_clipped_1e-3": lambda: create_named_weight_fn(
            lambda z: np.maximum(z, 1e-3),
            "max_clipped_1e-3",
            "Clipped Linear (>= 1e-3)",
        ),
        "inv": lambda: create_named_weight_fn(
            lambda z: 1 / np.maximum(z, 1e-2),
            "inv",
            "Inverse (<= 1e2)",
        ),
    }
    if name == "inverse":
        name = "inv"
    if name == "negexp":
        name = "exp_neg"
    return weight_builders[name]()


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
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="yacht")
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--p-init", type=int, help="Override the number of initial labeled points.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed; each run adds the run index.")
    parser.add_argument("--info-update-every", type=int, default=5)
    parser.add_argument(
        "--weight",
        choices=["exp", "exp_neg", "negexp", "squared", "max_clipped_1e-3", "inv", "inverse"],
        default="max_clipped_1e-3",
    )
    parser.add_argument("--transform", choices=["none", "log10"], default="log10")
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
    parser.add_argument("--viz-runs", type=int, default=1)
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
    weight_fn = build_weight_fn(args.weight)
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

        random_strategy = Random(build_gp_model(problem, run_seed))
        run_active_learning_with_eval_reg_std_update(
            problem,
            random_strategy,
            args.n_steps,
            args.n_samples,
            viz=False,
            seed=run_seed,
            info_update_every=args.info_update_every,
        )

        variance_strategy = RegressionVarianceReduction(build_gp_model(problem, run_seed))
        variance_viz = run_active_learning_with_eval_reg_std_update(
            problem,
            variance_strategy,
            args.n_steps,
            args.n_samples,
            viz=run_idx < args.viz_runs,
            seed=run_seed,
            info_update_every=args.info_update_every,
        )

        weighted_strategy = WeightedRegressionVarianceReduction(
            build_gp_model(problem, run_seed),
            weight_fn,
        )
        weighted_viz = run_active_learning_with_eval_reg_std_update(
            problem,
            weighted_strategy,
            args.n_steps,
            args.n_samples,
            viz=run_idx < args.viz_runs,
            seed=run_seed,
            info_update_every=args.info_update_every,
        )

        if run_idx < args.viz_runs:
            create_reg1d_acquisition_plots_with_visdata(
                problem,
                [variance_strategy, weighted_strategy],
                [variance_viz, weighted_viz],
                run_seed,
                n_samples=args.n_samples,
            )

    strategies = [random_strategy, variance_strategy, weighted_strategy]
    weight_id = getattr(weight_fn, "name", args.weight)
    primary_transform = transform
    if primary_transform is None:
        secondary_transform = "log10"
        secondary_filename = f"reg_metric_{weight_id}_mean_log10.svg"
    else:
        secondary_transform = None
        secondary_filename = f"reg_metric_{weight_id}_mean_raw.svg"

    create_metric_comparison_plots_reg(
        problem=problem,
        strategy_list=strategies,
        n_evals=args.n_samples,
        stat_type="mean",
        transform=primary_transform,
        filename=f"reg_metric_{weight_id}_mean.svg",
    )
    create_metric_comparison_plots_reg(
        problem=problem,
        strategy_list=strategies,
        n_evals=args.n_samples,
        stat_type="mean",
        transform=secondary_transform,
        filename=secondary_filename,
    )


if __name__ == "__main__":
    main()
