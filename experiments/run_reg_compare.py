"""Compare regression acquisition strategies on the synthetic 1D benchmark."""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from acquisition.base import run_active_learning_with_eval_reg
from acquisition.model import GPRegressorWrapper
from acquisition.strategies import Random, RegressionVarianceReduction, WeightedRegressionVarianceReduction
from acquisition.utils import create_named_weight_fn
from acquisition.utils.progress import trange
from acquisition.visualisation import create_metric_comparison_plots_reg
from data.problems import RegressionProblem


def build_weight_fn(name: str):
    weight_builders = {
        "exp": lambda: create_named_weight_fn(lambda z: np.exp(z), "exp", "Exponential"),
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
    return weight_builders[name]()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--n-steps", type=int, default=25)
    parser.add_argument("--n-samples", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weight", choices=["exp", "squared", "max_clipped_1e-3", "inv", "inverse"], default="exp")
    parser.add_argument("--transform", choices=["none", "log10"], default="log10")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transform = None if args.transform == "none" else args.transform
    true_mean = lambda x: 2 * np.sin(2 * x) + 10 * np.exp(-(x - 2.5) ** 2 / 0.5**2 / 2) - 8 * np.exp(-(x + 4.5) ** 2 / 0.5**2 / 2)
    true_std = 0.1
    kernel = RBF(length_scale=1, length_scale_bounds="fixed") + WhiteKernel(
        noise_level=4 * true_std**2,
        noise_level_bounds="fixed",
    )
    weight_fn = build_weight_fn(args.weight)

    for run_idx in trange(args.n_runs, desc="Runs"):
        rng = np.random.default_rng(seed=args.seed + run_idx)
        problem = RegressionProblem(
            name="pm_bump",
            true_args=(true_mean, true_std),
            initial=np.array([-1, 1]),
            targ=np.linspace(-8, 8, 49),
            pool_args=(-8, 8, 65),
            test_args=(-8, 8, 97),
            plot_args=(-8, 8, 193),
            rng=rng,
        )

        random_strategy = Random(GPRegressorWrapper(kernel=kernel, random_state=args.seed + run_idx))
        run_active_learning_with_eval_reg(
            problem,
            random_strategy,
            args.n_steps,
            args.n_samples,
            viz=False,
            seed=args.seed + run_idx,
        )

        variance_strategy = RegressionVarianceReduction(
            GPRegressorWrapper(kernel=kernel, random_state=args.seed + run_idx)
        )
        run_active_learning_with_eval_reg(
            problem,
            variance_strategy,
            args.n_steps,
            args.n_samples,
            viz=False,
            seed=args.seed + run_idx,
        )

        weighted_strategy = WeightedRegressionVarianceReduction(
            GPRegressorWrapper(kernel=kernel, random_state=args.seed + run_idx),
            weight_fn,
        )
        run_active_learning_with_eval_reg(
            problem,
            weighted_strategy,
            args.n_steps,
            args.n_samples,
            viz=False,
            seed=args.seed + run_idx,
        )

    create_metric_comparison_plots_reg(
        problem=problem,
        strategy_list=[random_strategy, variance_strategy, weighted_strategy],
        n_evals=args.n_samples,
        stat_type="mean",
        transform=transform,
    )


if __name__ == "__main__":
    main()
