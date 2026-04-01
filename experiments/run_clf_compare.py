"""Visualize entropy and weighted-entropy acquisition on synthetic classification."""

from __future__ import annotations

import argparse

import numpy as np

from acquisition.base import run_active_learning_with_eval
from acquisition.model import create_gp_classifier
from acquisition.strategies import ClfEntropyReduction, ClfWeightedEntropyReduction
from acquisition.utils.progress import trange
from acquisition.visualisation import (
    create_clf2d_acquisition_plots_with_visdata,
    create_metric_comparison_plots,
)
from data.problems import TernaryAngularProblem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=10_000)
    parser.add_argument("--lengthscale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--weight",
        default="1,50,50",
        help="Comma-separated class weights for weighted entropy.",
    )
    parser.add_argument(
        "--viz-runs",
        type=int,
        default=2,
        help="Number of runs for which acquisition snapshots are saved.",
    )
    parser.add_argument(
        "--plot-accuracy",
        action="store_true",
        help="Include accuracy when metric plots are enabled.",
    )
    parser.add_argument(
        "--plot-metrics",
        action="store_true",
        help="Also generate aggregate metric plots after the visualization run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weight = [int(token) for token in args.weight.split(",") if token.strip()]

    for run_idx in trange(args.n_runs, desc="Runs"):
        rng = np.random.default_rng(args.seed + run_idx)
        problem = TernaryAngularProblem(
            initial=rng.normal(loc=0.0, scale=1.0, size=(15, 2)),
            target=rng.normal(loc=0.0, scale=2.0, size=(150, 2)),
            test=rng.normal(loc=0.0, scale=2.0, size=(150, 2)),
            pool_args=(-4, 4, 20),
        )
        if len(weight) != problem.num_classes:
            raise ValueError(
                f"Expected {problem.num_classes} class weights, got {len(weight)}: {weight}"
            )

        entropy_model = create_gp_classifier(
            num_classes=problem.num_classes,
            ls=args.lengthscale,
            random_seed=args.seed + run_idx,
        )
        entropy_strategy = ClfEntropyReduction(entropy_model)
        entropy_viz = run_active_learning_with_eval(
            problem,
            entropy_strategy,
            n_steps=args.n_steps,
            weight=weight,
            n_samples=args.n_samples,
            viz=run_idx < args.viz_runs,
        )

        weighted_model = create_gp_classifier(
            num_classes=problem.num_classes,
            ls=args.lengthscale,
            random_seed=args.seed + run_idx,
        )
        weighted_strategy = ClfWeightedEntropyReduction(weighted_model, weight=weight)
        weighted_viz = run_active_learning_with_eval(
            problem,
            weighted_strategy,
            n_steps=args.n_steps,
            weight=weight,
            n_samples=args.n_samples,
            viz=run_idx < args.viz_runs,
        )

        if run_idx < args.viz_runs:
            create_clf2d_acquisition_plots_with_visdata(
                problem,
                [entropy_strategy, weighted_strategy],
                [entropy_viz, weighted_viz],
                run_idx,
                n_samples=args.n_samples,
            )

    if args.plot_metrics:
        create_metric_comparison_plots(
            problem=problem,
            strategy_list=[entropy_strategy, weighted_strategy],
            metric_list=["nll", "acc"] if args.plot_accuracy else ["nll"],
            stat_type="mean",
        )


if __name__ == "__main__":
    main()
