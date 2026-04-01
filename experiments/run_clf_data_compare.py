"""Compare classification acquisition strategies on real datasets."""

from __future__ import annotations

import argparse

from acquisition.base import run_active_learning_with_eval
from acquisition.model import RandomForestClassifierWrapper
from acquisition.strategies import ClfEntropyReduction, ClfWeightedEntropyReduction, Random
from acquisition.utils.progress import trange
from acquisition.visualisation import (
    create_clf2d_acquisition_plots_with_visdata,
    create_clf2d_proportion_plots,
    create_metric_comparison_plots,
)
from data.problems import UCIClassificationProblem

DATASETS = {
    "vehicle": {"dataset": 149, "p_init": 5, "p_targ": 45, "p_test": 45, "prefer": ["ucimlrepo"]},
    "image_seg": {"dataset": 147, "p_init": 5, "p_targ": 50, "p_test": 50, "prefer": ["ucimlrepo"]},
    "landsat": {"dataset": 146, "p_init": 5, "p_targ": 100, "p_test": 200, "prefer": ["ucimlrepo"]},
    "vowel": {"dataset": "vowel", "p_init": 5, "p_targ": 15, "p_test": 15, "prefer": ["openml"], "version": 2},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="vehicle")
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--n-samples", type=int, default=10_000)
    parser.add_argument("--n-estimators", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--weight",
        default="50,1,1,50",
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
        help="Also generate accuracy plots in addition to the default NLL plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config = dict(DATASETS[args.dataset])
    weight = [int(token) for token in args.weight.split(",") if token.strip()]

    for run_idx in trange(args.n_runs, desc="Runs"):
        run_seed = args.seed + run_idx
        problem = UCIClassificationProblem(**dataset_config, seed=run_seed)
        if len(weight) != problem.num_classes:
            raise ValueError(
                f"Expected {problem.num_classes} class weights, got {len(weight)}: {weight}"
            )

        random_model = RandomForestClassifierWrapper(
            n_estimators=args.n_estimators,
            random_state=run_seed,
        )
        random_strategy = Random(random_model)
        run_active_learning_with_eval(
            problem,
            random_strategy,
            n_steps=args.n_steps,
            weight=weight,
            n_samples=args.n_samples,
            viz=False,
        )

        entropy_model = RandomForestClassifierWrapper(
            n_estimators=args.n_estimators,
            random_state=run_seed,
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

        weighted_model = RandomForestClassifierWrapper(
            n_estimators=args.n_estimators,
            random_state=run_seed,
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

    strategies = [random_strategy, entropy_strategy, weighted_strategy]
    create_metric_comparison_plots(
        problem=problem,
        strategy_list=strategies,
        metric_list=["nll", "acc"] if args.plot_accuracy else ["nll"],
        stat_type="mean",
    )
    create_clf2d_proportion_plots(problem=problem, strategy_list=strategies, stat_type="mean")


if __name__ == "__main__":
    main()
