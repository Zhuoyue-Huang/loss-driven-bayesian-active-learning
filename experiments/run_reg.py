"""Generate the synthetic regression figures used in the paper."""

import numpy as np
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from acquisition.utils import create_named_weight_fn
from acquisition.model import GPRegressorWrapper
from acquisition.strategies import RegressionVarianceReduction, WeightedRegressionVarianceReduction
from acquisition.visualisation import (
    create_reg1d_acquisition_plots_live_by_weight,
    create_reg1d_branch_comparison_plot,
)
from data.problems import RegressionProblem

# weight choices
# weight_fn = create_named_weight_fn(
#     lambda z: np.maximum(z, 1e-3),
#     "max_clipped_1e-3",
#     "Clipped Linear (≥ 10⁻³)"
# )
# weight_fn = create_named_weight_fn(
#     lambda z: np.exp(-z),
#     "inv_exp",
#     "exp(-z)"
# )

def build_problem(seed: int, true_mean, true_std) -> RegressionProblem:
    return RegressionProblem(
        name="pm_bump",
        true_args=(true_mean, true_std),
        initial=np.array([-1, 1]),
        targ=np.linspace(-8, 8, 49),
        pool_args=(-8, 8, 65),
        test_args=(-8, 8, 97),
        plot_args=(-8, 8, 193),
        rng=np.random.default_rng(seed=seed),
    )


def build_kernel(true_std: float):
    return RBF(length_scale=1, length_scale_bounds="fixed") + WhiteKernel(
        noise_level=4 * true_std**2,
        noise_level_bounds="fixed",
    )


def build_weight_fn(name: str):
    builders = {
        "exp": lambda: create_named_weight_fn(
            lambda z: np.exp(z),
            "exp",
            "exp(z)",
        ),
        "inv_exp": lambda: create_named_weight_fn(
            lambda z: np.exp(-z),
            "inv_exp",
            "exp(-z)",
        ),
    }
    return builders[name]()


def main():
    seed = 0
    true_mean = lambda x: 2*np.sin(2*x) + 10*np.exp(-(x-2.5)**2/0.5**2/2) - 8*np.exp(-(x+4.5)**2/0.5**2/2)
    true_std = 0.1
    n_samples = 500
    kernel = build_kernel(true_std)

    weight_fn_exp = build_weight_fn("exp")
    weight_fn_inv_exp = build_weight_fn("inv_exp")

    prob_intro = build_problem(seed, true_mean, true_std)
    intro_var = RegressionVarianceReduction(GPRegressorWrapper(kernel=kernel, random_state=seed))
    intro_wvar_exp = WeightedRegressionVarianceReduction(
        GPRegressorWrapper(kernel=kernel, random_state=seed),
        weight_fn_exp,
    )
    intro_wvar_inv_exp = WeightedRegressionVarianceReduction(
        GPRegressorWrapper(kernel=kernel, random_state=seed),
        weight_fn_inv_exp,
    )

    create_reg1d_branch_comparison_plot(
        prob_intro,
        intro_var,
        intro_wvar_exp,
        intro_wvar_inv_exp,
        branch_iteration=7,
        n_samples=n_samples,
    )

    prob = build_problem(seed, true_mean, true_std)
    strat1 = RegressionVarianceReduction(GPRegressorWrapper(kernel=kernel, random_state=seed))
    plot_iters = [0, 1, 3, 6, 10, 15]
    create_reg1d_acquisition_plots_live_by_weight(
        prob,
        strat1,
        weight_fn=weight_fn_exp,
        n_steps=plot_iters[-1] + 1,
        n_samples=n_samples,
        plot_iters=plot_iters,
    )

if __name__ == "__main__":
    main()
