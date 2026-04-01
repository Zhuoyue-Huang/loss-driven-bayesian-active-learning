from acquisition.utils.classification import create_grid2d
from acquisition.base import run_active_learning
from acquisition.model import create_gp_classifier
from acquisition.strategies import ClfEntropyReduction, ClfWeightedEntropyReduction
from acquisition.visualisation import create_clf2d_acquisition_plots
from data.problems import QuadDiagProblem


# choice of data generator
# Specific grid: np.array([[-2,-2], [-2,2], [2,-2], [2,2], [-2,0], [0,-2], [2,0], [0,2]])
# Specific grid: np.array([[-1,-1], [-1,1], [1,-1], [1,1], [-1,0], [0,-1], [1,0], [0,1]])
# Grid: create_grid2d((-4, 4, 8))
# Angle grid: create_even_angle_grid(3)
# Normal distribution: rng.normal(loc=0, scale=2.5, size=(75, 2))
# Uniform distribution: np.random.uniform(-4, 4, size=(50, 2))


def main(l=2):
    rng = np.random.default_rng(seed=0)
    n_samples = 10000

    prob = QuadDiagProblem(
        initial = create_grid2d((-2, 2, 4)),
        target = create_grid2d((-4, 4, 15)),
        pool_args = (-4, 4, 25),
        plot_args = (-4, 4, 150),
    )

    model  = create_gp_classifier(num_classes=prob.num_classes, ls=l, random_seed=0)
    # model = RandomForestClassifierWrapper(n_estimators=2000, random_state=0)
    # model = NeuralNetClassifierWrapper(input_dim=prob.X0.shape[1], num_classes=prob.num_classes)

    strat1 = ClfEntropyReduction(model, name='entropy')
    run_active_learning(prob, strat1, n_steps=4, n_samples=n_samples)

    weight = [1, 50, 1, 50]
    model  = create_gp_classifier(num_classes=prob.num_classes, ls=l, random_seed=0)
    strat2 = ClfWeightedEntropyReduction(model, weight=weight, name='entropy_w')
    run_active_learning(prob, strat2, n_steps=4, n_samples=n_samples)
    create_clf2d_acquisition_plots(problem=prob, strategy_list=[strat1, strat2], n_samples=n_samples)

if __name__ == "__main__":
    main()
