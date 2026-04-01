from sklearn.datasets import load_iris

from data.problems import RealClassificationProblem
from data.generators import sample_per_class_indices
from acquisition.base import run_active_learning
from acquisition.model import create_gp_classifier
from acquisition.strategies import ClfEntropyReduction, ClfWeightedEntropyReduction
from acquisition.visualisation import create_clf2d_acquisition_plots

def main(l=1):
    n_samples = 10000
    # data = spiral(n=500, k=3, turns=0.75, seed=0)
    data = load_iris()
    prob = RealClassificationProblem(
        name = "iris",
        data = data,
        init_idxs = sample_per_class_indices(data, p=5, seed=0),
        targ_idxs = sample_per_class_indices(data, p=10, seed=0),
    )

    model  = create_gp_classifier(num_classes=prob.num_classes, ls=l, random_seed=0)
    # model = RandomForestClassifierWrapper(n_estimators=50, random_state=0)
    # model = NeuralNetClassifierWrapper(input_dim=2, num_classes=prob.num_classes, dropout_rate=0.7)

    strat1 = ClfEntropyReduction(model, name="entropy")
    run_active_learning(prob, strat1, n_steps=4, n_samples=n_samples)

    for weight in [[50, 1, 50],
                   [1, 50, 50]]:
        model  = create_gp_classifier(num_classes=prob.num_classes, ls=l, random_seed=0)
        strat2 = ClfWeightedEntropyReduction(model, weight=weight, name=f"entropy_w")
        run_active_learning(prob, strat2, n_steps=4, n_samples=n_samples)
        create_clf2d_acquisition_plots(problem=prob, strategy_list=[strat1, strat2], n_samples=n_samples)

if __name__ == "__main__":
    main()
