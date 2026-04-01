from data.problems import UCIClassificationProblem
from acquisition.base import run_active_learning_with_branching
from acquisition.model import RandomForestClassifierWrapper
from acquisition.strategies import ClfEntropyReduction, ClfWeightedEntropyReduction
from acquisition.utils.progress import trange
from acquisition.visualisation import create_branching_metric_comparison_plots


# model choice
# model  = create_gp_classifier(num_classes=prob.num_classes, ls=l)
# model = RandomForestClassifierWrapper(n_estimators=l)
# model = NeuralNetClassifierWrapper(input_dim=prob.X0.shape[1], num_classes=prob.num_classes, dropout_rate=l)
# model = BayesianNNClassifierWrapper(input_dim=prob.X0.shape[1], num_classes=prob.num_classes, prior_var=l)

# problem choice
# prob = RealClassificationProblem(
#         name = "spiral4",
#         data = spiral(n=500, k=4, turns=0.6),
#         p_init=5,
#         p_targ=25,
#         p_test=25,
#         seed=0
#     )

# prob = UCIClassificationProblem(**wine)
vehicle = {"dataset":149, "p_init":5, "p_targ":45, "p_test":45, "seed":0, "prefer":["ucimlrepo"]}  # 100 steps
image_seg = {"dataset":147, "p_init":5, "p_targ":50, "p_test":50, "seed":0, "prefer":["ucimlrepo"]}  # 100 steps
landsat = {"dataset":146, "p_init":5, "p_targ":100, "p_test":200, "seed":0, "prefer":["ucimlrepo"]}  # 100 steps
vowels = {"dataset":"vowel", "p_init":5, "p_targ":15, "p_test":15, "seed":0, "prefer":["openml"], "version":2}  # 100 steps
# iris = {"dataset":"iris", "p_init":2, "p_targ":10, "p_test":10, "seed":0, "prefer":["sklearn"]}  # 50 steps
# wine = {"dataset":"wine", "p_init":2, "p_targ":10, "p_test":10, "seed":0, "prefer":["sklearn"]}  # 50 steps


def main(l=1):
    n_runs = 50
    initial_steps = 50
    branch_steps = 50
    n_samples = 10000

    prob = UCIClassificationProblem(**vowels)
    weight = [1, 1, 1, 1, 1, 1, 1, 1, 50, 50, 1]
    
    # Run branching experiments
    for i in trange(n_runs):
        # Entropy with entropy and weighted entropy branches
        model1 = RandomForestClassifierWrapper(n_estimators=l, random_state=i)
        model2 = RandomForestClassifierWrapper(n_estimators=l, random_state=i)
        strat1 = ClfEntropyReduction(model1)
        strat2 = ClfWeightedEntropyReduction(model2, weight=weight)

        run_active_learning_with_branching(prob, strat1, strat2, initial_steps, branch_steps, weight, n_samples)
        
        # Weighted entropy with entropy and weighted entropy branches
        model3 = RandomForestClassifierWrapper(n_estimators=l, random_state=i)
        model4 = RandomForestClassifierWrapper(n_estimators=l, random_state=i)
        strat3 = ClfWeightedEntropyReduction(model3, weight=weight)
        strat4 = ClfEntropyReduction(model4)
        
        run_active_learning_with_branching(prob, strat3, strat4, initial_steps, branch_steps, weight, n_samples)

    # Create visualization
    branching_configs = [
        {
            'initial': strat1, 
            'branches': [strat1, strat2], 
            'initial_steps': initial_steps, 
            'branch_steps': branch_steps
        },
        {
            'initial': strat3, 
            'branches': [strat4, strat3], 
            'initial_steps': initial_steps, 
            'branch_steps': branch_steps
        }
    ]
    
    create_branching_metric_comparison_plots(prob, branching_configs)

if __name__ == "__main__":
    for l in [1000]:
        main(l=l)
