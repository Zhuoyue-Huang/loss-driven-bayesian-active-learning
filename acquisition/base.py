from types import SimpleNamespace
import torch
import numpy as np
import h5py

from acquisition.utils.regression import suggest_initial_lengthscale_noise_and_mean
from acquisition.paths import ensure_checkpoint_dir, ensure_results_dir

def _append_observation(X, y, x_new, y_new):
    """Append a newly acquired observation while preserving 1D and 2D feature layouts."""
    if X.ndim == 1:
        X_next = np.append(X, x_new)
    else:
        X_next = np.vstack([X, np.asarray(x_new).reshape(1, -1)])
    y_next = np.append(y, y_new)
    return X_next, y_next


def run_active_learning(problem, strategy, n_steps: int, n_samples: int = 100):
    """
    Generic loop: fit → score pool → select → acquire → repeat.
    Returns a history list of dicts with X, y, scores, selected…
    """
    X, y = problem.X0.copy(), problem.y0.copy()

    for i in range(n_steps):
        scores = strategy.score(X, y, problem.X_pool, problem.X_targ, n_samples=n_samples, is_weight=problem.targ_weight)
        idx    = strategy.select_best(scores, X, problem.X_pool)
        x_new  = problem.X_pool[idx]
        y_new  = problem.acquire(idx)

        X, y = _append_observation(X, y, x_new, y_new)

        ckpt = {
            "model_state":   strategy.model.get_model_state(),
            "idx":          idx,
            "rng_state":     strategy.model._rng,
        }
        froot = ensure_checkpoint_dir(problem.task, problem.name, strategy.model.identifier)
        torch.save(ckpt, froot / f"{strategy.name}_iter{i}.pth")

def run_active_learning_with_eval(problem, strategy, n_steps: int, weight: np.ndarray, n_samples: int = 100, viz: bool = True):
    """
    Run active learning with evaluation.
    Returns a dictionary containing visualization data.
    """
    X, y = problem.X0.copy(), problem.y0.copy()
    pred_probs_history = []
    if viz:
        viz_steps = np.linspace(n_steps // 5, n_steps - 1, 5, dtype=int)
        viz_data = {
            'steps': [],
            'scores': [],
            'X_data': [],
            'y_data': [],
            'model_state': [],
            'rng_state': [],
        }

    for i in range(n_steps):
        scores = strategy.score(X, y, problem.X_pool, problem.X_targ, n_samples=n_samples, is_weight=problem.targ_weight)
        idx = strategy.select_best(scores, X, problem.X_pool)
        x_new  = problem.X_pool[idx]
        y_new  = problem.acquire(idx)

        # Record visualization data every 5 steps
        if viz and i in viz_steps:
            viz_data['steps'].append(i)
            viz_data['scores'].append(scores.copy())
            viz_data['X_data'].append(X.copy())
            viz_data['y_data'].append(y.copy())
            viz_data['model_state'].append(strategy.model.get_model_state())
            viz_data['rng_state'].append(strategy.model._rng)

        X, y = _append_observation(X, y, x_new, y_new)
        strategy.model.fit(X, y)
        pred_probs_history.append(strategy.model.predict_proba(problem.X_test))

    froot = ensure_results_dir(problem.task, problem.name, strategy.model.identifier, "eval")
    file_path = froot / f"{strategy.name}_{'_'.join(map(str, weight))}.h5"
    pred_probs_array = np.array(pred_probs_history)

    with h5py.File(file_path, 'a') as f:
        y_test_current = np.asarray(problem.y_test)
        if 'y_test' not in f:
            # Create dataset: shape [n_runs, n_steps, n_test_samples, num_classes]
            f.create_dataset('pred_test_probs', data=[pred_probs_array], 
                           maxshape=(None, pred_probs_array.shape[0], pred_probs_array.shape[1], pred_probs_array.shape[2]))
            f.create_dataset('y_test', data=[y_test_current], maxshape=(None, y_test_current.shape[0]))
            f.create_dataset('y_acquired', data=[y], maxshape=(None, len(y)))
        else:
            # Append new run
            dset = f['pred_test_probs']
            current_runs = dset.shape[0]
            dset.resize(current_runs + 1, axis=0)
            dset[current_runs] = pred_probs_array

            y_test_ds = f['y_test']
            if y_test_ds.ndim == 1:
                existing = y_test_ds[()]
                del f['y_test']
                y_test_ds = f.create_dataset(
                    'y_test', data=[existing], maxshape=(None, existing.shape[0])
                )
            if y_test_ds.shape[1] != y_test_current.shape[0]:
                raise ValueError(
                    f"Test target length mismatch for {strategy.name}: "
                    f"stored={y_test_ds.shape[1]}, current={y_test_current.shape[0]}"
                )
            y_test_ds.resize(y_test_ds.shape[0] + 1, axis=0)
            y_test_ds[-1] = y_test_current

            dset = f['y_acquired']
            current_runs = dset.shape[0]
            dset.resize(current_runs + 1, axis=0)
            dset[current_runs] = y

    if viz:
        return viz_data

def run_active_learning_with_branching(problem, initial_strategy, branch_strategy, 
                                     initial_steps: int, branch_steps: int, 
                                     weight: np.ndarray, n_samples: int = 100):
    """
    Run active learning with strategy branching.
    
    First runs initial_strategy for initial_steps, then creates two branches:
    - Branch 1: continues with initial_strategy for branch_steps more steps
    - Branch 2: switches to branch_strategy for branch_steps steps
    
    Args:
        problem: The active learning problem
        initial_strategy: Strategy to use for initial steps and branch 1
        branch_strategy: Strategy to use for branch 2
        initial_steps: Number of steps to run initial strategy before branching
        branch_steps: Number of steps to run each branch
        weight: Weight array for evaluation
        n_samples: Number of samples for scoring
    """
    # Phase 1: Run initial strategy for initial_steps
    X, y = problem.X0.copy(), problem.y0.copy()
    initial_pred_probs = []
    
    for i in range(initial_steps):
        scores = initial_strategy.score(X, y, problem.X_pool, problem.X_targ, 
                                       n_samples=n_samples, is_weight=problem.targ_weight)
        idx = initial_strategy.select_best(scores, X, problem.X_pool)
        x_new = problem.X_pool[idx]
        y_new = problem.acquire(idx)

        X, y = _append_observation(X, y, x_new, y_new)
        initial_strategy.model.fit(X, y)
        initial_pred_probs.append(initial_strategy.model.predict_proba(problem.X_test))
    
    # Phase 2: Create two branches from current state
    # Branch 1: Continue with initial strategy
    X_branch1, y_branch1 = X.copy(), y.copy()
    branch1_pred_probs = []
    
    for i in range(branch_steps):
        scores = initial_strategy.score(X_branch1, y_branch1, problem.X_pool, problem.X_targ, 
                                       n_samples=n_samples, is_weight=problem.targ_weight)
        idx = initial_strategy.select_best(scores, X_branch1, problem.X_pool)
        x_new = problem.X_pool[idx]
        y_new = problem.acquire(idx)

        X_branch1, y_branch1 = _append_observation(X_branch1, y_branch1, x_new, y_new)
        initial_strategy.model.fit(X_branch1, y_branch1)
        branch1_pred_probs.append(initial_strategy.model.predict_proba(problem.X_test))
    
    # Branch 2: Switch to branch strategy
    X_branch2, y_branch2 = X.copy(), y.copy()
    branch2_pred_probs = []
    
    # Initialize branch strategy with current data
    branch_strategy.model.fit(X_branch2, y_branch2)
    
    for i in range(branch_steps):
        scores = branch_strategy.score(X_branch2, y_branch2, problem.X_pool, problem.X_targ, 
                                      n_samples=n_samples, is_weight=problem.targ_weight)
        idx = branch_strategy.select_best(scores, X_branch2, problem.X_pool)
        x_new = problem.X_pool[idx]
        y_new = problem.acquire(idx)

        X_branch2, y_branch2 = _append_observation(X_branch2, y_branch2, x_new, y_new)
        branch_strategy.model.fit(X_branch2, y_branch2)
        branch2_pred_probs.append(branch_strategy.model.predict_proba(problem.X_test))
    
    # Combine results for each branch
    branch1_all_probs = np.array(initial_pred_probs + branch1_pred_probs)
    branch2_all_probs = np.array(initial_pred_probs + branch2_pred_probs)
    
    # Save results with descriptive filenames
    froot = ensure_results_dir(problem.task, problem.name, initial_strategy.model.identifier, "branching")
    
    # Simplified naming: initial-branch format
    branch1_name = f"{initial_strategy.name}-{initial_strategy.name}"
    file_path_1 = froot / f"{branch1_name}_{initial_steps}_{branch_steps}_{'_'.join(map(str, weight))}.h5"
    
    branch2_name = f"{initial_strategy.name}-{branch_strategy.name}"
    file_path_2 = froot / f"{branch2_name}_{initial_steps}_{branch_steps}_{'_'.join(map(str, weight))}.h5"
    
    def save_branch_results(file_path, pred_probs_array, y_acquired, is_continued):
        with h5py.File(file_path, 'a') as f:
            y_test_current = np.asarray(problem.y_test)
            if 'y_test' not in f:
                f.create_dataset('pred_test_probs', data=[pred_probs_array], 
                               maxshape=(None, pred_probs_array.shape[0], pred_probs_array.shape[1], pred_probs_array.shape[2]))
                f.create_dataset('y_test', data=[y_test_current], maxshape=(None, y_test_current.shape[0]))
                f.create_dataset('y_acquired', data=[y_acquired], maxshape=(None, len(y_acquired)))
                # Store minimal metadata
                f.attrs['is_continued'] = is_continued
                f.attrs['initial_steps'] = initial_steps
                f.attrs['branch_steps'] = branch_steps
            else:
                # Append new run
                dset = f['pred_test_probs']
                current_runs = dset.shape[0]
                dset.resize(current_runs + 1, axis=0)
                dset[current_runs] = pred_probs_array

                y_test_ds = f['y_test']
                if y_test_ds.ndim == 1:
                    existing = y_test_ds[()]
                    del f['y_test']
                    y_test_ds = f.create_dataset(
                        'y_test', data=[existing], maxshape=(None, existing.shape[0])
                    )
                if y_test_ds.shape[1] != y_test_current.shape[0]:
                    raise ValueError(
                        f"Test target length mismatch for {file_path.name}: "
                        f"stored={y_test_ds.shape[1]}, current={y_test_current.shape[0]}"
                    )
                y_test_ds.resize(y_test_ds.shape[0] + 1, axis=0)
                y_test_ds[-1] = y_test_current

                dset = f['y_acquired']
                current_runs = dset.shape[0]
                dset.resize(current_runs + 1, axis=0)
                dset[current_runs] = y_acquired
    
    # Save both branches
    save_branch_results(file_path_1, branch1_all_probs, y_branch1, True)   # continued
    save_branch_results(file_path_2, branch2_all_probs, y_branch2, False)  # switched

def run_active_learning_with_eval_reg(problem, strategy, n_steps: int, n_samples: int = 100, seed: int = 0, viz: bool = True):
    """
    Run active learning with evaluation for regression.
    Returns a dictionary containing visualization data.
    """
    X, y = problem.X0.copy(), problem.y0.copy()
    
    # Check model
    is_gp_model = "gp" in strategy.model.identifier.lower()
    is_rf_model = "rf" in strategy.model.identifier.lower()
    
    # Unified prediction history - store everything as samples
    pred_history = []

    if viz:
        viz_steps = np.linspace(n_steps // 5, n_steps - 1, 5, dtype=int)
        viz_data = {
            'steps': [],
            'scores': [],
            'X_data': [],
            'y_data': [],
            'model_state': [],
            'rng_state': [],
        }

    for i in range(n_steps):
        scores = strategy.score(X, y, problem.X_pool, problem.X_targ, n_samples=n_samples, std=0.5, is_weight=problem.targ_weight,
                                seed=seed, is_rf_model=is_rf_model, is_gp_model=is_gp_model)
        idx    = strategy.select_best(scores, X, problem.X_pool)
        x_new  = problem.X_pool[idx]
        y_new  = problem.acquire(idx)

        if viz and i in viz_steps:
            viz_data['steps'].append(i)
            viz_data['scores'].append(scores.copy())
            viz_data['X_data'].append(X.copy())
            viz_data['y_data'].append(y.copy())
            viz_data['model_state'].append(strategy.model.get_model_state())
            viz_data['rng_state'].append(strategy.model._rng)

        X, y = _append_observation(X, y, x_new, y_new)
        strategy.model.fit(X, y)

        if is_gp_model:
            y_pred_mean, y_pred_std = strategy.model.predict(problem.X_test, return_std=True)
            pred_data = np.stack([
                y_pred_mean,
                y_pred_std,
            ], axis=1)
        else:
            samples = strategy.model.predict_samples(problem.X_test, n_samples=n_samples, std=0.5)
            pred_data = samples

        pred_history.append(pred_data)

    froot = ensure_results_dir(problem.task, problem.name, strategy.model.identifier, "eval")
    file_path = froot / f"{strategy.name}.h5"

    pred_array = np.array(pred_history)

    with h5py.File(file_path, 'a') as f:
        y_test_current = problem.y_test

        if 'pred_data' not in f:
            f.create_dataset(
                'pred_data', data=[pred_array],
                maxshape=(None, pred_array.shape[0], pred_array.shape[1], pred_array.shape[2])
            )
            f.create_dataset(
                'y_test', data=[y_test_current],
                maxshape=(None, y_test_current.shape[0])
            )
            f.create_dataset('y_acquired', data=[y], maxshape=(None, len(y)))
            f.attrs['is_gp_model'] = is_gp_model
            f.attrs['n_samples'] = n_samples if not is_gp_model else 2  # 2 for mean,std
        else:
            # Append new run predictions
            dset = f['pred_data']
            current_runs = dset.shape[0]
            dset.resize(current_runs + 1, axis=0)
            dset[current_runs] = pred_array

            # Append corresponding test targets
            y_test_ds = f['y_test']
            if y_test_ds.ndim == 1:
                existing = y_test_ds[()]
                del f['y_test']
                y_test_ds = f.create_dataset(
                    'y_test', data=[existing], maxshape=(None, existing.shape[0])
                )
            if y_test_ds.shape[1] != y_test_current.shape[0]:
                raise ValueError(
                    f"Test target length mismatch for {strategy.name}: "
                    f"stored={y_test_ds.shape[1]}, current={y_test_current.shape[0]}"
                )
            y_test_ds.resize(y_test_ds.shape[0] + 1, axis=0)
            y_test_ds[-1] = y_test_current

            # Update acquired labels
            dset = f['y_acquired']
            current_runs = dset.shape[0]
            dset.resize(current_runs + 1, axis=0)
            dset[current_runs] = y

    if viz:
        return viz_data


def run_active_learning_with_eval_reg_std_update(
    problem,
    strategy,
    n_steps: int,
    n_samples: int = 100,
    seed: int = 0,
    viz: bool = True,
    info_update_every: int = 1,
    min_std: float = 1e-6,
):
    """Run active learning with periodic GP hyperparameter refresh."""

    X, y = problem.X0.copy(), problem.y0.copy()

    is_gp_model = "gp" in strategy.model.identifier.lower()
    is_rf_model = "rf" in strategy.model.identifier.lower()

    pred_history = []
    current_std = float(getattr(strategy.model, "true_std", min_std))

    if viz:
        viz_steps = np.linspace(n_steps // 5, n_steps - 1, 5, dtype=int)
        viz_data = {
            'steps': [],
            'scores': [],
            'X_data': [],
            'y_data': [],
            'model_state': [],
            'rng_state': [],
        }

    update_interval = max(1, int(info_update_every))

    def _make_refresh_problem():
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        y_arr = np.asarray(y, dtype=float)
        pool = problem.X_pool.copy() if hasattr(problem, "X_pool") else np.empty((0, X_arr.shape[1]))
        return SimpleNamespace(
            X0=X_arr,
            X_pool=pool,
            y0=y_arr,
            X_all=X_arr,
            y_all=y_arr,
        )

    def _refresh_gp_params():
        if not hasattr(strategy.model, "reg"):
            return None
        params = suggest_initial_lengthscale_noise_and_mean(_make_refresh_problem())
        ls, sigma_n, mean_val, sigma_lin, sigma_f, kernel, info = params
        model = strategy.model
        model.mean_init = float(mean_val)
        model.true_std = float(max(sigma_n, min_std))
        if hasattr(model, "sigma_lin"):
            model.sigma_lin = float(sigma_lin)
        if hasattr(model, "sigma_f"):
            model.sigma_f = float(sigma_f)
        if hasattr(model, "nu") and info.get("nu") is not None:
            model.nu = info["nu"]
        model.ls = ls
        model.custom_kernel = kernel
        model.reg.kernel = kernel
        return model.true_std

    refreshed = _refresh_gp_params()
    if refreshed is not None:
        current_std = refreshed

    for i in range(n_steps):
        if i % update_interval == 0 and i > 0 and is_gp_model:
            refreshed = _refresh_gp_params()
            if refreshed is not None:
                current_std = max(refreshed, min_std)

        strategy.model.fit(X, y)

        scores = strategy.score(
            X,
            y,
            problem.X_pool,
            problem.X_targ,
            n_samples=n_samples,
            std=current_std,
            is_weight=problem.targ_weight,
            seed=seed,
            is_rf_model=is_rf_model,
            is_gp_model=is_gp_model,
            fitted=True,
        )
        idx = strategy.select_best(scores, X, problem.X_pool)
        x_new = problem.X_pool[idx]
        y_new = problem.acquire(idx)

        if viz and i in viz_steps:
            viz_data['steps'].append(i)
            viz_data['scores'].append(scores.copy())
            viz_data['X_data'].append(X.copy())
            viz_data['y_data'].append(y.copy())
            viz_data['model_state'].append(strategy.model.get_model_state())
            viz_data['rng_state'].append(strategy.model._rng)

        X, y = _append_observation(X, y, x_new, y_new)
        strategy.model.fit(X, y)

        if is_gp_model:
            y_pred_mean, y_pred_std = strategy.model.predict(problem.X_test, return_std=True)
            pred_data = np.stack([
                y_pred_mean,
                y_pred_std,
            ], axis=1)
        elif is_rf_model:
            samples = strategy.model.predict_samples(
                problem.X_test,
                n_samples=n_samples,
                std=current_std,
            )
            pred_data = samples
        else:
            samples = strategy.model.predict_samples(
                problem.X_test,
                n_samples=n_samples,
            )
            pred_data = samples

        pred_history.append(pred_data)

    froot = ensure_results_dir(problem.task, problem.name, strategy.model.identifier, "eval")
    file_path = froot / f"{strategy.name}.h5"

    pred_array = np.array(pred_history)

    with h5py.File(file_path, 'a') as f:
        y_test_current = problem.y_test

        if 'pred_data' not in f:
            f.create_dataset(
                'pred_data',
                data=[pred_array],
                maxshape=(None, pred_array.shape[0], pred_array.shape[1], pred_array.shape[2]),
            )
            f.create_dataset(
                'y_test', data=[y_test_current], maxshape=(None, y_test_current.shape[0])
            )
            f.create_dataset('y_acquired', data=[y], maxshape=(None, len(y)))
            f.attrs['is_gp_model'] = is_gp_model
            f.attrs['n_samples'] = n_samples if not is_gp_model else 2
        else:
            dset = f['pred_data']
            current_runs = dset.shape[0]
            dset.resize(current_runs + 1, axis=0)
            dset[current_runs] = pred_array

            y_test_ds = f['y_test']
            if y_test_ds.ndim == 1:
                existing = y_test_ds[()]
                del f['y_test']
                y_test_ds = f.create_dataset(
                    'y_test', data=[existing], maxshape=(None, existing.shape[0])
                )
            if y_test_ds.shape[1] != y_test_current.shape[0]:
                raise ValueError(
                    f"Test target length mismatch for {strategy.name}: "
                    f"stored={y_test_ds.shape[1]}, current={y_test_current.shape[0]}"
                )
            y_test_ds.resize(y_test_ds.shape[0] + 1, axis=0)
            y_test_ds[-1] = y_test_current

            dset = f['y_acquired']
            current_runs = dset.shape[0]
            dset.resize(current_runs + 1, axis=0)
            dset[current_runs] = y

    if viz:
        return viz_data
