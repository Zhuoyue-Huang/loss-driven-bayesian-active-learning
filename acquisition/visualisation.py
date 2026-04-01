import copy
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py

from scipy.stats import sem

from acquisition.utils import get_weight_identifier
from acquisition.utils.classification import compute_nll_metrics_from_saved_data, compute_accuracy_metrics_from_saved_data
from acquisition.utils.progress import trange
from acquisition.utils.regression import (
    compute_sqloss_unweighted_from_saved_data,
    compute_sqloss_weighted_from_saved_data,
    compute_linex_metrics_from_saved_data,
    _broadcast_y_test,
)
from acquisition.paths import ensure_results_dir, checkpoint_path
from data.generators import simulate_regression_labels


def _compute_sem(data, axis):
    """Compute standard error of the mean while avoiding NaNs for small samples."""
    return np.nan_to_num(sem(data, axis=axis, ddof=1, nan_policy='propagate'), nan=0.0)

def _scaler_pca_transform(X, pipe):
    if pipe is None:
        return X
    else:
        return pipe.transform(X)

def _scaler_pca_inverse_transform(X, pipe):
    if pipe is None:
        return X
    else:
        return pipe.inverse_transform(X)

def _get_clf_plot_args(problem):
    if hasattr(problem, 'plot_args'):
        plot_args = problem.plot_args
    else:
        Xpool = _scaler_pca_transform(problem.X_pool, problem.pipe)
        min = Xpool[:,:2].min() * 1.1
        max = Xpool[:,:2].max() * 1.1
        plot_args = (min, max, 150)
    return plot_args

def _get_reg_plot_args(problem, n_points=100):
    """
    Get plot arguments for regression plots (1D projection).
    For multi-dimensional inputs, always use PCA to reduce to 1D.
    """
    # Check if we have multi-dimensional data
    if hasattr(problem, 'X_pool') and problem.X_pool.ndim > 1 and problem.X_pool.shape[1] > 1:
        # Multi-dimensional case: always use PCA if available
        if hasattr(problem, 'pipe') and problem.pipe is not None:
            # Use PCA projection to first PC
            Xpool = _scaler_pca_transform(problem.X_pool, problem.pipe)
            min_pc1 = Xpool[:, 0].min() * 1.1
            max_pc1 = Xpool[:, 0].max() * 1.1
            plot_args = (min_pc1, max_pc1, n_points)
            return plot_args, problem.pipe
        else:
            # Multi-dimensional without PCA: use first dimension
            min_x = problem.X_pool[:, 0].min() * 1.1
            max_x = problem.X_pool[:, 0].max() * 1.1
            plot_args = (min_x, max_x, n_points)
            return plot_args, None
    else:
        # 1D case: use plot_args if available, otherwise derive from pool
        if hasattr(problem, 'plot_args'):
            return problem.plot_args, None
        else:
            # 1D without plot_args
            X_pool = problem.X_pool if hasattr(problem, 'X_pool') else np.array([])
            if len(X_pool) > 0:
                min_x = X_pool.min() * 1.1
                max_x = X_pool.max() * 1.1
                plot_args = (min_x, max_x, n_points)
            else:
                plot_args = (-1, 1, n_points)  # Default range
            return plot_args, None

def _get_class_colormaps(num_classes, base_cmap='hsv'):
    """
    Generate colormap names for multiple classes with equal spacing.
    
    Args:
        num_classes: Number of classes to generate colors for
        base_cmap: Base colormap to use for sampling colors
        
    Returns:
        List of colormap names or color values
    """
    if num_classes <= 3:
        # Use the original hard-coded colormaps for small number of classes
        return ["Oranges", "Blues", "Greens", "Reds", "Purples"][:num_classes]
    else:
        # Sample colors with equal spacing from a colormap
        cmap = cm.get_cmap(base_cmap)
        colors = []
        for i in range(num_classes):
            # Sample with equal spacing, avoiding very light colors (0.8-1.0 range)
            color_val = 0.2 + (i / (num_classes - 1)) * 0.6 if num_classes > 1 else 0.5
            rgba = cmap(color_val)

            # Create a custom colormap for this class
            # Use lighter version of the color for the colormap
            light_color = tuple(min(1.0, c + 0.3) for c in rgba[:3]) + (rgba[3],)
            custom_cmap = mcolors.LinearSegmentedColormap.from_list(
                f'class_{i}', ['white', rgba, light_color], N=256
            )
            colors.append(custom_cmap)
    return colors

def _count_checkpoints(root_dir, strategy_name):
    """
    Count how many checkpoint files exist for a given problem and strategy
    
    Args:
        problem_name: Name of the problem
        strategy_name: specific strategy name to check
        
    Returns:
        int: Number of checkpoint files matching the criteria
    """
    pattern = f"{strategy_name}_iter*.pth"
    
    # Count the files
    files = list(root_dir.glob(pattern))
    return len(files)

def _load_checkpoint_into_strategy(strategy, checkpoint_path, X=None, y=None):
    """
    Load a checkpoint into a strategy
    
    Args:
        strategy: AcquisitionStrategy instance
        checkpoint_path: Path to checkpoint file
        X: Optional training data to initialize model if needed
        y: Optional training labels to initialize model if needed
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = ckpt["model_state"]

    # For GP models, initialize if needed
    if (hasattr(strategy.model, 'initialized') and 
        not strategy.model.initialized and 
        model_state and model_state.get("model_type") == "gpytorch"):
        if X is None:
            raise ValueError("Training data X is required to initialize GP model")
        try:
            strategy.model.initialize_model(X)
        except TypeError:
            strategy.model.initialize_model(X, y)
    strategy.model.load_model_state(model_state)

    # Restore RNG
    rng = ckpt["rng_state"]
    strategy.model._rng = rng
    strategy.model.restore_rng_state()

def _load_model_data_into_strategy(strategy, model_data, i, X=None, y=None):
    """
    Load model data into a strategy

    Args:
        strategy: AcquisitionStrategy instance
        checkpoint_path: Path to checkpoint file
        X: Optional training data to initialize model if needed
        y: Optional training labels to initialize model if needed
    """

    model_state = model_data["model_state"][i]

    # For GP models, initialize if needed
    if (hasattr(strategy.model, 'initialized') and 
        not strategy.model.initialized and 
        model_state and model_state.get("model_type") == "gpytorch"):
        if X is None:
            raise ValueError("Training data X is required to initialize GP model")
        try:
            strategy.model.initialize_model(X)
        except TypeError:
            strategy.model.initialize_model(X, y)
    strategy.model.load_model_state(model_state)

    # Restore RNG
    rng = model_data["rng_state"][i]
    strategy.model._rng = rng
    strategy.model.restore_rng_state()

def _reconstruct_training_set(problem, checkpoint_dir, strategy_name, target_iter):
    """
    Returns:
      X_train_i : np.ndarray of shape (N_i, D)
      y_train_i : np.ndarray of shape (N_i,)
    at iteration target_iter, by starting from (problem.X0, problem.y0)
    and replaying all intermediate "idx" from iter=0..target_iter inclusive.

    Arguments:
      - problem: your Problem object, which has .X0, .y0, .X_pool, .acquire(idx)
      - checkpoint_dir: root folder, e.g. "checkpoint/{problem.name}/"
      - strategy_name: string, e.g. strategy.name exactly as used in run_active_learning
      - target_iter: integer, 0-based iteration index.
    """
    # Start from initial labeled set
    X_train = problem.X0.copy()
    y_train = problem.y0.copy()

    # For each iter = 0..target_iter, open the file
    for t in range(target_iter):
        fname = Path(checkpoint_dir) / f"{strategy_name}_iter{t}.pth"
        ckpt = torch.load(str(fname), map_location="cpu", weights_only=False)
        idx_new = ckpt["idx"]
        # Acquire labels for those indices out of the pool:
        x_new = problem.X_pool[idx_new]
        y_new = problem.acquire(idx_new)

        if X_train.ndim == 1:
            X_train = np.append(X_train, x_new)
        else:
            X_train = np.vstack([X_train, x_new.reshape(1, -1)])
        # X_train = np.vstack([X_train, x_new.reshape(-1, problem.X0.shape[1])])
        y_train = np.append(y_train, y_new)
    return X_train, y_train

def plot_multiclass_proba_contours_with_boundary(
    ax,
    problem,
    model,
    current_inputs,
    current_labels,
    class_names=None,
    targ_show=False
):
    
    np.seterr(divide='ignore', invalid='ignore', over='ignore')

    # Project training inputs to 2D if needed
    pipe = problem.pipe
    X2d = _scaler_pca_transform(current_inputs, pipe)

    # Build grid in 2D
    plot_args = _get_clf_plot_args(problem)
    xs = np.linspace(*plot_args)
    ys = np.linspace(*plot_args)
    xx2d, yy2d = np.meshgrid(xs, ys)
    grid2d = np.column_stack([xx2d.ravel(), yy2d.ravel()])

    # Inverse-project to original space if PCA used
    grid_full = _scaler_pca_inverse_transform(grid2d, pipe)

    # Predict probabilities
    proba = model.predict_proba(grid_full)
    K = proba.shape[1]
    P = proba.reshape(xx2d.shape + (K,))
    Z = np.argmax(P, axis=2)

    # Colormap defaults
    cmaps = _get_class_colormaps(num_classes=K)
    if class_names is None:
        class_names = [f"Class {k}" for k in range(K)]

    vmin = 1.0 / K
    vmax = 1.0 + 0.2
    levels = np.arange(vmin, vmax, 0.08)

    # Filled contours per class
    for k, cmap_name in enumerate(cmaps):
        Pk = np.where(Z == k, P[..., k], np.nan)
        ax.contourf(xx2d, yy2d, Pk,
                    levels=levels, cmap=cmap_name,
                    vmin=vmin, vmax=vmax)

    # Predicted & true boundaries
    ax.contour(xx2d, yy2d, Z,
               levels=np.arange(K + 1) - 0.5,
               colors='k', linewidths=1.5)
    if hasattr(problem, 'decision_fn'):
        true_flat = problem.decision_fn(grid_full)
        Z_true = true_flat.reshape(xx2d.shape)
        ax.contour(xx2d, yy2d, Z_true,
                   levels=np.arange(K + 1) - 0.5,
                   colors='grey', linestyles='--', linewidths=1.0)

    # Scatter train points
    for k in range(K):
        mask = (current_labels == k)
        col = mcolors.to_rgba(plt.get_cmap(cmaps[k])(0.8))
        ax.scatter(X2d[mask, 0], X2d[mask, 1], s=50,
                   edgecolor='k', color=col,
                   label=f"Train {class_names[k]}")

    # Star for last acquisition
    if current_inputs.shape[0] > problem.X0.shape[0]:
        pt = X2d[-1]
        ax.scatter(pt[0], pt[1], marker='*', s=150,
                   color='gold', edgecolor='k', linewidth=1.5,
                   label="New Acquisition", zorder=5)

    # Scatter test points
    if targ_show:
        X_test = problem.X_targ
        y_test = problem.y_targ
        X2d_test = _scaler_pca_transform(X_test, pipe)
        for k in range(K):
            mask = (y_test == k)
            col = mcolors.to_rgba(plt.get_cmap(cmaps[k])(0.8))
            ax.scatter(X2d_test[mask, 0], X2d_test[mask, 1], marker='D', s=10,
                       edgecolor='k', color=col,
                       label=f"Test {class_names[k]}")

    ax.set_xlim(plot_args[:2])
    ax.set_ylim(plot_args[:2])

def plot_uncertainty_contour(ax, problem, expected_reduction):
    # Calculate local min and max for this specific plot
    local_vmin = np.min(expected_reduction)
    local_vmax = np.max(expected_reduction)
    # Determine if we have grid data (2D) or scattered (1D)
    if hasattr(problem, 'pool_args'):
        # grid case: derive bounds and coordinate mesh
        pool_args = problem.pool_args
        x_pool = np.linspace(*pool_args)
        y_pool = np.linspace(*pool_args)
        xx_pool, yy_pool = np.meshgrid(x_pool, y_pool)
        
        reduction_grid = expected_reduction.reshape(xx_pool.shape)
        contour = ax.contourf(xx_pool, yy_pool, reduction_grid,
                              levels=10, cmap='viridis',
                              vmin=local_vmin, vmax=local_vmax)

        ax.set_xlim(pool_args[:2])
        ax.set_ylim(pool_args[:2])

    else:
        # scattered case: use triangulation contour
        Xpool = np.asarray(problem.X_pool)
        # project to 2D if needed
        X2d_pool = _scaler_pca_transform(Xpool, problem.pipe)
        contour = ax.tricontourf(
                    X2d_pool[:, 0], X2d_pool[:, 1], expected_reduction,
                    levels=20, cmap='viridis',
                    vmin=local_vmin, vmax=local_vmax)
        xmin, xmax = X2d_pool[:, 0].min(), X2d_pool[:, 0].max()
        ymin, ymax = X2d_pool[:, 1].min(), X2d_pool[:, 1].max()
        # pad bounds slightly
        dx = (xmax - xmin) * 0.05
        dy = (ymax - ymin) * 0.05
        xmin, xmax = xmin - dx, xmax + dx
        ymin, ymax = ymin - dy, ymax + dy
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    return contour

def create_clf2d_acquisition_plots(problem, strategy_list, n_samples=100, targ_show=False):
    """
    Create a figure showing acquisition process; auto-PCA if features >2.
    """
    strat1, strat2 = strategy_list
    main_root = f"{problem.task}/{problem.name}/{strat1.model.identifier}"
    ckpt_root = checkpoint_path(main_root)
    plot_root = ensure_results_dir(main_root)

    train_start = len(problem.X0)
    n_steps = _count_checkpoints(ckpt_root, strat1.name)
    X_train1, y_train1 = _reconstruct_training_set(
        problem,
        ckpt_root,
        strategy_name=strat1.name,
        target_iter=n_steps
    )
    X_train2, y_train2 = _reconstruct_training_set(
        problem,
        ckpt_root,
        strategy_name=strat2.name,
        target_iter=n_steps
    )
    
    # 1) Create a 6×n_steps grid of axes:
    figsize = (4 * n_steps, 6 * 3)
    fig, axes = plt.subplots(6, n_steps, figsize=figsize)
    plt.subplots_adjust(wspace=0.4, hspace=0.2, top=0.9)

    # Add y‐labels for each block of three rows:
    axes[0, 0].set_ylabel("M1 decision boundary")
    axes[1, 0].set_ylabel("EUR by M1")
    axes[2, 0].set_ylabel("EUR by M2 on M1 decision")
    axes[3, 0].set_ylabel("M2 decision boundary")
    axes[4, 0].set_ylabel("EUR by M2")
    axes[5, 0].set_ylabel("EUR by M1 on M2 decision")

    for i in range(n_steps):
        # 2a) Find exactly two checkpoint files for iteration t:
        strat1_path = ckpt_root / f"{strat1.name}_iter{i}.pth"
        strat2_path = ckpt_root / f"{strat2.name}_iter{i}.pth"
        X_train1i = X_train1[:train_start + i]
        y_train1i = y_train1[:train_start + i]
        X_train2i = X_train2[:train_start + i]
        y_train2i = y_train2[:train_start + i]

        # 2b) Row 0
        ax_db1 = axes[0, i]
        _load_checkpoint_into_strategy(strat1, strat1_path, X_train1i)
        plot_multiclass_proba_contours_with_boundary(
            ax_db1,
            problem,
            strat1.model,
            X_train1i,
            y_train1i,
            targ_show=targ_show
        )
        ax_db1.set_title(f"Iteration {i}")

        # 2c) Row 1
        ax_a11 = axes[1, i]
        _load_checkpoint_into_strategy(strat1, strat1_path, X_train1i)
        scores11 = strat1.score(X_train1i, y_train1i, problem.X_pool, problem.X_targ, n_samples=n_samples, is_weight=problem.targ_weight, fitted=True)
        c11 = plot_uncertainty_contour(ax_a11, problem, scores11)
        divider_u = make_axes_locatable(ax_a11)
        cax_a11 = divider_u.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(c11, cax=cax_a11)

        # 2d) Row 2
        ax_a21 = axes[2, i]
        _load_checkpoint_into_strategy(strat2, strat1_path, X_train1i)
        scores21 = strat2.score(X_train1i, y_train1i, problem.X_pool, problem.X_targ, n_samples=n_samples, is_weight=problem.targ_weight, fitted=True)
        c21 = plot_uncertainty_contour(ax_a21, problem, scores21)
        divider_u = make_axes_locatable(ax_a21)
        cax_a21 = divider_u.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(c21, cax=cax_a21)

        # 2e) Row 3
        ax_db2 = axes[3, i]
        _load_checkpoint_into_strategy(strat2, strat2_path, X_train2i)
        plot_multiclass_proba_contours_with_boundary(
            ax_db2,
            problem,
            strat2.model,
            X_train2i,
            y_train2i,
            targ_show=targ_show
        )

        # 2f) Row 4
        ax_a22 = axes[4, i]
        _load_checkpoint_into_strategy(strat2, strat2_path, X_train2i)
        scores22 = strat2.score(X_train2i, y_train2i, problem.X_pool, problem.X_targ, n_samples=n_samples, is_weight=problem.targ_weight, fitted=True)
        c22 = plot_uncertainty_contour(ax_a22, problem, scores22)
        divider_u = make_axes_locatable(ax_a22)
        cax_a22 = divider_u.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(c22, cax=cax_a22)

        # 2g) Row 5
        ax_a21 = axes[5, i]
        _load_checkpoint_into_strategy(strat1, strat2_path, X_train2i)
        scores21 = strat1.score(X_train2i, y_train2i, problem.X_pool, problem.X_targ, n_samples=n_samples, is_weight=problem.targ_weight, fitted=True)
        c21 = plot_uncertainty_contour(ax_a21, problem, scores21)
        divider_u = make_axes_locatable(ax_a21)
        cax_a21 = divider_u.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(c21, cax=cax_a21)

    # Legend proxies
    classes = np.unique(y_train1)
    default_cmaps = _get_class_colormaps(num_classes=len(classes))
    proxies = [Line2D([0],[0],color='k',lw=2,linestyle='-',label='Predicted Boundary')]
    if hasattr(problem, 'decision_fn'):
        proxies.append(Line2D([0],[0],color='gray',lw=1,linestyle='--',label='True Decision'))
    for idx in classes:
        col = mcolors.to_rgba(plt.get_cmap(default_cmaps[int(idx)])(0.8))
        proxies.append(Line2D([0],[0],marker='o',color='w',markerfacecolor=col,
                               markeredgecolor='k',markersize=8,
                               label=f"Train Class {int(idx)}"))
    proxies.append(Line2D([0],[0],marker='*',color='w',markerfacecolor='gold',
                           markeredgecolor='k',markersize=12,label='New Acquisition'))
    if targ_show:
        for idx in classes:
            col = mcolors.to_rgba(plt.get_cmap(default_cmaps[int(idx)])(0.8))
            proxies.append(Line2D([0],[0],marker='D',color='w',markerfacecolor=col,
                                markeredgecolor='k',markersize=4,
                                label=f"Test Class {int(idx)}"))
    axes[0, -1].legend(handles=proxies, loc='center left', bbox_to_anchor=(1.05, 0.5))

    fig.suptitle(f"M1: entropy-based acquisition\nM2: weighted entropy-based acquisition {strat2.weight.numpy()}", fontsize=20)
    fig.savefig(plot_root / f"clf_{strat2.name}.svg", bbox_inches='tight')

def create_clf2d_acquisition_plots_with_visdata(problem, strategy_list, vis_data, num, n_samples=100, targ_show=False):
    """
    Create a figure showing acquisition process; auto-PCA if features >2.
    """
    strat1, strat2 = strategy_list
    vis_data1, vis_data2 = vis_data
    plot_root = ensure_results_dir(problem.task, problem.name, strat1.model.identifier, "eval")

    n_steps = len(vis_data1['steps'])
    
    # 1) Create a 6×n_steps grid of axes:
    figsize = (4 * n_steps, 6 * 3)
    fig, axes = plt.subplots(6, n_steps, figsize=figsize)
    plt.subplots_adjust(wspace=0.4, hspace=0.2, top=0.9)

    # Add y‐labels for each block of three rows:
    axes[0, 0].set_ylabel("M1 decision boundary")
    axes[1, 0].set_ylabel("EUR by M1")
    axes[2, 0].set_ylabel("EUR by M2 on M1 decision")
    axes[3, 0].set_ylabel("M2 decision boundary")
    axes[4, 0].set_ylabel("EUR by M2")
    axes[5, 0].set_ylabel("EUR by M1 on M2 decision")

    for i in range(n_steps):
        # 2a) Find exactly two checkpoint files for iteration t:

        X_train1i = vis_data1['X_data'][i]
        y_train1i = vis_data1['y_data'][i]
        X_train2i = vis_data2['X_data'][i]
        y_train2i = vis_data2['y_data'][i]

        # 2b) Row 0
        ax_db1 = axes[0, i]
        _load_model_data_into_strategy(strat1, vis_data1, i, X_train1i, y_train1i)
        plot_multiclass_proba_contours_with_boundary(
            ax_db1,
            problem,
            strat1.model,
            X_train1i,
            y_train1i,
            targ_show=targ_show
        )
        ax_db1.set_title(f"Iteration {vis_data1['steps'][i]}")

        # 2c) Row 1
        ax_a11 = axes[1, i]
        c11 = plot_uncertainty_contour(ax_a11, problem, vis_data1['scores'][i])
        divider_u = make_axes_locatable(ax_a11)
        cax_a11 = divider_u.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(c11, cax=cax_a11)

        # 2d) Row 2
        ax_a21 = axes[2, i]
        _load_model_data_into_strategy(strat2, vis_data1, i, X_train1i, y_train1i)
        scores21 = strat2.score(X_train1i, y_train1i, problem.X_pool, problem.X_targ, n_samples=n_samples, is_weight=problem.targ_weight, fitted=True)
        c21 = plot_uncertainty_contour(ax_a21, problem, scores21)
        divider_u = make_axes_locatable(ax_a21)
        cax_a21 = divider_u.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(c21, cax=cax_a21)

        # 2e) Row 3
        ax_db2 = axes[3, i]
        _load_model_data_into_strategy(strat2, vis_data2, i, X_train2i, y_train2i)
        plot_multiclass_proba_contours_with_boundary(
            ax_db2,
            problem,
            strat2.model,
            X_train2i,
            y_train2i,
            targ_show=targ_show
        )

        # 2f) Row 4
        ax_a22 = axes[4, i]
        c22 = plot_uncertainty_contour(ax_a22, problem, vis_data2['scores'][i])
        divider_u = make_axes_locatable(ax_a22)
        cax_a22 = divider_u.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(c22, cax=cax_a22)

        # 2g) Row 5
        ax_a21 = axes[5, i]
        _load_model_data_into_strategy(strat1, vis_data2, i, X_train2i, y_train2i)
        scores21 = strat1.score(X_train2i, y_train2i, problem.X_pool, problem.X_targ, n_samples=n_samples, is_weight=problem.targ_weight, fitted=True)
        c21 = plot_uncertainty_contour(ax_a21, problem, scores21)
        divider_u = make_axes_locatable(ax_a21)
        cax_a21 = divider_u.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(c21, cax=cax_a21)

    # Legend proxies
    classes = np.unique(y_train1i)
    default_cmaps = _get_class_colormaps(num_classes=len(classes))
    proxies = [Line2D([0],[0],color='k',lw=2,linestyle='-',label='Predicted Boundary')]
    if hasattr(problem, 'decision_fn'):
        proxies.append(Line2D([0],[0],color='gray',lw=1,linestyle='--',label='True Decision'))
    for idx in classes:
        col = mcolors.to_rgba(plt.get_cmap(default_cmaps[int(idx)])(0.8))
        proxies.append(Line2D([0],[0],marker='o',color='w',markerfacecolor=col,
                               markeredgecolor='k',markersize=8,
                               label=f"Train Class {int(idx)}"))
    proxies.append(Line2D([0],[0],marker='*',color='w',markerfacecolor='gold',
                           markeredgecolor='k',markersize=12,label='New Acquisition'))
    if targ_show:
        for idx in classes:
            col = mcolors.to_rgba(plt.get_cmap(default_cmaps[int(idx)])(0.8))
            proxies.append(Line2D([0],[0],marker='D',color='w',markerfacecolor=col,
                                markeredgecolor='k',markersize=4,
                                label=f"Test Class {int(idx)}"))
    axes[0, -1].legend(handles=proxies, loc='center left', bbox_to_anchor=(1.05, 0.5))

    fig.suptitle(f"M1: entropy-based acquisition\nM2: weighted entropy-based acquisition {strat2.weight.numpy()}", fontsize=20)
    fig.savefig(plot_root / f"clf_{strat2.name}_iter{num}.svg", bbox_inches='tight')

def create_clf2d_proportion_plots(problem, strategy_list, stat_type='median'):
    main_dir = ensure_results_dir(problem.task, problem.name, strategy_list[0].model.identifier)
    eval_dir = ensure_results_dir(problem.task, problem.name, strategy_list[0].model.identifier, "eval")

    # Read y data from h5py files instead of vis_data
    weight_str = '_'.join(map(str, strategy_list[-1].weight.cpu().numpy().tolist()))
    n_strategies = len(strategy_list)
    strategy_names = ["Random", "EPIG", "EPIG_w"][-n_strategies:]

    # Load data for all strategies
    all_y_data = []

    for strategy in strategy_list:
        with h5py.File(eval_dir / f"{strategy.name}_{weight_str}.h5", 'r') as f:
            all_y_data.append(f['y_acquired'][:])  # Shape: [n_runs, n_steps]

    fig, axes = plt.subplots(nrows=1, ncols=n_strategies, figsize=(3*n_strategies, 2.5))
    # Handle single subplot case
    if n_strategies == 1:
        axes = [axes]
    
    # Get all unique classes from initial data
    all_classes = np.unique(problem.y0)
    n_classes = len(all_classes)
    initial_size = len(problem.y0)
    # Get colormap for consistency
    default_cmaps = _get_class_colormaps(num_classes=n_classes)
    colors = [plt.get_cmap(default_cmaps[int(idx)])(0.8) for idx in all_classes]
    
    legend_handles = None
    legend_labels = None

    for strategy_idx, (y_data_all_runs, strategy) in enumerate(zip(all_y_data, strategy_list)):
        # Calculate subplot position
        ax = axes[strategy_idx]
        
        # Get dimensions
        n_steps = y_data_all_runs.shape[1] - initial_size
        n_runs = y_data_all_runs.shape[0]
        
        # Initialize proportions arrays for all runs and steps
        # Shape: [n_classes, n_steps+1, n_runs]
        class_proportions_all_runs = np.zeros((n_classes, n_steps+1, n_runs))
        
        # Calculate initial class counts from problem.y0
        initial_counts = np.array([np.sum(problem.y0 == cls) for cls in all_classes])
        
        # For each run, calculate proportions incrementally
        for run_idx in range(n_runs):
            # Start with initial counts
            current_counts = initial_counts.copy()
            
            # Calculate initial proportions (step 0)
            proportions = current_counts / np.sum(current_counts)
            class_proportions_all_runs[:, 0, run_idx] = proportions
            
            # For each acquisition step
            for step in range(1, n_steps+1):
                # Add the newly acquired label
                new_label = y_data_all_runs[run_idx, initial_size + step - 1]
                new_label_idx = np.where(all_classes == new_label)[0][0]
                current_counts[new_label_idx] += 1
                
                proportions = current_counts / np.sum(current_counts)
                class_proportions_all_runs[:, step, run_idx] = proportions

        # Plot with error bars/regions
        for cls_idx, cls in enumerate(all_classes):
            proportions_over_time = class_proportions_all_runs[cls_idx]  # Shape: [n_steps+1, n_runs]
            
            if stat_type == 'mean':
                # Compute mean and standard error across runs
                central_data = np.mean(proportions_over_time, axis=1)
                error_data = _compute_sem(proportions_over_time, axis=1)
                # Plot mean line
                ax.plot(np.arange(n_steps+1), central_data, color=colors[cls_idx], linewidth=0.75, 
                       marker='o', markersize=0.5, label=f'Class {int(cls)}')
                # Plot error region (mean ± SEM)
                ax.fill_between(np.arange(n_steps+1), central_data - error_data, central_data + error_data, 
                               color=colors[cls_idx], alpha=0.3)
            
            elif stat_type == 'median':
                # Compute median and quantile range across runs
                central_data = np.median(proportions_over_time, axis=1)
                q25 = np.percentile(proportions_over_time, 25, axis=1)
                q75 = np.percentile(proportions_over_time, 75, axis=1)
                # Plot median line
                ax.plot(np.arange(n_steps+1), central_data, color=colors[cls_idx], linewidth=0.75, 
                       marker='o', markersize=0.5, label=f'Class {int(cls)}')
                # Plot quantile range (25th to 75th percentile)
                ax.fill_between(np.arange(n_steps+1), q25, q75, color=colors[cls_idx], alpha=0.3)
            else:
                raise ValueError("stat_type must be either 'mean' or 'median'")
        
        # Formatting
        ax.set_xlabel('Acquisition Step')
        ax.set_ylabel('Class Proportion')
        ax.set_title(f'{strategy_names[strategy_idx]}')
        ax.set_ylim(0, 2 / n_classes)
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at initial proportions for reference
        for cls_idx, cls in enumerate(all_classes):
            initial_prop = initial_counts[cls_idx] / np.sum(initial_counts)
            ax.axhline(y=initial_prop, color=colors[cls_idx], 
                       linestyle='--', alpha=0.5, linewidth=1)

        if strategy_idx == n_strategies - 1:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    if legend_handles is not None and legend_labels is not None:
        axes[-1].legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1.02, 0.5))
    
    # Add overall title
    if hasattr(strategy_list[-1], 'weight'):
        weight_str_title = f"w={strategy_list[-1].weight.cpu().numpy().tolist()}"
    else:
        weight_str_title = "weighted"
    
    # Update title to reflect the statistic type
    if stat_type == 'mean':
        title_stat = "Mean ± SEM"
    else:
        title_stat = "Median ± IQR"
    
    # Create title with strategy names
    fig.suptitle(f'{title_stat} Class Proportion Evolution: ({weight_str_title})', fontsize=13, x=0.47)
    fig.tight_layout()
    plt.savefig(main_dir / f"clf_prop_{strategy_list[-1].name}_{stat_type}.svg", bbox_inches="tight")
    plt.close(fig)

def _plot_probability_contours(ax, problem, pool_probs):
    """
    Plot probability contours for pool data without any point labels.
    
    Args:
        ax: matplotlib axis
        problem: Problem object with pool data and optional transformations
        pool_probs: numpy array of shape [N_pool, num_classes] with predicted probabilities
    """
    np.seterr(divide='ignore', invalid='ignore', over='ignore')
    
    # Get pool data in 2D (transformed if needed)
    Xpool = np.asarray(problem.X_pool)
    X2d_pool = _scaler_pca_transform(Xpool, problem.pipe)
    
    K = pool_probs.shape[1]  # number of classes
    
    # Colormap defaults
    cmaps = _get_class_colormaps(num_classes=K)
    
    vmin = 1.0 / K
    vmax = 1.1
    levels = np.arange(vmin, vmax, 0.08)
    
    # Determine if we have grid data or scattered data
    if hasattr(problem, 'pool_args'):
        # Grid case: reshape probabilities to grid
        pool_args = problem.pool_args
        x_pool = np.linspace(*pool_args)
        y_pool = np.linspace(*pool_args)
        xx_pool, yy_pool = np.meshgrid(x_pool, y_pool)
        
        # Get class predictions for boundary
        Z = np.argmax(pool_probs, axis=1).reshape(xx_pool.shape)
        P = pool_probs.reshape(xx_pool.shape + (K,))
        
        # Filled contours per class
        for k, cmap_name in enumerate(cmaps):
            Pk = np.where(Z == k, P[..., k], np.nan)
            ax.contourf(xx_pool, yy_pool, Pk,
                        levels=levels, cmap=cmap_name,
                        vmin=vmin, vmax=vmax)
        
        # Predicted boundaries
        ax.contour(xx_pool, yy_pool, Z,
               levels=np.arange(K + 1) - 0.5,
               colors='k', linewidths=1.5)
        
        ax.set_xlim(pool_args[:2])
        ax.set_ylim(pool_args[:2])
        
    else:
        # Scattered case: use triangulation
        from matplotlib.tri import Triangulation
        
        # Get class predictions for boundary
        Z_pool = np.argmax(pool_probs, axis=1)
        
        # Create triangulation
        tri = Triangulation(X2d_pool[:, 0], X2d_pool[:, 1])
        
        # Filled contours per class
        for k, cmap_name in enumerate(cmaps):
            # Create mask for this class
            mask_k = (Z_pool == k)
            if np.any(mask_k):
                # Get probabilities for this class
                prob_k = pool_probs[:, k].copy()
                # Set probabilities to NaN where this class is not dominant
                prob_k[~mask_k] = np.nan
                
                ax.tricontourf(tri, prob_k,
                               levels=levels, cmap=cmap_name,
                               vmin=vmin, vmax=vmax)
        
        # Predicted boundaries using tricontour
        ax.tricontour(tri, Z_pool.astype(float),
                      levels=np.arange(K + 1) - 0.5,
                      colors='k', linewidths=1.5)
        
        # Set axis limits based on data
        xmin, xmax = X2d_pool[:, 0].min(), X2d_pool[:, 0].max()
        ymin, ymax = X2d_pool[:, 1].min(), X2d_pool[:, 1].max()
        # Pad bounds slightly
        dx = (xmax - xmin) * 0.05
        dy = (ymax - ymin) * 0.05
        xmin, xmax = xmin - dx, xmax + dx
        ymin, ymax = ymin - dy, ymax + dy
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

def _create_entropy_decomposition_plots(problem, strategy_list, n_samples=100, targ_show=False):
    """
    Create a 9x4 figure showing entropy decomposition:
    Row 0: Decision boundaries for unweighted case
    Row 1: Pool probability contour (unweighted)
    Row 2: Pool probability contour (weighted)
    Row 3: Pool entropy (unweighted)
    Row 4: Pool entropy (weighted)
    Row 5: Posterior entropy (unweighted)
    Row 6: Posterior entropy (weighted)
    Row 7: EUR (unweighted)
    Row 8: EUR (weighted)
    """
    strat1, strat2 = strategy_list  # strat1: unweighted, strat2: weighted
    main_root = f"{problem.task}/{problem.name}/{strat1.model.identifier}"
    ckpt_root = checkpoint_path(main_root)
    plot_root = ensure_results_dir(main_root)

    train_start = len(problem.X0)
    n_steps = min(4, _count_checkpoints(ckpt_root, strat1.name))  # Limit to 4 steps for 4 columns
    
    # Reconstruct training sets
    X_train1, y_train1 = _reconstruct_training_set(problem, ckpt_root, strat1.name, n_steps)

    # Initialize table to collect MSE values
    mse_table = {
        'step': [],
        'entropy_pool_mse': [],
        'entropy_targ_mse': [],
        'entropy_joint_mse': [],
        'posterior_entropy_mse': []
    }

    # Create 9x4 grid
    figsize = (4 * n_steps, 9 * 3)
    fig, axes = plt.subplots(9, n_steps, figsize=figsize)
    plt.subplots_adjust(wspace=0.4, hspace=0.3, top=0.9)

    # Row labels
    row_labels = [
        "M1 decision boundary",
        "M1 pool probability",
        "M2 pool probability",
        "M1 prior entropy",
        "M2 prior entropy",
        "M1 posterior entropy",
        "M2 posterior entropy",
        "M1 EUR",
        "M2 EUR"
    ]
    
    for i, label in enumerate(row_labels):
        axes[i, 0].set_ylabel(label)

    for step in range(n_steps):
        # Get training data for this step
        X_train1_step = X_train1[:train_start + step]
        y_train1_step = y_train1[:train_start + step]

        # Load checkpoints
        strat1_path = ckpt_root / f"{strat1.name}_iter{step}.pth"

        axes[0, step].set_title(f"Step {step}")
        # Row 0: Decision boundary (unweighted)
        ax_db = axes[0, step]
        _load_checkpoint_into_strategy(strat1, strat1_path, X_train1_step)
        plot_multiclass_proba_contours_with_boundary(
            ax_db, problem, strat1.model, X_train1_step, y_train1_step, targ_show=targ_show
        )

        # Get entropy components
        _load_checkpoint_into_strategy(strat1, strat1_path, X_train1_step)
        components_unweighted = strat1._score_components(
            X_train1_step, y_train1_step, problem.X_pool, problem.X_targ, n_samples=n_samples, is_weight=problem.targ_weight, fitted=True
        )
        _load_checkpoint_into_strategy(strat2, strat1_path, X_train1_step)
        components_weighted = strat2._score_components(
            X_train1_step, y_train1_step, problem.X_pool, problem.X_targ, n_samples=n_samples, is_weight=problem.targ_weight, fitted=True
        )

        # Collect MSE values for table
        mse_table['step'].append(step)
        mse_table['entropy_pool_mse'].append(((components_unweighted['entropy_pool']-components_weighted['entropy_pool'])**2).mean())
        mse_table['entropy_targ_mse'].append(((components_unweighted['entropy_targ']-components_weighted['entropy_targ'])**2).mean())
        mse_table['entropy_joint_mse'].append(((components_unweighted['entropy_joint']-components_weighted['entropy_joint'])**2).mean())
        mse_table['posterior_entropy_mse'].append(((components_unweighted['posterior_entropy']-components_weighted['posterior_entropy'])**2).mean())

        # Row 1: Pool probability contour (unweighted) - show max probability
        ax_prob1 = axes[1, step]
        _plot_probability_contours(ax_prob1, problem, components_unweighted['prob_pool'])
        divider = make_axes_locatable(ax_prob1)

        # Row 2: Pool probability contour (weighted) - show max probability
        ax_prob2 = axes[2, step]
        _plot_probability_contours(ax_prob2, problem, components_weighted['prob_pool'])
        divider = make_axes_locatable(ax_prob2)
        
        # Row 3: Pool entropy (unweighted)
        ax_pe1 = axes[3, step]
        c_pe1 = plot_uncertainty_contour(ax_pe1, problem, components_unweighted['entropy_pool'])
        divider = make_axes_locatable(ax_pe1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(c_pe1, cax=cax)

        # Row 4: Pool entropy (weighted)
        ax_pe2 = axes[4, step]
        c_pe2 = plot_uncertainty_contour(ax_pe2, problem, components_weighted['entropy_pool'])
        divider = make_axes_locatable(ax_pe2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(c_pe2, cax=cax)

        # Row 5: Posterior entropy (unweighted)
        ax_post1 = axes[5, step]
        c_post1 = plot_uncertainty_contour(ax_post1, problem, components_unweighted['posterior_entropy'])
        divider = make_axes_locatable(ax_post1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(c_post1, cax=cax)

        # Row 6: Posterior entropy (weighted)
        ax_post2 = axes[6, step]
        c_post2 = plot_uncertainty_contour(ax_post2, problem, components_weighted['posterior_entropy'])
        divider = make_axes_locatable(ax_post2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(c_post2, cax=cax)

        # Row 7: EUR (unweighted) - pool entropy + posterior entropy
        ax_eur1 = axes[7, step]
        eur_unweighted = components_unweighted['entropy_pool'] - components_unweighted['posterior_entropy']
        c_eur1 = plot_uncertainty_contour(ax_eur1, problem, eur_unweighted)
        divider = make_axes_locatable(ax_eur1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(c_eur1, cax=cax)

        # Row 8: EUR (weighted) - pool entropy + posterior entropy
        ax_eur2 = axes[8, step]
        eur_weighted = components_weighted['entropy_pool'] - components_weighted['posterior_entropy']
        c_eur2 = plot_uncertainty_contour(ax_eur2, problem, eur_weighted)
        divider = make_axes_locatable(ax_eur2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(c_eur2, cax=cax)

    # Print MSE comparison table at the end
    print("\nMSE Comparison Table (Unweighted vs Weighted):")
    print("=" * 80)
    print(f"{'Step':<6} {'Entropy Pool':<15} {'Entropy Targ':<15} {'Entropy Joint':<15} {'Posterior Entropy':<18}")
    print("-" * 80)
    for i in range(len(mse_table['step'])):
        print(f"{mse_table['step'][i]:<6} "
              f"{mse_table['entropy_pool_mse'][i]:<15.6f} "
              f"{mse_table['entropy_targ_mse'][i]:<15.6f} "
              f"{mse_table['entropy_joint_mse'][i]:<15.6f} "
              f"{mse_table['posterior_entropy_mse'][i]:<18.6f}")
    print("=" * 80)

    # Add legend (similar to existing function)
    classes = np.unique(y_train1)
    default_cmaps = _get_class_colormaps(num_classes=len(classes))
    proxies = [Line2D([0],[0],color='k',lw=2,linestyle='-',label='Predicted Boundary')]
    if hasattr(problem, 'decision_fn'):
        proxies.append(Line2D([0],[0],color='gray',lw=1,linestyle='--',label='True Decision'))
    
    for idx in classes:
        col = mcolors.to_rgba(plt.get_cmap(default_cmaps[int(idx)])(0.8))
        proxies.append(Line2D([0],[0],marker='o',color='w',markerfacecolor=col,
                               markeredgecolor='k',markersize=8,
                               label=f"Train Class {int(idx)}"))
    
    axes[0, -1].legend(handles=proxies, loc='center left', bbox_to_anchor=(1.05, 0.5))

    fig.suptitle(f"Entropy Decomposition: Unweighted vs Weighted (w={strat2.weight.numpy()})", fontsize=16)
    fig.savefig(plot_root / f"clf_{strat2.name}.svg", bbox_inches='tight')
    plt.close(fig)

def create_metric_comparison_plots(problem, strategy_list, metric_list=None, stat_type='median'):
    """
    Create metric comparison plots using saved evaluation files.
    
    Args:
        problem: Problem instance
        strategy_list: List of strategies to compare
        metric_list: Metrics to plot. Defaults to ["nll"].
        stat_type: str, either 'mean' (mean ± SEM) or 'median' (median with quantile range)
    """
    if metric_list is None:
        metric_list = ['nll']
    if not metric_list:
        raise ValueError("metric_list must contain at least one metric")
    
    main_dir = ensure_results_dir(problem.task, problem.name, strategy_list[0].model.identifier)
    eval_dir = ensure_results_dir(problem.task, problem.name, strategy_list[0].model.identifier, "eval")

    # Get weight string from the last strategy (assumed to be weighted)
    weight_str = '_'.join(map(str, strategy_list[-1].weight.cpu().numpy().tolist()))
    strategy_names = ["Random", "EPIG", "EPIG_w"][-len(strategy_list):]

    # Load all prediction data
    all_pred_data = []
    y_test_runs = None
    for strategy in strategy_list:
        with h5py.File(eval_dir / f"{strategy.name}_{weight_str}.h5", 'r') as f:
            all_pred_data.append(f['pred_test_probs'][:])
            current_y_test = np.asarray(f['y_test'][:])
            if y_test_runs is None:
                y_test_runs = current_y_test
            else:
                if current_y_test.shape != y_test_runs.shape:
                    raise ValueError(
                        "Mismatch in stored y_test shapes across strategies; "
                        "ensure all strategies were evaluated with identical runs."
                    )
                if not np.array_equal(current_y_test, y_test_runs):
                    raise ValueError(
                        "Stored y_test values differ between strategies; "
                        "ensure identical problem initialisation per run."
                    )
    
    # Get weight from the weighted strategy
    weight = strategy_list[-1].weight.cpu().numpy()
    # Compute metrics efficiently
    metric_data = {}
    if 'nll' in metric_list:
        all_nll_data, all_nll_w_data = compute_nll_metrics_from_saved_data(all_pred_data, y_test_runs, weight)
        metric_data['nll'] = [all_nll_data, all_nll_w_data]
    if 'acc' in metric_list:
        all_acc_data, all_acc_w_data = compute_accuracy_metrics_from_saved_data(all_pred_data, y_test_runs, weight)
        metric_data['acc'] = [all_acc_data, all_acc_w_data]

    # Plot metrics
    n_metrics = len(metric_list)
    fig, axes = plt.subplots(nrows=n_metrics, ncols=2, figsize=(6, 2.5*n_metrics))
    if n_metrics == 1:
        axes = axes.reshape(1, -1)
    
    colors = [plt.get_cmap(_get_class_colormaps(num_classes=len(strategy_list))[i])(0.7) 
              for i in range(len(strategy_list))]
    
    for metric_idx, metric_name in enumerate(metric_list):
        data_groups = metric_data[metric_name]
        plot_names = [f"{metric_name.upper()}", f"Weighted {metric_name.upper()}"]
        
        for plot_idx, (data_group, plot_name) in enumerate(zip(data_groups, plot_names)):
            ax = axes[metric_idx, plot_idx]
            n_steps = data_group[0].shape[1]

            for (data, strategy_name, color) in zip(data_group, strategy_names, colors):
                if stat_type == 'mean':
                    central_data = np.mean(data, axis=0)
                    error_data = _compute_sem(data, axis=0)
                    ax.fill_between(np.arange(n_steps), central_data - error_data, 
                                   central_data + error_data, alpha=0.3, color=color)
                    print(
                        f"{strategy_name} {plot_name} mean at final step: "
                        f"{central_data[-1]:.4f} \\SEM{{{error_data[-1]:.4f}}}"
                    )
                elif stat_type == 'median':
                    central_data = np.median(data, axis=0)
                    q25 = np.percentile(data, 25, axis=0)
                    q75 = np.percentile(data, 75, axis=0)
                    ax.fill_between(np.arange(n_steps), q25, q75, alpha=0.3, color=color)
                    print(f"{strategy_name} {plot_name} median at final step: {central_data[-1]:.4f} ({q25[-1]:.4f}-{q75[-1]:.4f})")
                else:
                    raise ValueError("stat_type must be either 'mean' or 'median'")
                
                ax.plot(np.arange(n_steps), central_data, marker='o', markersize=0.5, 
                       label=strategy_name, color=color, linewidth=0.75)
            
            ax.set_xlabel("Acquisition Step")
            ax.set_ylabel(plot_name)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
    
    # Update title to reflect the statistic type
    if stat_type == 'mean':
        title_stat = "Mean ± SEM"
    else:
        title_stat = "Median ± IQR"

    # fig.suptitle(f"{title_stat} (w={weight.tolist()})", fontsize=14)
    plt.tight_layout()
    plt.savefig(main_dir / f"clf_metric_{strategy_list[-1].name}_{stat_type}.svg", bbox_inches="tight")
    plt.close(fig)

def create_branching_metric_comparison_plots(problem, branching_configs, 
                                           metric_list=['nll', 'acc'], stat_type='median'):
    """
    Create metric comparison plots for branching strategies with proper color scheme.
    Shows complete routes for continued strategies and only branching portion for switched strategies.
    
    Args:
        problem: Problem instance
        branching_configs: List of dictionaries with branching configuration:
                          [{'initial': strat1, 'branches': [strat1, strat2], 'initial_steps': 25, 'branch_steps': 10},
                           {'initial': strat2, 'branches': [strat1, strat2], 'initial_steps': 25, 'branch_steps': 10}]
        metric_list: List of metrics to plot ['nll', 'acc']
        stat_type: str, either 'mean' (mean ± SEM) or 'median' (median with quantile range)
    """
    
    # Get weight from first strategy in first config
    weight = branching_configs[0]['branches'][1].weight.cpu().numpy()
    weight_str = '_'.join(map(str, weight))
    
    main_dir = ensure_results_dir(problem.task, problem.name, branching_configs[0]['initial'].model.identifier, "branching")
    
    # Collect all strategies and create color mapping
    all_strategies = set()
    for config in branching_configs:
        all_strategies.add(config['initial'].name)
        for branch in config['branches']:
            all_strategies.add(branch.name)
    
    strategy_list = sorted(list(all_strategies))
    base_colors = {}
    colormaps = _get_class_colormaps(num_classes=len(strategy_list))
    
    for i, strategy in enumerate(strategy_list):
        base_colors[strategy] = plt.get_cmap(colormaps[i])(0.7)
    
    # Collect all data - separate continued and switched
    continued_data = []
    continued_names = []
    continued_colors = []
    
    switched_data = []
    switched_names = []
    switched_colors = []
    switched_initial_steps = []
    
    # Track unique combinations to avoid duplicates
    processed_combinations = set()
    
    for config in branching_configs:
        initial_strategy = config['initial']
        initial_steps = config['initial_steps']
        branch_steps = config['branch_steps']
        
        for branch_strategy in config['branches']:
            is_continued = (initial_strategy.name == branch_strategy.name)
            
            # Create unique identifier to avoid duplicates
            combo_id = f"{initial_strategy.name}-{branch_strategy.name}-{initial_steps}-{branch_steps}"
            if combo_id in processed_combinations:
                continue
            processed_combinations.add(combo_id)
            
            # Construct filename
            branch_name = f"{initial_strategy.name}-{branch_strategy.name}"
            file_path = main_dir / f"{branch_name}_{initial_steps}_{branch_steps}_{weight_str}.h5"
            
            try:
                with h5py.File(file_path, 'r') as f:
                    pred_probs = f['pred_test_probs'][:]
                    y_test = f['y_test'][:]  # Same for all files
                
                # Color assignment: lighter for switched strategies
                base_color = base_colors[branch_strategy.name]
                
                if is_continued:
                    # Full route for continued strategies
                    continued_data.append(pred_probs)
                    continued_names.append(f"{initial_strategy.name} (full route)")
                    continued_colors.append(base_color)
                else:
                    # Only branching portion for switched strategies
                    branching_portion = pred_probs[:, initial_steps:]
                    switched_data.append(branching_portion)
                    switched_names.append(f"{initial_strategy.name}→{branch_strategy.name}")
                    switched_initial_steps.append(initial_steps)
                    
                    # Lighter color for switched strategies
                    rgba = list(base_color)
                    if len(rgba) == 3:
                        rgba.append(1.0)  # Add alpha if not present
                    # Make lighter by interpolating with white
                    lighter_rgb = tuple(min(1.0, c + 0.3) for c in rgba[:3])
                    color = lighter_rgb + (rgba[3],)
                    switched_colors.append(color)
                    
            except FileNotFoundError:
                print(f"Warning: File not found: {file_path}")
                continue
    
    if not continued_data and not switched_data:
        print("No data files found!")
        return
    
    # Compute metrics for continued strategies (full route)
    continued_metric_data = {}
    if continued_data:
        if 'nll' in metric_list:
            all_nll_data, all_nll_w_data = compute_nll_metrics_from_saved_data(continued_data, y_test, weight)
            continued_metric_data['nll'] = [all_nll_data, all_nll_w_data]
        if 'acc' in metric_list:
            all_acc_data, all_acc_w_data = compute_accuracy_metrics_from_saved_data(continued_data, y_test, weight)
            continued_metric_data['acc'] = [all_acc_data, all_acc_w_data]
    
    # Compute metrics for switched strategies (branching portion only)
    switched_metric_data = {}
    if switched_data:
        if 'nll' in metric_list:
            all_nll_data, all_nll_w_data = compute_nll_metrics_from_saved_data(switched_data, y_test, weight)
            switched_metric_data['nll'] = [all_nll_data, all_nll_w_data]
        if 'acc' in metric_list:
            all_acc_data, all_acc_w_data = compute_accuracy_metrics_from_saved_data(switched_data, y_test, weight)
            switched_metric_data['acc'] = [all_acc_data, all_acc_w_data]

    # Plot metrics with much wider figure for long legend names
    n_metrics = len(metric_list)
    fig, axes = plt.subplots(nrows=n_metrics, ncols=2, figsize=(22, 5*n_metrics))
    if n_metrics == 1:
        axes = axes.reshape(1, -1)
    
    for metric_idx, metric_name in enumerate(metric_list):
        plot_names = [f"{metric_name.upper()}", f"Weighted {metric_name.upper()}"]
        
        for plot_idx, plot_name in enumerate(plot_names):
            ax = axes[metric_idx, plot_idx]
            
            # Plot continued strategies (full route)
            if continued_data:
                continued_data_groups = continued_metric_data[metric_name]
                data_group = continued_data_groups[plot_idx]
                n_steps_full = data_group[0].shape[1]
                
                for (data, display_name, color) in zip(data_group, continued_names, continued_colors):
                    if stat_type == 'mean':
                        central_data = np.mean(data, axis=0)
                        error_data = _compute_sem(data, axis=0)
                        ax.fill_between(np.arange(n_steps_full), central_data - error_data, 
                                       central_data + error_data, alpha=0.3, color=color)
                    elif stat_type == 'median':
                        central_data = np.median(data, axis=0)
                        q25 = np.percentile(data, 25, axis=0)
                        q75 = np.percentile(data, 75, axis=0)
                        ax.fill_between(np.arange(n_steps_full), q25, q75, alpha=0.3, color=color)
                    else:
                        raise ValueError("stat_type must be either 'mean' or 'median'")
                    
                    ax.plot(np.arange(n_steps_full), central_data, marker='o', markersize=2.5, 
                           label=display_name, color=color, linestyle='-', linewidth=2)
            
            # Plot switched strategies (branching portion only)
            if switched_data:
                switched_data_groups = switched_metric_data[metric_name]
                data_group = switched_data_groups[plot_idx]
                
                for i, (data, display_name, color) in enumerate(zip(data_group, switched_names, switched_colors)):
                    initial_steps = switched_initial_steps[i]
                    n_branch_steps = data.shape[1]
                    
                    # X-axis starts from initial_steps for switched strategies
                    x_axis = np.arange(initial_steps, initial_steps + n_branch_steps)
                    
                    if stat_type == 'mean':
                        central_data = np.mean(data, axis=0)
                        error_data = _compute_sem(data, axis=0)
                        ax.fill_between(x_axis, central_data - error_data, 
                                       central_data + error_data, alpha=0.3, color=color)
                    elif stat_type == 'median':
                        central_data = np.median(data, axis=0)
                        q25 = np.percentile(data, 25, axis=0)
                        q75 = np.percentile(data, 75, axis=0)
                        ax.fill_between(x_axis, q25, q75, alpha=0.3, color=color)
                    else:
                        raise ValueError("stat_type must be either 'mean' or 'median'")
                    
                    ax.plot(x_axis, central_data, marker='s', markersize=2.5, 
                           label=display_name, color=color, linestyle='--', linewidth=1.5)
            
            # Add vertical line at branching point
            if switched_data:
                branching_point = switched_initial_steps[0]  # Assuming all have same initial_steps
                ax.axvline(x=branching_point, color='red', linestyle=':', alpha=0.7, linewidth=2, 
                          label='Branching Point' if plot_idx == 0 and metric_idx == 0 else "")
            
            ax.set_xlabel("Acquisition Step")
            ax.set_ylabel(plot_name)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Place legend outside with even more space and smaller font
            ax.legend(bbox_to_anchor=(1.25, 1), loc='upper left', fontsize=9, 
                     frameon=True, fancybox=True, shadow=True)
    
    # Update title to reflect the statistic type
    if stat_type == 'mean':
        title_stat = "Mean ± SEM"
    else:
        title_stat = "Median ± IQR"

    # Get initial_steps for title (from first switched strategy if available, otherwise from config)
    if switched_initial_steps:
        initial_steps = switched_initial_steps[0]
    else:
        initial_steps = branching_configs[0]['initial_steps']
    
    fig.suptitle(f"Branching Strategy Comparison - {title_stat}\n"
                f"Full routes (solid) vs Branching after step {initial_steps} (dashed)\n"
                f"Weight: {weight.tolist()}", fontsize=14)
    
    # Adjust layout to accommodate much wider legend space
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # Make even more room for legend
    
    # Save with higher DPI for better quality
    plt.savefig(main_dir / f"branching_metric_comparison_{stat_type}.svg", 
               bbox_inches="tight", dpi=300, facecolor='white')
    plt.close(fig)

def plot_regression_with_uncertainty(
    ax,
    problem,
    model,
    current_inputs,
    current_labels,
    show_legend=True,
    highlight_last_acquisition=True,
    model_color="C0",
    point_color="#003169",
    new_point_color="gold",
    true_color="gray",
    uncertainty_alpha=0.1,
    uncertainty_std_scale=1.0,
    model_linewidth=1.8,
    true_linewidth=1.0,
    point_markersize=7,
    new_point_markersize=10,
):
    # Get plot arguments and PCA pipe (if any)
    plot_args, pipe = _get_reg_plot_args(problem)
    plot_inputs_1d = np.linspace(*plot_args)

    # Handle multi-dimensional inputs with PCA projection
    if pipe is not None:
        # Multi-dimensional case: project training inputs to first PC
        current_inputs_1d = _scaler_pca_transform(current_inputs, pipe)[:, 0]

        # Create full-dimensional plot inputs by inverse transforming
        # For visualization, we create points along the first PC direction
        # Get the number of PC components by transforming a sample
        sample_pca = _scaler_pca_transform(current_inputs[:1], pipe)
        n_components = sample_pca.shape[1]

        plot_inputs_pca = np.zeros((len(plot_inputs_1d), n_components))
        plot_inputs_pca[:, 0] = plot_inputs_1d  # Set first PC values
        # Set other components to mean values from current data
        if n_components > 1:
            current_pca = _scaler_pca_transform(current_inputs, pipe)
            for i in range(1, n_components):
                plot_inputs_pca[:, i] = np.mean(current_pca[:, i])

        plot_inputs_full = _scaler_pca_inverse_transform(plot_inputs_pca, pipe)
    else:
        # 1D case or multi-dimensional without PCA
        if current_inputs.ndim == 1:
            current_inputs_1d = current_inputs.flatten()
            plot_inputs_full = plot_inputs_1d
        else:
            current_inputs_1d = current_inputs[:, 0]
            # Multi-dimensional without PCA: fix other dimensions to mean
            plot_inputs_full = np.zeros((len(plot_inputs_1d), current_inputs.shape[1]))
            plot_inputs_full[:, 0] = plot_inputs_1d
            for i in range(1, current_inputs.shape[1]):
                plot_inputs_full[:, i] = np.mean(current_inputs[:, i])

    # Ensure model is fitted on the correct data before making predictions
    model.fit(current_inputs, current_labels)
    pred_mean_quad, pred_std_quad = model.predict(plot_inputs_full, return_std=True)

    # Store legend elements to return
    legend_elements = []

    # Plot true function if available (only for 1D case)
    if hasattr(problem, 'true_mean') and hasattr(problem, 'true_std') and pipe is None:
        true_mean, true_std = problem.true_mean, problem.true_std
        true_line, = ax.plot(
            plot_inputs_1d,
            true_mean(plot_inputs_1d),
            color=true_color,
            linestyle=":",
            linewidth=true_linewidth,
            label="True function",
        )
        ax.fill_between(
            plot_inputs_1d,
            true_mean(plot_inputs_1d) - true_std,
            true_mean(plot_inputs_1d) + true_std,
            color=true_color,
            alpha=0.08,
        )
        legend_elements.append(true_line)

    # Plot model predictions
    pred_line, = ax.plot(
        plot_inputs_1d,
        pred_mean_quad,
        color=model_color,
        linewidth=model_linewidth,
        label="Model prediction",
    )
    ax.fill_between(
        plot_inputs_1d,
        pred_mean_quad - uncertainty_std_scale * pred_std_quad,
        pred_mean_quad + uncertainty_std_scale * pred_std_quad,
        color=model_color,
        alpha=uncertainty_alpha,
    )
    legend_elements.append(pred_line)

    num_start = len(problem.X0)
    num_current = len(current_inputs)
    # Plot training points (projected to 1D if needed)
    if num_current > 1:
        if num_current == num_start or not highlight_last_acquisition:
            initial_points = ax.plot(
                current_inputs_1d,
                current_labels,
                ".",
                color=point_color,
                markersize=point_markersize,
                label="Current points",
            )
            legend_elements.append(initial_points[0])
        else:
            initial_points = ax.plot(
                current_inputs_1d[:-1],
                current_labels[:-1],
                ".",
                color=point_color,
                markersize=point_markersize,
                label="Current points",
            )
            new_point = ax.plot(
                current_inputs_1d[-1],
                current_labels[-1],
                "*",
                color=new_point_color,
                markersize=new_point_markersize,
                markeredgecolor='black',
                markeredgewidth=1,
                label="New acquisition",
            )
            legend_elements.extend([initial_points[0], new_point[0]])

    # Show legend only if requested
    if show_legend and legend_elements:
        ax.legend(handles=legend_elements)
    
    return legend_elements

def plot_reg_pair_eur(ax, problem, eur1, eur2, show_legend=True):
    # Get pool inputs for plotting - handle multi-dimensional case with PCA
    # Check if we have multi-dimensional data first
    if hasattr(problem, 'X_pool') and problem.X_pool.ndim > 1 and problem.X_pool.shape[1] > 1:
        # Multi-dimensional case: always use PCA if available
        if hasattr(problem, 'pipe') and problem.pipe is not None:
            # Use PCA projection to first PC
            pool_projected = _scaler_pca_transform(problem.X_pool, problem.pipe)
            pool_inputs_1d = pool_projected[:, 0]
        else:
            # Multi-dimensional without PCA: use first dimension
            pool_inputs_1d = problem.X_pool[:, 0]
        use_sorting = True
    else:
        # 1D case: check if we have structured pool_args
        if hasattr(problem, 'pool_args'):
            # 1D case with linspace: use existing pool_args
            pool_inputs_1d = np.linspace(*problem.pool_args)
            use_sorting = False
        else:
            # 1D case without pool_args: use X_pool directly
            pool_inputs_1d = problem.X_pool
            use_sorting = True

    # Sort by x-axis for better line plotting when needed
    if use_sorting:
        # For pool case, sort by projected values
        sort_indices = np.argsort(pool_inputs_1d)
        x_values = pool_inputs_1d[sort_indices]
        eur1_sorted = eur1[sort_indices]
        eur2_sorted = eur2[sort_indices]
    else:
        # For linspace case, already sorted
        x_values = pool_inputs_1d
        eur1_sorted = eur1
        eur2_sorted = eur2

    # Create twin axis for the second score
    ax_twin = ax.twinx()

    # Plot the first score on the left y-axis (blue)
    line1, = ax.plot(x_values, eur1_sorted, 'b-', label="Var")
    ax.tick_params(axis='y', labelcolor='b')

    # Plot the second score on the right y-axis (red)
    line2, = ax_twin.plot(x_values, eur2_sorted, 'r-', label="Var_w")
    ax_twin.tick_params(axis='y', labelcolor='r')

    # Add a legend only if requested
    if show_legend:
        lines = [line1, line2]
        ax.legend(lines, [l.get_label() for l in lines], loc='upper center')
    
    # Return the lines for external legend creation
    return line1, line2


def _append_regression_observation(X, y, x_new, y_new):
    """Append a scalar or vector observation to a regression design matrix."""
    if X.ndim == 1:
        X_next = np.append(X, x_new)
    else:
        X_next = np.vstack([X, np.asarray(x_new).reshape(1, -1)])
    y_next = np.append(y, y_new)
    return X_next, y_next


def _sample_counterfactual_regression_label(problem, x_new, rng_state):
    """
    Sample a reproducible counterfactual observation for a candidate point.

    Both branches use the same copied RNG state so the two updates are coupled by
    the same noise draw while still respecting the stochastic observation model.
    """
    branch_rng = np.random.default_rng()
    branch_rng.bit_generator.state = copy.deepcopy(rng_state)

    x_query = np.asarray(x_new)
    if x_query.ndim == 0:
        x_query = np.array([float(x_query)])
    elif x_query.ndim == 1 and np.asarray(problem.X0).ndim > 1:
        x_query = x_query.reshape(1, -1)

    observed = simulate_regression_labels(problem.true_mean(x_query), problem.true_std, rng=branch_rng)
    return float(np.ravel(observed)[0])


def _normalize_acquisition_scores(scores):
    scores = np.asarray(scores, dtype=float)
    score_min = np.min(scores)
    score_max = np.max(scores)
    if score_max - score_min <= 1e-12:
        return np.zeros_like(scores)
    return (scores - score_min) / (score_max - score_min)


def _add_response_weight_background(
    ax,
    x_min,
    x_max,
    y_min,
    y_max,
    color=None,
    direction="neutral",
    max_alpha=0.16,
    fade_power=2.2,
):
    """Add a faint response-space tint to indicate the weighting emphasis."""
    if direction == "neutral" or color is None:
        return

    rgba = np.ones((256, 2, 4), dtype=float)
    rgba[..., :3] = mcolors.to_rgb(color)
    alpha = max_alpha * np.linspace(0.0, 1.0, 256) ** fade_power
    if direction == "down":
        alpha = alpha[::-1]
    rgba[..., 3] = alpha[:, None]
    ax.imshow(
        rgba,
        extent=(x_min, x_max, y_min, y_max),
        origin="lower",
        interpolation="bicubic",
        aspect="auto",
        zorder=0,
    )


def _plot_acquisition_strip(
    ax,
    x_values,
    scores,
    selected_x,
    color,
    label,
    show_xlabels=False,
    label_fontsize=7.8,
):
    """Render a thin acquisition-score strip aligned to the main posterior panel."""
    ax.set_facecolor("white")
    ax.plot(x_values, scores, color=color, linewidth=1.55, alpha=0.95, zorder=2)
    selected_y = np.interp(selected_x, x_values, scores)
    ax.scatter(
        [selected_x],
        [selected_y],
        s=18,
        color=color,
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )
    ax.set_title(
        label,
        loc="left",
        fontsize=label_fontsize,
        color=color,
        pad=0.8,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([])
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.tick_params(
        axis="x",
        which="both",
        length=0,
        labelbottom=show_xlabels,
        pad=1.5,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axhline(0.0, color="#D9DDE1", linewidth=0.8, zorder=1)


def _style_objective_box(ax, color):
    """Style a subplot with a subtle objective-matched border."""

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(mcolors.to_rgba(color, alpha=0.55))
        spine.set_linewidth(0.95)


def create_reg1d_branch_comparison_plot(
    problem,
    variance_strategy,
    weighted_exp_strategy,
    weighted_invexp_strategy,
    branch_iteration: int = 7,
    n_samples: int = 100,
    filename: str = "reg_var_exp_intro.svg",
):
    """
    Create a landscape one-step branch comparison from a shared current state.

    The current state is obtained by following standard variance reduction for
    ``branch_iteration`` acquisitions. From that shared state, the figure compares
    the posterior updates that would result from adding the variance-selected point
    or the weighted-variance-selected points under exponential growth and
    exponential decay beliefs,
    and reports weighted expected loss reduction under both beliefs.
    """
    if branch_iteration < 0:
        raise ValueError("branch_iteration must be non-negative")
    if not hasattr(weighted_exp_strategy, "weight_fn"):
        raise ValueError("weighted_exp_strategy must define weight_fn")
    if not hasattr(weighted_invexp_strategy, "weight_fn"):
        raise ValueError("weighted_invexp_strategy must define weight_fn")

    plot_root = ensure_results_dir(problem.task, problem.name, variance_strategy.model.identifier)
    exp_weight_fn = weighted_exp_strategy.weight_fn
    invexp_weight_fn = weighted_invexp_strategy.weight_fn
    is_gp_model = "gp" in variance_strategy.model.identifier.lower()
    is_rf_model = "rf" in variance_strategy.model.identifier.lower()
    palette = {
        "true": "#9AA3AB",
        "pred_line": "#3568B0",
        "pred_point": "#244C80",
        "var_obj": "#1B9E77",
        "exp_obj": "#C98A00",
        "inv_obj": "#7B61C7",
    }

    # Roll out the shared current state using standard variance reduction only.
    X_current = problem.X0.copy()
    y_current = problem.y0.copy()
    for _ in range(branch_iteration):
        scores_var_step, _ = variance_strategy.score_compare(
            X_current,
            y_current,
            problem.X_pool,
            problem.X_targ,
            exp_weight_fn,
            n_samples=n_samples,
            is_gp_model=is_gp_model,
            is_rf_model=is_rf_model,
        )
        idx_step = variance_strategy.select_best(scores_var_step, X_current, problem.X_pool)
        x_step = problem.X_pool[idx_step]
        y_step = problem.acquire(idx_step)
        X_current, y_current = _append_regression_observation(X_current, y_current, x_step, y_step)

    # Score all acquisition rules at the shared current state.
    scores_var, scores_exp = variance_strategy.score_compare(
        X_current,
        y_current,
        problem.X_pool,
        problem.X_targ,
        exp_weight_fn,
        n_samples=n_samples,
        is_gp_model=is_gp_model,
        is_rf_model=is_rf_model,
    )
    _, scores_inv = variance_strategy.score_compare(
        X_current,
        y_current,
        problem.X_pool,
        problem.X_targ,
        invexp_weight_fn,
        n_samples=n_samples,
        is_gp_model=is_gp_model,
        is_rf_model=is_rf_model,
    )

    idx_var = variance_strategy.select_best(scores_var, X_current, problem.X_pool)
    idx_exp = weighted_exp_strategy.select_best(scores_exp, X_current, problem.X_pool)
    idx_inv = weighted_invexp_strategy.select_best(scores_inv, X_current, problem.X_pool)
    x_var = problem.X_pool[idx_var]
    x_exp = problem.X_pool[idx_exp]
    x_inv = problem.X_pool[idx_inv]

    # Couple both hypothetical updates with the same noise draw for a fair visual comparison.
    branch_rng_state = copy.deepcopy(problem.rng.bit_generator.state)
    y_var = _sample_counterfactual_regression_label(problem, x_var, branch_rng_state)
    y_exp = _sample_counterfactual_regression_label(problem, x_exp, branch_rng_state)
    y_inv = _sample_counterfactual_regression_label(problem, x_inv, branch_rng_state)

    X_branch_var, y_branch_var = _append_regression_observation(X_current, y_current, x_var, y_var)
    X_branch_exp, y_branch_exp = _append_regression_observation(X_current, y_current, x_exp, y_exp)
    X_branch_inv, y_branch_inv = _append_regression_observation(X_current, y_current, x_inv, y_inv)

    plot_args, pipe = _get_reg_plot_args(problem)
    plot_inputs = np.linspace(*plot_args)
    true_curve = np.asarray(problem.true_mean(plot_inputs), dtype=float)
    true_std = np.broadcast_to(np.asarray(problem.true_std, dtype=float), true_curve.shape)
    y_lower = np.min(
        np.concatenate([true_curve - 2.0 * true_std, y_branch_var, y_branch_exp, y_branch_inv])
    )
    y_upper = np.max(
        np.concatenate([true_curve + 2.0 * true_std, y_branch_var, y_branch_exp, y_branch_inv])
    )
    y_margin = 0.08 * max(1e-6, y_upper - y_lower)

    fig = plt.figure(figsize=(4.45, 6.35))
    outer_grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=-0.15)
    current_grid = outer_grid[0, 0].subgridspec(
        6,
        1,
        height_ratios=[0.01, 1.0, 0.17, 0.17, 0.17, 0.69],
        hspace=0.32,
    )
    update_grid = outer_grid[0, 1].subgridspec(3, 1, hspace=0.34)
    ax_current = fig.add_subplot(current_grid[1, 0])
    ax_score_var = fig.add_subplot(current_grid[2, 0], sharex=ax_current)
    ax_score_exp = fig.add_subplot(current_grid[3, 0], sharex=ax_current)
    ax_score_inv = fig.add_subplot(current_grid[4, 0], sharex=ax_current)
    ax_var_update = fig.add_subplot(update_grid[0, 0], sharex=ax_current, sharey=ax_current)
    ax_exp_update = fig.add_subplot(update_grid[1, 0], sharex=ax_current, sharey=ax_current)
    ax_inv_update = fig.add_subplot(update_grid[2, 0], sharex=ax_current, sharey=ax_current)
    posterior_box_aspect = 0.92
    for ax in (ax_current, ax_var_update, ax_exp_update, ax_inv_update):
        ax.set_box_aspect(posterior_box_aspect)
        ax.set_anchor("N")
    ax_current.set_anchor("S")

    plot_regression_with_uncertainty(
        ax_current,
        problem,
        variance_strategy.model,
        X_current,
        y_current,
        show_legend=False,
        highlight_last_acquisition=False,
        model_color=palette["pred_line"],
        point_color=palette["pred_point"],
        true_color=palette["true"],
        uncertainty_alpha=0.24,
        uncertainty_std_scale=2.0,
        model_linewidth=1.35,
        true_linewidth=0.95,
        point_markersize=6.6,
    )
    ax_current.set_title(
        "Initial predictions and\nacquisition objectives",
        fontsize=8.2,
        pad=2.0,
    )
    ax_current.set_ylabel("Output ($z$)", labelpad=2)
    ax_current.tick_params(axis="both", labelsize=7.0, length=0)
    ax_current.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_current.set_ylim(y_lower - y_margin, y_upper + y_margin)
    ax_current.tick_params(labelbottom=False, labelleft=False)
    ax_current.spines["top"].set_visible(False)
    ax_current.spines["right"].set_visible(False)
    ax_current.spines["left"].set_color("#C9CED3")
    ax_current.spines["bottom"].set_visible(False)
    ax_current.spines["left"].set_linewidth(0.8)

    if hasattr(problem, "X_pool") and problem.X_pool.ndim > 1 and problem.X_pool.shape[1] > 1:
        if hasattr(problem, "pipe") and problem.pipe is not None:
            pool_inputs = _scaler_pca_transform(problem.X_pool, problem.pipe)[:, 0]
        else:
            pool_inputs = problem.X_pool[:, 0]
        sort_idx = np.argsort(pool_inputs)
        x_scores = pool_inputs[sort_idx]
        scores_var_plot = scores_var[sort_idx]
        scores_exp_plot = scores_exp[sort_idx]
        scores_inv_plot = scores_inv[sort_idx]
    elif hasattr(problem, "pool_args"):
        x_scores = np.linspace(*problem.pool_args)
        scores_var_plot = scores_var
        scores_exp_plot = scores_exp
        scores_inv_plot = scores_inv
    else:
        x_scores = np.asarray(problem.X_pool)
        sort_idx = np.argsort(x_scores)
        x_scores = x_scores[sort_idx]
        scores_var_plot = scores_var[sort_idx]
        scores_exp_plot = scores_exp[sort_idx]
        scores_inv_plot = scores_inv[sort_idx]

    scores_var_plot = _normalize_acquisition_scores(scores_var_plot)
    scores_exp_plot = _normalize_acquisition_scores(scores_exp_plot)
    scores_inv_plot = _normalize_acquisition_scores(scores_inv_plot)

    def _candidate_x_coord(x_value):
        x_value = np.asarray(x_value)
        if pipe is not None:
            return float(_scaler_pca_transform(x_value.reshape(1, -1), pipe)[0, 0])
        return float(x_value.reshape(-1)[0])

    candidate_specs = (
        (_candidate_x_coord(x_var), palette["var_obj"]),
        (_candidate_x_coord(x_exp), palette["exp_obj"]),
        (_candidate_x_coord(x_inv), palette["inv_obj"]),
    )

    for x_coord, color in candidate_specs:
        ax_current.axvline(
            x_coord,
            color=color,
            linestyle=(0, (4, 3)),
            linewidth=1.0,
            alpha=0.34,
            zorder=0.9,
        )

    _plot_acquisition_strip(
        ax_score_var,
        x_scores,
        scores_var_plot,
        _candidate_x_coord(x_var),
        palette["var_obj"],
        r"EUR with $\ell(z,a)=(z-a)^2$",
        label_fontsize=6,
    )
    _plot_acquisition_strip(
        ax_score_exp,
        x_scores,
        scores_exp_plot,
        _candidate_x_coord(x_exp),
        palette["exp_obj"],
        r"EUR with $\ell(z,a)=\exp(z)(z-a)^2$",
        label_fontsize=6,
    )
    _plot_acquisition_strip(
        ax_score_inv,
        x_scores,
        scores_inv_plot,
        _candidate_x_coord(x_inv),
        palette["inv_obj"],
        r"EUR with $\ell(z,a)=\exp(-z)(z-a)^2$",
        show_xlabels=True,
        label_fontsize=6,
    )
    for strip_ax in (ax_score_var, ax_score_exp, ax_score_inv):
        strip_ax.tick_params(axis="x", labelsize=6.8, labelbottom=False, length=0)
    ax_score_inv.set_xlabel("Input", labelpad=1.5)

    x_min, x_max = float(plot_inputs.min()), float(plot_inputs.max())
    panel_y_min = y_lower - y_margin
    panel_y_max = y_upper + y_margin

    _add_response_weight_background(
        ax_var_update,
        x_min,
        x_max,
        panel_y_min,
        panel_y_max,
        direction="neutral",
    )
    plot_regression_with_uncertainty(
        ax_var_update,
        problem,
        variance_strategy.model,
        X_branch_var,
        y_branch_var,
        show_legend=False,
        highlight_last_acquisition=True,
        model_color=palette["pred_line"],
        point_color=palette["pred_point"],
        new_point_color=palette["var_obj"],
        true_color=palette["true"],
        uncertainty_alpha=0.24,
        uncertainty_std_scale=2.0,
        model_linewidth=1.55,
        true_linewidth=0.95,
        point_markersize=6.6,
        new_point_markersize=9.2,
    )
    ax_var_update.set_title(
        r"EUR with" "\n" r"$\ell(z,a)=(z-a)^2$",
        fontsize=7.65,
        pad=2.2,
    )
    ax_var_update.set_ylabel("")
    ax_var_update.tick_params(axis="both", labelsize=7.0, length=0)
    ax_var_update.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_var_update.set_ylim(panel_y_min, panel_y_max)
    ax_var_update.tick_params(labelbottom=False, labelleft=False)
    _style_objective_box(ax_var_update, palette["var_obj"])

    _add_response_weight_background(
        ax_exp_update,
        x_min,
        x_max,
        panel_y_min,
        panel_y_max,
        color=palette["exp_obj"],
        direction="up",
    )
    plot_regression_with_uncertainty(
        ax_exp_update,
        problem,
        weighted_exp_strategy.model,
        X_branch_exp,
        y_branch_exp,
        show_legend=False,
        highlight_last_acquisition=True,
        model_color=palette["pred_line"],
        point_color=palette["pred_point"],
        new_point_color=palette["exp_obj"],
        true_color=palette["true"],
        uncertainty_alpha=0.24,
        uncertainty_std_scale=2.0,
        model_linewidth=1.55,
        true_linewidth=0.95,
        point_markersize=6.6,
        new_point_markersize=9.2,
    )
    ax_exp_update.set_title(
        r"EUR with" "\n" r"$\ell(z,a)=\exp(z)(z-a)^2$",
        fontsize=7.65,
        pad=2.2,
    )
    ax_exp_update.set_ylabel("")
    ax_exp_update.tick_params(axis="both", labelsize=7.0, length=0)
    ax_exp_update.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_exp_update.set_ylim(panel_y_min, panel_y_max)
    ax_exp_update.tick_params(labelbottom=False, labelleft=False)
    _style_objective_box(ax_exp_update, palette["exp_obj"])

    _add_response_weight_background(
        ax_inv_update,
        x_min,
        x_max,
        panel_y_min,
        panel_y_max,
        color=palette["inv_obj"],
        direction="down",
    )
    plot_regression_with_uncertainty(
        ax_inv_update,
        problem,
        weighted_invexp_strategy.model,
        X_branch_inv,
        y_branch_inv,
        show_legend=False,
        highlight_last_acquisition=True,
        model_color=palette["pred_line"],
        point_color=palette["pred_point"],
        new_point_color=palette["inv_obj"],
        true_color=palette["true"],
        uncertainty_alpha=0.24,
        uncertainty_std_scale=2.0,
        model_linewidth=1.55,
        true_linewidth=0.95,
        point_markersize=6.6,
        new_point_markersize=9.2,
    )
    ax_inv_update.set_title(
        r"EUR with" "\n" r"$\ell(z,a)=\exp(-z)(z-a)^2$",
        fontsize=7.65,
        pad=2.2,
    )
    ax_inv_update.set_ylabel("")
    ax_inv_update.tick_params(axis="both", labelsize=7.0, length=0)
    ax_inv_update.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_inv_update.set_ylim(panel_y_min, panel_y_max)
    ax_inv_update.tick_params(labelbottom=False, labelleft=False)
    _style_objective_box(ax_inv_update, palette["inv_obj"])

    true_handle = Line2D([0], [0], color=palette["true"], linestyle=":", linewidth=0.95)
    posterior_handle = Line2D([0], [0], color=palette["pred_line"], linewidth=1.55)
    current_data_handle = Line2D([0], [0], linestyle="None", marker="o", color=palette["pred_point"], markersize=5.2)
    acquired_point_handle = (
        Line2D([0], [0], linestyle="None", marker="*", color="w", markerfacecolor=palette["var_obj"], markeredgecolor="black", markersize=8.6),
        Line2D([0], [0], linestyle="None", marker="*", color="w", markerfacecolor=palette["exp_obj"], markeredgecolor="black", markersize=8.6),
        Line2D([0], [0], linestyle="None", marker="*", color="w", markerfacecolor=palette["inv_obj"], markeredgecolor="black", markersize=8.6),
    )
    acquisition_handle = (
        Line2D([0], [0], color=palette["var_obj"], linewidth=1.55),
        Line2D([0], [0], color=palette["exp_obj"], linewidth=1.55),
        Line2D([0], [0], color=palette["inv_obj"], linewidth=1.55),
    )
    fig.subplots_adjust(bottom=0.15, top=0.855, left=0.09, right=0.99)

    right_box = ax_var_update.get_position()
    current_box = ax_current.get_position()
    ax_current.set_position([
        0.5 * (current_box.x0 + current_box.x1) - 0.5 * right_box.width,
        0.5 * (current_box.y0 + current_box.y1) - 0.5 * right_box.height,
        right_box.width,
        right_box.height,
    ])
    current_box = ax_current.get_position()
    for strip_ax in (ax_score_var, ax_score_exp, ax_score_inv):
        strip_box = strip_ax.get_position()
        strip_ax.set_position([current_box.x0, strip_box.y0, current_box.width, strip_box.height])

    left_box = ax_current.get_position()
    left_center = 0.5 * (left_box.x0 + left_box.x1)
    right_center = 0.5 * (right_box.x0 + right_box.x1)
    header_y = min(0.975, max(left_box.y1, right_box.y1) + 0.052)
    fig.text(
        left_center,
        header_y,
        "Before Acquisition",
        ha="center",
        va="bottom",
        fontsize=8.8,
        fontweight="semibold",
        color="#25313A",
    )
    fig.text(
        right_center,
        header_y,
        "After One Acquisition Step",
        ha="center",
        va="bottom",
        fontsize=8.8,
        fontweight="semibold",
        color="#25313A",
    )
    fig.add_artist(Line2D(
        [left_box.x0, left_box.x1],
        [header_y - 0.006, header_y - 0.006],
        transform=fig.transFigure,
        color="#CCD2D8",
        linewidth=0.8,
    ))
    fig.add_artist(Line2D(
        [right_box.x0, right_box.x1],
        [header_y - 0.006, header_y - 0.006],
        transform=fig.transFigure,
        color="#CCD2D8",
        linewidth=0.8,
    ))
    fig.legend(
        [true_handle, posterior_handle, current_data_handle, acquired_point_handle, acquisition_handle],
        ["True function", "Predictive mean", "Current data", "Acquired point", "Acquisition score"],
        loc="lower center",
        bbox_to_anchor=(left_center, ax_inv_update.get_position().y0),
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="#B7BDC4",
        fancybox=False,
        framealpha=1.0,
        fontsize=6.6,
        columnspacing=0.9,
        handlelength=2.8,
        handletextpad=0.9,
        borderpad=0.35,
        labelspacing=0.35,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.45)},
    )
    fig.savefig(plot_root / filename, bbox_inches="tight")
    plt.close(fig)

def create_reg1d_acquisition_plots_live_by_weight(
    problem,
    strategy,
    weight_fn,
    n_steps: int,
    n_samples: int = 100,
    plot_iters=None,
):
    """
    Run live active learning for a single strategy and render only selected iterations.

    Args:
        problem: Problem definition containing pools and initial data.
        strategy: Acquisition strategy instance to drive active learning.
        weight_fn: Weighting function passed to `strategy.score_compare`.
        n_steps: Total number of acquisition iterations to execute.
        n_samples: Number of MC samples to use when scoring EUR curves.
        plot_iters: Iterable of iteration indices to visualize. Defaults to every step.
    """
    main_root = f"{problem.task}/{problem.name}/{strategy.model.identifier}"
    plot_root = ensure_results_dir(main_root)
    is_gp_model = "gp" in strategy.model.identifier.lower()
    is_rf_model = "rf" in strategy.model.identifier.lower()

    X1 = problem.X0.copy()
    y1 = problem.y0.copy()
    X2 = problem.X0.copy()
    y2 = problem.y0.copy()

    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")

    # Determine which iterations to plot
    if plot_iters is None:
        plot_iter_list = list(range(n_steps))
    else:
        try:
            plot_iter_list = sorted({int(it) for it in plot_iters})
        except TypeError as exc:
            raise TypeError("plot_iters must be an iterable of integers") from exc
        if any(it < 0 for it in plot_iter_list):
            raise ValueError("plot_iters must contain non-negative iteration indices.")
        if plot_iter_list and plot_iter_list[-1] >= n_steps:
            raise ValueError("plot_iters contains iterations beyond n_steps.")
    if not plot_iter_list:
        raise ValueError("At least one iteration must be provided for plotting.")
    total_steps = n_steps
    n_plot_cols = len(plot_iter_list)
    iter_to_col = {iteration: idx for idx, iteration in enumerate(plot_iter_list)}

    # 1) Create a 4×n_plot_cols grid of axes (only for requested iterations)
    figsize = (2.6 * n_plot_cols, 2.3 * 3)
    fig, axes = plt.subplots(4, n_plot_cols, figsize=figsize)
    if n_plot_cols == 1:
        axes = axes.reshape(4, 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.2, top=0.9)

    # Add y‐labels for each block of three rows:
    axes[0, 0].set_ylabel("M1 data and predictions")
    axes[1, 0].set_ylabel("EUR on M1 decision")
    axes[2, 0].set_ylabel("M2 data and predictions")
    axes[3, 0].set_ylabel("EUR on M2 decision")

    # Store legend elements from first plot for figure-level legend
    all_legend_elements = []
    eur_legend_lines = None
    reg_legend_captured = False
    plot_iter_set = set(plot_iter_list)
    
    for i in trange(total_steps):
        scores1, scores2 = strategy.score_compare(
            X1, y1, problem.X_pool, problem.X_targ, weight_fn,
            n_samples=n_samples, is_gp_model=is_gp_model, is_rf_model=is_rf_model)
        should_plot = i in plot_iter_set
        if should_plot:
            col_idx = iter_to_col[i]
            ax_db1 = axes[0, col_idx]
            reg_legend_elements = plot_regression_with_uncertainty(
                ax_db1, problem, strategy.model, X1, y1, show_legend=False
            )
            if not reg_legend_captured:
                all_legend_elements.extend(reg_legend_elements)
                reg_legend_captured = True
            ax_db1.set_title(f"Iteration {i}")

        # Row 1 - plot decision boundary for M1 and update dataset
        idx1 = strategy.select_best(scores1, X1, problem.X_pool)
        x_new1 = problem.X_pool[idx1]
        y_new1 = problem.acquire(idx1)
        if X1.ndim == 1:
            X1 = np.append(X1, x_new1)
        else:
            X1 = np.vstack([X1, x_new1.reshape(1, -1)])
        y1 = np.append(y1, y_new1)
        if should_plot:
            ax_a1 = axes[1, col_idx]
            line1, line2 = plot_reg_pair_eur(ax_a1, problem, scores1, scores2, show_legend=False)
            if eur_legend_lines is None:
                eur_legend_lines = [line1, line2]

        # Row 2
        scores1, scores2 = strategy.score_compare(
            X2, y2, problem.X_pool, problem.X_targ, weight_fn,
            n_samples=n_samples, is_gp_model=is_gp_model, is_rf_model=is_rf_model)
        if should_plot:
            ax_db2 = axes[2, col_idx]
            plot_regression_with_uncertainty(
                ax_db2,
                problem,
                strategy.model,
                X2,
                y2,
                show_legend=False
            )

        # Row 3 - Plot decision boundary for M2 and update dataset
        idx2 = strategy.select_best(scores2, X2, problem.X_pool)
        x_new2 = problem.X_pool[idx2]
        y_new2 = problem.acquire(idx2)
        if X2.ndim == 1:
            X2 = np.append(X2, x_new2)
        else:
            X2 = np.vstack([X2, x_new2.reshape(1, -1)])
        y2 = np.append(y2, y_new2)
        if should_plot:
            ax_a2 = axes[3, col_idx]
            plot_reg_pair_eur(ax_a2, problem, scores1, scores2, show_legend=False)

    # Combine all legend elements
    combined_legend_elements = []
    combined_legend_labels = []
    # Add regression elements
    if all_legend_elements:
        for element in all_legend_elements:
            combined_legend_elements.append(element)
            combined_legend_labels.append(element.get_label())
    
    # Add EUR elements
    if eur_legend_lines:
        for line in eur_legend_lines:
            combined_legend_elements.append(line)
            combined_legend_labels.append(line.get_label())

    # Add single legend at the bottom center of the figure
    if combined_legend_elements:
        fig.legend(combined_legend_elements, combined_legend_labels,
                  loc='lower center', bbox_to_anchor=(0.5, 0.025), 
                  ncol=len(combined_legend_elements), frameon=True)
    # fig.suptitle(f"M1: variance-based acquisition\nM2: weighted variance-based acquisition ({get_weight_display_name(weight_fn)})", fontsize=20)
    fig.savefig(plot_root / f"reg_{strategy.name}_{get_weight_identifier(weight_fn)}.svg", bbox_inches='tight')

def create_reg1d_acquisition_plots_with_visdata(problem, strategy_list, vis_data, num, n_samples=100):
    strat1, strat2 = strategy_list
    vis_data1, vis_data2 = vis_data
    plot_root = ensure_results_dir(problem.task, problem.name, strat1.model.identifier, "eval")

    n_steps = len(vis_data1['steps'])
    is_gp_model = "gp" in strat1.model.identifier.lower()
    is_rf_model = "rf" in strat1.model.identifier.lower()


    # 1) Create a 4×n_steps grid of axes:
    figsize = (5 * n_steps, 4 * 3)
    fig, axes = plt.subplots(4, n_steps, figsize=figsize)
    plt.subplots_adjust(wspace=0.4, hspace=0.2, top=0.9)

    # Add y‐labels for each block of three rows:
    axes[0, 0].set_ylabel("M1 data and predictions")
    axes[1, 0].set_ylabel("EUR on M1 decision")
    axes[2, 0].set_ylabel("M2 data and predictions")
    axes[3, 0].set_ylabel("EUR on M2 decision")

    for i in range(n_steps):
        X_train1i = vis_data1['X_data'][i]
        y_train1i = vis_data1['y_data'][i]
        X_train2i = vis_data2['X_data'][i]
        y_train2i = vis_data2['y_data'][i]

        # Row 0
        ax_db1 = axes[0, i]
        _load_model_data_into_strategy(strat1, vis_data1, i, X_train1i, y_train1i)
        scores1, scores2 = strat1.score_compare(X_train1i, y_train1i, problem.X_pool, problem.X_targ, strat2.weight_fn,
                                                n_samples=n_samples, is_gp_model=is_gp_model, is_rf_model=is_rf_model)
        plot_regression_with_uncertainty(
            ax_db1,
            problem,
            strat1.model,
            X_train1i,
            y_train1i
        )
        ax_db1.set_title(f"Iteration {vis_data1['steps'][i]}")

        # Row 1 - plot decision boundary for M1
        ax_a1 = axes[1, i]
        plot_reg_pair_eur(ax_a1, problem, scores1, scores2)

        # Row 2
        ax_db2 = axes[2, i]
        _load_model_data_into_strategy(strat2, vis_data2, i, X_train2i, y_train2i)
        scores1, scores2 = strat2.score_compare(X_train2i, y_train2i, problem.X_pool, problem.X_targ, strat2.weight_fn,
                                                n_samples=n_samples, is_gp_model=is_gp_model, is_rf_model=is_rf_model)
        plot_regression_with_uncertainty(
            ax_db2,
            problem,
            strat2.model,
            X_train2i,
            y_train2i
        )

        # Row 3 - Plot decision boundary for M2
        ax_a2 = axes[3, i]
        plot_reg_pair_eur(ax_a2, problem, scores1, scores2)

    fig.suptitle(f"M1: variance-based acquisition\nM2: weighted variance-based acquisition", fontsize=20)
    fig.savefig(plot_root / f"reg_{strat2.name}_iter{num}.svg", bbox_inches='tight')

def create_metric_comparison_plots_reg(
    problem,
    strategy_list,
    n_evals=100,
    metric_list=['sqloss'],
    stat_type='mean',
    transform: str | None = "log10",
    filename: str | None = None,
    max_steps: int | None = None,
):
    """
    Create metric comparison plots using saved evaluation files for regression.
    
    Args:
        problem: Problem instance
        strategy_list: List of strategies to compare
        weight_fn: Weight function for weighted metrics
        n_evals: Number of evaluations for weighted loss computation
        metric_list: List of metrics to plot ['sqloss']
        stat_type: str, either 'mean' (mean ± SEM) or 'median' (median with quantile range)
        transform: None for raw metrics, or "log10" to plot log10-transformed metrics
        filename: Optional output filename. Defaults to reg_metric_<weight>_<stat>.svg
        max_steps: Optional maximum number of acquisition steps to plot.
    """
    
    main_dir = ensure_results_dir(problem.task, problem.name, strategy_list[0].model.identifier)
    eval_dir = ensure_results_dir(problem.task, problem.name, strategy_list[0].model.identifier, "eval")

    weighted_strategies = [strategy for strategy in strategy_list if hasattr(strategy, "weight_fn")]
    if not weighted_strategies:
        raise ValueError("create_metric_comparison_plots_reg requires at least one weighted strategy.")
    weight_fn = weighted_strategies[-1].weight_fn
    strategy_names = [strategy.name for strategy in strategy_list]
    display_strategy_names = [name.replace("_exp_neg", "") for name in strategy_names]

    # Load all prediction and uncertainty data
    all_pred_data = []
    is_gp_models = []
    y_test_runs = None
    for strategy in strategy_list:
        with h5py.File(eval_dir / f"{strategy.name}.h5", 'r') as f:
            all_pred_data.append(f['pred_data'][:])
            is_gp_models.append(f.attrs['is_gp_model'])
            current_y_test = np.asarray(f['y_test'][:])
            if y_test_runs is None:
                y_test_runs = current_y_test
            else:
                if current_y_test.shape != y_test_runs.shape:
                    raise ValueError(
                        "Mismatch in stored y_test shapes across strategies; "
                        "ensure all strategies were evaluated with consistent runs."
                    )
                if not np.allclose(current_y_test, y_test_runs):
                    raise ValueError(
                        "Stored y_test values differ between strategies; "
                        "ensure identical problem initialisation per run."
                    )

    # Compute metrics efficiently
    metric_data = {}
    if 'sqloss' in metric_list:
        all_sqloss_data = compute_sqloss_unweighted_from_saved_data(all_pred_data, y_test_runs, is_gp_models)
        all_sqloss_w_data = compute_sqloss_weighted_from_saved_data(
            all_pred_data, y_test_runs, weight_fn, is_gp_models, n_evals=n_evals
        )
        metric_data['sqloss'] = [all_sqloss_data, all_sqloss_w_data]

    # Plot metrics
    n_metrics = len(metric_list)
    fig, axes = plt.subplots(nrows=n_metrics, ncols=2, figsize=(6, 2.5*n_metrics))
    if n_metrics == 1:
        axes = axes.reshape(1, -1)
    
    # Use consistent colors from classification function
    colors = [plt.get_cmap(_get_class_colormaps(num_classes=len(strategy_list))[i])(0.7) 
              for i in range(len(strategy_list))]
    
    for metric_idx, metric_name in enumerate(metric_list):
        data_groups = metric_data[metric_name]
        plot_names = ["SEL", "Weighted SEL"]
        
        for plot_idx, (data_group, plot_name) in enumerate(zip(data_groups, plot_names)):
            ax = axes[metric_idx, plot_idx]
            n_steps = data_group[0].shape[1]
            if max_steps is not None:
                n_steps = min(n_steps, max_steps)

            def _apply_transform(arr: np.ndarray) -> np.ndarray:
                if transform is None:
                    return arr
                if transform == "log10":
                    tiny = np.finfo(arr.dtype).tiny
                    return np.log10(np.maximum(arr, tiny))
                raise ValueError(f"Unsupported transform: {transform}")

            for (data, strategy_name, display_name, color) in zip(
                data_group, strategy_names, display_strategy_names, colors
            ):
                transformed = _apply_transform(data)
                if stat_type == 'mean':
                    central_data = np.mean(transformed, axis=0)[:n_steps]
                    error_data = _compute_sem(transformed, axis=0)[:n_steps]
                    ax.fill_between(np.arange(n_steps), central_data - error_data, 
                                   central_data + error_data, alpha=0.3, color=color)
                    print(f"{strategy_name} {plot_name} mean at final step: "
                          f"{central_data[-1]:.4f} \\SEM{{{error_data[-1]:.4f}}}")
                elif stat_type == 'median':
                    central_data = np.median(transformed, axis=0)[:n_steps]
                    q25 = np.percentile(transformed, 25, axis=0)[:n_steps]
                    q75 = np.percentile(transformed, 75, axis=0)[:n_steps]
                    ax.fill_between(np.arange(n_steps), q25, q75, alpha=0.3, color=color)
                    print(f"Strategy: {strategy_name}, Median final {plot_name}: {central_data[-1]:.4f} ({q25[-1]:.4f}-{q75[-1]:.4f})")
                else:
                    raise ValueError("stat_type must be either 'mean' or 'median'")
                
                ax.plot(np.arange(n_steps), central_data, marker='o', markersize=0.5, 
                       label=display_name, color=color, linewidth=0.75)
            
            ax.set_xlabel("Acquisition Step")
            y_label = plot_name if transform is None else f"{transform} {plot_name}"
            ax.set_ylabel(y_label)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
    
    # Update title to reflect the statistic type
    if stat_type == 'mean':
        title_stat = "Mean ± SEM"
    else:
        title_stat = "Median ± IQR"

    # Create informative title
    # fig.suptitle(f"{title_stat} Regression Metrics {get_weight_display_name(weight_fn)}", fontsize=14)
    if filename is None:
        filename = f"reg_metric_{get_weight_identifier(weight_fn)}_{stat_type}.svg"

    output_path = main_dir / filename
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved regression metric comparison plot to {output_path}")
    plt.close(fig)

def _infer_loss_type_and_weight(strategy):
    """Infer loss type identifier and weight_fn (if needed) from strategy."""
    name = strategy.__class__.__name__.lower()
    strat_name = getattr(strategy, "name", "").lower()

    if "weightedregressionposteriorlinex" in name or "linex_w" in strat_name:
        return "w_linex", getattr(strategy, "weight_fn", None)
    if "regressionposteriorlinex" in name or "linex" in strat_name:
        return "linex", None
    if "weightedregressionvariancereduction" in name or "var_w" in strat_name:
        return "w_sq", getattr(strategy, "weight_fn", None)
    return "sq", None


def _compute_loss_for_type(loss_type, pred_data, is_gp, y_test_runs, alpha_linex, weight_fn=None):
    """Compute loss curves for a single strategy and loss definition."""
    n_runs, n_steps, n_test = pred_data.shape[:3]
    y_runs = _broadcast_y_test(y_test_runs, n_runs, n_test)

    if loss_type == "sq":
        unweighted = compute_sqloss_unweighted_from_saved_data(
            [pred_data],
            y_runs,
            [is_gp],
        )
        return unweighted[0]

    if loss_type == "w_sq":
        if weight_fn is None:
            raise ValueError("weight_fn is required for weighted squared loss")
        weighted = compute_sqloss_weighted_from_saved_data(
            [pred_data],
            y_runs,
            weight_fn,
            [is_gp],
        )
        return weighted[0]

    if loss_type == "linex":
        return compute_linex_metrics_from_saved_data([pred_data], y_runs, [is_gp], alpha=alpha_linex)[0]

    if loss_type == "w_linex":
        if weight_fn is None:
            raise ValueError("weight_fn is required for weighted LINEX loss")
        return compute_linex_metrics_from_saved_data(
            [pred_data], y_runs, [is_gp], alpha=alpha_linex, weight_fn=weight_fn
        )[0]

    raise ValueError(f"Unsupported loss type: {loss_type}")


def create_loss_comparison_plots_reg(
    problem,
    strategies,
    *,
    ref_indices=(1, 2),
    alpha_linex: float = 1.0,
    stat_type: str = "mean",
    filename: str = "reg_loss_compare.svg",
    transform: str | None = None,
):
    """
    Compare strategies under losses implied by reference acquisitions.

    - First subplot: loss corresponding to strategies[ref_indices[0]], applied to all strategies' predictions.
    - Second subplot: loss corresponding to strategies[ref_indices[1]], applied likewise.

    transform:
        None (default) for raw losses, or "log10" to plot log10 losses. Any other
        value raises a ValueError.
    """
    if len(ref_indices) != 2:
        raise ValueError("ref_indices must contain exactly two indices")
    if max(ref_indices) >= len(strategies):
        raise ValueError("ref_indices refer to non-existent strategies")

    main_dir = ensure_results_dir(problem.task, problem.name, strategies[0].model.identifier)
    eval_dir = ensure_results_dir(problem.task, problem.name, strategies[0].model.identifier, "eval")

    # Load prediction data
    all_pred_data = []
    is_gp_models = []
    y_test_runs = None
    for strat in strategies:
        with h5py.File(eval_dir / f"{strat.name}.h5", "r") as f:
            # Materialise datasets before closing file to avoid invalid handles
            all_pred_data.append(np.asarray(f["pred_data"]))
            is_gp_models.append(f.attrs["is_gp_model"])
            current_y_test = np.asarray(f["y_test"])
            if y_test_runs is None:
                y_test_runs = current_y_test
            else:
                if current_y_test.shape != y_test_runs.shape:
                    raise ValueError("Mismatch in stored y_test shapes across strategies.")
                if not np.allclose(current_y_test, y_test_runs):
                    raise ValueError("Stored y_test values differ; ensure identical runs.")

    # Reference loss specs
    ref_specs = []
    for idx in ref_indices:
        loss_type, weight_fn = _infer_loss_type_and_weight(strategies[idx])
        ref_specs.append((loss_type, weight_fn, strategies[idx].name))

    figsize = (6, 2.5)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    colors = [plt.get_cmap(_get_class_colormaps(num_classes=len(strategies))[i])(0.7) for i in range(len(strategies))]

    for ax, (loss_type, weight_fn, ref_name) in zip(axes, ref_specs):
        loss_curves = []
        for pred_data, is_gp in zip(all_pred_data, is_gp_models):
            loss_curves.append(
                _compute_loss_for_type(loss_type, pred_data, is_gp, y_test_runs, alpha_linex, weight_fn)
            )

        n_steps = loss_curves[0].shape[1]

        # Optional transform (e.g., log scaling) for stability/visualisation
        def _apply_transform(arr: np.ndarray) -> np.ndarray:
            if transform is None:
                return arr
            if transform == "log10":
                tiny = np.finfo(arr.dtype).tiny
                return np.log10(np.maximum(arr, tiny))
            raise ValueError(f"Unsupported transform: {transform}")

        for losses, strat, color in zip(loss_curves, strategies, colors):
            transformed = _apply_transform(losses)
            if stat_type == "mean":
                central = np.mean(transformed, axis=0)[:n_steps]
                error = _compute_sem(transformed, axis=0)[:n_steps]
                ax.fill_between(np.arange(n_steps), central - error, central + error, alpha=0.3, color=color)
                # Log every 10 steps (including final)
                step_idxs = list(range(9, n_steps, 5))
                if (n_steps - 1) not in step_idxs:
                    step_idxs.append(n_steps - 1)
                for s_idx in step_idxs:
                    print(f"{strat.name} {ref_name} mean at step {s_idx}: "
                          f"{central[s_idx]:.4f} \\SEM{{{error[s_idx]:.4f}}}")
            elif stat_type == "median":
                central = np.median(transformed, axis=0)[:n_steps]
                q25 = np.percentile(transformed, 25, axis=0)[:n_steps]
                q75 = np.percentile(transformed, 75, axis=0)[:n_steps]
                ax.fill_between(np.arange(n_steps), q25, q75, alpha=0.3, color=color)
                step_idxs = list(range(0, n_steps, 10))
                if (n_steps - 1) not in step_idxs:
                    step_idxs.append(n_steps - 1)
                for s_idx in step_idxs:
                    print(f"{strat.name} {ref_name} median at step {s_idx}: "
                          f"{central[s_idx]:.4f} \\IQR{{{q25[s_idx]:.4f}, {q75[s_idx]:.4f}}}")
            else:
                raise ValueError("stat_type must be either 'mean' or 'median'")

            ax.plot(np.arange(n_steps), central, marker="o", markersize=0.5, label=strat.name, color=color, linewidth=0.75)

        ax.set_xlabel("Acquisition Step")
        y_label = f"Loss ({ref_name})" if transform is None else f"{transform} Loss ({ref_name})"
        ax.set_ylabel(y_label)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)

    fig.tight_layout()
    fig.savefig(main_dir / filename, bbox_inches="tight")
    plt.close(fig)
