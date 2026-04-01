import numpy as np
from typing import Callable
from scipy.stats import norm
from scipy.special import logsumexp
from scipy.linalg import cho_solve
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct, Matern, ConstantKernel, RationalQuadratic


from acquisition.model import RegressorWrapper


def _infer_dtype_eps(arr: np.ndarray) -> float:
    dtype = np.dtype(arr.dtype if isinstance(arr, np.ndarray) else float)
    return np.finfo(dtype).eps


def _ensure_2d_input(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def build_fixed_gp_kernel(
    prob,
    *,
    family: str = "m32",
    add_linear: bool = True,
    add_rq: bool = False,
    min_std: float = 1e-6,
    ridge_alpha: float = 1e-6,
):
    """Construct a fixed GP kernel with robust hyperparameters."""

    X_all = getattr(prob, "X_all", None)
    y_all = getattr(prob, "y_all", None)

    if X_all is None or not np.size(X_all):
        X0 = _ensure_2d_input(prob.X0)
        Xpool = _ensure_2d_input(prob.X_pool)
        X_all = np.vstack([X0, Xpool]) if Xpool.size else X0
    else:
        X_all = np.asarray(X_all, dtype=float)

    if y_all is None or not np.size(y_all):
        y_all = np.asarray(prob.y0, dtype=float).reshape(-1)
    else:
        y_all = np.asarray(y_all, dtype=float).reshape(-1)

    n = min(X_all.shape[0], y_all.shape[0])
    if n == 0:
        sigma_n = min_std
        sigma_lin = 0.0
        sigma_f = 1.0
        lengthscale = 1.0
        kernel = ConstantKernel(sigma_f**2, constant_value_bounds="fixed") * \
            RBF(length_scale=lengthscale, length_scale_bounds="fixed")
        kernel += WhiteKernel(noise_level=sigma_n**2, noise_level_bounds="fixed")
        return {
            "kernel": kernel,
            "lengthscale": lengthscale,
            "sigma_n": sigma_n,
            "mean": 0.0,
            "sigma_lin": sigma_lin,
            "sigma_f": sigma_f,
            "family": family,
            "nu": 1.5,
            "add_rq": add_rq,
        }

    X_used = np.asarray(X_all[:n], dtype=float)
    y_used = np.asarray(y_all[:n], dtype=float)

    mean_level = float(np.median(y_used))
    centered_y = y_used - mean_level

    feat_mean = X_used.mean(axis=0, keepdims=True)
    feat_std = X_used.std(axis=0, ddof=0, keepdims=True)
    feat_std[feat_std < 1e-12] = 1.0
    Xz = (X_used - feat_mean) / feat_std

    if n > 1:
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(Xz)
        dists, idx = nn.kneighbors(Xz)
        r0 = float(np.median(dists[:, 1]))
        nn_dy = np.abs(y_used - y_used[idx[:, 1]])
        sigma_n = max(1.4826 * np.median(nn_dy) / np.sqrt(2.0), min_std)
    else:
        r0 = 1.0
        sigma_n = min_std

    sy = 1.4826 * np.median(np.abs(centered_y)) if n > 0 else 0.0

    sigma_lin = 0.0
    sigma_f = max(np.sqrt(max(sy**2 - sigma_n**2, 0.0)), 1e-8)

    if add_linear and Xz.shape[1] > 0 and n > 1:
        ridge = Ridge(alpha=ridge_alpha)
        ridge.fit(Xz, centered_y)
        y_lin = ridge.predict(Xz)
        sigma_lin = float(np.std(y_lin))
        sigma_f = max(float(np.std(centered_y - y_lin)), 1e-8)

    coef_map = {"rbf": 0.8493218, "m32": 1.0319980, "m52": 0.9595803}
    family_key = family if family in coef_map else "m32"
    lengthscale = max(coef_map[family_key] * max(r0, 1e-12), 1e-6)

    parts = []
    if add_linear and sigma_lin > 1e-12:
        parts.append(
            ConstantKernel(sigma_lin**2, constant_value_bounds="fixed") *
            DotProduct(sigma_0=0.0, sigma_0_bounds="fixed")
        )

    nu_val = None
    if add_rq:
        parts.append(
            ConstantKernel(sigma_f**2, constant_value_bounds="fixed") *
            RationalQuadratic(length_scale=lengthscale, alpha=1.0,
                               length_scale_bounds="fixed", alpha_bounds="fixed")
        )
    else:
        if family == "rbf":
            parts.append(
                ConstantKernel(sigma_f**2, constant_value_bounds="fixed") *
                RBF(length_scale=lengthscale, length_scale_bounds="fixed")
            )
        else:
            nu_map = {"m32": 1.5, "m52": 2.5}
            nu_val = nu_map.get(family, 1.5)
            parts.append(
                ConstantKernel(sigma_f**2, constant_value_bounds="fixed") *
                Matern(length_scale=lengthscale, nu=nu_val, length_scale_bounds="fixed")
            )

    parts.append(WhiteKernel(noise_level=sigma_n**2, noise_level_bounds="fixed"))

    kernel = parts[0]
    for p in parts[1:]:
        kernel = kernel + p

    return {
        "kernel": kernel,
        "lengthscale": lengthscale,
        "sigma_n": sigma_n,
        "mean": mean_level,
        "sigma_lin": sigma_lin,
        "sigma_f": sigma_f,
        "family": family,
        "nu": nu_val,
        "add_rq": add_rq,
    }


def suggest_initial_lengthscale_noise_and_mean(
    prob,
    *,
    family: str = "m32",
    add_linear: bool = True,
    add_rq: bool = False,
    min_std: float = 1e-6,
):
    """Estimate hyperparameters and kernel using robust heuristics."""

    info = build_fixed_gp_kernel(
        prob,
        family=family,
        add_linear=add_linear,
        add_rq=add_rq,
        min_std=min_std,
    )

    return (
        info["lengthscale"],
        info["sigma_n"],
        info["mean"],
        info["sigma_lin"],
        info["sigma_f"],
        info["kernel"],
        info,
    )


def suggest_initial_lengthscale_and_noise(prob, **kwargs):
    ls, noise, *_ = suggest_initial_lengthscale_noise_and_mean(prob, **kwargs)
    return ls, noise


def _collect_tree_predictions(estimators, X: np.ndarray) -> np.ndarray:
    return np.stack([tree.predict(X) for tree in estimators], axis=0)


def _rf_target_mean_chunks(
    model: RegressorWrapper,
    cand_inputs: np.ndarray,
    targ_inputs: np.ndarray,
    n_samples: int,
    std: float,
    rng: np.random.Generator,
    *,
    chunk_size: int,
):
    base_model = getattr(model, "reg", None)
    if base_model is None or not hasattr(base_model, "estimators_"):
        raise ValueError("Model must wrap a RandomForestRegressor with fitted estimators_")

    estimators = base_model.estimators_
    if not estimators:
        raise ValueError("Random forest has no fitted estimators")

    cand_inputs_2d = _ensure_2d_input(cand_inputs)
    targ_inputs_2d = _ensure_2d_input(targ_inputs)

    # tree_preds_*: (n_trees, n_points)
    tree_preds_cand = _collect_tree_predictions(estimators, cand_inputs_2d)
    tree_preds_targ = _collect_tree_predictions(estimators, targ_inputs_2d)

    n_trees, n_cand = tree_preds_cand.shape
    n_targ = targ_inputs_2d.shape[0]

    cand_indices = np.arange(n_cand)[:, None]

    tiny = np.finfo(tree_preds_cand.dtype).tiny

    for start in range(0, n_samples, chunk_size):
        size = min(chunk_size, n_samples - start)
        breakpoint()
        prior_tree_indices = rng.integers(0, n_trees, size=(n_cand, size))
        prior_tree_means = tree_preds_cand[prior_tree_indices, cand_indices]
        cand_samples = prior_tree_means + rng.normal(0.0, std, size=(n_cand, size))

        diffs = (cand_samples[np.newaxis, :, :] - tree_preds_cand[:, :, np.newaxis]) / std
        log_resp = -0.5 * diffs**2
        log_resp -= logsumexp(log_resp, axis=0, keepdims=True)
        tree_posteriors = np.exp(log_resp).transpose(1, 2, 0)
        denom = tree_posteriors.sum(axis=2, keepdims=True)
        denom = np.maximum(denom, tiny)
        tree_posteriors /= denom

        cdf = np.cumsum(tree_posteriors, axis=2)
        cdf[..., -1] = 1.0
        uniform_samples = rng.random(size=(n_cand, size))
        selected_tree_indices = (cdf >= uniform_samples[..., None]).argmax(axis=2)

        targ_means = tree_preds_targ[selected_tree_indices.reshape(-1), :]
        yield targ_means.reshape(n_cand, size, n_targ), start, size


def compute_expected_variance_reduction(
    targ_inputs: np.ndarray,
    cand_inputs: np.ndarray,
    model: RegressorWrapper,
    *,
    n_samples: int = 100,
    std: float = 1.0,
    seed: int | None = None,
    chunk_size: int | None = None,
) -> np.ndarray:
    """Draw Monte Carlo samples and return per-candidate expectation estimates.

    Returns
    -------
    np.ndarray
        Array of shape (n_cand,) with the mean over target inputs of the
        product of the two independent target-sample means.
    """
    if std <= 0:
        raise ValueError("std must be positive")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    rng = np.random.default_rng(seed)
    chunk = chunk_size or min(n_samples, 64)
    n_targ = _ensure_2d_input(targ_inputs).shape[0]

    sum_first: np.ndarray | None = None
    sum_second: np.ndarray | None = None
    processed = 0

    for targ_means_chunk, start, size in _rf_target_mean_chunks(
        model,
        cand_inputs,
        targ_inputs,
        n_samples,
        std,
        rng,
        chunk_size=chunk,
    ):
        if sum_first is None:
            n_cand, _, n_targ = targ_means_chunk.shape
            sum_first = np.zeros((n_cand, n_targ), dtype=targ_means_chunk.dtype)
            sum_second = np.zeros_like(sum_first)

        noise = rng.normal(0.0, std, size=(2, *targ_means_chunk.shape))
        samples = targ_means_chunk[np.newaxis, ...] + noise

        sum_first += samples[0].sum(axis=1)
        sum_second += samples[1].sum(axis=1)
        processed += size

    if processed != n_samples:
        raise RuntimeError("Mismatch between processed samples and requested n_samples")

    if sum_first is None or sum_second is None:
        raise RuntimeError("No samples processed")

    mean_first = sum_first / n_samples
    mean_second = sum_second / n_samples
    product_means = mean_first * mean_second  # (n_cand, n_targ)

    return product_means.mean(axis=1)


def compute_expected_weighted_variance_reduction(
    targ_inputs: np.ndarray,
    cand_inputs: np.ndarray,
    model: RegressorWrapper,
    weight_fn: Callable,
    *,
    n_samples: int = 100,
    std: float = 1.0,
    seed: int | None = None,
    chunk_size: int | None = None,
) -> np.ndarray:
    """Monte Carlo estimator for weighted variance reduction.

    Nested sampling uses ``m = max(1, floor(sqrt(n_samples)))`` draws per outer sample
    to approximate the two weighted expectations before taking their product.
    """
    if std <= 0:
        raise ValueError("std must be positive")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    rng = np.random.default_rng(seed)
    chunk = chunk_size or min(n_samples, 4)
    n_targ = _ensure_2d_input(targ_inputs).shape[0]

    m = max(1, int(np.sqrt(n_samples)))
    total_num = None
    total_den = None
    processed = 0

    for targ_means_chunk, start, size in _rf_target_mean_chunks(
        model,
        cand_inputs,
        targ_inputs,
        n_samples,
        std,
        rng,
        chunk_size=chunk,
    ):
        n_cand, _, n_targ = targ_means_chunk.shape

        if total_num is None:
            total_num = np.zeros((n_cand, n_samples, n_targ), dtype=targ_means_chunk.dtype)
            total_den = np.zeros((n_cand, n_samples, n_targ), dtype=targ_means_chunk.dtype)

        noise = rng.normal(0.0, std, size=(m, *targ_means_chunk.shape))
        samples = targ_means_chunk[np.newaxis, ...] + noise
        weights = np.asarray(weight_fn(samples), dtype=samples.dtype)
        chunk_num = np.sum(weights * samples, axis=0)
        chunk_den = np.sum(weights, axis=0)
        total_num[:, start:start + size, :] += chunk_num
        total_den[:, start:start + size, :] += chunk_den
        processed += size
    tiny = np.finfo(total_den.dtype).tiny
    denominator = np.maximum(total_den, tiny)
    weighted_mean = total_num**2 / denominator / n_samples
        

    if processed != n_samples:
        raise RuntimeError("Mismatch between processed samples and requested n_samples")

    if total_num is None:
        raise RuntimeError("No samples processed")

    return weighted_mean.mean(axis=(1, 2))


def compute_posterior_variance(
    targ_inputs: np.ndarray,
    inputs: np.ndarray,
    labels: np.ndarray,
    model: RegressorWrapper,
) -> float:
    model.fit(inputs, labels)
    _, std = model.predict(targ_inputs, return_std=True)
    return np.mean(std**2)

def compute_posterior_variance_cached(
    targ_inputs: np.ndarray,
    inputs: np.ndarray,
    labels: np.ndarray,
    model: RegressorWrapper,
    restore_state_fn=None,
) -> float:
    """
    Compute posterior variance with optional state restoration.

    Args:
        targ_inputs: Target inputs for prediction
        inputs: Training inputs
        labels: Training labels
        model: Model to use
        restore_state_fn: Optional function to restore model state after computation
    """
    model.fit(inputs, labels)
    _, std = model.predict(targ_inputs, return_std=True)
    result = np.mean(std**2)

    # Restore original model state if function provided
    if restore_state_fn is not None:
        restore_state_fn()

    return result

def estimate_expected_posterior_variance(
    targ_inputs: np.ndarray,
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    cand_input: float,
    cand_labels: np.ndarray,
    model: RegressorWrapper,
) -> float:
    if train_inputs.ndim == 1:
        inputs = np.append(train_inputs, cand_input)
    else:
        inputs = np.concatenate((train_inputs, cand_input.reshape(1, -1)), axis=0)
    vals = []
    for y in cand_labels:
        labels = np.append(train_labels, y)
        vals.append(compute_posterior_variance(targ_inputs, inputs, labels, model))
    return np.mean(vals)

def estimate_expected_posterior_variance_cached(
    targ_inputs: np.ndarray,
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    cand_input: float,
    cand_labels: np.ndarray,
    model: RegressorWrapper,
) -> float:
    """
    Estimate expected posterior variance with model state caching to avoid unnecessary refits.

    This version saves the original model state and restores it after each computation,
    avoiding the need to refit the model thousands of times.
    """
    # Save original model state
    original_state = model.get_model_state()

    # Create restore function
    def restore_original_state():
        model.load_model_state(original_state)

    if train_inputs.ndim == 1:
        inputs = np.append(train_inputs, cand_input)
    else:
        inputs = np.concatenate((train_inputs, cand_input.reshape(1, -1)), axis=0)

    vals = []
    for y in cand_labels:
        labels = np.append(train_labels, y)
        vals.append(compute_posterior_variance_cached(
            targ_inputs, inputs, labels, model, restore_original_state
        ))

    # Ensure original state is restored at the end
    restore_original_state()

    return np.mean(vals)

def estimate_wquad_uncertainty(
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    weight_fn: Callable = lambda z: np.maximum(z, 1e-3),
    n_evals: int | None = None,
    quantile: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    w = weight_fn
    z_w = lambda z: z * w(z)
    z_sq_w = lambda z: z**2 * w(z)

    uncertainty = []

    for pred_mean_i, pred_std_i in zip(pred_mean, pred_std):
        z_dist = norm(loc=pred_mean_i, scale=pred_std_i)

        if n_evals is None:
            uncertainty_i = z_dist.expect(z_sq_w) - z_dist.expect(z_w) ** 2 / z_dist.expect(w)
        elif quantile is None:
            zs = z_dist.rvs(size=n_evals, random_state=seed)
            uncertainty_i = np.mean(z_sq_w(zs)) - np.mean(z_w(zs)) ** 2 / np.mean(w(zs))
        else:
            zs = np.linspace(z_dist.ppf(quantile), z_dist.ppf(1 - quantile), n_evals)

            densities = z_dist.pdf(zs)
            densities /= np.sum(densities)

            uncertainty_i = np.sum(z_sq_w(zs) * densities)
            uncertainty_i -= np.sum(z_w(zs) * densities) ** 2 / np.sum(w(zs) * densities)

        uncertainty += [uncertainty_i]

    return np.array(uncertainty)

def estimate_wquad_uncertainty_samples(
    pred_samples: np.ndarray,
    weight_fn: Callable = lambda z: np.maximum(z, 1e-3),
) -> np.ndarray:
    w = weight_fn
    z_w = lambda z: z * w(z)
    z_sq_w = lambda z: z**2 * w(z)

    uncertainty = []

    for samples_i in pred_samples:
        uncertainty_i = np.mean(z_sq_w(samples_i)) - np.mean(z_w(samples_i)) ** 2 / np.mean(w(samples_i))
        uncertainty += [uncertainty_i]

    return np.array(uncertainty)

def estimate_posterior_wquad_uncertainty(
    targ_inputs: np.ndarray,
    inputs: np.ndarray,
    labels: np.ndarray,
    model: RegressorWrapper,
    weight_fn: Callable = lambda z: np.maximum(z, 1e-3),
    n_evals: int | None = None,
    is_gp_model: bool = False,
    seed: int = 0,
) -> float:
    model.fit(inputs, labels)
    if is_gp_model:
        targ_mean, targ_std = model.predict(targ_inputs, return_std=True)
        return np.mean(estimate_wquad_uncertainty(targ_mean, targ_std, weight_fn=weight_fn, n_evals=n_evals, seed=seed))
    else:
        targ_samples = model.predict_samples(targ_inputs, n_samples=n_evals, seed=seed)
        return np.mean(estimate_wquad_uncertainty_samples(targ_samples, weight_fn=weight_fn))

def estimate_expected_posterior_wquad_uncertainty(
    targ_inputs: np.ndarray,
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    cand_input: float,
    cand_labels: np.ndarray,
    model: RegressorWrapper,
    weight_fn: Callable = lambda z: np.maximum(z, 1e-3),
    n_evals: int | None = None,
    is_gp_model: bool = False,
    seed: int = 0,
) -> float:
    if train_inputs.ndim == 1:
        inputs = np.append(train_inputs, cand_input)
    else:
        inputs = np.concatenate((train_inputs, cand_input.reshape(1, -1)), axis=0)
    vals = []
    for y in cand_labels:
        labels = np.append(train_labels, y)
        vals.append(
            estimate_posterior_wquad_uncertainty(targ_inputs, inputs, labels, model, weight_fn, n_evals, is_gp_model, seed=seed)
        )
    return np.mean(vals)

def compute_posterior_wquad_uncertainty_cached(
    targ_inputs: np.ndarray,
    inputs: np.ndarray,
    labels: np.ndarray,
    model: RegressorWrapper,
    weight_fn: Callable = lambda z: np.maximum(z, 1e-3),
    n_evals: int | None = None,
    is_gp_model: bool = False,
    seed: int = 0,
    restore_state_fn=None,
) -> float:
    """
    Compute posterior weighted quadratic uncertainty with optional state restoration.

    Args:
        targ_inputs: Target inputs for prediction
        inputs: Training inputs
        labels: Training labels
        model: Model to use
        weight_fn: Weight function for weighted uncertainty
        n_evals: Number of evaluations for uncertainty computation
        is_gp_model: Whether the model is GP-based
        seed: Random seed
        restore_state_fn: Optional function to restore model state after computation
    """
    model.fit(inputs, labels)

    # Compute weighted uncertainty
    if is_gp_model:
        targ_mean, targ_std = model.predict(targ_inputs, return_std=True)
        result = np.mean(estimate_wquad_uncertainty(targ_mean, targ_std, weight_fn=weight_fn, n_evals=n_evals, seed=seed))
    else:
        targ_samples = model.predict_samples(targ_inputs, n_samples=n_evals, seed=seed)
        result = np.mean(estimate_wquad_uncertainty_samples(targ_samples, weight_fn=weight_fn))

    # Restore original model state if function provided
    if restore_state_fn is not None:
        restore_state_fn()

    return result

def estimate_expected_posterior_wquad_uncertainty_cached(
    targ_inputs: np.ndarray,
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    cand_input: float,
    cand_labels: np.ndarray,
    model: RegressorWrapper,
    weight_fn: Callable = lambda z: np.maximum(z, 1e-3),
    n_evals: int | None = None,
    is_gp_model: bool = False,
    seed: int = 0,
) -> float:
    """
    Estimate expected posterior weighted quadratic uncertainty with model state caching.

    This version saves the original model state and restores it after each computation,
    avoiding the need to refit the model thousands of times.
    """
    # Save original model state
    original_state = model.get_model_state()

    # Create restore function
    def restore_original_state():
        model.load_model_state(original_state)

    if train_inputs.ndim == 1:
        inputs = np.append(train_inputs, cand_input)
    else:
        inputs = np.concatenate((train_inputs, cand_input.reshape(1, -1)), axis=0)

    vals = []
    for y in cand_labels:
        labels = np.append(train_labels, y)
        vals.append(compute_posterior_wquad_uncertainty_cached(
            targ_inputs, inputs, labels, model,
            weight_fn=weight_fn, n_evals=n_evals, is_gp_model=is_gp_model,
            seed=seed, restore_state_fn=restore_original_state
        ))

    # Ensure original state is restored at the end
    restore_original_state()

    return np.mean(vals)
#
# Predictive-posterior log-exp utilities
#

def _aggregate_target_scores(scores: np.ndarray, targ_weight=None) -> float:
    """Average per-target scores with optional weights."""
    if targ_weight is None:
        return float(np.mean(scores))

    if hasattr(targ_weight, "detach"):
        targ_weight = targ_weight.detach().cpu().numpy()
    weights = np.asarray(targ_weight, dtype=float).reshape(-1)
    if weights.shape[0] != scores.shape[0]:
        raise ValueError(
            f"Target weight length {weights.shape[0]} does not match scores {scores.shape[0]}"
        )
    total = float(np.sum(weights))
    if total <= 0:
        return float(np.mean(scores))
    return float(np.sum(scores * weights) / total)


def predictive_posterior_linex_score(
    samples: np.ndarray,
    targ_weight=None,
) -> float:
    """Compute E[log Z] - log E[Z] from posterior samples."""
    samples = np.asarray(samples, dtype=float)
    tiny = np.finfo(samples.dtype).tiny
    mean_log = np.mean(np.log(np.maximum(samples, tiny)), axis=1)
    mean_val = np.mean(samples, axis=1)
    scores = mean_log - np.log(np.maximum(mean_val, tiny))
    # mean = np.mean(samples, axis=1)
    # mean_negexp = np.mean(np.exp(-samples), axis=1)
    # scores = - mean - np.log(mean_negexp)
    return _aggregate_target_scores(scores, targ_weight)


def predictive_posterior_linex_score_weighted(
    samples: np.ndarray,
    weight_fn: Callable,
    targ_weight=None,
) -> float:
    """
    Weighted score:
        E_w[log Z] - log E_w[Z]
    """
    samples = np.asarray(samples, dtype=float)
    weights = np.asarray(weight_fn(samples), dtype=float)
    if weights.shape != samples.shape:
        raise ValueError("weight_fn must return the same shape as samples")

    tiny = np.finfo(samples.dtype).tiny

    sum_w = np.sum(weights, axis=1)
    denom = np.maximum(sum_w, tiny)
    mean_log = np.sum(weights * np.log(np.maximum(samples, tiny)), axis=1) / denom
    mean_val = np.sum(weights * samples, axis=1) / denom

    scores = mean_log - np.log(np.maximum(mean_val, tiny))

    return _aggregate_target_scores(scores, targ_weight)


def _append_candidate(train_inputs: np.ndarray, cand_input: float) -> np.ndarray:
    """Append a candidate input to training inputs, handling 1D/2D shapes."""
    if train_inputs.ndim == 1:
        return np.append(train_inputs, cand_input)
    return np.concatenate((train_inputs, np.asarray(cand_input).reshape(1, -1)), axis=0)


def _gaussian_expectation_log(mean: float, std: float, *, n_evals: int | None = None, seed: int | None = None) -> float:
    """E[log Z] for Z ~ N(mean, std^2); falls back to sampling if requested."""
    tiny = np.finfo(float).tiny
    dist = norm(loc=mean, scale=std)
    if n_evals is None:
        return float(dist.expect(lambda z: np.log(np.maximum(z, tiny))))
    rng = np.random.default_rng(seed)
    samples = dist.rvs(size=max(1, n_evals), random_state=rng)
    return float(np.mean(np.log(np.maximum(samples, tiny))))


def estimate_expected_posterior_linex_score_cached(
    targ_inputs: np.ndarray,
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    cand_input: float,
    cand_labels: np.ndarray,
    model: RegressorWrapper,
    *,
    n_post_samples: int = 100,
    targ_weight=None,
    is_rf_model: bool = False,
    is_gp_model: bool = False,
    std: float = 0.1,
    seed: int | None = None,
) -> float:
    """
    Expected score over candidate labels using predictive posterior samples.

    Uses sampling for all model types (including GP) since E[log Z] is not
    available in closed form for a Gaussian predictive over Z itself.
    """
    original_state = model.get_model_state()

    def restore_original_state():
        model.load_model_state(original_state)

    inputs = _append_candidate(train_inputs, cand_input)
    rng = np.random.default_rng(seed)
    label_seeds = rng.integers(0, 2**31 - 1, size=len(cand_labels))

    scores = []
    for idx, y in enumerate(cand_labels):
        labels = np.append(train_labels, y)
        model.fit(inputs, labels)

        sample_seed = int(label_seeds[idx])
        if is_gp_model:
            pred_mean, pred_std = model.predict(targ_inputs, return_std=True)
            tiny = np.finfo(float).tiny
            mean_log = np.array(
                [
                    _gaussian_expectation_log(m, s, n_evals=n_post_samples, seed=sample_seed)
                    for m, s in zip(np.atleast_1d(pred_mean), np.atleast_1d(pred_std))
                ]
            )
            mean_val = np.asarray(pred_mean, dtype=float)
            per_target_scores = mean_log - np.log(np.maximum(mean_val, tiny))
            # pred_mean = np.asarray(pred_mean, dtype=float)
            # pred_var = np.asarray(pred_std, dtype=float) ** 2
            # log_mean_negexp = -pred_mean + 0.5 * pred_var
            # per_target_scores = -pred_mean - log_mean_negexp
            scores.append(_aggregate_target_scores(per_target_scores, targ_weight))
        else:
            if is_rf_model:
                pred_samples = model.predict_samples(
                    targ_inputs,
                    std=std,
                    n_samples=n_post_samples,
                    seed=sample_seed,
                )
            else:
                pred_samples = model.predict_samples(
                    targ_inputs,
                    n_samples=n_post_samples,
                    seed=sample_seed,
                )
            scores.append(predictive_posterior_linex_score(pred_samples, targ_weight))
        restore_original_state()

    return float(np.mean(scores))


def estimate_expected_posterior_linex_score_weighted_cached(
    targ_inputs: np.ndarray,
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    cand_input: float,
    cand_labels: np.ndarray,
    model: RegressorWrapper,
    weight_fn: Callable,
    *,
    n_post_samples: int = 100,
    targ_weight=None,
    is_rf_model: bool = False,
    is_gp_model: bool = False,
    std: float = 0.1,
    seed: int | None = None,
) -> float:
    """
    Weighted expected score over candidate labels using predictive posterior samples.
    """
    original_state = model.get_model_state()

    def restore_original_state():
        model.load_model_state(original_state)

    inputs = _append_candidate(train_inputs, cand_input)
    rng = np.random.default_rng(seed)
    label_seeds = rng.integers(0, 2**31 - 1, size=len(cand_labels))

    scores = []
    for idx, y in enumerate(cand_labels):
        labels = np.append(train_labels, y)
        model.fit(inputs, labels)

        sample_seed = int(label_seeds[idx])
        if is_gp_model:
            pred_samples = model.predict_samples(
                targ_inputs,
                n_samples=n_post_samples,
                seed=sample_seed,
            )
        else:
            if is_rf_model:
                pred_samples = model.predict_samples(
                    targ_inputs,
                    std=std,
                    n_samples=n_post_samples,
                    seed=sample_seed,
                )
            else:
                pred_samples = model.predict_samples(
                    targ_inputs,
                    n_samples=n_post_samples,
                    seed=sample_seed,
                )

        scores.append(
            predictive_posterior_linex_score_weighted(
                pred_samples, weight_fn, targ_weight
            )
        )
        restore_original_state()

    return float(np.mean(scores))


def prepare_gp_posterior_statistics(
    model: RegressorWrapper,
    targ_inputs: np.ndarray,
    pool_inputs: np.ndarray,
) -> dict[str, np.ndarray]:
    """Pre-compute GP posterior update quantities for variance/weighted metrics."""
    gpr = getattr(model, "reg", None)
    if gpr is None or not hasattr(gpr, "kernel_"):
        raise ValueError("Model must wrap a fitted GaussianProcessRegressor")
    if not hasattr(gpr, "L_"):
        raise ValueError("GaussianProcessRegressor appears to be unfitted (missing L_)")

    targ_inputs_2d = _ensure_2d_input(targ_inputs)
    pool_inputs_2d = _ensure_2d_input(pool_inputs)

    kernel = gpr.kernel_
    X_train = gpr.X_train_
    L = gpr.L_

    K_xt = kernel(X_train, targ_inputs_2d)
    K_xs = kernel(X_train, pool_inputs_2d)
    K_ts = kernel(targ_inputs_2d, pool_inputs_2d)

    solve = lambda rhs: cho_solve((L, True), rhs)
    A_inv_K_xs = solve(K_xs)

    cross = K_ts - K_xt.T @ A_inv_K_xs  # (n_targ, n_pool)

    prior_mean, prior_std = model.predict(targ_inputs, return_std=True)
    pool_mean, pool_std = model.predict(pool_inputs, return_std=True)

    prior_var = prior_std**2
    pool_var = pool_std**2

    tiny = _infer_dtype_eps(prior_var)
    denom = np.maximum(pool_var, tiny)

    posterior_var = prior_var[:, None] - (cross**2) / denom[None, :]
    posterior_var = np.maximum(posterior_var, 0.0)

    gain = cross / denom[None, :]

    return {
        "prior_mean": prior_mean,
        "prior_std": prior_std,
        "prior_var": prior_var,
        "pool_mean": pool_mean,
        "pool_var": pool_var,
        "gain": gain,
        "posterior_var": posterior_var,
        "posterior_std": np.sqrt(posterior_var),
    }


def compute_gp_expected_posterior_variance(stats: dict[str, np.ndarray]) -> np.ndarray:
    """Return per-candidate expected posterior variance averaged over targets."""
    posterior_var = stats["posterior_var"]
    return posterior_var.mean(axis=0)


def compute_gp_linex_score(
    stats: dict[str, np.ndarray],
    cand_samples: np.ndarray,
    *,
    targ_weight=None,
    n_evals: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Expected E[log Z] - log E[Z] per candidate using analytic GP updates with shared MC eps."""
    prior_mean = stats["prior_mean"]
    pool_mean = stats["pool_mean"]
    gain = stats["gain"]
    posterior_std = stats["posterior_std"]

    n_cand, n_label = cand_samples.shape
    tiny = np.finfo(float).tiny
    scores = np.empty(n_cand)

    n_draws = max(1, n_evals or 0)
    rng = np.random.default_rng(seed)
    eps_base = rng.standard_normal(size=(prior_mean.shape[0], n_draws))

    for i in range(n_cand):
        g = gain[:, i]
        std_vec = posterior_std[:, i]
        deltas = cand_samples[i] - pool_mean[i]
        per_label = []
        for delta in deltas:
            mean_vec = prior_mean + g * delta
            Z = mean_vec[:, None] + std_vec[:, None] * eps_base
            mean_log = np.log(np.maximum(Z, tiny)).mean(axis=1)
            score_targets = mean_log - np.log(np.maximum(mean_vec, tiny))
            # log_mean_negexp = -mean_vec + 0.5 * (std_vec**2)
            # score_targets = -mean_vec - log_mean_negexp
            per_label.append(_aggregate_target_scores(score_targets, targ_weight))
        scores[i] = float(np.mean(per_label))
    return scores


def compute_gp_weighted_linex_score(
    stats: dict[str, np.ndarray],
    cand_samples: np.ndarray,
    weight_fn: Callable,
    *,
    targ_weight=None,
    n_evals: int = 100,
    seed: int | None = None,
) -> np.ndarray:
    """Weighted E_w[log Z] - log E_w[Z] per candidate using GP updates with sampling."""
    prior_mean = stats["prior_mean"]
    pool_mean = stats["pool_mean"]
    gain = stats["gain"]
    posterior_std = stats["posterior_std"]

    n_cand, n_label = cand_samples.shape
    rng = np.random.default_rng(seed)
    eps_base = rng.standard_normal(size=(prior_mean.shape[0], max(1, n_evals)))

    scores = np.empty(n_cand)
    for i in range(n_cand):
        g = gain[:, i]
        std_vec = posterior_std[:, i]
        deltas = cand_samples[i] - pool_mean[i]
        per_label = []
        for delta in deltas:
            mean_vec = prior_mean + g * delta  # (n_targ,)
            # Generate posterior samples using shared eps_base
            Z = mean_vec[:, None] + std_vec[:, None] * eps_base
            per_label.append(
                predictive_posterior_linex_score_weighted(
                    Z,
                    weight_fn,
                    targ_weight=targ_weight,
                )
            )
        scores[i] = float(np.mean(per_label))
    return scores


def _weighted_uncertainty_from_samples(
    samples: np.ndarray,
    weight_fn: Callable,
    tiny: float,
) -> np.ndarray:
    weights = np.asarray(weight_fn(samples))
    numerator = (samples**2 * weights).mean(axis=-1)
    first_moment = (samples * weights).mean(axis=-1)
    denom = np.maximum(weights.mean(axis=-1), tiny)
    return numerator - (first_moment**2) / denom


def compute_gp_prior_weighted_uncertainty(
    stats: dict[str, np.ndarray],
    weight_fn: Callable,
    n_evals: int,
    eps_base: np.ndarray,
) -> float:
    if eps_base.shape[1] != n_evals:
        raise ValueError("eps_base must have n_evals columns")
    tiny = _infer_dtype_eps(stats["prior_var"])
    Z = stats["prior_mean"][:, None] + stats["prior_std"][:, None] * eps_base
    unc = _weighted_uncertainty_from_samples(Z, weight_fn, tiny)
    return float(unc.mean())


def compute_gp_expected_posterior_weighted_uncertainty(
    stats: dict[str, np.ndarray],
    cand_samples: np.ndarray,
    weight_fn: Callable,
    eps_base: np.ndarray,
) -> np.ndarray:
    gain = stats["gain"]
    prior_mean = stats["prior_mean"]
    pool_mean = stats["pool_mean"]
    posterior_std = stats["posterior_std"]

    if eps_base.shape[0] != prior_mean.shape[0]:
        raise ValueError("eps_base has incompatible number of rows")

    tiny = _infer_dtype_eps(stats["posterior_var"])

    n_cand = cand_samples.shape[0]
    expected = np.empty(n_cand)
    for idx in range(n_cand):
        deltas = cand_samples[idx] - pool_mean[idx]
        mu_new = prior_mean[:, None] + gain[:, idx][:, None] * deltas[None, :]
        Z = mu_new[:, :, None] + posterior_std[:, idx][:, None, None] * eps_base[:, None, :]
        unc = _weighted_uncertainty_from_samples(Z, weight_fn, tiny)
        expected[idx] = unc.mean(axis=0).mean()
    return expected

def estimate_true_expected_loss(
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    true_mean: np.ndarray,
    true_std: np.ndarray,
    weight_fn: Callable = lambda z: np.maximum(z, 1e-3),
    n_evals: int | None = None,
    quantile: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    w = weight_fn
    z_w = lambda z: z * w(z)

    expected_loss = []

    dists = zip(pred_mean, pred_std, true_mean, true_std)

    for pred_mean_i, pred_std_i, true_mean_i, true_std_i in dists:
        pred_z_dist = norm(loc=pred_mean_i, scale=pred_std_i)
        true_z_dist = norm(loc=true_mean_i, scale=true_std_i)

        if n_evals is None:
            optimal_pred = pred_z_dist.expect(z_w) / pred_z_dist.expect(w)
            optimal_loss = lambda z: w(z) * (z - optimal_pred) ** 2

            expected_loss_i = true_z_dist.expect(optimal_loss)
        elif quantile is None:
            pred_zs = pred_z_dist.rvs(size=n_evals, random_state=seed)
            true_zs = true_z_dist.rvs(size=n_evals, random_state=seed)

            optimal_pred = np.mean(z_w(pred_zs)) / np.mean(w(pred_zs))
            optimal_loss = lambda z: w(z) * (z - optimal_pred) ** 2

            expected_loss_i = np.mean(optimal_loss(true_zs))
        else:
            raise NotImplementedError

        expected_loss += [expected_loss_i]

    return np.array(expected_loss)

# for evaluation
def estimate_sqloss_w_gaussian(
    true_z: np.ndarray,
    pred_z: np.ndarray,
    pred_std: np.ndarray,
    weight_fn: Callable = lambda z: np.maximum(z, 1e-3),
    n_evals: int | None = None,
    quantile: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    w = weight_fn
    z_w = lambda z: z * w(z)

    sqloss = []

    for true_z_i, pred_z_i, pred_std_i in zip(true_z, pred_z, pred_std):
        if w(true_z_i) <= 1e-2 and w(pred_z_i) <= 1e-2:
            sqloss += [w(true_z_i) * (true_z_i - pred_z_i) ** 2]
            continue
        z_dist = norm(loc=pred_z_i, scale=pred_std_i)

        if n_evals is None:
            pred_z_w_i = z_dist.expect(z_w) / z_dist.expect(w)
            sqloss_i = w(true_z_i) * (true_z_i - pred_z_w_i) **2
        elif quantile is None:
            zs = z_dist.rvs(size=n_evals, random_state=seed)
            pred_z_w_i = np.mean(z_w(zs)) / np.mean(w(zs))
            sqloss_i = w(true_z_i) * (true_z_i - pred_z_w_i) ** 2
        else:
            zs = np.linspace(z_dist.ppf(quantile), z_dist.ppf(1 - quantile), n_evals)

            densities = z_dist.pdf(zs)
            densities /= np.sum(densities)

            pred_z_w_i = np.sum(z_w(zs) * densities) / np.sum(w(zs) * densities)
            sqloss_i = w(true_z_i) * (true_z_i - pred_z_w_i) ** 2

        sqloss += [sqloss_i]

    return np.mean(sqloss)

def estimate_sqloss_w_samples(
    true_z: np.ndarray,
    pred_samples: np.ndarray,  # Shape: [n_test_points, n_samples] - samples from the predictive distribution
    weight_fn: Callable = lambda z: np.maximum(z, 1e-3),
) -> float:
    """
    Estimate weighted squared loss using samples from the predictive distribution.
    
    Args:
        true_z: True values, shape [n_test_points]
        pred_samples: Samples from predictive distribution, shape [n_test_points, n_samples]
        weight_fn: Weight function to apply
    
    Returns:
        Mean weighted squared loss across all test points
    """
    w = weight_fn
    z_w = lambda z: z * w(z)
    
    sqloss = []
    
    for i, true_z_i in enumerate(true_z):
        samples_i = pred_samples[i, :]  # All samples for test point i
        weights_i = w(samples_i)
        
        # Handle case where both true and predicted values have very low weight
        if w(true_z_i) <= 1e-2 and np.all(weights_i <= 1e-2):
            # Add small regularization to weights
            regularized_weights = weights_i + 1e-6
            weighted_samples_i = samples_i * regularized_weights
            pred_z_w_i = np.sum(weighted_samples_i) / np.sum(regularized_weights)
        else:
            # Compute weighted expectation using samples
            
            weighted_samples_i = z_w(samples_i)
            
            # Weighted mean: E[z * w(z)] / E[w(z)]
            pred_z_w_i = np.mean(weighted_samples_i) / np.mean(weights_i)

        sqloss_i = w(true_z_i) * (true_z_i - pred_z_w_i) ** 2
        sqloss.append(sqloss_i)
    
    return np.mean(sqloss)

def compute_sqloss_unweighted_from_saved_data(all_pred_data, y_test, is_gp_models):
    """
    Compute unweighted squared loss from saved prediction data.
    """
    all_sqloss_unweighted = []

    for strategy_pred_data, is_gp in zip(all_pred_data, is_gp_models):
        n_runs, n_steps, n_test = strategy_pred_data.shape[:3]
        y_test_runs = _broadcast_y_test(y_test, n_runs, n_test)

        if is_gp:
            pred_means = strategy_pred_data[:, :, :, 0]
            sqloss_unweighted = np.mean((y_test_runs[:, None, :] - pred_means) ** 2, axis=2)
        else:
            pred_means = np.mean(strategy_pred_data, axis=3)
            sqloss_unweighted = np.mean((y_test_runs[:, None, :] - pred_means) ** 2, axis=2)

        all_sqloss_unweighted.append(sqloss_unweighted)

    return all_sqloss_unweighted


def compute_sqloss_weighted_from_saved_data(all_pred_data, y_test, weight_fn, is_gp_models, n_evals=100):
    """
    Compute weighted squared loss from saved prediction data.
    """
    all_sqloss_weighted = []

    for strategy_pred_data, is_gp in zip(all_pred_data, is_gp_models):
        n_runs, n_steps, n_test = strategy_pred_data.shape[:3]
        y_test_runs = _broadcast_y_test(y_test, n_runs, n_test)

        if is_gp:
            pred_means = strategy_pred_data[:, :, :, 0]
            pred_stds = strategy_pred_data[:, :, :, 1]
            sqloss_weighted = np.zeros((n_runs, n_steps))
            for run_idx in range(n_runs):
                for step_idx in range(n_steps):
                    sqloss_weighted[run_idx, step_idx] = estimate_sqloss_w_gaussian(
                        y_test_runs[run_idx],
                        pred_means[run_idx, step_idx, :],
                        pred_stds[run_idx, step_idx, :],
                        weight_fn,
                        n_evals,
                    )
        else:
            sqloss_weighted = np.zeros((n_runs, n_steps))
            for run_idx in range(n_runs):
                for step_idx in range(n_steps):
                    samples = strategy_pred_data[run_idx, step_idx, :, :]
                    sqloss_weighted[run_idx, step_idx] = estimate_sqloss_w_samples(
                        y_test_runs[run_idx],
                        samples,
                        weight_fn,
                    )

        all_sqloss_weighted.append(sqloss_weighted)

    return all_sqloss_weighted


def _broadcast_y_test(y_test, n_runs: int, n_test: int) -> np.ndarray:
    """Ensure y_test has shape (n_runs, n_test) for metric helpers."""
    y_test_arr = np.asarray(y_test)
    if y_test_arr.ndim == 1:
        if y_test_arr.shape[0] != n_test:
            raise ValueError(f"Expected {n_test} test targets, got {y_test_arr.shape[0]}")
        return np.broadcast_to(y_test_arr, (n_runs, n_test))
    if y_test_arr.ndim == 2:
        if y_test_arr.shape[1] != n_test:
            raise ValueError(
                f"Test target shape mismatch: expected {n_test}, got {y_test_arr.shape[1]}"
            )
        if y_test_arr.shape[0] < n_runs:
            raise ValueError(
                f"Not enough test target runs: have {y_test_arr.shape[0]}, need {n_runs}"
            )
        return y_test_arr[:n_runs]
    raise ValueError("y_test must be 1D or 2D array")


def compute_linex_metrics_from_saved_data(
    all_pred_data,
    y_test,
    is_gp_models,
    alpha: float = 1.0,
    weight_fn=None,
):
    """
    Compute LINEX-style metrics:
        L = y + exp(-y) / E[exp(-Z)] + log(E[exp(-Z)]) - 1
    where Z is the predictive posterior. For GP, E[exp(-Z)] is analytic; for
    samples, it is estimated via Monte Carlo.
    """
    all_linex = []
    for pred_data, is_gp in zip(all_pred_data, is_gp_models):
        n_runs, n_steps, n_test = pred_data.shape[:3]
        y_runs = _broadcast_y_test(y_test, n_runs, n_test)
        weights = None
        if weight_fn is not None:
            weights = np.asarray(weight_fn(y_runs), dtype=float)
            if weights.shape != y_runs.shape:
                raise ValueError("weight_fn must return the same shape as y_test")

        losses = np.zeros((n_runs, n_steps))
        if is_gp:
            pred_means = pred_data[:, :, :, 0]
            pred_stds = pred_data[:, :, :, 1]
            # Ratio-style implementation:
            # tiny = np.finfo(float).tiny
            # ratio = np.maximum(y_runs[:, None, :], tiny) / np.maximum(pred_means, tiny)
            # losses = ratio - np.log(ratio) - 1.0
            # Use the log-domain form to avoid overflow in exp(...) followed by log(...).
            log_exp_neg = -pred_means + 0.5 * (pred_stds**2)
            reciprocal = np.exp(np.clip(-y_runs[:, None, :] - log_exp_neg, a_min=None, a_max=700))
            losses = (
                y_runs[:, None, :]
                + reciprocal
                + log_exp_neg
                - 1.0
            )
            if weights is not None:
                losses = losses * weights[:, None, :]
            losses = np.mean(losses, axis=2)
        else:
            # samples: [n_runs, n_steps, n_test, n_samples]
            exp_neg = np.mean(np.exp(-pred_data), axis=3)
            # Ratio-style implementation:
            # tiny = np.finfo(float).tiny
            # pred_means = np.mean(pred_data, axis=3)
            # ratio = np.maximum(y_runs[:, None, :], tiny) / np.maximum(pred_means, tiny)
            # linex_terms = ratio - np.log(ratio) - 1.0
            linex_terms = (
                y_runs[:, None, :]
                + np.exp(-y_runs[:, None, :]) / exp_neg
                - np.log(exp_neg)
                - 1.0
            )
            if weights is not None:
                linex_terms = linex_terms * weights[:, None, :]
            losses = np.mean(linex_terms, axis=2)
        all_linex.append(losses)

    return all_linex
