import logging
import math
import numpy as np
import torch
from torch import Tensor

def create_grid2d(data_args: tuple) -> np.ndarray:
    xs = np.linspace(*data_args)
    ys = np.linspace(*data_args)
    xx2d, yy2d = np.meshgrid(xs, ys)
    return np.column_stack([xx2d.ravel(), yy2d.ravel()])

def create_even_angle_grid(n: int, scale: float = 1) -> np.ndarray:
    """
    Create a grid of n points evenly spaced on the unit circle.
    
    Args:
        n: Integer number of points
        scale: Float scale factor to apply to the coordinates
        
    Returns:
        Array of shape (n, 2) with coordinates of points on the unit circle,
        starting at angle π/n and spaced evenly by 2π/n
    """
    # Generate n angles starting at π/n with step size 2π/n
    angles = np.pi/n + np.arange(n) * (2*np.pi/n)
    
    # Convert angles to cartesian coordinates on unit circle
    x = np.cos(angles) * scale
    y = np.sin(angles) * scale
    
    # Combine x,y coordinates
    return np.column_stack((x, y))

def gaussian_weights_torch(points, mean=None, cov=None):
    """
    Calculate weights for points according to a Gaussian distribution using PyTorch.
    
    Args:
        points: Tensor of shape (n, d) containing d-dimensional points
        mean: Mean vector of the Gaussian, default zeros
        cov: Covariance matrix of the Gaussian, default 2I
        
    Returns:
        Tensor of weights for each point, normalized to sum to 1
    """
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points, dtype=torch.float32)
    
    n_dims = points.shape[1]
    
    if mean is None:
        mean = torch.zeros(n_dims, device=points.device)
    elif not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, dtype=torch.float32, device=points.device)
        
    if cov is None:
        cov = 2.5 * torch.eye(n_dims, device=points.device)
    elif not isinstance(cov, torch.Tensor):
        cov = torch.tensor(cov, dtype=torch.float32, device=points.device)
    
    # Calculate determinant and inverse of covariance matrix
    det_cov = torch.det(cov)
    inv_cov = torch.inverse(cov)
    
    # Calculate Mahalanobis distance for each point
    x_minus_mu = points - mean
    
    # Matrix multiplication approach for Mahalanobis distance
    mahalanobis = torch.sum(torch.matmul(x_minus_mu, inv_cov) * x_minus_mu, dim=1)
    
    # Calculate Gaussian density
    normalization = 1.0 / torch.sqrt((2 * torch.pi) ** n_dims * det_cov)
    densities = normalization * torch.exp(-0.5 * mahalanobis)
    
    # Normalize weights to sum to 1
    weights = densities / torch.sum(densities)
    
    return weights

def median_heuristic(inputs: np.ndarray) -> float:
    """
    Compute the median heuristic for the RBF/Gaussian kernel length scale,
    defined as the median of all pairwise Euclidean distances between samples in X.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data matrix.
    
    Returns
    -------
    float
        The median of pairwise distances.
    """
    n_samples = inputs.shape[0]
    # Compute all pairwise distances
    diffs = inputs[:, np.newaxis, :] - inputs[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    # Extract upper-triangular (i < j) distances to avoid duplicates and zeros
    i_upper = np.triu_indices(n_samples, k=1)
    pairwise_dists = dists[i_upper]
    return np.median(pairwise_dists)

def silverman_length_scale(X):
    """
    Compute Silverman’s rule-of-thumb for the RBF/Gaussian kernel length scale,
    using the average of per-feature standard deviations.

    ℓ = (4 / (d + 2))^(1/(d+4)) * n^(-1/(d+4)) * sigma_avg

    where
    - n = number of samples
    - d = number of features
    - sigma_avg = mean of standard deviations of each feature

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data.

    Returns
    -------
    float
        Silverman’s length scale.
    """
    X = np.asarray(X, dtype=float)
    n_samples, n_features = X.shape
    
    # Compute per-feature standard deviations (ddof=1 for sample std)
    feature_stds = np.std(X, axis=0, ddof=1)
    sigma_avg = np.mean(feature_stds)
    
    # Silverman coefficient
    coef = (4.0 / (n_features + 2.0)) ** (1.0 / (n_features + 4.0))
    exponent = -1.0 / (n_features + 4.0)
    
    length_scale = coef * (n_samples ** exponent) * sigma_avg
    return length_scale

def check(
    scores: Tensor, min_value: float = 0.0, max_value: float = math.inf, score_type: str = ""
) -> Tensor:
    """
    Warn if any element of scores is a NaN or lies outside the range [min_value, max_value].
    """
    epsilon = 100 * torch.finfo(scores.dtype).eps

    if not torch.all((scores >= min_value - epsilon) & (scores <= max_value + epsilon)):
        min_score = torch.min(scores).item()
        max_score = torch.max(scores).item()

        logging.warning(
            f"Invalid score (type = {score_type}, min = {min_score}, max = {max_score})"
        )

    return scores

def compute_joint_matmul(
    probs_pool: Tensor, probs_targ: Tensor
) -> Tensor:
    """
    Compute the joint probability distribution p(y,y_*|x,x_*) from p(y|x) and p(y_*|x_*).

    Arguments:
        probs_pool: Tensor[float], [N_p, K, Cl]
        probs_targ: Tensor[float], [N_t, K, Cl]

    Returns:
        probs_joint: Tensor[float], [N_p, Cl, N_t, Cl]
    """
    assert probs_pool.ndim == probs_targ.ndim == 3

    _, K, _ = probs_targ.shape
    probs_joint = torch.einsum('i k n,  j k m -> i n j m', probs_pool, probs_targ) / K  # [N_p, Cl, N_t, Cl]

    return probs_joint

def epig_from_probs(probs_pool: Tensor, probs_targ: Tensor, is_weight=None) -> Tensor:

    _, K, Cl = probs_targ.shape

    probs_joint = compute_joint_matmul(probs_pool, probs_targ)  # [N_p, Cl, N_t, Cl]

    log_joint = torch.log(probs_joint + 1e-12)  # [N_p, Cl, N_t, Cl]
    log_pz = torch.log(probs_targ.mean(dim=1) + 1e-12)  # [N_t, Cl]
    log_py = torch.log(probs_joint.sum(dim=-1) + 1e-12)  # [N_p, Cl, N_t]

    mutual_info = torch.sum(probs_joint * (log_joint - log_pz[None, None, :, :] - log_py[:, :, :, None]), dim=(-3, -1))  # [N_p, N_t]

    if is_weight is None:
        scores = mutual_info.mean(dim=-1)
    else:
        scores = (mutual_info * is_weight[None, :]).sum(dim=-1)

    scores = check(scores, max_value=math.log(Cl**2), score_type="EPIG")  # [N_p,]

    return scores.cpu().numpy()  # [N_p,]

def epig_from_probs_w(probs_pool: Tensor, probs_targ: Tensor, weights: Tensor, is_weight=None) -> Tensor:

    _, _, Cl = probs_targ.shape

    probs_targ_w = probs_targ.mean(dim=1) * weights[None, :]  # [N_t, Cl]
    expected_w = probs_targ_w.sum(-1)  # [N_t]
    probs_targ_w = probs_targ_w / expected_w[:, None]  # [N_t, Cl]

    probs_joint = compute_joint_matmul(probs_pool, probs_targ)  # [N_p, Cl, N_t, Cl]
    probs_joint_w = weights[None, None, None, :] * probs_joint  # [N_p, Cl, N_t, Cl]

    log_joint_w = torch.log(probs_joint_w + 1e-12)  # [N_p, Cl, N_t, Cl]
    log_pz_w = torch.log(probs_targ_w + 1e-12)  # [N_t, Cl]
    log_py_w = torch.log(probs_joint_w.sum(dim=-1) + 1e-12)  # [N_p, Cl, N_t]

    mutual_info_w = torch.sum(probs_joint_w * (log_joint_w-log_pz_w[None, None, :, :]-log_py_w[:, :, :, None]), dim=(-3, -1))  # [N_p, N_t]

    if is_weight is None:
        scores = mutual_info_w.mean(dim=-1) / expected_w.mean()
    else:
        scores = (mutual_info_w * is_weight[None, :]).sum(dim=-1) / (expected_w * is_weight).sum()

    scores = check(scores, max_value=math.log(Cl**2), score_type="EPIG_w")  # [N_p,]

    return scores.cpu().numpy()  # [N_p,]

def _probs_weighted(
    probs: Tensor, weight: Tensor
) -> Tensor:
    """
    Compute weighted probabilities from pred_probs and weight.
    """
    assert probs.ndim == 3
    N, K, Cl = probs.shape
    assert Cl == weight.shape[0]
    
    un = probs * weight[None, None, :]  # [N, K, Cl]
    norm_factor = torch.sum(un, dim=-1, keepdims=True)  # [N, K, 1]
    return un / norm_factor  # [N, K, Cl]

def _compute_joint_weighted(
    probs_pool: Tensor, probs_targ: Tensor, weights: Tensor
) -> Tensor:
    """
    Compute the joint probability distribution p(y,y_*|x,x_*) from p(y|x) and p(y_*|x_*) with weights.

    Arguments:
        probs_pool: Tensor[float], [N_p, K, Cl]
        probs_targ: Tensor[float], [N_t, K, Cl]
        weights: Tensor[float], [Cl,]

    Returns:
        probs_joint: Tensor[float], [N_p, Cl, N_t, Cl]
    """
    assert probs_pool.ndim == probs_targ.ndim == 3
    assert weights.ndim == 1

    N_p, _, _ = probs_pool.shape
    N_t, _, _ = probs_targ.shape

    probs_joint = compute_joint_matmul(probs_pool, probs_targ)  # [N_p, Cl, N_t, Cl]
    numerator = weights[None, None, None, :] * probs_joint  # [N_p, Cl, N_t, Cl]
    denominator = weights[None, :] * probs_targ.mean(dim=1)  # [N_t, Cl]
    probs_joint_w = numerator / denominator.sum(dim=-1, keepdim=True)[None, None, :]  # [N_p, Cl, N_t, Cl]
    
    check(probs_joint_w.sum(dim=(1, 2, 3)) / N_t, min_value=1.0, max_value=1.0, score_type="MP")
    check(probs_joint_w.sum(dim=(0, 1, 3)) / N_p, min_value=1.0, max_value=1.0, score_type="MP")

    return probs_joint_w  # [N_p, Cl, N_t, Cl]

def _compute_marginal_weighted(probs_pool: Tensor, probs_targ: Tensor, weights: Tensor) -> Tensor:
    """
    Compute q(y)=E_q(p(y_pool | y_target)) from samples
    
    Args:
        probs_pool: [N_p, K, Cl]
        probs_targ: [N_t, K, Cl]
        weights: [Cl,]
    
    Returns:
        probs_marg_p: [N_p, Cl, N_t] tensor of p(y_pool | y_target)
    """

    probs_joint = compute_joint_matmul(probs_pool, probs_targ)  # [N_p, Cl, N_t, Cl]
    numerator = weights[None, None, None, :] * probs_joint  # [N_p, Cl, N_t, Cl]
    denominator = weights[None, :] * probs_targ.mean(dim=1)  # [N_t, Cl]

    probs_marg_p = numerator.sum(-1) / denominator.sum(-1)[None, None, :]  # [N_p, Cl, N_t]

    check(probs_marg_p.sum(dim=(-1, -2))/probs_targ.shape[0], min_value=1.0, max_value=1.0, score_type="MP")

    return probs_marg_p

def _marginal_entropy_from_probs(probs: Tensor) -> Tensor:
    """
    H[E_{p(θ)}[p(y|x,θ)]]

    Arguments:
        probs: Tensor[float], [N, K, Cl] or [N, Cl]

    Returns:
        Tensor[float], [N,]
    """
    if probs.ndim == 3:
        probs = torch.mean(probs, dim=1)  # [N, Cl]
    elif probs.ndim != 2:
        raise ValueError(f"Invalid probs shape: {probs.shape}")

    scores = -torch.sum(torch.xlogy(probs, probs), dim=-1)  # [N,]
    check(scores, max_value=math.log(probs.shape[-1]**2), score_type="ME")  # [N,]

    return scores  # [N,]

def _marginal_entropy_from_probs_w(probs: Tensor) -> Tensor:
    """
    H[E_{p(θ)}[p(y|x,θ,x_*)]]

    Arguments:
        probs: Tensor[float], [N_p, Cl, N_t]

    Returns:
        Tensor[float], [N_p, N_t]
    """

    _, Cl, _ = probs.shape

    scores = -torch.sum(torch.xlogy(probs, probs), dim=1)  # [N_p, N_t]

    check(scores.mean(0), max_value=math.log(Cl**2), score_type="ME")
    check(scores.mean(1), max_value=math.log(Cl**2), score_type="ME")

    return scores  # [N_p, N_t]

def _joint_entropy_from_probs(probs_pool: Tensor, probs_targ: Tensor) -> Tensor:
    """
    H[p(y,y_*|x,x_*)]

    References:
        https://github.com/baal-org/baal/pull/270#discussion_r1271487205

    Arguments:
        probs_pool: Tensor[float], [N_p, K, Cl]
        probs_targ: Tensor[float], [N_t, K, Cl]
        is_weight: Tensor[float], [N_t,] or None

    Returns:
        Tensor[float], [N_p, N_t]
    """
    assert probs_pool.ndim == probs_targ.ndim == 3

    _, _, Cl = probs_targ.shape

    probs_joint = compute_joint_matmul(probs_pool, probs_targ)  # [N_p, Cl, N_t, Cl]

    scores = -torch.sum(torch.xlogy(probs_joint, probs_joint), dim=(-3, -1)) # [N_p, N_t]

    check(scores.mean(0), max_value=math.log(Cl**2), score_type="ME")
    check(scores.mean(1), max_value=math.log(Cl**2), score_type="ME")

    return scores  # [N_p, N_t]

def _joint_entropy_from_probs_w(probs_joint_w: Tensor) -> Tensor:
    """
    H[p(y,y_*|x,x_*)]

    References:
        https://github.com/baal-org/baal/pull/270#discussion_r1271487205

    Arguments:
        probs_pool: Tensor[float], [N_p, K, Cl]
        probs_targ: Tensor[float], [N_t, K, Cl]
        is_weight: Tensor[float], [N_t,] or None

    Returns:
        Tensor[float], [N_p, N_t]
    """
    assert probs_joint_w.ndim == 4

    _, Cl, _, _ = probs_joint_w.shape

    scores = -torch.sum(torch.xlogy(probs_joint_w, probs_joint_w), dim=(-3, -1)) # [N_p, N_t]

    check(scores.mean(0), max_value=math.log(Cl**2), score_type="ME")
    check(scores.mean(1), max_value=math.log(Cl**2), score_type="ME")

    return scores  # [N_p, N_t]

def _epig_from_probs(probs_pool: Tensor, probs_targ: Tensor, is_weight=None) -> Tensor:
    """
    EPIG(x) = E_{p_*(x_*)}[I(y;y_*|x,x_*)]
            = H[p(y|x)] + E_{p_*(x_*)}[H[p(y_*|x_*)]] - E_{p_*(x_*)}[H[p(y,y_*|x,x_*)]]

    This uses the fact that I(A;B) = H(A) + H(B) - H(A,B).

    References:
        https://en.wikipedia.org/wiki/Mutual_information#Relation_to_conditional_and_joint_entropy

    Arguments:
        probs_pool: Tensor[float], [N_p, K, Cl]
        probs_targ: Tensor[float], [N_t, K, Cl]
        is_weight: Tensor[float], [N_t,] or None

    Returns:
        Tensor[float], [N_p,]
    """
    _, _, Cl = probs_targ.shape

    entropy_pool = _marginal_entropy_from_probs(probs_pool)  # [N_p,]
    entropy_targ = _marginal_entropy_from_probs(probs_targ)  # [N_t,]
    entropy_joint = _joint_entropy_from_probs(probs_pool, probs_targ)  # [N_p, N_t]

    if is_weight is None:
        scores = entropy_pool + torch.mean(entropy_targ)\
                    - torch.mean(entropy_joint, dim=-1)  # [N_p,]
    else:
        scores = entropy_pool + torch.sum(entropy_targ * is_weight)\
                    - torch.sum(entropy_joint * is_weight[None, :], dim=-1)  # [N_p,]
    scores = check(scores, max_value=math.log(Cl**2), score_type="EPIG")  # [N_p,]

    return scores.cpu().numpy()  # [N_p,]

def _epig_from_probs_w(probs_pool: Tensor, probs_targ: Tensor, weights: Tensor, is_weight=None) -> Tensor:
    """
    EPIG_w(x) = E_{q_*(x_*)}[I(y;y_*|x,x_*)]
            = H[q(y|x)] + E_{q_*(x_*)}[H[p(y_*|x_*)]] - E_{q_*(x_*)}[H[p(y,y_*|x,x_*)]]

    This uses the fact that I(A;B) = H(A) + H(B) - H(A,B).

    Arguments:
        probs_pool: Tensor[float], [N_p, K, Cl]
        probs_targ: Tensor[float], [N_t, K, Cl]
        weights: Tensor[float], [Cl,]

    Returns:
        Tensor[float], [N_p,]
    """
    _, K, Cl = probs_targ.shape

    probs_joint_w = _compute_joint_weighted(probs_pool, probs_targ, weights)  # [N_p, Cl, N_t, Cl]
    probs_pool_weighted = _compute_marginal_weighted(probs_pool, probs_targ, weights)  # [N_p, Cl, N_t]

    entropy_pool = _marginal_entropy_from_probs_w(probs_pool_weighted)  # [N_p, N_t]

    probs_targ_weighted = _probs_weighted(probs_targ, weights)  # [N_t, K, Cl]
    entropy_targ = _marginal_entropy_from_probs(probs_targ_weighted)  # [N_t,]
    entropy_joint = _joint_entropy_from_probs_w(probs_joint_w)  # [N_p, N_t]

    total_weight = torch.sum(probs_targ * weights[None, None, :], dim=(1, 2)) / K  # [N_t,]
    if is_weight:
        total_weight *= is_weight
    total_weight /= torch.sum(total_weight)  # [N_t,]
    scores = torch.sum((entropy_pool - entropy_joint) * total_weight[None, :], dim=-1)\
            + torch.sum(entropy_targ * total_weight)  # [N_p,]

    scores = check(scores, max_value=math.log(Cl**2), score_type="EPIG_w")  # [N_p,]

    return scores.cpu().numpy()  # [N_p,]

def _epig_components_from_probs(probs_pool: Tensor, probs_targ: Tensor, is_weight=None) -> dict:
    """
    Return individual entropy components for EPIG calculation and visualization.
    
    Returns:
        dict with keys: 'entropy_pool', 'entropy_targ', 'entropy_joint', 'posterior_entropy'
    """

    entropy_pool = _marginal_entropy_from_probs(probs_pool)  # [N_p,]
    entropy_targ = _marginal_entropy_from_probs(probs_targ)  # [N_t,]
    entropy_joint = _joint_entropy_from_probs(probs_pool, probs_targ)  # [N_p, N_t]
    
    if is_weight is None:
        posterior_entropy = -torch.mean(entropy_targ) + torch.mean(entropy_joint, dim=-1)  # [N_p,]
    else:
        posterior_entropy = -torch.sum(entropy_targ * is_weight) + torch.sum(entropy_joint * is_weight[None, :], dim=-1)  # [N_p,]

    return {
        'prob_pool': probs_pool.mean(dim=1).cpu().numpy(),
        'entropy_pool': entropy_pool.cpu().numpy(),
        'entropy_targ': entropy_targ.cpu().numpy(),
        'entropy_joint': entropy_joint.mean(dim=-1).cpu().numpy(),
        'posterior_entropy': posterior_entropy.cpu().numpy()
    }

def _epig_components_from_probs_w(probs_pool: Tensor, probs_targ: Tensor, weights: Tensor, is_weight=None) -> dict:
    """
    Return individual entropy components for weighted EPIG calculation and visualization.
    
    Returns:
        dict with keys: 'entropy_pool', 'entropy_targ', 'entropy_joint', 'posterior_entropy'
    """
    _, _, Cl = probs_targ.shape

    probs_joint_w = _compute_joint_weighted(probs_pool, probs_targ, weights)  # [N_p, Cl, N_t, Cl]
    probs_pool_weighted = _compute_marginal_weighted(probs_pool, probs_targ, weights)  # [N_p, Cl, N_t]

    entropy_pool = _marginal_entropy_from_probs_w(probs_pool_weighted)  # [N_p, N_t]

    probs_targ_weighted = _probs_weighted(probs_targ, weights)  # [N_t, K, Cl]
    entropy_targ = _marginal_entropy_from_probs(probs_targ_weighted)  # [N_t,]
    entropy_joint = _joint_entropy_from_probs_w(probs_joint_w)  # [N_p, N_t]

    total_weight = torch.sum(probs_targ.mean(dim=1) * weights[None, :], dim=1)  # [N_t,]
    total_weight = total_weight / torch.sum(total_weight)  # [N_t,]
    if is_weight:
        total_weight = total_weight * is_weight
    posterior_entropy = torch.sum(entropy_joint * total_weight[None, :], dim=-1)\
            - torch.sum(entropy_targ * total_weight)  # [N_p,]


    return {
        'prob_pool': torch.sum(probs_pool_weighted * total_weight[None, None, :], dim=-1).cpu().numpy(),
        'entropy_pool': torch.sum(entropy_pool * total_weight[None, :], dim=-1).cpu().numpy(),
        'entropy_targ': entropy_targ.cpu().numpy(),
        'entropy_joint': torch.sum(entropy_joint * total_weight[None, :], dim=-1).cpu().numpy(),
        'posterior_entropy': posterior_entropy.cpu().numpy()
    }


def _broadcast_y_test(y_test, n_runs: int, n_test: int) -> np.ndarray:
    """Ensure classification labels have shape (n_runs, n_test)."""
    y_test_arr = np.asarray(y_test)
    if y_test_arr.ndim == 1:
        if y_test_arr.shape[0] != n_test:
            raise ValueError(f"Expected {n_test} test labels, got {y_test_arr.shape[0]}")
        return np.broadcast_to(y_test_arr, (n_runs, n_test))
    if y_test_arr.ndim == 2:
        if y_test_arr.shape[1] != n_test:
            raise ValueError(
                f"Test label shape mismatch: expected {n_test}, got {y_test_arr.shape[1]}"
            )
        if y_test_arr.shape[0] < n_runs:
            raise ValueError(
                f"Not enough test-label runs: have {y_test_arr.shape[0]}, need {n_runs}"
            )
        return y_test_arr[:n_runs]
    raise ValueError("y_test must be a 1D or 2D array")

def compute_nll_metrics_from_saved_data(all_pred_data, y_test, weight):
    """
    Compute NLL metrics from saved prediction data.
    
    Args:
        all_pred_data: List of arrays, each with shape [n_runs, n_steps, n_test_samples, n_classes]
        y_test: True test labels, shape [n_test_samples]
        weight: Class weights, shape [n_classes]
    
    Returns:
        all_nll_data: List of arrays with shape [n_runs, n_steps] for unweighted NLL
        all_nll_w_data: List of arrays with shape [n_runs, n_steps] for weighted NLL
    """
    weight = np.array(weight)
    all_nll_data = []
    all_nll_w_data = []
    
    for strategy_pred_data in all_pred_data:
        n_runs, n_steps, n_test_samples, _ = strategy_pred_data.shape
        y_test_runs = _broadcast_y_test(y_test, n_runs, n_test_samples)

        true_class_probs = np.empty((n_runs, n_steps, n_test_samples))
        weighted_true_probs = np.empty((n_runs, n_steps, n_test_samples))
        true_class_weights = np.empty((n_runs, n_test_samples))

        weighted_probs = strategy_pred_data * weight[None, None, None, :]
        weighted_probs = weighted_probs / np.sum(weighted_probs, axis=3, keepdims=True)

        for run_idx in range(n_runs):
            labels = y_test_runs[run_idx]
            gather_idx = labels[None, :, None]
            true_class_probs[run_idx] = np.take_along_axis(
                strategy_pred_data[run_idx],
                gather_idx,
                axis=2,
            ).squeeze(-1)
            weighted_true_probs[run_idx] = np.take_along_axis(
                weighted_probs[run_idx],
                gather_idx,
                axis=2,
            ).squeeze(-1)
            true_class_weights[run_idx] = weight[labels]

        nll_unweighted = -np.mean(np.log(true_class_probs + 1e-12), axis=2)
        nll_weighted = -np.sum(
            true_class_weights[:, None, :] * np.log(weighted_true_probs + 1e-12),
            axis=2,
        ) / np.sum(true_class_weights, axis=1, keepdims=True)
        
        all_nll_data.append(nll_unweighted)
        all_nll_w_data.append(nll_weighted)
    
    return all_nll_data, all_nll_w_data

def compute_accuracy_metrics_from_saved_data(all_pred_data, y_test, weight):
    """
    Compute accuracy metrics from saved prediction data.
    
    Args:
        all_pred_data: List of arrays, each with shape [n_runs, n_steps, n_test_samples, n_classes]
        y_test: True test labels, shape [n_test_samples]
        weight: Class weights, shape [n_classes]
    
    Returns:
        all_acc_data: List of arrays with shape [n_runs, n_steps] for unweighted accuracy
        all_acc_w_data: List of arrays with shape [n_runs, n_steps] for weighted accuracy
    """
    weight = np.array(weight)
    all_acc_data = []
    all_acc_w_data = []
    
    for strategy_pred_data in all_pred_data:
        n_runs, _, n_test_samples, _ = strategy_pred_data.shape
        y_test_runs = _broadcast_y_test(y_test, n_runs, n_test_samples)

        pred_classes = np.argmax(strategy_pred_data, axis=3)
        correct_unweighted = (pred_classes == y_test_runs[:, None, :])
        acc_unweighted = np.mean(correct_unweighted, axis=2)

        weighted_probs = strategy_pred_data * weight[None, None, None, :]
        weighted_probs = weighted_probs / np.sum(weighted_probs, axis=3, keepdims=True)
        pred_classes_weighted = np.argmax(weighted_probs, axis=3)

        true_class_weights = weight[y_test_runs]
        correct_weighted = (pred_classes_weighted == y_test_runs[:, None, :]).astype(float)
        acc_weighted = np.sum(
            true_class_weights[:, None, :] * correct_weighted,
            axis=2,
        ) / np.sum(true_class_weights, axis=1, keepdims=True)
        
        all_acc_data.append(acc_unweighted)
        all_acc_w_data.append(acc_weighted)
    
    return all_acc_data, all_acc_w_data
