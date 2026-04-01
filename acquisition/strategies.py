import numpy as np
import torch
from abc import ABC, abstractmethod

from acquisition.utils import get_weight_identifier
from acquisition.utils.classification import (
    epig_from_probs, epig_from_probs_w,
    _epig_components_from_probs, _epig_components_from_probs_w,
)
from acquisition.utils.regression import (
    estimate_expected_posterior_variance_cached,
    estimate_wquad_uncertainty,
    estimate_expected_posterior_wquad_uncertainty_cached,
    compute_expected_variance_reduction,
    compute_expected_weighted_variance_reduction,
    prepare_gp_posterior_statistics,
    compute_gp_expected_posterior_variance,
    compute_gp_prior_weighted_uncertainty,
    compute_gp_expected_posterior_weighted_uncertainty,
    compute_gp_linex_score,
    compute_gp_weighted_linex_score,
    estimate_expected_posterior_linex_score_cached,
    estimate_expected_posterior_linex_score_weighted_cached,
)

class AcquisitionStrategy(ABC):
    def __init__(self, model, name=None):
        self.model = model
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def score(self, X, y, pool, targ): ...

    def select_best(self, scores, X, pool):
        acq_scores = np.copy(scores)
        # mask out already-seen points, support 1D or multi-D pools
        if pool.ndim == 1:
            seen = np.isin(pool, X)
        else:
            seen = np.any(np.all(pool[:, None, :] == X[None, :, :], axis=2), axis=1)
        acq_scores[seen] = -np.inf
        return int(np.argmax(acq_scores))


class Random(AcquisitionStrategy):
    def __init__(self, model, name="random"):
        super().__init__(model, name)
    def score(self, X, y, pool, targ, n_samples=100, is_weight=None, fitted=False, **kwargs):
        if not fitted:
            self.model.fit(X, y)
        return np.zeros(len(pool))
    def select_best(self, scores, X, pool):
        # mask out already-seen points, support 1D or multi-D pools
        if pool.ndim == 1:
            seen = np.isin(pool, X)
        else:
            seen = np.any(np.all(pool[:, None, :] == X[None, :, :], axis=2), axis=1)
        # get indices of unseen points
        unseen_indices = np.where(~seen)[0]
        if len(unseen_indices) == 0:
            raise ValueError("No unseen points available in pool")
        # randomly select from unseen points
        return int(np.random.choice(unseen_indices))


class ClfEntropyReduction(AcquisitionStrategy):
    def __init__(self, model, name="entropy"):
        super().__init__(model, name)
    def score(self, X, y, pool, targ, n_samples=100, is_weight=None, fitted=False):
        if not fitted:
            self.model.fit(X, y)
        probs_pool, probs_targ = self.model.get_samples(pool, targ, n_samples=n_samples)
        return epig_from_probs(probs_pool, probs_targ, is_weight)
    def _score_components(self, X, y, pool, targ, n_samples=100, is_weight=None, fitted=False):
        if not fitted:
            self.model.fit(X, y)
        probs_pool, probs_targ = self.model.get_samples(pool, targ, n_samples=n_samples)   
        return _epig_components_from_probs(probs_pool, probs_targ, is_weight)


class ClfWeightedEntropyReduction(AcquisitionStrategy):
    def __init__(self, model, weight, name="entropy_w"):
        super().__init__(model, name)
        self.weight = torch.tensor(weight)
        self.name = f"{name}_{'_'.join(map(str, weight))}"
    def score(self, X, y, pool, targ, n_samples=100, is_weight=None, fitted=False):
        if not fitted:
            self.model.fit(X, y)
        probs_pool, probs_targ = self.model.get_samples(pool, targ, n_samples=n_samples)
        return epig_from_probs_w(probs_pool, probs_targ, self.weight, is_weight)
    def _score_components(self, X, y, pool, targ, n_samples=100, is_weight=None, fitted=False):
        if not fitted:
            self.model.fit(X, y)
        probs_pool, probs_targ = self.model.get_samples(pool, targ, n_samples=n_samples)
        return _epig_components_from_probs_w(probs_pool, probs_targ, self.weight, is_weight)


class RegressionVarianceReduction(AcquisitionStrategy):
    def __init__(self, model, name='var'):
        super().__init__(model, name)
    def score(self, X, y, pool, targ, fitted=False, n_samples=100, seed=1, std=0.1, **kwargs):
        if not fitted:
            self.model.fit(X, y)
        if kwargs.get("is_rf_model", False):
            return compute_expected_variance_reduction(targ, pool, self.model, n_samples=n_samples, std=std, seed=seed)
        if kwargs.get("is_gp_model", False):
            stats = prepare_gp_posterior_statistics(self.model, targ, pool)
            prior_scores = np.mean(stats["prior_var"])
            expected_postr_scores = compute_gp_expected_posterior_variance(stats)
            return prior_scores - expected_postr_scores
        _, stds = self.model.predict(targ, return_std=True)
        prior_scores = np.mean(stds**2)

        expected_postr_scores  = np.empty(len(pool))
        cand_labels = self.model.predict_samples(pool, n_samples=n_samples, seed=seed)

        for i in range(len(pool)):
            expected_postr_scores[i] = estimate_expected_posterior_variance_cached(
                targ, X, y, pool[i], cand_labels[i], self.model)

        return prior_scores - expected_postr_scores
    def score_compare(self, X, y, pool, targ, weight_fn, n_samples=100, seed=1, std=0.1, **kwargs):
        self.model.fit(X, y)
        if kwargs.get("is_rf_model", False):
            return compute_expected_variance_reduction(targ, pool, self.model, n_samples=n_samples, std=std, seed=seed),\
                compute_expected_weighted_variance_reduction(targ, pool, self.model, weight_fn, n_samples=n_samples, std=std, seed=seed)
        if kwargs.get("is_gp_model", False):
            stats = prepare_gp_posterior_statistics(self.model, targ, pool)
            prior_scores1 = np.mean(stats["prior_var"])

            n_targ = stats["prior_mean"].shape[0]
            n_evals = max(1, n_samples)
            rng = np.random.default_rng(seed)
            eps_base = rng.standard_normal(size=(n_targ, n_evals))
            prior_scores2 = compute_gp_prior_weighted_uncertainty(stats, weight_fn, n_evals, eps_base)

            expected_postr_scores1 = compute_gp_expected_posterior_variance(stats)
            cand_labels = self.model.predict_samples(pool, n_samples=n_samples, seed=seed)
            expected_postr_scores2 = compute_gp_expected_posterior_weighted_uncertainty(
                stats, cand_labels, weight_fn, eps_base
            )
            score1 = prior_scores1 - expected_postr_scores1
            score2 = prior_scores2 - expected_postr_scores2
            return score1, score2
        _, stds = self.model.predict(targ, return_std=True)
        prior_scores1 = np.mean(stds**2)

        means, stds = self.model.predict(targ, return_std=True)
        prior_scores2 = np.mean(estimate_wquad_uncertainty(means, stds, weight_fn=weight_fn, n_evals=n_samples))

        # pool_mean, pool_std = self.model.predict(pool, return_std=True)
        cand_labels = self.model.predict_samples(pool, n_samples=n_samples, seed=seed)
        expected_postr_scores1  = np.empty(len(pool))
        expected_postr_scores2  = np.empty(len(pool))
        for i in range(len(pool)):
            # cand_labels = rng.normal(loc=pool_mean[i], scale=pool_std[i], size=n_samples)
            expected_postr_scores1[i] = estimate_expected_posterior_variance_cached(targ, X, y, pool[i], cand_labels[i], self.model)
            expected_postr_scores2[i] = estimate_expected_posterior_wquad_uncertainty_cached(
                targ, X, y, pool[i], cand_labels[i],
                self.model, weight_fn=weight_fn,
                n_evals=n_samples, is_gp_model=kwargs.get("is_gp_model", False))
        score1 = prior_scores1 - expected_postr_scores1
        score2 = prior_scores2 - expected_postr_scores2
        return score1, score2


class RegressionPosteriorLinex(AcquisitionStrategy):
    def __init__(self, model, name="linex"):
        super().__init__(model, name)

    def score(self, X, y, pool, targ, fitted=False, n_samples=100, seed=1, std=0.1, is_weight=None, **kwargs):
        if not fitted:
            self.model.fit(X, y)

        if kwargs.get("is_gp_model", False):
            stats = prepare_gp_posterior_statistics(self.model, targ, pool)
            cand_labels = self.model.predict_samples(pool, n_samples=n_samples, seed=seed)
            return compute_gp_linex_score(
                stats,
                cand_labels,
                targ_weight=is_weight,
                n_evals=max(1, n_samples),
                seed=seed,
            )
        if kwargs.get("is_rf_model", False):
            cand_labels = self.model.predict_samples(pool, std=std, n_samples=n_samples, seed=seed)
        else:
            cand_labels = self.model.predict_samples(pool, n_samples=n_samples, seed=seed)

        scores = np.empty(len(pool))
        for i in range(len(pool)):
            score_seed = None if seed is None else seed + i
            scores[i] = estimate_expected_posterior_linex_score_cached(
                targ,
                X,
                y,
                pool[i],
                cand_labels[i],
                self.model,
                n_post_samples=n_samples,
                targ_weight=is_weight,
                is_rf_model=kwargs.get("is_rf_model", False),
                is_gp_model=kwargs.get("is_gp_model", False),
                std=std,
                seed=score_seed,
            )
        return scores

    def score_compare(self, X, y, pool, targ, weight_fn, n_samples=100, seed=1, std=0.1, is_weight=None, **kwargs):
        """Return (linex, weighted-linex) scores using provided weight_fn."""
        if not kwargs.get("fitted", False):
            self.model.fit(X, y)

        if weight_fn is None:
            raise ValueError("weight_fn is required for score_compare in RegressionPosteriorlinex")

        if kwargs.get("is_gp_model", False):
            stats = prepare_gp_posterior_statistics(self.model, targ, pool)
            cand_labels = self.model.predict_samples(pool, n_samples=n_samples, seed=seed)
            scores1 = compute_gp_linex_score(
                stats,
                cand_labels,
                targ_weight=is_weight,
                n_evals=max(1, n_samples),
                seed=seed,
            )
            scores2 = compute_gp_weighted_linex_score(
                stats,
                cand_labels,
                weight_fn,
                targ_weight=is_weight,
                n_evals=n_samples,
                seed=seed,
            )
            return scores1, scores2
        if kwargs.get("is_rf_model", False):
            cand_labels = self.model.predict_samples(pool, std=std, n_samples=n_samples, seed=seed)
        else:
            cand_labels = self.model.predict_samples(pool, n_samples=n_samples, seed=seed)

        scores1 = np.empty(len(pool))
        scores2 = np.empty(len(pool))
        for i in range(len(pool)):
            score_seed = None if seed is None else seed + i
            scores1[i] = estimate_expected_posterior_linex_score_cached(
                targ,
                X,
                y,
                pool[i],
                cand_labels[i],
                self.model,
                n_post_samples=n_samples,
                targ_weight=is_weight,
                is_rf_model=kwargs.get("is_rf_model", False),
                is_gp_model=kwargs.get("is_gp_model", False),
                std=std,
                seed=score_seed,
            )
            scores2[i] = estimate_expected_posterior_linex_score_weighted_cached(
                targ,
                X,
                y,
                pool[i],
                cand_labels[i],
                self.model,
                weight_fn,
                n_post_samples=n_samples,
                targ_weight=is_weight,
                is_rf_model=kwargs.get("is_rf_model", False),
                is_gp_model=kwargs.get("is_gp_model", False),
                std=std,
                seed=score_seed,
            )
        return scores1, scores2


class WeightedRegressionPosteriorLinex(AcquisitionStrategy):
    def __init__(self, model, weight_fn, name="linex_w"):
        super().__init__(model, name)
        self.weight_fn = weight_fn
        self.name = f"{name}_{get_weight_identifier(weight_fn)}"

    def score(self, X, y, pool, targ, fitted=False, n_samples=100, seed=1, std=0.1, is_weight=None, **kwargs):
        if not fitted:
            self.model.fit(X, y)

        if kwargs.get("is_rf_model", False):
            cand_labels = self.model.predict_samples(pool, std=std, n_samples=n_samples, seed=seed)
        else:
            cand_labels = self.model.predict_samples(pool, n_samples=n_samples, seed=seed)

        scores = np.empty(len(pool))
        for i in range(len(pool)):
            score_seed = None if seed is None else seed + i
            scores[i] = estimate_expected_posterior_linex_score_weighted_cached(
                targ,
                X,
                y,
                pool[i],
                cand_labels[i],
                self.model,
                self.weight_fn,
                n_post_samples=n_samples,
                targ_weight=is_weight,
                is_rf_model=kwargs.get("is_rf_model", False),
                is_gp_model=kwargs.get("is_gp_model", False),
                std=std,
                seed=score_seed,
            )
        return scores

    def score_compare(self, X, y, pool, targ, weight_fn, n_samples=100, seed=1, std=0.1, is_weight=None, **kwargs):
        """Return (linex, weighted-linex) scores; weighted uses provided weight_fn (defaults to self.weight_fn)."""
        if not kwargs.get("fitted", False):
            self.model.fit(X, y)

        weight_fn_use = weight_fn if weight_fn is not None else self.weight_fn

        if kwargs.get("is_gp_model", False):
            stats = prepare_gp_posterior_statistics(self.model, targ, pool)
            cand_labels = self.model.predict_samples(pool, n_samples=n_samples, seed=seed)
            scores1 = compute_gp_linex_score(
                stats,
                cand_labels,
                targ_weight=is_weight,
                n_evals=None,
                seed=seed,
            )
            scores2 = compute_gp_weighted_linex_score(
                stats,
                cand_labels,
                weight_fn_use,
                targ_weight=is_weight,
                n_evals=n_samples,
                seed=seed,
            )
            return scores1, scores2
        if kwargs.get("is_rf_model", False):
            cand_labels = self.model.predict_samples(pool, std=std, n_samples=n_samples, seed=seed)
        else:
            cand_labels = self.model.predict_samples(pool, n_samples=n_samples, seed=seed)

        scores1 = np.empty(len(pool))
        scores2 = np.empty(len(pool))
        for i in range(len(pool)):
            score_seed = None if seed is None else seed + i
            scores1[i] = estimate_expected_posterior_linex_score_cached(
                targ,
                X,
                y,
                pool[i],
                cand_labels[i],
                self.model,
                n_post_samples=n_samples,
                targ_weight=is_weight,
                is_rf_model=kwargs.get("is_rf_model", False),
                is_gp_model=kwargs.get("is_gp_model", False),
                std=std,
                seed=score_seed,
            )
            scores2[i] = estimate_expected_posterior_linex_score_weighted_cached(
                targ,
                X,
                y,
                pool[i],
                cand_labels[i],
                self.model,
                weight_fn_use,
                n_post_samples=n_samples,
                targ_weight=is_weight,
                is_rf_model=kwargs.get("is_rf_model", False),
                is_gp_model=kwargs.get("is_gp_model", False),
                std=std,
                seed=score_seed,
            )
        return scores1, scores2


class WeightedRegressionVarianceReduction(AcquisitionStrategy):
    def __init__(self, model, weight_fn, name='var_w'):
        super().__init__(model, name)
        self.weight_fn = weight_fn
        self.name = f"{name}_{get_weight_identifier(weight_fn)}"
    def score(self, X, y, pool, targ, fitted=False, n_samples=100, seed=1, std=0.1, **kwargs):
        if not fitted:
            self.model.fit(X, y)
        if kwargs.get("is_rf_model", False):
            return compute_expected_weighted_variance_reduction(targ, pool, self.model, self.weight_fn, n_samples=n_samples, std=std, seed=seed)
        if kwargs.get("is_gp_model", False):
            stats = prepare_gp_posterior_statistics(self.model, targ, pool)
            n_targ = stats["prior_mean"].shape[0]
            n_evals = max(1, n_samples)
            rng = np.random.default_rng(seed)
            eps_base = rng.standard_normal(size=(n_targ, n_evals))

            prior_scores = compute_gp_prior_weighted_uncertainty(stats, self.weight_fn, n_evals, eps_base)
            cand_labels = self.model.predict_samples(pool, n_samples=n_samples, seed=seed)
            expected_postr_scores = compute_gp_expected_posterior_weighted_uncertainty(
                stats, cand_labels, self.weight_fn, eps_base
            )
            return prior_scores - expected_postr_scores
        means, stds = self.model.predict(targ, return_std=True)
        prior_scores = np.mean(estimate_wquad_uncertainty(means, stds, weight_fn=self.weight_fn, n_evals=n_samples))

        expected_postr_scores  = np.empty(len(pool))
        cand_labels = self.model.predict_samples(pool, n_samples=n_samples, seed=seed)

        for i in range(len(pool)):
            expected_postr_scores[i] = estimate_expected_posterior_wquad_uncertainty_cached(
                targ, X, y, pool[i], cand_labels[i], self.model,
                weight_fn=self.weight_fn, n_evals=n_samples,
                is_gp_model=kwargs.get("is_gp_model", False))

        return prior_scores - expected_postr_scores
    def score_compare(self, X, y, pool, targ, weight_fn, n_samples=100, seed=1, std=0.1, **kwargs):
        self.model.fit(X, y)
        if kwargs.get("is_rf_model", False):
            return compute_expected_variance_reduction(targ, pool, self.model, n_samples=n_samples, std=std, seed=seed),\
                compute_expected_weighted_variance_reduction(targ, pool, self.model, weight_fn, n_samples=n_samples, std=std, seed=seed)
        if kwargs.get("is_gp_model", False):
            stats = prepare_gp_posterior_statistics(self.model, targ, pool)
            prior_scores1 = np.mean(stats["prior_var"])

            n_targ = stats["prior_mean"].shape[0]
            n_evals = max(1, n_samples)
            rng = np.random.default_rng(seed)
            eps_base = rng.standard_normal(size=(n_targ, n_evals))
            prior_scores2 = compute_gp_prior_weighted_uncertainty(stats, weight_fn, n_evals, eps_base)

            expected_postr_scores1 = compute_gp_expected_posterior_variance(stats)
            cand_labels = self.model.predict_samples(pool, n_samples=n_samples, seed=seed)
            expected_postr_scores2 = compute_gp_expected_posterior_weighted_uncertainty(
                stats, cand_labels, weight_fn, eps_base
            )
            score1 = prior_scores1 - expected_postr_scores1
            score2 = prior_scores2 - expected_postr_scores2
            return score1, score2
        _, stds = self.model.predict(targ, return_std=True)
        prior_scores1 = np.mean(stds**2)

        means, stds = self.model.predict(targ, return_std=True)
        prior_scores2 = np.mean(estimate_wquad_uncertainty(means, stds, weight_fn=weight_fn, n_evals=n_samples))

        # pool_mean, pool_std = self.model.predict(pool, return_std=True)
        cand_labels = self.model.predict_samples(pool, n_samples=n_samples, seed=seed)
        expected_postr_scores1  = np.empty(len(pool))
        expected_postr_scores2  = np.empty(len(pool))
        for i in range(len(pool)):
            # cand_labels = rng.normal(loc=pool_mean[i], scale=pool_std[i], size=n_samples)
            expected_postr_scores1[i] = estimate_expected_posterior_variance_cached(targ, X, y, pool[i], cand_labels[i], self.model)
            expected_postr_scores2[i] = estimate_expected_posterior_wquad_uncertainty_cached(
                targ, X, y, pool[i], cand_labels[i],
                self.model, weight_fn=weight_fn,
                n_evals=n_samples, is_gp_model=kwargs.get("is_gp_model", False))
        score1 = prior_scores1 - expected_postr_scores1
        score2 = prior_scores2 - expected_postr_scores2
        return score1, score2
