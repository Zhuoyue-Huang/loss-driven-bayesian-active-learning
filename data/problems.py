import numpy as np
import pandas as pd
from typing import Optional, Sequence, Union, Literal, Tuple
from abc import ABC, abstractmethod

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from data.generators import grid_points, simulate_regression_labels, binary_arctan, ternary_angular, ternary_parabola, quadrant, quad_diag, fetch_any_uci
from acquisition.utils.classification import gaussian_weights_torch


class StdOnlyScaler:
    """Scale targets by their fitted standard deviation without centering."""

    def __init__(self, min_scale: float = 1e-12):
        self.min_scale = float(min_scale)
        self.scale_ = 1.0

    def fit(self, y):
        arr = np.asarray(y, dtype=float).reshape(-1, 1)
        scale = float(arr.std(ddof=0))
        self.scale_ = max(scale, self.min_scale)
        return self

    def transform(self, y):
        arr = np.asarray(y, dtype=float)
        return arr / self.scale_

    def inverse_transform(self, y):
        arr = np.asarray(y, dtype=float)
        return arr * self.scale_


class Problem(ABC):
    @property
    @abstractmethod
    def X0(self): ...
    @property
    @abstractmethod
    def y0(self): ...
    @property
    @abstractmethod
    def X_pool(self): ...
    @property
    @abstractmethod
    def X_targ(self): ...
    @property
    @abstractmethod
    def y_targ(self): ...
    @abstractmethod
    def acquire(self, pool_idx: int): ...

class ClassificationProblem(Problem):
    def __init__(self, decision_fn, initial, target, pool_args, test=None, plot_args=None, targ_weight=False):
        self.task = "classification"
        self.decision_fn = decision_fn
        self._X  = grid_points(*pool_args)
        self._X0 = initial
        self._y0 = self.decision_fn(self._X0)
        self._X_targ = target
        self._X_test = test if test is not None else self._X_targ
        self.num_classes = len(np.unique(self._y0))

        self.pool_args = pool_args
        if plot_args is None:
            self.plot_args = pool_args
        else:
            self.plot_args = plot_args
        if self.X0.shape[1] > 2:
            self.pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=2))
            ])
            self.pipe.fit(self._X0)
        else:
            self.pipe = None
        if targ_weight:
            self.targ_weight = gaussian_weights_torch(self._X_targ)
        else:
            self.targ_weight = None

    @property
    def X0(self):
        return self._X0
    @property
    def y0(self):
        return self._y0
    @property
    def X_pool(self):
        return self._X
    @property
    def X_targ(self):
        return self._X_targ
    @property
    def y_targ(self):
        return self.decision_fn(self._X_targ)
    @property
    def X_test(self):
        return self._X_test
    @property
    def y_test(self):
        return self.decision_fn(self._X_test)

    def acquire(self, idx):
        return self.decision_fn(self._X[idx])

class BinaryArctanProblem(ClassificationProblem):
    def __init__(self, initial, target, pool_args, test=None, plot_args=None, targ_weight=False):
        super().__init__(decision_fn=binary_arctan, initial=initial, target=target, pool_args=pool_args,
                         test=test, plot_args=plot_args, targ_weight=targ_weight)
        self.name = "binary_arctan"
        self.num_classes = 2

class TernaryAngularProblem(ClassificationProblem):
    def __init__(self, initial, target, pool_args, test=None, plot_args=None, targ_weight=False):
        super().__init__(decision_fn=ternary_angular, initial=initial, target=target, pool_args=pool_args,
                         test=test, plot_args=plot_args, targ_weight=targ_weight)
        self.name = "ternary_angular"
        self.num_classes = 3

class TernaryParabolaProblem(ClassificationProblem):
    def __init__(self, initial, target, pool_args, test=None, plot_args=None, targ_weight=False):
        super().__init__(decision_fn=ternary_parabola, initial=initial, target=target, pool_args=pool_args,
                         test=test, plot_args=plot_args, targ_weight=targ_weight)
        self.name = "ternary_parabola"
        self.num_classes = 3

class QuadrantProblem(ClassificationProblem):
    def __init__(self, initial, target, pool_args, test=None, plot_args=None, targ_weight=False):
        super().__init__(decision_fn=quadrant, initial=initial, target=target, pool_args=pool_args,
                         test=test, plot_args=plot_args, targ_weight=targ_weight)
        self.name = "quadrant"
        self.num_classes = 4

class QuadDiagProblem(ClassificationProblem):
    def __init__(self, initial, target, pool_args, test=None, plot_args=None, targ_weight=False):
        super().__init__(decision_fn=quad_diag, initial=initial, target=target, pool_args=pool_args,
                         test=test, plot_args=plot_args, targ_weight=targ_weight)
        self.name = "quad_diag"
        self.num_classes = 4

FitScope = Literal["train", "train+pool", "all"]  # "all" for debugging only

class RealClassificationProblem(Problem):
    def __init__(
        self,
        name: str,
        data: Union[Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.Series]],
                    np.ndarray, object],
        n_test: Optional[int] = None,
        init_idxs: Optional[Sequence[int]] = None,
        targ_idxs: Optional[Sequence[int]] = None,
        test_idxs: Optional[Sequence[int]] = None,
        *,
        # optional per-class sampling (used if indices are None)
        p_init: Optional[Union[int, Sequence[int]]] = None,
        p_targ: Optional[Union[int, Sequence[int]]] = None,
        p_test: Optional[Union[int, Sequence[int]]] = None,
        # preprocessing
        dropna: bool = True,
        scale_numeric: bool = True,
        one_hot: bool = True,
        preprocess_fit_scope: FitScope = "train+pool",
        seed: Optional[int] = None,
    ):
        self.task = "classification"
        self.name = name

        # ---- Parse input into (X_df, y_series) --------------------------------
        X_df, y_series = self._coerce_to_dataframe_and_labels(data, dropna=dropna)

        # ---- Encode labels to ints (robust to strings/categoricals) -----------
        self.label_encoder_ = LabelEncoder()
        y_all = self.label_encoder_.fit_transform(np.asarray(y_series)).astype(int)
        self.class_names_ = list(self.label_encoder_.classes_)

        # ---- Build/fit preprocessor (numeric scale + categorical encode) ------
        num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in X_df.columns if c not in num_cols]

        transformers = []
        if num_cols:
            steps = [('variance_filter', VarianceThreshold(threshold=1e-8))]
            if scale_numeric:
                steps.append(('scaler', StandardScaler(with_mean=True, with_std=True)))
            num_pipeline = Pipeline(steps)
            transformers.append(('num', num_pipeline, num_cols))
        if cat_cols and one_hot:
            try:
                ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            except TypeError:  # sklearn < 1.2
                ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
            transformers.append(('cat', ohe, cat_cols))
        elif cat_cols and not one_hot:
            transformers.append(('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols))

        self.preprocess_ = ColumnTransformer(transformers, remainder='drop') if transformers else None

        # ---- Handle predefined test set if n_test is provided ------------------
        n = len(y_all)
        all_idx = np.arange(n)
        classes = np.unique(y_all)
        rng = np.random.default_rng(seed)
        
        if n_test is not None and test_idxs is None:
            # Extract predefined test set (last n_test samples)
            test_idxs = all_idx[-n_test:]
            available_idx = all_idx[:-n_test]
            y_available = y_all[:-n_test]
            y_test_predefined = y_all[-n_test:]
        else:
            available_idx = all_idx
            y_available = y_all
            y_test_predefined = None

        # ---- Print dataset statistics ------------------------------------------
        print(f"Total number of data points: {n}")
        if n_test is not None:
            print(f"Predefined test set size: {n_test}")
            print("Training data distribution:")
            for c in classes:
                count_train = np.sum(y_available == c)
                class_name = self.label_encoder_.classes_[c]
                print(f"  Class {c} ({class_name}): {count_train} samples")
            
            if y_test_predefined is not None:
                print("Test data distribution:")
                for c in classes:
                    count_test = np.sum(y_test_predefined == c)
                    class_name = self.label_encoder_.classes_[c]
                    print(f"  Class {c} ({class_name}): {count_test} samples")
        else:
            print("Sample distribution:")
            for c in classes:
                count = np.sum(y_all == c)
                class_name = self.label_encoder_.classes_[c]
                print(f"  Class {c} ({class_name}): {count} samples")

        # ---- Enhanced per-class sampling function -----------------------------
        def per_class_sample(y: np.ndarray, idx_pool: np.ndarray, 
                           p: Optional[Union[int, Sequence[int]]]) -> Optional[np.ndarray]:
            if p is None:
                return None
            
            # Handle both int and list of ints
            if isinstance(p, int):
                p_per_class = [p] * len(classes)
            else:
                p_per_class = list(p)
                if len(p_per_class) != len(classes):
                    raise ValueError(f"Length of p ({len(p_per_class)}) must match number of classes ({len(classes)})")
            
            picks = []
            for i, c in enumerate(classes):
                cls_idx = idx_pool[y[idx_pool] == c]
                if len(cls_idx) < p_per_class[i]:
                    class_name = self.label_encoder_.classes_[c]
                    raise ValueError(f"Class {c} ({class_name}) has {len(cls_idx)} samples; requested {p_per_class[i]}.")
                if p_per_class[i] > 0:
                    picks.append(rng.choice(cls_idx, size=p_per_class[i], replace=False))
            
            return np.hstack(picks) if picks else np.array([], dtype=int)

        # ---- Build (or infer) splits ------------------------------------------
        if init_idxs is None:
            init_idxs = per_class_sample(y_all, available_idx, p_init)
            if init_idxs is not None:
                available_idx = available_idx[~np.isin(available_idx, init_idxs)]
        if targ_idxs is None:
            targ_idxs = per_class_sample(y_all, available_idx, p_targ)
            if targ_idxs is not None:
                available_idx = available_idx[~np.isin(available_idx, targ_idxs)]
        if test_idxs is None and p_test is not None:
            test_idxs = per_class_sample(y_all, available_idx, p_test)
            if test_idxs is not None:
                available_idx = available_idx[~np.isin(available_idx, test_idxs)]

        if init_idxs is None or targ_idxs is None:
            raise ValueError("Provide (init_idxs & targ_idxs) or (p_init & p_targ) for stratified sampling.")

        self.labeled = np.array(init_idxs, dtype=int)
        self.targ    = np.array(targ_idxs, dtype=int)
        self.test    = [] if test_idxs is None else list(np.array(test_idxs, dtype=int))
        self.pool = list(available_idx)

        # ---- Fit preprocessor on leak-safe scope; transform all features ------
        if self.preprocess_ is not None:
            if preprocess_fit_scope == "train":
                fit_idx = self.labeled
            elif preprocess_fit_scope == "train+pool":
                fit_idx = np.array(self.labeled.tolist() + self.pool, dtype=int)
            elif preprocess_fit_scope == "all":
                fit_idx = all_idx
            else:
                raise ValueError("preprocess_fit_scope must be 'train', 'train+pool', or 'all'.")

            self.preprocess_.fit(X_df.iloc[fit_idx])
            X_all_scaled = self.preprocess_.transform(X_df)
            if hasattr(X_all_scaled, "toarray"):
                X_all_scaled = X_all_scaled.toarray()
        else:
            X_all_scaled = X_df.to_numpy()

        # ---- Store scaled features + encoded labels ---------------------------
        self.X_all = np.asarray(X_all_scaled)
        self.y_all = np.asarray(y_all, dtype=int)
        self.num_classes = int(len(np.unique(self.y_all)))

        # Optional small viz helper (kept like your other problems)
        if self.X_all.shape[1] > 2:
            self.pipe = Pipeline([('pca', PCA(n_components=2))])
            self.pipe.fit(self.X_all)
        else:
            self.pipe = None

        self.targ_weight = None  # keep API parity

    # ---- Properties: all return **scaled/encoded** data -----------------------
    @property
    def X0(self):
        return self.X_all[self.labeled]
    @property
    def y0(self):
        return self.y_all[self.labeled]
    @property
    def X_pool(self):
        return self.X_all[self.pool]
    @property
    def X_targ(self):
        return self.X_all[self.targ]
    @property
    def y_targ(self):
        return self.y_all[self.targ]
    @property
    def X_test(self):
        return self.X_all[self.test]
    @property
    def y_test(self):
        return self.y_all[self.test]

    def acquire(self, idx: int):
        real_idx = self.pool[idx]
        return int(self.y_all[real_idx])
    
    def invert_labels(self, y_int: np.ndarray) -> np.ndarray:
        return self.label_encoder_.inverse_transform(y_int.astype(int))

    # ---- Helpers (internal) ---------------------------------------------------
    @staticmethod
    def _coerce_to_dataframe_and_labels(
        data: Union[Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.Series]],
                    np.ndarray, object],
        dropna: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Accept (X, y), a single array [X|y], or a sklearn Bunch with .data/.target."""
        # (X, y) tuple
        if isinstance(data, tuple) and len(data) == 2:
            X, y = data
        else:
            # sklearn Bunch or object with .data/.target
            if hasattr(data, "data") and hasattr(data, "target"):
                X, y = data.data, data.target
            else:
                # single array: last column is y
                arr = np.asarray(data)
                X, y = arr[:, :-1], arr[:, -1]

        # Convert to DataFrame/Series to retain dtypes
        X_df = pd.DataFrame(X)
        y_series = pd.Series(y)

        if dropna:
            df = X_df.copy()
            df['__target__'] = y_series
            df = df.dropna(axis=0).reset_index(drop=True)
            X_df = df.drop(columns='__target__')
            y_series = df['__target__'].reset_index(drop=True)

        return X_df, y_series

class UCIClassificationProblem(RealClassificationProblem):
    def __init__(self, dataset, init_idxs=None, targ_idxs=None, test_idxs=None, *,
                 p_init=None, p_targ=None, p_test=None,
                 dropna=True, scale_numeric=True, one_hot=True,
                 preprocess_fit_scope="train+pool", seed=None,
                 version=None, data_home=None,
                 prefer=("sklearn", "pmlb", "ucimlrepo", "openml")):

        X_df, y_series, src, n_test = fetch_any_uci(dataset, version=version, data_home=data_home, prefer=prefer)
        super().__init__(
            name=src,
            data=(X_df, y_series),
            n_test=n_test,
            init_idxs=init_idxs, targ_idxs=targ_idxs, test_idxs=test_idxs,
            p_init=p_init, p_targ=p_targ, p_test=p_test,
            dropna=dropna, scale_numeric=scale_numeric, one_hot=one_hot,
            preprocess_fit_scope=preprocess_fit_scope, seed=seed,
        )

class RegressionProblem(Problem):
    def __init__(self, name, true_args, initial, targ, pool_args, test_args, plot_args=None, rng=np.random.default_rng(0)):
        self.task = "regression"
        self.name = name
        self.true_mean, self.true_std = true_args
        self.pool_args = pool_args
        self.test_args = test_args
        if plot_args is None:
            self.plot_args = pool_args
        else:
            self.plot_args = plot_args
        self.rng = rng
        self._X0 = initial
        self._y0 = simulate_regression_labels(self.true_mean(initial), self.true_std, rng=self.rng)
        self._pool = np.linspace(*pool_args)
        self._targ = targ
        self._y_targ = simulate_regression_labels(self.true_mean(self._targ), self.true_std, rng=self.rng)
        self._test = np.linspace(*test_args)
        self._y_test = self.true_mean(self._test)  # noiseless test labels
        self.targ_weight = None

    @property
    def X0(self):
        return np.array(self._X0)
    @property
    def y0(self):
        return np.array(self._y0)
    @property
    def X_pool(self):
        return np.array(self._pool)
    @property
    def X_targ(self):
        return np.array(self._targ)
    @property
    def y_targ(self):
        return np.array(self._y_targ)
    @property
    def X_test(self):
        return np.array(self._test)
    @property
    def y_test(self):
        return np.array(self._y_test)

    def acquire(self, idx):
        return simulate_regression_labels(self.true_mean(self._pool[idx]), self.true_std, rng=self.rng)

class RealRegressionProblem(Problem):
    def __init__(
        self,
        name: str,
        data: Union[Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.Series]],
                    np.ndarray, object],
        n_test: Optional[int] = None,
        init_idxs: Optional[Sequence[int]] = None,
        targ_idxs: Optional[Sequence[int]] = None,
        test_idxs: Optional[Sequence[int]] = None,
        *,
        # optional sampling (used if indices are None)
        p_init: Optional[int] = None,
        p_targ: Optional[int] = None,
        p_test: Optional[int] = None,
        # preprocessing
        dropna: bool = True,
        scale_numeric: bool = True,
        scale_target: bool = False,
        target_scale_std_only: bool = False,
        target_scale_range: Optional[Tuple[float, float]] = None,
        one_hot: bool = True,
        preprocess_fit_scope: FitScope = "train+pool",
        seed: Optional[int] = None,
    ):
        self.task = "regression"
        self.name = name

        # ---- Parse input into (X_df, y_series) --------------------------------
        X_df, y_series = self._coerce_to_dataframe_and_labels(data, dropna=dropna)
        
        # Convert target to float and handle any remaining non-numeric values
        y_all = pd.to_numeric(y_series, errors='coerce').astype(float)
        if y_all.isna().any():
            print(f"Warning: {y_all.isna().sum()} non-numeric target values converted to NaN")
            if dropna:
                valid_mask = ~y_all.isna()
                X_df = X_df[valid_mask].reset_index(drop=True)
                y_all = y_all[valid_mask].reset_index(drop=True)

        y_all = np.asarray(y_all)

        # ---- Build/fit preprocessor (numeric scale + categorical encode) ------
        num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in X_df.columns if c not in num_cols]

        transformers = []
        if num_cols:
            steps = [('variance_filter', VarianceThreshold(threshold=1e-8))]
            if scale_numeric:
                steps.append(('scaler', StandardScaler(with_mean=True, with_std=True)))
            num_pipeline = Pipeline(steps)
            transformers.append(('num', num_pipeline, num_cols))
        if cat_cols and one_hot:
            try:
                ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            except TypeError:  # sklearn < 1.2
                ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
            transformers.append(('cat', ohe, cat_cols))
        elif cat_cols and not one_hot:
            transformers.append(('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols))

        self.preprocess_ = ColumnTransformer(transformers, remainder='drop') if transformers else None

        # ---- Target scaling (optional) -----------------------------------------
        target_scale_modes = int(bool(scale_target)) + int(bool(target_scale_std_only)) + int(target_scale_range is not None)
        if target_scale_modes > 1:
            raise ValueError(
                "Use at most one of scale_target=True, "
                "target_scale_std_only=True, or target_scale_range=(low, high)."
            )
        if target_scale_range is not None:
            lower, upper = (float(target_scale_range[0]), float(target_scale_range[1]))
            if not np.isfinite(lower) or not np.isfinite(upper):
                raise ValueError("target_scale_range bounds must be finite.")
            if lower >= upper:
                raise ValueError("target_scale_range requires lower < upper.")
            self.target_scaler_ = MinMaxScaler(feature_range=(lower, upper))
            self.target_scale_range = (lower, upper)
            self.target_scale_mode = "range"
        elif target_scale_std_only:
            self.target_scaler_ = StdOnlyScaler()
            self.target_scale_range = None
            self.target_scale_mode = "std_only"
        elif scale_target:
            self.target_scaler_ = StandardScaler()
            self.target_scale_range = None
            self.target_scale_mode = "standard"
        else:
            self.target_scaler_ = None
            self.target_scale_range = None
            self.target_scale_mode = None

        # ---- Handle predefined test set if n_test is provided ------------------
        n = len(y_all)
        all_idx = np.arange(n)
        rng = np.random.default_rng(seed)
        
        if n_test is not None and test_idxs is None and p_test is not None:
            # Only carve out a predefined test split when the caller explicitly
            # requests a separate test set via p_test.
            test_idxs = all_idx[-n_test:]
            available_idx = all_idx[:-n_test]
        else:
            available_idx = all_idx

        # ---- Print dataset statistics ------------------------------------------
        # print(f"Total number of data points: {n}")
        # print(f"Number of features: {X_df.shape[1]}")
        # if n_test is not None:
        #     print(f"Predefined test set size: {n_test}")

        # ---- Simple random sampling function for regression -------------------
        def random_sample(idx_pool: np.ndarray, p: Optional[int]) -> Optional[np.ndarray]:
            if p is None:
                return None
            if len(idx_pool) < p:
                raise ValueError(f"Pool has {len(idx_pool)} samples; requested {p}.")
            if p > 0:
                return rng.choice(idx_pool, size=p, replace=False)
            return np.array([], dtype=int)

        if init_idxs is None:
            init_idxs = random_sample(available_idx, p_init)
            if init_idxs is not None:
                available_idx = available_idx[~np.isin(available_idx, init_idxs)]
        
        if targ_idxs is None:
            targ_idxs = random_sample(available_idx, p_targ)
            if targ_idxs is not None:
                available_idx = available_idx[~np.isin(available_idx, targ_idxs)]
        
        if test_idxs is None and p_test is not None:
            test_idxs = random_sample(available_idx, p_test)
            if test_idxs is not None:
                available_idx = available_idx[~np.isin(available_idx, test_idxs)]

        if init_idxs is None or targ_idxs is None:
            raise ValueError("Provide (init_idxs & targ_idxs) or (p_init & p_targ) for sampling.")

        self.labeled = np.array(init_idxs, dtype=int)
        self.targ    = np.array(targ_idxs, dtype=int)
        self.test    = [] if test_idxs is None else list(np.array(test_idxs, dtype=int))
        self.pool = list(available_idx)

        # ---- Fit preprocessor on leak-safe scope; transform all features ------
        if self.preprocess_ is not None:
            if preprocess_fit_scope == "train":
                fit_idx = self.labeled
            elif preprocess_fit_scope == "train+pool":
                fit_idx = np.array(self.labeled.tolist() + self.pool, dtype=int)
            elif preprocess_fit_scope == "all":
                fit_idx = all_idx
            else:
                raise ValueError("preprocess_fit_scope must be 'train', 'train+pool', or 'all'.")

            self.preprocess_.fit(X_df.iloc[fit_idx])
            X_all_scaled = self.preprocess_.transform(X_df)
            if hasattr(X_all_scaled, "toarray"):
                X_all_scaled = X_all_scaled.toarray()
        else:
            X_all_scaled = X_df.to_numpy()

        # ---- Fit target scaler if requested ------------------------------------
        if self.target_scaler_ is not None:
            # Only fit on training data to prevent data leakage
            self.target_scaler_.fit(y_all[self.labeled].reshape(-1, 1))
            y_all_scaled = self.target_scaler_.transform(y_all.reshape(-1, 1)).flatten()
        else:
            y_all_scaled = y_all

        # ---- Store scaled features + targets ----------------------------------
        self.X_all = np.asarray(X_all_scaled)
        self.y_all = np.asarray(y_all_scaled, dtype=float)

        # Optional small viz helper (kept like your other problems)
        if self.X_all.shape[1] > 1:
            self.pipe = Pipeline([('pca', PCA(n_components=1))])
            self.pipe.fit(self.X_all)
        else:
            self.pipe = None

        self.targ_weight = None  # keep API parity

    # ---- Properties: all return **scaled** data ------------------------------
    @property
    def X0(self):
        return self.X_all[self.labeled]
    @property
    def y0(self):
        return self.y_all[self.labeled]
    @property
    def X_pool(self):
        return self.X_all[self.pool]
    @property
    def X_targ(self):
        return self.X_all[self.targ]
    @property
    def y_targ(self):
        return self.y_all[self.targ]
    @property
    def X_test(self):
        if len(self.test) == 0:
            return self.X_targ
        return self.X_all[self.test]
    @property
    def y_test(self):
        if len(self.test) == 0:
            return self.y_targ
        return self.y_all[self.test]

    def acquire(self, idx: int):
        real_idx = self.pool[idx]
        return float(self.y_all[real_idx])
    
    def invert_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        """Convert scaled targets back to original scale"""
        if self.target_scaler_ is not None:
            return self.target_scaler_.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
        return y_scaled

    # ---- Helpers (internal) ---------------------------------------------------
    @staticmethod
    def _coerce_to_dataframe_and_labels(
        data: Union[Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.Series]],
                    np.ndarray, object],
        dropna: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Accept (X, y), a single array [X|y], or a sklearn Bunch with .data/.target."""
        # (X, y) tuple
        if isinstance(data, tuple) and len(data) == 2:
            X, y = data
        else:
            # sklearn Bunch or object with .data/.target
            if hasattr(data, "data") and hasattr(data, "target"):
                X, y = data.data, data.target
            else:
                # single array: last column is y
                arr = np.asarray(data)
                X, y = arr[:, :-1], arr[:, -1]

        # Convert to DataFrame/Series to retain dtypes
        X_df = pd.DataFrame(X)
        y_series = pd.Series(y)

        if dropna:
            df = X_df.copy()
            df['__target__'] = y_series
            df = df.dropna(axis=0).reset_index(drop=True)
            X_df = df.drop(columns='__target__')
            y_series = df['__target__'].reset_index(drop=True)

        return X_df, y_series

class UCIRegressionProblem(RealRegressionProblem):
    def __init__(self, dataset, init_idxs=None, targ_idxs=None, test_idxs=None, *,
                 p_init=None, p_targ=None, p_test=None,
                 dropna=True, scale_numeric=True, scale_target=False, target_scale_std_only=False,
                 target_scale_range=None, one_hot=True,
                 preprocess_fit_scope="train+pool", seed=None,
                 version=None, data_home=None,
                 prefer=("sklearn", "pmlb", "ucimlrepo", "openml")):

        X_df, y_series, src, n_test = fetch_any_uci(dataset, version=version, data_home=data_home, prefer=prefer)
        super().__init__(
            name=src,
            data=(X_df, y_series),
            n_test=n_test,
            init_idxs=init_idxs, targ_idxs=targ_idxs, test_idxs=test_idxs,
            p_init=p_init, p_targ=p_targ, p_test=p_test,
            dropna=dropna, scale_numeric=scale_numeric, scale_target=scale_target,
            target_scale_std_only=target_scale_std_only,
            target_scale_range=target_scale_range, one_hot=one_hot,
            preprocess_fit_scope=preprocess_fit_scope, seed=seed,
        )
