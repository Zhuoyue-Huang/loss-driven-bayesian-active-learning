from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, DotProduct, Matern, ConstantKernel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from abc import ABC, abstractmethod
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
import gpytorch
from gpytorch.models import ExactGP, ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.likelihoods import GaussianLikelihood, SoftmaxLikelihood
from gpytorch.means import ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel
import numpy as np
import pickle
import random
from typing import Any, Dict, Optional

def _lengthscale_identifier(ls: Any) -> str:
    if ls is None:
        return "auto"
    try:
        arr = np.asarray(ls, dtype=float)
    except (TypeError, ValueError):
        return str(ls)
    if arr.size == 0:
        return "auto"
    first = float(arr.flat[0])
    return f"{first:g}"


# regression
class RegressorWrapper(ABC):
    """Abstract base class for deterministic or probabilistic regressors."""

    def __init__(self, model: Any, device: Optional[str] = None) -> None:
        self.reg = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._rng: Optional[Dict[str, Any]] = None

    # ------------------------------- API -------------------------------
    @abstractmethod
    def fit(self, X, y, *args, **kwargs):
        """Fit the model."""

    @abstractmethod
    def predict(self, X, *args, **kwargs):
        """Predict point estimates or distributions."""

    @abstractmethod
    def get_model_state(self) -> Dict[str, Any]:
        """Return a JSON‑serialisable snapshot for checkpointing."""

    @abstractmethod
    def load_model_state(self, state: Dict[str, Any]) -> None:
        """Restore the model from *state*."""

    # ----------------------------- RNG handling -----------------------------
    def snapshot_rng_state(self) -> None:
        """Capture Python, NumPy and Torch RNG states so runs are repeatable."""
        self._rng = {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }

    def restore_rng_state(self) -> None:
        """Restore RNG state captured by :py:meth:`snapshot_rng_state`."""
        if self._rng is None:
            raise RuntimeError("snapshot_rng_state() must be called first.")

        torch.set_rng_state(self._rng["torch_cpu"])
        if torch.cuda.is_available() and self._rng["torch_cuda"] is not None:
            for idx, st in enumerate(self._rng["torch_cuda"]):
                torch.cuda.set_rng_state(st, device=idx)

        np.random.set_state(self._rng["numpy"])
        random.setstate(self._rng["python"])

    # ------------------------------ utilities ------------------------------
    def _as_tensor(self, arr, *, dtype=torch.float32):
        """Convert *arr* to a tensor on the correct device and dtype."""
        if torch.is_tensor(arr):
            return arr.to(self.device, dtype=dtype)
        return torch.as_tensor(arr, dtype=dtype, device=self.device)


class GPyTorchExactGPModel(ExactGP):
    """Base GPyTorch model for regression"""
    
    def __init__(self, train_x, train_y, likelihood, ls=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel())
        self.ls = ls
        
        if ls is not None:
            ls_tensor = torch.as_tensor(ls, dtype=train_x.dtype, device=train_x.device)
            self.covar_module.base_kernel.lengthscale = ls_tensor
            self.covar_module.base_kernel.raw_lengthscale.requires_grad = False
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPyTorchRegressorWrapper(RegressorWrapper):
    """Wrapper for GPyTorch regression model"""
    
    def __init__(self, ls=1.0, true_std=1.0, n_iterations=500, learning_rate=0.01, 
                 use_lr_schedule=False, lr_min=0.0001, random_seed=None):
        """
        Args:
            ls: Lengthscale for RBF kernel
            true_std: Standard deviation (noise) for Gaussian likelihood
            n_iterations: Number of training iterations
            learning_rate: Learning rate for optimization
            use_lr_schedule: Whether to use learning rate scheduler
            lr_min: Minimum learning rate when using scheduler
            random_seed: Random seed for reproducibility
        """
        super().__init__(None)  # No scikit-learn model to wrap
        self.ls = ls
        self.true_std = true_std
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.use_lr_schedule = use_lr_schedule
        self.lr_min = lr_min
        self.identifier = f"gpy/{_lengthscale_identifier(ls)}"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialized = False
        
        # Set random seed if provided
        if random_seed is not None:
            self._set_random_seed(random_seed)
            self._initial_seed = random_seed

    def _set_random_seed(self, seed):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def initialize_model(self, X, y=None):
        """Initialize model without training"""
        if not self.initialized:
            # Convert numpy to tensor if necessary
            train_x = X if torch.is_tensor(X) else torch.tensor(X, dtype=torch.float32).to(self.device)
            train_y = y if torch.is_tensor(y) else torch.tensor(y, dtype=torch.float32).to(self.device)
            
            # Create likelihood with fixed noise if true_std provided
            self.likelihood = GaussianLikelihood()
            if self.true_std is not None:
                self.likelihood.noise = self.true_std ** 2
                self.likelihood.raw_noise.requires_grad = False
                
            # Create model
            self.reg = GPyTorchExactGPModel(train_x, train_y, self.likelihood, ls=self.ls)
            
            # Move to device
            self.reg = self.reg.to(self.device)
            self.likelihood = self.likelihood.to(self.device)
            
            self.initialized = True
        return self

    def fit(self, X, y):
        """
        Train the GP model on the given data
        
        Args:
            X: Features (numpy array or torch tensor)
            y: Target values (numpy array or torch tensor)
            
        Returns:
            self
        """
        # First initialize the model if not already done
        self.initialize_model(X, y)
        
        # Convert numpy to tensor if necessary
        train_x = X if torch.is_tensor(X) else torch.tensor(X, dtype=torch.float32).to(self.device)
        train_y = y if torch.is_tensor(y) else torch.tensor(y, dtype=torch.float32).to(self.device)
        
        # Update the training data
        self.reg.set_train_data(train_x, train_y, strict=False)

        # Train the model
        self.reg.train()
        self.likelihood.train()
        
        # Use Adam optimizer
        optimizer = torch.optim.Adam([
            {'params': self.reg.parameters()},  # This already includes likelihood parameters
        ], lr=self.learning_rate)
        
        # Add learning rate scheduler if enabled
        if self.use_lr_schedule:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.n_iterations, 
                eta_min=self.lr_min
            )
        
        # Define the loss function (Marginal Log Likelihood)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.reg)

        # Training loop
        for i in range(self.n_iterations):
            optimizer.zero_grad()
            output = self.reg(train_x)
            loss = -mll(output, train_y).mean()
            loss.backward()
            optimizer.step()
            
            # Update learning rate if using scheduler
            if self.use_lr_schedule:
                scheduler.step()

        # Set model to evaluation mode
        self.reg.eval()
        self.likelihood.eval()
        self.snapshot_rng_state()
        return self

    def predict(self, X, return_std=False):
        """
        Predict mean and optionally standard deviation for X
        
        Args:
            X: Features (numpy array or torch tensor)
            return_std: If True, return both mean and standard deviation
            
        Returns:
            mean predictions, or (mean predictions, standard deviations) if return_std=True
        """
        # Convert numpy to tensor if necessary
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            
        # Ensure model is in evaluation mode
        self.reg.eval()
        self.likelihood.eval()
        
        # Get predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Get function distribution
            function_dist = self.reg(X)
            # Get predictive distribution (includes observation noise)
            predictive_dist = self.likelihood(function_dist)
            
            # Get mean and standard deviation
            mean = predictive_dist.mean.cpu().numpy()
            
            if return_std:
                # Get standard deviation
                std = predictive_dist.stddev.cpu().numpy()
                return mean, std
            else:
                return mean

    def predict_samples(self, X, n_samples: int = 100, seed: Optional[int] = None):
        """
        Generate samples from the predictive distribution
        
        Args:
            X: Features (numpy array or torch tensor)
            n_samples: Number of samples to generate
            seed: Random seed for reproducible sampling
            
        Returns:
            Samples with shape (X.shape[0], n_samples)
        """
        # Convert numpy to tensor if necessary
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            
        # Ensure model is in evaluation mode
        self.reg.eval()
        self.likelihood.eval()
        
        # Handle seeding if provided
        if seed is not None:
            # Store current RNG state
            current_state = torch.get_rng_state()
            if torch.cuda.is_available():
                cuda_state = torch.cuda.get_rng_state()
            # Set the seed
            torch.manual_seed(seed)
        
        try:
            # Get predictions
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                # Get function distribution
                function_dist = self.reg(X)
                # Get predictive distribution (includes observation noise)
                predictive_dist = self.likelihood(function_dist)
                
                # Sample from the predictive distribution
                samples = predictive_dist.sample(torch.Size([n_samples]))  # (n_samples, N)
                samples = samples.T  # Transpose to (N, n_samples)
                
            return samples.cpu().numpy()
        finally:
            # Restore RNG state if we changed it
            if seed is not None:
                torch.set_rng_state(current_state)
                if torch.cuda.is_available():
                    torch.cuda.set_rng_state(cuda_state)

    def get_model_state(self):
        """Get model state for checkpointing"""
        if not self.initialized:
            raise ValueError("Model must be initialized before getting checkpoint")
            
        return {
            "model_state": self.reg.state_dict(),
            "likelihood_state": self.likelihood.state_dict(),
            "model_type": "gpytorch",
            "ls": self.ls,
            "true_std": self.true_std,
            "device": str(self.device)
        }
    
    def load_model_state(self, state):
        """Load model state from checkpoint"""
        if state is None:
            return
            
        if not self.initialized:
            raise ValueError("Model must be initialized before loading checkpoint")
            
        self.reg.load_state_dict(state["model_state"])
        self.likelihood.load_state_dict(state["likelihood_state"])
            
        self.reg.eval()
        self.likelihood.eval()


class GPRegressorWrapper(RegressorWrapper):
    """Wrapper for scikit-learn GaussianProcessRegressor with enhanced functionality"""
    
    def __init__(self, ls=1.0, nu=1.5, true_std=1.0, mean_init: float = 0.0,
                 sigma_lin: float = 1.0, sigma_f: float = 1.0,
                 kernel=None,
                 random_state=None):
        """
        Args:
            ls: Length scale parameter
            nu: Smoothness parameter for the Matern kernel
            true_std: True standard deviation of the target variable
            mean_init: Constant mean offset applied to predictions
            random_state: Random seed for reproducibility
        """

        self.custom_kernel = kernel
        if kernel is None:
            dot_part = ConstantKernel(sigma_lin**2, constant_value_bounds="fixed") * \
                DotProduct(sigma_0=0.0, sigma_0_bounds="fixed")
            matern_part = ConstantKernel(sigma_f**2, constant_value_bounds="fixed") * \
                Matern(length_scale=ls, length_scale_bounds="fixed", nu=nu)
            noise_part = WhiteKernel(noise_level=true_std**2, noise_level_bounds="fixed")
            kernel = dot_part + matern_part + noise_part
        
        # Create the scikit-learn GP regressor
        reg = GaussianProcessRegressor(kernel=kernel, random_state=random_state)
        super().__init__(reg)
        
        # Store parameters for identification and checkpointing
        self.ls = ls
        self.nu = nu
        self.true_std = true_std
        self.mean_init = float(mean_init)
        self.sigma_lin = float(sigma_lin)
        self.sigma_f = float(sigma_f)
        self.random_state = random_state
        self.identifier = f"sklearn_gp/{_lengthscale_identifier(ls)}" if kernel is None else "sklearn_gp/custom"
        self.fitted = False
        
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

    def fit(self, X, y):
        """
        Fit the Gaussian Process model
        
        Args:
            X: Features (numpy array or torch tensor)
            y: Target values (numpy array or torch tensor)
            
        Returns:
            self
        """
        # Convert tensors to numpy if necessary
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        if torch.is_tensor(y):
            y = y.cpu().numpy()
            
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        if self.mean_init != 0.0:
            y = y - self.mean_init
        
        # Fit the model
        self.reg.fit(X, y)
        self.fitted = True
        self.snapshot_rng_state()
        return self

    def predict(self, X, return_std=False):
        """
        Predict mean and optionally uncertainty for X
        
        Args:
            X: Features (numpy array or torch tensor)
            return_std: If True, return standard deviations
            return_cov: If True, return full covariance matrix
            
        Returns:
            Predictions, or (predictions, uncertainty) if return_std/return_cov=True
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before making predictions")
            
        # Convert tensor to numpy if necessary
        if torch.is_tensor(X):
            X = X.cpu().numpy()
            
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        # Make predictions
        if return_std:
            mean, std = self.reg.predict(X, return_std=True)
            return mean + self.mean_init, std
        else:
            return self.reg.predict(X) + self.mean_init

    def predict_samples(self, X, n_samples=100, seed=None):
        """
        Generate samples from the predictive distribution
        
        Args:
            X: Features (numpy array or torch tensor)
            n_samples: Number of samples to generate
            seed: Random seed for reproducible sampling
            
        Returns:
            Samples with shape (X.shape[0], n_samples)
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before making predictions")
            
        # Convert tensor to numpy if necessary
        if torch.is_tensor(X):
            X = X.cpu().numpy()
            
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        # Handle seeding
        if seed is not None:
            current_state = np.random.get_state()
            np.random.seed(seed)
            rng = np.random.default_rng(seed=seed)
        else:
            rng = np.random.default_rng()

        # Get mean and covariance
        mean, std = self.predict(X, return_std=True)

        # Sample from multivariate normal
        samples = rng.normal(loc=mean.reshape(-1, 1), scale=std.reshape(-1, 1), size=(X.shape[0], n_samples))

        if seed is not None:
            np.random.set_state(current_state)
        
        return samples

    def get_model_state(self):
        """Get model state for checkpointing"""
        if not self.fitted:
            raise ValueError("Model must be fitted before getting checkpoint")
            
        return {
            "model_state": pickle.dumps(self.reg),
            "random_state": self.random_state,
            "model_type": "sklearn_gp",
            "ls": self.ls,
            "true_std": self.true_std,
            "nu": self.nu,
            "mean_init": self.mean_init,
            "sigma_lin": self.sigma_lin,
            "sigma_f": self.sigma_f,
        }
    
    def load_model_state(self, state):
        """Load model state from checkpoint"""
        if state is None:
            return

        self.reg = pickle.loads(state["model_state"])
        self.random_state = state["random_state"]
        self.ls = state.get("ls", self.ls)
        self.true_std = state.get("true_std", self.true_std)
        self.nu = state.get("nu", self.nu)
        self.mean_init = state.get("mean_init", self.mean_init)
        self.sigma_lin = state.get("sigma_lin", self.sigma_lin)
        self.sigma_f = state.get("sigma_f", self.sigma_f)
        self.custom_kernel = self.reg.kernel
        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)


class RandomForestRegressorWrapper(RegressorWrapper):
    """Wrapper for scikit-learn RandomForestRegressor with enhanced functionality"""
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, max_features='sqrt', bootstrap=True, 
                 oob_score=False, n_jobs=None, random_state=None, **kwargs):
        """
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None for unlimited)
            min_samples_split: Minimum samples required to split node
            min_samples_leaf: Minimum samples required at leaf node
            max_features: Number of features to consider when looking for best split
            bootstrap: Whether bootstrap samples are used when building trees
            oob_score: Whether to use out-of-bag samples for R² estimation
            n_jobs: Number of jobs to run in parallel (-1 for all processors)
            random_state: Random seed for reproducibility
            **kwargs: Additional arguments passed to RandomForestRegressor
        """
        
        reg = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            **kwargs
        )
        super().__init__(reg)
        
        # Store parameters for identification and checkpointing
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.identifier = f"rf/{n_estimators}"
        self.fitted = False
        
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

    def fit(self, X, y):
        """
        Fit the Random Forest model
        
        Args:
            X: Features (numpy array or torch tensor)
            y: Target values (numpy array or torch tensor)
            
        Returns:
            self
        """
        # Convert tensors to numpy if necessary
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        if torch.is_tensor(y):
            y = y.cpu().numpy()
            
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        # Fit the model
        self.reg.fit(X, y)
        self.fitted = True
        self.snapshot_rng_state()
        return self

    def predict(self, X, return_std=False):
        """
        Predict mean and optionally uncertainty estimate for X
        
        Args:
            X: Features (numpy array or torch tensor)
            return_std: If True, return standard deviations from tree predictions
            
        Returns:
            Predictions, or (predictions, std_deviations) if return_std=True
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before making predictions")
            
        # Convert tensor to numpy if necessary
        if torch.is_tensor(X):
            X = X.cpu().numpy()
            
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        # Make predictions
        if return_std:
            # Get predictions from all trees
            tree_predictions = np.array([tree.predict(X) for tree in self.reg.estimators_])
            # Calculate mean and standard deviation
            mean_pred = np.mean(tree_predictions, axis=0)
            std_pred = np.std(tree_predictions, axis=0)
            return mean_pred, std_pred
        else:
            return self.reg.predict(X)

    def predict_samples(self, X, std=1, n_samples=100, seed=None):
        """
        Generate samples from the predictive distribution using individual tree predictions
        
        Args:
            X: Features (numpy array or torch tensor)
            std: Standard deviation for Gaussian bump in each leaf
            n_samples: Number of samples to generate
            seed: Random seed for reproducible sampling
            
        Returns:
            Samples with shape (X.shape[0], n_samples)
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before making predictions")
            
        # Convert tensor to numpy if necessary
        if torch.is_tensor(X):
            X = X.cpu().numpy()
            
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        # Handle seeding
        if seed is not None:
            current_state = np.random.get_state()
            np.random.seed(seed)
            
        try:
            # Get predictions from all trees
            tree_predictions = np.array([tree.predict(X) for tree in self.reg.estimators_])
            n_trees = tree_predictions.shape[0]  # T
            n_points = X.shape[0]  # Number of data points
            
            # Sample tree indices for each sample (same across all data points)
            sampled_tree_indices = np.random.choice(n_trees, size=n_samples, replace=True)
            
            # Get the corresponding tree predictions for each sample
            # Shape: (n_samples, n_points)
            selected_tree_preds = tree_predictions[sampled_tree_indices]  # Broadcasting magic!
            
            # Generate Gaussian noise with std deviation
            # Shape: (n_samples, n_points)
            noise = np.random.normal(0, std, size=(n_samples, n_points))
            
            # Add noise to tree predictions
            samples = selected_tree_preds + noise
            
            # Transpose to get shape (n_points, n_samples)
            return samples.T
        finally:
            if seed is not None:
                np.random.set_state(current_state)

    def get_model_state(self):
        """Get model state for checkpointing"""
        if not self.fitted:
            raise ValueError("Model must be fitted before getting checkpoint")
            
        return {
            "model_state": pickle.dumps(self.reg),
            "random_state": self.random_state,
            "model_type": "random_forest_regressor",
            "n_estimators": self.n_estimators,
        }
    
    def load_model_state(self, state):
        """Load model state from checkpoint"""
        if state is None:
            return

        self.reg = pickle.loads(state["model_state"])
        self.random_state = state["random_state"]
        self.fitted = True
        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)


# classification
class ClassifierWrapper(ABC):
    def __init__(self, model):
        self.model = model
        self._rng = None

    @abstractmethod
    def fit(self, X, y, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def get_samples(self, X_pool, X_target, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_model_state(self):
        """Get model state for checkpointing"""
        raise NotImplementedError
    
    @abstractmethod
    def load_model_state(self, state):
        """Load model state from checkpoint"""
        raise NotImplementedError
    
    def snapshot_rng_state(self):
        import random
        
        self._rng = {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            # Store only the seed values instead of full NumPy state
            "numpy_seed": np.random.get_state()[1][:5].copy(),  # First 5 seeds are sufficient
            "python_seed": random.getstate()[1][0],  # Extract just the seed value
        }
        
        # For model-specific states (like scikit-learn)
        if hasattr(self, 'model') and hasattr(self.model, 'random_state'):
            if self.model.random_state is not None:
                self._rng["sklearn_random_state"] = int(self.model.random_state)

    def restore_rng_state(self):
        """Restore RNG state from snapshot"""
        if self._rng is None:
            raise ValueError("RNG state has not been saved. Call snapshot_rng_state() first.")

        # Restore PyTorch states
        torch.set_rng_state(self._rng["cpu"])
        if torch.cuda.is_available() and self._rng["cuda"]:
            for device_idx, cuda_state in enumerate(self._rng["cuda"]):
                torch.cuda.set_rng_state(cuda_state, device=device_idx)
        
        # Restore NumPy state using seed
        if "numpy_seed" in self._rng:
            # Use the first seed value to reset NumPy's state
            np.random.seed(self._rng["numpy_seed"][0])
        
        # Restore Python random state using seed
        if "python_seed" in self._rng:
            random.seed(self._rng["python_seed"])
        
        # Restore sklearn random state if available
        if ("sklearn_random_state" in self._rng and 
            hasattr(self, 'model') and 
            hasattr(self.model, 'random_state')):
            self.model.random_state = self._rng["sklearn_random_state"]


class GPBaseClassifierModel(ApproximateGP):
    """Base GP model for classification using variational inference"""
    
    def __init__(self, train_x, ls=None):
        self.train_x = train_x
        self.ls = ls
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPBinaryClassifierModel(GPBaseClassifierModel):
    """GP model for binary classification using variational inference"""
    
    def __init__(self, train_x, ls=None):
        n_inputs = train_x.shape[0]
        inducing_points = train_x.clone()  # Use training points as inducing points
        
        # Single variational distribution for binary classification
        variational_distribution = CholeskyVariationalDistribution(n_inputs)
        
        # Variational strategy connects variational parameters to the model
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, 
            learn_inducing_locations=False
        )
        
        ApproximateGP.__init__(self, variational_strategy)
        
        # Mean and covariance functions for the GP - no batch shape
        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel())

        if ls is not None:
            # Set the lengthscale if provided
            self.covar_module.base_kernel.lengthscale = ls
            self.covar_module.base_kernel.raw_lengthscale.requires_grad = False


class GPMultiClassClassifierModel(GPBaseClassifierModel):
    """GP model for multi-class classification using variational inference"""
    
    def __init__(self, train_x, num_classes, ls=None):
        n_inputs = train_x.shape[0]
        inducing_points = train_x.clone()  # Use training points as inducing points
        
        # Variational distribution with one distribution per latent GP (one per class)
        variational_distribution = CholeskyVariationalDistribution(
            n_inputs, batch_shape=torch.Size([num_classes])
        )
        
        # Variational strategy connects variational parameters to the model
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, 
            learn_inducing_locations=False
        )
        
        ApproximateGP.__init__(self, variational_strategy)
        
        # Mean and covariance functions for the GP
        self.mean_module = ZeroMean(batch_shape=torch.Size([num_classes]))
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=torch.Size([num_classes])),
            batch_shape=torch.Size([num_classes])
        )

        if ls is not None:
            # Set the lengthscale if provided
            self.covar_module.base_kernel.lengthscale = torch.full(
                self.covar_module.base_kernel.lengthscale.shape,
                ls,
                dtype=self.covar_module.base_kernel.lengthscale.dtype
            )
            self.covar_module.base_kernel.raw_lengthscale.requires_grad = False


class GPyTorchBaseClassifierWrapper(ClassifierWrapper):
    """Base wrapper for GPyTorch classifiers"""
    
    def __init__(self, num_classes, n_iterations=500, learning_rate=0.01, ls=None, 
                 use_lr_schedule=False, lr_min=0.0001, random_seed=None):
        super().__init__(None)
        self.num_classes = num_classes
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.ls = ls
        self.model = None
        self.likelihood = None
        self.initialized = False
        self.identifier = f"gpy/{_lengthscale_identifier(ls)}"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_lr_schedule = use_lr_schedule
        self.lr_min = lr_min
        if random_seed is not None:
            self._set_random_seed(random_seed)

    def _set_random_seed(self, seed):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        self._initial_seed = seed

    def _create_model_and_likelihood(self, train_x):
        """To be implemented by subclasses"""
        raise NotImplementedError

    def initialize_model(self, X):
        """Initialize model and likelihood without training"""
        if not self.initialized:
            if hasattr(self, '_initial_seed'):
                self._set_random_seed(self._initial_seed)
            # Convert numpy to tensor if necessary
            train_x = X if torch.is_tensor(X) else torch.tensor(X, dtype=torch.float32).to(self.device)
            
            # Create model and likelihood
            self.model, self.likelihood = self._create_model_and_likelihood(train_x)
            
            # Move to device
            self.model = self.model.to(self.device)
            self.likelihood = self.likelihood.to(self.device)
            
            self.initialized = True
        return self
        
    def fit(self, X, y):
        # First initialize the model if not already done
        self.initialize_model(X)
        # Convert numpy to tensor if necessary
        train_x = torch.tensor(X, dtype=torch.float32).to(self.device)
        train_y = torch.tensor(y, dtype=torch.long).to(self.device)

        # Train the model
        self.model.train()
        self.likelihood.train()
        
        # Use Adam optimizer
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=self.learning_rate)
        
        # Add cosine learning rate scheduler if enabled
        if self.use_lr_schedule:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.n_iterations, 
                eta_min=self.lr_min
            )
        
        # Define the loss function (ELBO for variational inference)
        mll = gpytorch.mlls.VariationalELBO(
            self.likelihood, self.model, num_data=train_y.size(0)
        )

        # Training loop
        for i in range(self.n_iterations):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y).mean()
            loss.backward()
            optimizer.step()
            
            # Update learning rate if using scheduler
            if self.use_lr_schedule:
                scheduler.step()

        # Set model to evaluation mode
        self.model.eval()
        self.likelihood.eval()
        self.snapshot_rng_state()
        return self
    
    def predict(self, X):
        """
        Predict class labels for X.
        
        Args:
            X: Features (numpy array or torch tensor)
            
        Returns:
            Predicted class labels (numpy array)
        """
        # Get class probabilities
        probs = self.predict_proba(X)
        
        # Return class with highest probability
        return np.argmax(probs, axis=1)

    def get_model_state(self):
        """Get model state for checkpointing"""
        if not self.initialized:
            raise ValueError("Model must be initialized before loading checkpoint")
        return {
            "model_state": self.model.state_dict(),
            "likelihood_state": self.likelihood.state_dict(),
            "model_type": "gpytorch",
            "num_classes": self.num_classes,
            "ls": self.ls
        }
    
    def load_model_state(self, state):
        """Load model state from checkpoint"""
        # Initialize model if needed
        if not self.initialized:
            raise ValueError("Model must be initialized before loading checkpoint")
            
        self.model.load_state_dict(state["model_state"])
        self.likelihood.load_state_dict(state["likelihood_state"])
        self.model.eval()
        self.likelihood.eval()


class GPyTorchBinaryClassifierWrapper(GPyTorchBaseClassifierWrapper):
    """Wrapper for GPyTorch binary classifier"""
    
    def __init__(self, n_iterations=500, learning_rate=0.01, ls=None, random_seed=None):
        super().__init__(num_classes=2, n_iterations=n_iterations, learning_rate=learning_rate, ls=ls, random_seed=random_seed)

    def _create_model_and_likelihood(self, train_x):
        model = GPBinaryClassifierModel(train_x, ls=self.ls)
        likelihood = gpytorch.likelihoods.BernoulliLikelihood()
        return model, likelihood
        
    def get_samples(self, X_pool, X_target, n_samples=100):
        """
        Generate K different samples from GP latent functions for pool and optionally target points.
        """
        # Convert to tensor if necessary
        if isinstance(X_pool, np.ndarray):
            X_pool = torch.tensor(X_pool, dtype=torch.float32).to(self.device)
        if isinstance(X_target, np.ndarray):
            X_target = torch.tensor(X_target, dtype=torch.float32).to(self.device)
            
        # Get all points
        X_all = torch.cat([X_pool, X_target], dim=0)
            
        n_pool = X_pool.shape[0]
        
        # Get samples from GP posterior
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Get function distribution
            function_dist = self.model(X_all)
            
            # Sample from the function distribution (latent values)
            # Shape: [K, N_p + N_t] for binary case
            f_samples = function_dist.sample(torch.Size([n_samples]))
            
            # Reshape to [N_p + N_t, K]
            f_samples = f_samples.permute(1, 0)
            
            # Apply sigmoid to get probability samples
            prob_samples = torch.sigmoid(f_samples)
            
            # Convert to [N_p + N_t, K, 2] for consistent interface
            prob_class_1 = prob_samples.unsqueeze(-1)
            prob_class_0 = 1 - prob_class_1
            prob_samples = torch.cat([prob_class_0, prob_class_1], dim=-1)
            
        # Split into pool and target
        probs_pool = prob_samples[:n_pool]
        probs_targ = prob_samples[n_pool:]

        return probs_pool, probs_targ
        
    def predict_proba(self, X):
        # Convert numpy to tensor if necessary
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            
        # Ensure model is in evaluation mode
        self.model.eval()
        self.likelihood.eval()
        
        # Get predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Get function distribution
            function_dist = self.model(X)
            # Apply sigmoid to get class probabilities
            probs_class_1 = torch.sigmoid(function_dist.mean).unsqueeze(-1)
            probs_class_0 = 1 - probs_class_1
            probs = torch.cat([probs_class_0, probs_class_1], dim=-1)
            
        return probs.cpu().numpy()


class GPyTorchMultiClassClassifierWrapper(GPyTorchBaseClassifierWrapper):
    """Wrapper for GPyTorch multi-class classifier"""
    
    def _create_model_and_likelihood(self, train_x):
        model = GPMultiClassClassifierModel(train_x, self.num_classes, ls=self.ls)
        likelihood = SoftmaxLikelihood(self.num_classes, self.num_classes)
        return model, likelihood
        
    def get_samples(self, X_pool, X_target, n_samples=100):
        """Generate K different samples from GP latent functions for pool and target points."""
        # Convert to tensor if necessary
        if isinstance(X_pool, np.ndarray):
            X_pool = torch.tensor(X_pool, dtype=torch.float32).to(self.device)
        if isinstance(X_target, np.ndarray):
            X_target = torch.tensor(X_target, dtype=torch.float32).to(self.device)
            
        # Get all points
        X_all = torch.cat([X_pool, X_target], dim=0)
            
        n_pool = X_pool.shape[0]
        
        # Get samples from GP posterior
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Get function distribution
            function_dist = self.model(X_all)
            
            # Get samples in a way that's consistent with predict_proba
            f_samples = function_dist.sample(torch.Size([n_samples]))
            # Reshape to [N_p + N_t, K, Cl]
            f_samples = f_samples.permute(2, 0, 1)
            # Then apply the likelihood's transformation to get probabilities
            prob_samples = self.likelihood(f_samples).probs
        
        # Split into pool and target
        probs_pool = prob_samples[:n_pool]
        probs_targ = prob_samples[n_pool:]

        return probs_pool, probs_targ
        
    def predict_proba(self, X):
        # Convert numpy to tensor if necessary
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            
        # Ensure model is in evaluation mode
        self.model.eval()
        self.likelihood.eval()
        
        # Get predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Get function distribution
            function_dist = self.model(X)
            # Get probability distribution with just 1 sample
            probs = self.likelihood(function_dist, num_samples=100).probs.mean(dim=0)
            
        assert X.shape[0]*(1-1e-5)<probs.sum()<X.shape[0]*(1+1e-5), "Probabilities do not sum to 1"
        return probs.cpu().numpy()


class RandomForestClassifierWrapper(ClassifierWrapper):

    """Wrapper for scikit-learn RandomForestClassifier with tensor support"""
    
    def __init__(self, n_estimators=100, random_state=None, **kwargs):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.kwargs = kwargs
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=random_state,
            **kwargs
        )
        self.identifier = f"rf/{n_estimators}"
        super().__init__(model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def fit(self, X, y):
        # Convert to numpy if tensor
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        if torch.is_tensor(y):
            y = y.cpu().numpy()
            
        self.model.fit(X, y)
        self.snapshot_rng_state()
        return self
    
    def predict_proba(self, X):
        # Convert to numpy if tensor
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        
        # Get raw probabilities
        probs = self.model.predict_proba(X)
        
        # Convert to logits and apply softmax with small temperature for numerical stability
        epsilon = 1e-6
        logits = np.log(np.clip(probs, epsilon, 1 - epsilon))
        
        # Apply softmax (automatically handles numerical stability)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        stable_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        return stable_probs
    
    def get_samples(self, X_pool, X_target, n_samples=100):
        """
        Approximate samples from the model by using individual trees.
        Each tree vote is treated as a "sample" from the distribution.
        
        Args:
            X_pool: Pool data points
            X_target: Target data points
            n_samples: Number of samples (uses min(n_trees, n_samples) if provided)
        """
        # Convert to numpy if tensor
        if torch.is_tensor(X_pool):
            X_pool = X_pool.cpu().numpy()
        if torch.is_tensor(X_target):
            X_target = X_target.cpu().numpy()
            
        # Concatenate pool and target
        X_all = np.vstack([X_pool, X_target])
        n_pool = X_pool.shape[0]
        
        # Get number of samples (use all trees if n_samples not specified)
        n_trees = self.model.n_estimators
        n_samples_to_use = min(n_trees, n_samples) if n_samples else n_trees
        
        # Get class counts from individual trees
        n_classes = len(self.model.classes_)
        
        # Initialize result arrays
        all_samples = np.zeros((X_all.shape[0], n_samples_to_use, n_classes))
        
        # Sample trees without replacement
        rng = np.random.RandomState(self.random_state)
        sampled_trees = rng.choice(n_trees, n_samples_to_use, replace=False)
        
        # For each sampled tree, get its predictions
        for i, tree_idx in enumerate(sampled_trees):
            tree = self.model.estimators_[tree_idx]
            # Get class probabilities
            tree_probs = tree.predict_proba(X_all)
            all_samples[:, i, :] = tree_probs
        
        # Create tensors for consistency with GP wrappers
        all_samples_tensor = torch.tensor(all_samples, dtype=torch.float32, device=self.device)
        
        # Split back into pool and target
        probs_pool = all_samples_tensor[:n_pool]
        probs_targ = all_samples_tensor[n_pool:]
        
        return probs_pool, probs_targ

    def get_model_state(self):
        """Get model state for checkpointing"""
        return {
            "model_state": pickle.dumps(self.model),
            "model_type": "random_forest",
            "n_estimators": self.n_estimators,
        }
    
    def load_model_state(self, state):
        """Load model state from checkpoint"""
        self.model = pickle.loads(state["model_state"])


class NNClassifierModel(nn.Module):
    """Neural network classifier model with dropout for uncertainty estimation"""
    
    def __init__(self, input_dim, num_classes, hidden_dims=[64, 32], dropout_rate=0.1):
        """
        Args:
            input_dim: Dimension of input features
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability for uncertainty estimation
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = hidden_dim
            
        # Output layer
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, num_classes)
        
    def forward(self, x, enable_dropout=False):
        """
        Forward pass with optional dropout activation in evaluation mode
        
        Args:
            x: Input tensor
            enable_dropout: Whether to enable dropout during inference
        
        Returns:
            Logits (pre-activation)
        """
        if enable_dropout:
            # Store the original mode
            train_mode = self.training
            # Enable dropout
            self.train()
            
            # Forward pass
            x = self.hidden_layers(x)
            x = self.output_layer(x)
            
            # Restore the original mode
            if not train_mode:
                self.eval()
        else:
            # Normal forward pass
            x = self.hidden_layers(x)
            x = self.output_layer(x)
            
        return x


class NeuralNetClassifierWrapper(ClassifierWrapper):
    """Wrapper for PyTorch neural network classifier"""
    
    def __init__(self, input_dim, num_classes=2, hidden_dims=[64, 32], 
                 n_epochs=100, batch_size=32, learning_rate=0.01, 
                 dropout_rate=0.1, weight_decay=1e-4):
        """
        Args:
            input_dim: Input dimensionality
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
            n_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
            dropout_rate: Dropout probability for uncertainty estimation
            weight_decay: L2 regularization strength
        """
        super().__init__(None)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.identifier = f"nn/{dropout_rate}"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create the model
        self.model = NNClassifierModel(
            input_dim=input_dim,
            num_classes=num_classes, 
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        )
        self.model = self.model.to(self.device)
        
    def fit(self, X, y):
        """
        Train the neural network on the given data
        
        Args:
            X: Features (numpy array or torch tensor)
            y: Labels (numpy array or torch tensor)
            
        Returns:
            self
        """
        # Convert to tensors if necessary
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.long).to(self.device)
        
        # Create optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Set to train mode
        self.model.train()
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        
        # Training loop
        for epoch in range(self.n_epochs):
            for batch_X, batch_y in dataloader:
                # Forward pass
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Set to evaluation mode
        self.model.eval()
        self.snapshot_rng_state()
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X
        
        Args:
            X: Features (numpy array or torch tensor)
            
        Returns:
            Class probabilities (numpy array)
        """
        # Convert to tensor if necessary
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Get predictions
        with torch.no_grad():
            logits = self.model(X)
            # Apply softmax for multiclass or sigmoid for binary
            probs = F.softmax(logits, dim=1)
            
        return probs.cpu().numpy()
    
    def get_samples(self, X_pool, X_target, n_samples=100):
        """
        Generate samples using MC dropout
        
        Args:
            X_pool: Pool data points
            X_target: Target data points
            n_samples: Number of Monte Carlo samples
            
        Returns:
            (pool_samples, target_samples) tuple of tensors with shape:
            - pool_samples: [n_pool, n_samples, n_classes]
            - target_samples: [n_target, n_samples, n_classes]
        """
        # Convert to tensor if necessary
        if isinstance(X_pool, np.ndarray):
            X_pool = torch.tensor(X_pool, dtype=torch.float32).to(self.device)
        if isinstance(X_target, np.ndarray):
            X_target = torch.tensor(X_target, dtype=torch.float32).to(self.device)
            
        # Get all points
        X_all = torch.cat([X_pool, X_target], dim=0)
        n_pool = X_pool.shape[0]
        
        # Initialize result tensor
        all_samples = torch.zeros(
            (X_all.shape[0], n_samples, self.num_classes),
            device=self.device
        )
        
        # Get MC dropout samples
        self.model.eval()  # Still in eval mode, but we'll force dropout
        with torch.no_grad():
            for i in range(n_samples):
                # Forward pass with dropout enabled
                logits = self.model(X_all, enable_dropout=True)
                
                # Apply appropriate activation
                if self.num_classes > 2:
                    probs = F.softmax(logits, dim=1)
                else:
                    # For binary classification
                    probs = torch.sigmoid(logits)
                    
                all_samples[:, i, :] = probs
        
        # Split into pool and target
        probs_pool = all_samples[:n_pool]
        probs_targ = all_samples[n_pool:]
        
        return probs_pool, probs_targ

    def get_model_state(self):
        """Get model state for checkpointing"""
        return {
            "model_state": self.model.state_dict(),
            "model_type": "neural_net",
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "hidden_dims": self.hidden_dims,
            "dropout_rate": self.dropout_rate,
            "device": str(self.device)
        }
    
    def load_model_state(self, state):
        """Load model state from checkpoint"""
        if state is None:
            return
        
        # Recreate model if needed
        if self.model is None:
            self.model = NNClassifierModel(
                state["input_dim"], 
                state["num_classes"],
                state["hidden_dims"],
                state["dropout_rate"]
            ).to(self.device)
        
        self.model.load_state_dict(state["model_state"])
        self.model.eval()


class BayesianLinear(nn.Module):
    """Bayesian linear layer with variational inference"""
    
    def __init__(self, in_features, out_features, prior_var=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_var = prior_var
        
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -3.0))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), -3.0))
        
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        # Initialize means
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.constant_(self.bias_mu, 0)
        
    def forward(self, x):
        # Sample weights and biases
        weight_std = F.softplus(self.weight_rho) + 1e-5
        weight = Normal(self.weight_mu, weight_std).rsample()
        
        bias_std   = F.softplus(self.bias_rho) + 1e-5
        bias = Normal(self.bias_mu, bias_std).rsample()
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self):
        """Compute KL divergence between posterior and prior"""
        # Create prior distributions
        weight_prior = Normal(torch.zeros_like(self.weight_mu), 
                             torch.full_like(self.weight_mu, math.sqrt(self.prior_var)))
        bias_prior = Normal(torch.zeros_like(self.bias_mu), 
                           torch.full_like(self.bias_mu, math.sqrt(self.prior_var)))
        
        # Create posterior distributions
        weight_std = F.softplus(self.weight_rho) + 1e-5
        bias_std   = F.softplus(self.bias_rho) + 1e-5
        
        weight_posterior = Normal(self.weight_mu, weight_std)
        bias_posterior = Normal(self.bias_mu, bias_std)
        
        # Compute KL divergences
        weight_kl = kl_divergence(weight_posterior, weight_prior).sum()
        bias_kl = kl_divergence(bias_posterior, bias_prior).sum()
        
        return weight_kl + bias_kl


class BayesianNNClassifierModel(nn.Module):
    """Bayesian neural network classifier with variational inference"""
    
    def __init__(self, input_dim, num_classes, hidden_dims=[64, 32], prior_var=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.prior_var = prior_var
        
        # Build Bayesian layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(BayesianLinear(prev_dim, hidden_dim, prior_var))
            prev_dim = hidden_dim
            
        # Output layer
        layers.append(BayesianLinear(prev_dim, num_classes, prior_var))
        
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        """Forward pass with ReLU activations between hidden layers"""
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        # Final layer (no activation, returns logits)
        x = self.layers[-1](x)
        return x
    
    def kl_divergence(self):
        """Compute total KL divergence for the network"""
        kl_sum = 0
        for layer in self.layers:
            kl_sum += layer.kl_divergence()
        return kl_sum
    
    def sample_predictions(self, x, n_samples=100):
        """Generate multiple predictions by sampling from weight distributions"""
        # Set to training mode to enable sampling
        training_mode = self.training
        self.train()
        
        predictions = []
        for _ in range(n_samples):
            logits = self.forward(x)
            predictions.append(F.softmax(logits, dim=1))
            
        # Restore original mode
        if not training_mode:
            self.eval()
            
        return torch.stack(predictions, dim=1)  # [batch_size, n_samples, num_classes]


class BayesianNNClassifierWrapper(ClassifierWrapper):
    """Bayesian neural network classifier wrapper with variational inference"""

    def __init__(self, input_dim, num_classes=2, hidden_dims=[32, 16],
                 n_epochs=200, batch_size=64, learning_rate=0.01,
                 prior_var=1.0, kl_weight=0.001, random_seed=None):
        """
        Args:
            input_dim: Input feature dimensionality
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
            n_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
            prior_var: Prior variance for Bayesian layers
            kl_weight: Weight for KL divergence term in ELBO (reduced default)
            random_seed: Random seed for reproducibility
        """
        super().__init__(None)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.prior_var = prior_var
        self.kl_weight = kl_weight
        self.identifier = f"bnn/{prior_var}_{kl_weight}"  # Updated identifier
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seed
        if random_seed is not None:
            self._set_random_seed(random_seed)
            self._initial_seed = random_seed
        
        # Create the Bayesian model
        self.model = BayesianNNClassifierModel(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            prior_var=prior_var
        )
        self.model = self.model.to(self.device)
        
    def _set_random_seed(self, seed):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        
    def fit(self, X, y):
        """
        Train the Bayesian neural network using variational inference
        
        Args:
            X: Features (numpy array or torch tensor)
            y: Labels (numpy array or torch tensor)
            
        Returns:
            self
        """
        # Convert to tensors if necessary
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.long).to(self.device)
        
        # Number of training samples for KL scaling
        n_train = X.shape[0]

        # Create optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Loss function
        criterion = nn.CrossEntropyLoss(reduction='mean')  # Explicit mean reduction
        
        # Set to train mode
        self.model.train()
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        
        # Training loop with better monitoring
        best_nll = float('inf')
        patience_counter = 0

        # Training loop with KL annealing
        for epoch in range(self.n_epochs):
            # KL annealing: start from 0 and gradually increase
            kl_weight_current = self.kl_weight * min(1.0, epoch / (self.n_epochs * 0.2))
            
            epoch_loss = 0
            epoch_nll = 0
            epoch_kl = 0
            for batch_X, batch_y in dataloader:
                # Forward pass
                logits = self.model(batch_X)
                
                # Compute losses
                nll_loss = criterion(logits, batch_y)
                kl_loss = self.model.kl_divergence()
                
                # Use annealed KL weight
                kl_loss_scaled = (1.0 / n_train) * kl_weight_current * kl_loss
                total_loss = nll_loss + kl_loss_scaled
                
                # Backward pass and optimization
                optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping with monitoring
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Check for exploding gradients
                if grad_norm > 10.0:
                    print(f"Warning: Large gradient norm {grad_norm:.2f} at epoch {epoch}")
                    continue  # Skip this batch
                
                optimizer.step()
                
                epoch_loss += total_loss.item()
                epoch_nll += nll_loss.item()
                epoch_kl += kl_loss.item()

            # Calculate averages
            # avg_total = epoch_loss / len(dataloader)
            avg_nll = epoch_nll / len(dataloader)
            # avg_kl = epoch_kl / len(dataloader)
            
            # Early stopping based on NLL (not total loss!)
            if avg_nll < best_nll:
                best_nll = avg_nll
                patience_counter = 0
                # Save best model state here if needed
            else:
                patience_counter += 1
            
            # Enhanced logging
            # if epoch % 20 == 0:
            #     print(f"Epoch {epoch}: Total={avg_total:.4f}, NLL={avg_nll:.4f} ({'↓' if avg_nll < best_nll else '↑'}), "
            #         f"KL={avg_kl:.1f}, KL_weight={kl_weight_current:.4f}")
            
            # Optional: Early stopping on NLL plateau
            if patience_counter > 50:  # Adjust patience as needed
                # print(f"Early stopping at epoch {epoch} (NLL plateau)")
                break
        
        # print("---------------------------------------------------------")
        # Set to evaluation mode
        self.model.eval()
        self.snapshot_rng_state()
        return self
    
    def predict_proba(self, X, n_samples=100):
        """
        Predict class probabilities using Monte Carlo sampling
        
        Args:
            X: Features (numpy array or torch tensor)
            n_samples: Number of Monte Carlo samples for prediction
            
        Returns:
            Class probabilities (numpy array)
        """
        # Convert to tensor if necessary
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Get Monte Carlo predictions
        with torch.no_grad():
            mc_predictions = self.model.sample_predictions(X, n_samples)
            # Average over samples
            probs = mc_predictions.mean(dim=1)
            
        return probs.cpu().numpy()
    
    def get_samples(self, X_pool, X_target, n_samples=100):
        """
        Generate samples for active learning using Monte Carlo sampling
        
        Args:
            X_pool: Pool data points
            X_target: Target data points  
            n_samples: Number of Monte Carlo samples
            
        Returns:
            (pool_samples, target_samples) tuple with shapes:
            - pool_samples: [n_pool, n_samples, n_classes]
            - target_samples: [n_target, n_samples, n_classes]
        """
        # Convert to tensor if necessary
        if isinstance(X_pool, np.ndarray):
            X_pool = torch.tensor(X_pool, dtype=torch.float32).to(self.device)
        if isinstance(X_target, np.ndarray):
            X_target = torch.tensor(X_target, dtype=torch.float32).to(self.device)
            
        # Get all points
        X_all = torch.cat([X_pool, X_target], dim=0)
        n_pool = X_pool.shape[0]
        
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Get Monte Carlo predictions for all points
        with torch.no_grad():
            all_samples = self.model.sample_predictions(X_all, n_samples)
        
        # Split into pool and target
        probs_pool = all_samples[:n_pool]
        probs_targ = all_samples[n_pool:]
        
        return probs_pool, probs_targ

    def get_model_state(self):
        """Get model state for checkpointing"""
        return {
            "model_state": self.model.state_dict(),
            "model_type": "bayesian_nn",
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "hidden_dims": self.hidden_dims,
            "prior_var": self.prior_var,
            "kl_weight": self.kl_weight,
            "device": str(self.device)
        }
    
    def load_model_state(self, state):
        """Load model state from checkpoint"""
        if state is None:
            return
        
        # Recreate model if needed
        if self.model is None:
            self.model = BayesianNNClassifierModel(
                state["input_dim"],
                state["num_classes"], 
                state["hidden_dims"],
                state["prior_var"]
            ).to(self.device)
        
        self.model.load_state_dict(state["model_state"])
        self.model.eval()


# Factory function to choose the appropriate model
def create_gp_classifier(num_classes=2, n_iterations=500, learning_rate=0.01, ls=None, 
                         use_lr_schedule=False, lr_min=1e-5, random_seed=None):
    """
    Factory function to create appropriate GP classifier based on number of classes
    
    Args:
        num_classes: Number of classes (2 for binary, >2 for multi-class)
        n_iterations: Number of training iterations
        learning_rate: Learning rate for optimization
        ls: Lengthscale for RBF kernel (optional)
        use_lr_schedule: Whether to use cosine learning rate schedule
        lr_min: Minimum learning rate for scheduler
        random_seed: Random seed for reproducibility
    
    Returns:
        A classifier wrapper instance
    """
    if num_classes == 2:
        return GPyTorchBinaryClassifierWrapper(n_iterations=n_iterations,
                                               learning_rate=learning_rate,
                                               ls=ls,
                                               use_lr_schedule=use_lr_schedule,
                                               lr_min=lr_min,
                                               random_seed=random_seed)
    elif num_classes > 2:
        return GPyTorchMultiClassClassifierWrapper(num_classes=num_classes,
                                                   n_iterations=n_iterations,
                                                   learning_rate=learning_rate,
                                                   ls=ls,
                                                   use_lr_schedule=use_lr_schedule,
                                                   lr_min=lr_min,
                                                   random_seed=random_seed)
    else:
        raise ValueError(f"Invalid number of classes: {num_classes}")
