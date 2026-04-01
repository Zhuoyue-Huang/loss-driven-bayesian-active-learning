import os
import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits, fetch_openml
from pmlb import fetch_data
from ucimlrepo import fetch_ucirepo, list_available_datasets


def grid_points(
    start: float,
    end: float,
    num: int,
) -> np.ndarray:
    x = np.linspace(start, end, num)
    xx, yy = np.meshgrid(x, x)
    return np.column_stack((xx.ravel(), yy.ravel()))

def binary_arctan(
    loc: np.ndarray
) -> float | np.ndarray:
    decision = lambda x: np.arctan(x)
    if loc.ndim == 1:
        loc = loc.reshape(1, -1)
    labels = (decision(loc[:, 0]) <= loc[:,1]).astype(int)
    return labels

def ternary_angular(
    loc: np.ndarray,
) -> float | np.ndarray:
    if loc.ndim == 1:
        loc = loc.reshape(1, -1)
    angles = np.arctan2(loc[:, 1], loc[:, 0])
    angles = (angles + 2 * np.pi) % (2 * np.pi)
    # Define sector boundaries at 0, 2pi/3, 4pi/3
    labels = np.zeros(len(loc), dtype=int)
    labels[angles < 2 * np.pi / 3] = 0
    labels[(angles >= 2 * np.pi / 3) & (angles < 4 * np.pi / 3)] = 1
    labels[angles >= 4 * np.pi / 3] = 2

    return labels

def ternary_parabola(
    loc: np.ndarray,
) -> float | np.ndarray:
    if loc.ndim == 1:
        loc = loc.reshape(1, -1)
    labels = np.zeros(len(loc), dtype=int)
    labels[loc[:, 1] >= loc[:, 0]**2-loc[:, 0]] = 1
    labels[(loc[:, 1] < loc[:, 0]**2-loc[:, 0]) & (loc[:, 1] <= -loc[:, 0]**2-loc[:, 0])] = 2
    return labels

def quadrant(
    loc: np.ndarray,
) -> float | np.ndarray:
    """Label points based on their quadrant (0=I, 1=II, 2=III, 3=IV)."""
    if loc.ndim == 1:
        loc = loc.reshape(1, -1)
        
    labels = np.zeros(len(loc), dtype=int)
    
    # Quadrant II: x < 0, y > 0
    labels[(loc[:, 0] <= 0) & (loc[:, 1] > 0)] = 1
    
    # Quadrant III: x < 0, y < 0
    labels[(loc[:, 0] < 0) & (loc[:, 1] <= 0)] = 2
    
    # Quadrant IV: x > 0, y < 0
    labels[(loc[:, 0] >= 0) & (loc[:, 1] < 0)] = 3
    
    return labels

def quad_diag(
    loc: np.ndarray,
) -> float | np.ndarray:
    """
    Label points based on regions defined by y=x and y=-x lines:
    0: Above both lines (upper region where y > x and y > -x)
    1: Right of y=-x but below y=x (right region where y < x and y > -x)
    2: Below both lines (lower region where y < x and y < -x)
    3: Left of y=x but below y=-x (left region where y > x and y < -x)
    """
    if loc.ndim == 1:
        loc = loc.reshape(1, -1)
        
    labels = np.zeros(len(loc), dtype=int)
    
    # Region 1: Right (y <= x and y > -x)
    labels[(loc[:, 1] <= loc[:, 0]) & (loc[:, 1] > -loc[:, 0])] = 1
    
    # Region 2: Lower (y < x and y <= -x)
    labels[(loc[:, 1] < loc[:, 0]) & (loc[:, 1] <= -loc[:, 0])] = 2
    
    # Region 3: Left (y >= x and y < -x)
    labels[(loc[:, 1] >= loc[:, 0]) & (loc[:, 1] < -loc[:, 0])] = 3
    
    # Region 0: Upper (y > x and y >= -x) - already labeled as 0
    
    return labels

def spiral(n, k, noise=0.2, turns=1.0, seed=None):
    """
    Generate a 2D k-class spiral dataset with configurable coil tightness.

    Parameters:
    - n (int): total number of samples
    - k (int): number of classes (spiral arms)
    - noise (float): standard deviation of Gaussian noise added to the angles
    - turns (float): number of revolutions for each spiral arm (smaller values = tighter spiral)
    - seed (int, optional): random seed for reproducibility

    Returns:
    - data (np.ndarray): shape (n, 3), columns are [x, y, class_label]
    """
    if seed is not None:
        np.random.seed(seed)

    # Determine samples per class (distribute remainder among first classes)
    n_per_class = n // k
    remainder = n % k

    X = np.zeros((n, 2), dtype=float)
    labels = np.zeros(n, dtype=int)

    idx = 0
    for class_label in range(k):
        # Number of samples for this class
        num = n_per_class + (1 if class_label < remainder else 0)

        # Radius values (linearly spaced from center to edge)
        r = np.linspace(0.0, 1.0, num)
        # Angle values: offset by class, spread by 'turns', plus noise
        theta = (
            class_label * 2 * np.pi / k
            + r * turns * 2 * np.pi
            + np.random.randn(num) * noise
        )

        # Cartesian coordinates
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # Assign into arrays
        X[idx : idx + num, 0] = x
        X[idx : idx + num, 1] = y
        labels[idx : idx + num] = class_label
        idx += num

    # Concatenate inputs and labels into one array of shape (n, 3)
    data = np.hstack((X, labels.reshape(-1, 1)))
    return data

def fetch_any_uci(dataset, *, version=None, data_home=None,
                  prefer=("sklearn", "pmlb", "ucimlrepo", "openml")):
    """
    dataset: str name or int id
    prefer: ordered tuple of sources to try
    Returns: (X_df, y_series, source_tag, n_test)
    where n_test is the number of test samples if pre-split exists, None otherwise
    If n_test is not None, the last n_test samples are the test set
    """

    ds_is_int = isinstance(dataset, int)
    ds_name = str(dataset).lower() if not ds_is_int else None
    last_err = None

    if data_home is None:
        env_home = os.environ.get("LDAL_DATA_HOME")
        if env_home:
            data_home = os.path.expanduser(env_home)

    for src in prefer:
        try:
            if src == "sklearn":
                print("--------fetch from sklearn-------")
                if not ds_is_int and ds_name in {"iris", "wine", "breast_cancer", "digits"}:
                    loader = {"iris": load_iris, "wine": load_wine,
                              "breast_cancer": load_breast_cancer, "digits": load_digits}[ds_name]
                    b = loader(as_frame=True)
                    # sklearn toy datasets don't have predefined splits
                    return b.data.copy(), b.target.copy(), f"sk:{ds_name}", None
                    
            elif src == "pmlb":
                print("--------fetch from pmlb-------")
                try:
                    X, y = fetch_data(dataset if not ds_is_int else str(dataset), return_X_y=True)
                    # PMLB datasets typically don't have predefined splits
                    return pd.DataFrame(X), pd.Series(y, name="target"), f"pmlb:{dataset}", None
                except Exception as e:
                    last_err = e

            elif src == "ucimlrepo":
                # print("--------fetch from ucimlrepo-------")
                # Works best with numeric IDs; name search is supported via metadata scan
                attempts = 3
                delay = 1.0
                for attempt in range(attempts):
                    try:
                        if ds_is_int:
                            uci = fetch_ucirepo(id=dataset)
                        else:
                            meta = list_available_datasets()
                            hit = meta[meta["name"].str.lower() == ds_name]
                            if hit.empty:
                                # try contains match
                                hit = meta[meta["name"].str.lower().str.contains(ds_name)]
                            if hit.empty:
                                raise ValueError(f"ucimlrepo: dataset '{dataset}' not found")
                            uci = fetch_ucirepo(id=int(hit.iloc[0]["id"]))
                        break
                    except Exception as e:
                        last_err = e
                        if attempt < attempts - 1:
                            time.sleep(delay)
                            delay *= 2
                            continue
                        raise
                
                X_df = uci.data.features.copy()
                y = uci.data.targets
                if hasattr(y, "iloc") and y.ndim > 1 and y.shape[1] > 1:
                    y = y.iloc[:, 0]
                y_series = pd.Series(np.asarray(y).squeeze(), name="target")
                
                # Check for train/test split in UCI data
                n_test = None
                if hasattr(uci.data, 'ids') and uci.data.ids is not None:
                    # Some UCI datasets have predefined splits indicated by IDs
                    ids = uci.data.ids
                    if 'fold' in ids.columns or 'split' in ids.columns:
                        # Check if there's a test set indicator
                        test_mask = None
                        if 'fold' in ids.columns:
                            test_mask = ids['fold'] == 'test'
                        elif 'split' in ids.columns:
                            test_mask = ids['split'] == 'test'
                        
                        if test_mask is not None and test_mask.any():
                            # Reorder data so test samples are at the end
                            train_mask = ~test_mask
                            train_indices = train_mask[train_mask].index
                            test_indices = test_mask[test_mask].index
                            
                            # Reorder both X and y
                            new_order = list(train_indices) + list(test_indices)
                            X_df = X_df.loc[new_order].reset_index(drop=True)
                            y_series = y_series.loc[new_order].reset_index(drop=True)
                            n_test = len(test_indices)
                
                return X_df, y_series, f"uci:{getattr(uci, 'metadata', {}).get('name', dataset)}", n_test

            elif src == "openml":
                # print("--------fetch from openml-------")
                fetch_kw = dict(as_frame=True)
                if data_home is not None:
                    fetch_kw["data_home"] = data_home
                if ds_is_int:
                    ds = fetch_openml(data_id=dataset, **fetch_kw)
                else:
                    ds = fetch_openml(name=str(dataset), **({} if version is None else {"version": version}), **fetch_kw)
                target = ds.target
                if hasattr(target, "shape") and target.ndim > 1 and target.shape[1] > 1:
                    target = target.iloc[:, 0]
                target = pd.Series(np.asarray(target).squeeze(), name="target")
                n_test = None
                return ds.data.copy(), target, f"openml:{dataset}", n_test

        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Could not fetch dataset '{dataset}' via {prefer}. Last error: {last_err}")

def sample_per_class_indices(data, p, seed=None):
    """
    Randomly sample indices of p points from each class in a dataset.

    Parameters:
    - data (np.ndarray): shape (n, 3), columns are [x, y, class_label]
    - p (int): number of samples to draw from each class
    - seed (int, optional): random seed for reproducibility

    Returns:
    - indices (np.ndarray): shape (p*k,), indices into the original data array
    """
    if seed is not None:
        np.random.seed(seed)

    # Identify unique classes
    try:
        y_arr = data.target
    except AttributeError:
        y_arr = data[:, -1]
    classes = np.unique(y_arr).astype(int)
    all_indices = []

    for cls in classes:
        # Get indices for this class
        class_indices = np.where(y_arr == cls)[0]
        n_cls = class_indices.shape[0]
        if n_cls < p:
            raise ValueError(f"Not enough samples in class {cls}: requested {p}, but only {n_cls} available.")
        # Randomly choose p indices without replacement
        chosen = np.random.choice(class_indices, p, replace=False)
        all_indices.append(chosen)

    # Concatenate and return flat array of indices
    return np.hstack(all_indices)

def simulate_regression_labels(
    mean: np.ndarray,
    std: float,
    rng = np.random.default_rng(0),
) -> np.ndarray:
    return rng.normal(loc=mean, scale=std)
