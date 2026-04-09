"""
General data loading and splitting utilities for JAG and borehole datasets.

JAG:
    - 5 inputs, 1 output
    - default path: "../../data/JAG_10k.csv"

Borehole:
    - 8 inputs, 1 output
    - default path: "../../data/borehole_10k.csv"

HST: 
    - 8 inputs, 1 output

"""

from typing import Optional, Tuple
import warnings
import os

import numpy as np
import pandas as pd

from scipy.spatial import cKDTree  # type: ignore
from scipy.stats import qmc

from sklearn.model_selection import train_test_split
# Dataset configuration
DATASET_CONFIG = {
    "JAG": {
        "path": "../../data/JAG_10k.csv",
        "n_inputs": 5,
        "n_outputs": 1,
    },
    "borehole": {
        "path": "../../data/borehole_10k.csv",
        "n_inputs": 8,
        "n_outputs": 1,
    },
    "HST": {
        "path": "../../data/hst_O2_10k.csv",
        "n_inputs": 8,
        "n_outputs": 1,
    }
}


# Loading data

def load_data(
    dataset: str = "JAG",
    n_samples: int = 10000,
    random: bool = True,
    path_to_csv: Optional[str] = None,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Load a subset of a dataset from CSV.

    Assumes:
        - CSV has exactly n_inputs + n_outputs columns
        - No header, or any header will be ignored and replaced

    Args:
        dataset: Which dataset to load, "JAG" or "borehole".
        path_to_csv: Optional explicit path; if None, use default from config.
        n_samples: Number of rows to load.
        random: If True, select rows randomly; else select first n_samples rows.
        seed: Random seed for reproducibility (used if random is True).

    Returns:
        pd.DataFrame:
            For JAG:
                columns: [x0, x1, x2, x3, x4, y]
            For borehole:
                columns: [rw, r, Tu, Hu, Tl, Hl, L, Kw, y]
    """
    if dataset not in DATASET_CONFIG:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. "
            f"Supported: {list(DATASET_CONFIG.keys())}"
        )

    cfg = DATASET_CONFIG[dataset]

    if path_to_csv is None:
        path_to_csv = cfg["path"]

    if not os.path.isfile(path_to_csv): # type: ignore
        raise FileNotFoundError(f"CSV file not found at: {path_to_csv}")

    df = pd.read_csv(path_to_csv) # type: ignore

    if dataset == "JAG":
        df.columns = ["x0", "x1", "x2", "x3", "x4", "y"]
    elif dataset == "borehole":
        df.columns = ["rw", "r", "Tu", "Hu", "Tl", "Hl", "L", "Kw", "y"]
    elif dataset == "HST":
        df.columns = ["Umag","Ts","Ta","alphan","sigmat","theta","phi","panang","Cd"]

    # Check and warn if n_samples is too large
    if n_samples > len(df):
        warnings.warn(
            "n_samples is greater than the number of rows in the dataset "
            f"({len(df)}). Using the full 10k dataset instead."
        )
        n_samples = len(df)

    # Select rows
    if random:
        print(
            f"Selecting {n_samples} samples at random from the {dataset} dataset (seed={seed}).\n"
        )
        df = df.sample(n=n_samples, random_state=seed)
    else:
        print(f"Selecting the first {n_samples} samples from the {dataset} dataset.\n")
        df = df.iloc[:n_samples]

    return df


def load_data_from_file(
    dataset: str = "HST",
    train_path: str = "train.csv",
    test_path: str = "test.csv",
    validation_path: str = "validation.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load training, test, and validation datasets from CSV files.

    Assumes:
        - Each CSV has exactly n_inputs + n_outputs columns
        - Headers may be missing or ignored, and will be replaced

    Args:
        dataset: Which dataset to load, "JAG", "borehole", or "HST".
        train_path: Path to the training CSV file.
        test_path: Path to the test CSV file.
        validation_path: Path to the validation CSV file.

    Returns:
        traindf: Training dataframe
        testdf: Test dataframe
        validationdf: Validation dataframe
    """
    if dataset not in DATASET_CONFIG:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. "
            f"Supported: {list(DATASET_CONFIG.keys())}"
        )

    for path in [train_path, test_path, validation_path]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"CSV file not found at: {path}")

    traindf = pd.read_csv(train_path)
    testdf = pd.read_csv(test_path)
    validationdf = pd.read_csv(validation_path)

    if dataset == "JAG":
        columns = ["x0", "x1", "x2", "x3", "x4", "y"]
    elif dataset == "borehole":
        columns = ["rw", "r", "Tu", "Hu", "Tl", "Hl", "L", "Kw", "y"]
    elif dataset == "HST":
        columns = ["Umag", "Ts", "Ta", "alphan", "sigmat", "theta", "phi", "panang", "Cd"]

    traindf.columns = columns
    testdf.columns = columns
    validationdf.columns = columns

    print(f"Loaded training data from:   {train_path}, shape={traindf.shape}")
    print(f"Loaded test data from:       {test_path}, shape={testdf.shape}")
    print(f"Loaded validation data from: {validation_path}, shape={validationdf.shape}")

    return traindf, testdf, validationdf


# Splitting data

def split_data(
    df: pd.DataFrame,
    LHD: bool = False,
    n_train: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets using either Latin Hypercube Design
    (LHD) or random split.

    Args:
        df: Input DataFrame where the last column is the output.
        LHD: If True, use Latin Hypercube Design for selecting training
            samples; if False, use random split.
        n_train: Number of training samples to select.
        seed: Random seed for reproducibility.

    Returns:
        x_train: Training features array.
        x_test: Testing features array.
        y_train: Training labels array (column vector).
        y_test: Testing labels array (column vector).

    Raises:
        ValueError: If n_train is greater than the total number of samples in df.
    """
    # Split the data into features (x) and labels (y)
    x = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()
    n_total, k = x.shape

    # Ensure n_train is not greater than total_samples
    if n_train > n_total:
        raise ValueError(
            f"n_train cannot be greater than the total number of samples "
            f"({n_total})."
        )

    if LHD:
        print(
            "Using n_train closest points to Latin Hypercube Design for "
            "training points.\n"
        )
        # Latin Hypercube Sampling for n_train points in k dimensions
        LHD_gen = qmc.LatinHypercube(d=k, seed=seed)  # type: ignore
        x_lhd = LHD_gen.random(n=n_train)

        # Scale LHD points to the range of x
        for i in range(k):
            x_lhd[:, i] = x_lhd[:, i] * (np.max(x[:, i]) - np.min(x[:, i])) + np.min(
                x[:, i]
            )

        # Build KDTree for nearest neighbor search
        tree = cKDTree(x)

        def query_unique(tree_obj, small_data):
            used_indices = set()
            unique_indices = []
            unique_distances = []

            for point in small_data:
                distances, indices = tree_obj.query(point, k=50)
                for dist, idx in zip(distances, indices):
                    if idx not in used_indices:
                        used_indices.add(idx)
                        unique_indices.append(idx)
                        unique_distances.append(dist)
                        break
            return np.array(unique_distances), np.array(unique_indices)

        # Query for unique nearest neighbors
        _, index = query_unique(tree, x_lhd)

        x_train = x[index, :]
        y_train = y[index].reshape(-1, 1)
        mask = np.ones(n_total, dtype=bool)
        mask[index] = False
        x_test = x[mask, :]
        y_test = y[mask].reshape(-1, 1)
    else:
        # Standard random split with exact n_train samples
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            train_size=n_train,
            test_size=None,
            random_state=seed,
        )
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

    print(f"x_train shape: {x_train.shape}")
    print(f"x_test shape:  {x_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape:  {y_test.shape}\n")

    return x_train, x_test, y_train, y_test


def split_data_val(
    df: pd.DataFrame,
    LHD: bool = False,
    n_train: int = 600,
    n_val: int = 200,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test sets using either Latin
    Hypercube Design (LHD) or random split.

    Args:
        df: Input DataFrame where the last column is the output.
        LHD: If True, use Latin Hypercube Design for selecting training and
            validation samples; if False, use random split.
        n_train: Number of training samples to select.
        n_val: Number of validation samples to select.
        seed: Random seed for reproducibility.

    Returns:
        x_train: Training features array.
        x_val: Validation features array.
        x_test: Testing features array.
        y_train: Training labels array (column vector).
        y_val: Validation labels array (column vector).
        y_test: Testing labels array (column vector).

    Raises:
        ValueError: If n_train + n_val is greater than the total number of
            samples in df.
    """
    # Split the data into features (x) and labels (y)
    x = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()
    n_total, k = x.shape

    if n_train + n_val > n_total:
        raise ValueError(
            f"n_train + n_val cannot be greater than the total number of "
            f"samples ({n_total})."
        )

    def query_unique(tree_obj, small_data, k_neighbors=50):
        used_indices = set()
        unique_indices = []
        unique_distances = []

        for point in small_data:
            distances, indices = tree_obj.query(point, k=min(k_neighbors, tree_obj.n))
            distances = np.atleast_1d(distances)
            indices = np.atleast_1d(indices)

            for dist, idx in zip(distances, indices):
                if idx not in used_indices:
                    used_indices.add(idx)
                    unique_indices.append(idx)
                    unique_distances.append(dist)
                    break

        return np.array(unique_distances), np.array(unique_indices)

    if LHD:
        print(
            "Using LHD-based selection for training and validation points.\n"
        )

        # Step 1: Select training points from full dataset
        lhd_train_gen = qmc.LatinHypercube(d=k, seed=seed)  # type: ignore
        x_lhd_train = lhd_train_gen.random(n=n_train)

        for i in range(k):
            x_lhd_train[:, i] = (
                x_lhd_train[:, i] * (np.max(x[:, i]) - np.min(x[:, i])) + np.min(x[:, i])
            )

        tree_full = cKDTree(x)
        _, train_idx = query_unique(tree_full, x_lhd_train)

        x_train = x[train_idx, :]
        y_train = y[train_idx].reshape(-1, 1)

        # Remaining data after removing training points
        mask_after_train = np.ones(n_total, dtype=bool)
        mask_after_train[train_idx] = False
        x_remaining = x[mask_after_train, :]
        y_remaining = y[mask_after_train]

        # Step 2: Select validation points from remaining dataset
        lhd_val_gen = qmc.LatinHypercube(d=k, seed=seed + 1)  # type: ignore
        x_lhd_val = lhd_val_gen.random(n=n_val)

        for i in range(k):
            x_lhd_val[:, i] = (
                x_lhd_val[:, i]
                * (np.max(x_remaining[:, i]) - np.min(x_remaining[:, i]))
                + np.min(x_remaining[:, i])
            )

        tree_remaining = cKDTree(x_remaining)
        _, val_idx_local = query_unique(tree_remaining, x_lhd_val)

        x_val = x_remaining[val_idx_local, :]
        y_val = y_remaining[val_idx_local].reshape(-1, 1)

        # Test set is whatever remains after validation selection
        mask_after_val = np.ones(x_remaining.shape[0], dtype=bool)
        mask_after_val[val_idx_local] = False
        x_test = x_remaining[mask_after_val, :]
        y_test = y_remaining[mask_after_val].reshape(-1, 1)

    else:
        # First split off training set
        x_train, x_temp, y_train, y_temp = train_test_split(
            x,
            y,
            train_size=n_train,
            random_state=seed,
        )

        # Then split remaining into validation and test
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp,
            y_temp,
            train_size=n_val,
            random_state=seed,
        )

        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

    print(f"x_train shape: {x_train.shape}")
    print(f"x_val shape:   {x_val.shape}")
    print(f"x_test shape:  {x_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_val shape:   {y_val.shape}")
    print(f"y_test shape:  {y_test.shape}\n")

    return x_train, x_val, x_test, y_train, y_val, y_test


def prepare_train_test_arrays(
    traindf: pd.DataFrame,
    testdf: pd.DataFrame,
    LHD: bool = False,
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert already-split train and test DataFrames into feature and target arrays,
    with an option to choose subsets of both training and test data.

    Assumes the last column is the target/output column.

    Args:
        traindf: Training dataframe.
        testdf: Test dataframe.
        LHD: If True, use Latin Hypercube Design to select samples from each
            dataframe; if False, use random selection.
        n_train: Number of samples to select from traindf. If None, use all rows.
        n_test: Number of samples to select from testdf. If None, use all rows.
        seed: Random seed for reproducibility.

    Returns:
        x_train: Training features array.
        x_test: Testing features array.
        y_train: Training targets as a column vector.
        y_test: Testing targets as a column vector.

    Raises:
        ValueError: If n_train or n_test is greater than the available number of
            rows in the corresponding dataframe.
    """
    def query_unique(tree_obj, small_data):
        used_indices = set()
        unique_indices = []
        unique_distances = []

        for point in small_data:
            distances, indices = tree_obj.query(point, k=min(50, tree_obj.n))
            distances = np.atleast_1d(distances)
            indices = np.atleast_1d(indices)

            for dist, idx in zip(distances, indices):
                if idx not in used_indices:
                    used_indices.add(idx)
                    unique_indices.append(idx)
                    unique_distances.append(dist)
                    break

        return np.array(unique_distances), np.array(unique_indices)

    def select_subset(
        df: pd.DataFrame,
        n_select: Optional[int],
        LHD: bool,
        seed: int,
        label: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()
        n_total, k = x.shape

        if n_select is None:
            n_select = n_total

        if n_select > n_total:
            raise ValueError(
                f"{label}: requested {n_select} samples, but only {n_total} "
                f"are available."
            )

        if n_select == n_total:
            x_out = x
            y_out = y.reshape(-1, 1)
            return x_out, y_out

        if LHD:
            print(
                f"Using {n_select} closest points to Latin Hypercube Design "
                f"for {label} samples.\n"
            )

            LHD_gen = qmc.LatinHypercube(d=k, seed=seed)  # type: ignore
            x_lhd = LHD_gen.random(n=n_select)

            for i in range(k):
                x_lhd[:, i] = (
                    x_lhd[:, i] * (np.max(x[:, i]) - np.min(x[:, i])) + np.min(x[:, i])
                )

            tree = cKDTree(x)
            _, index = query_unique(tree, x_lhd)

            x_out = x[index, :]
            y_out = y[index].reshape(-1, 1)
        else:
            x_out, _, y_out, _ = train_test_split(
                x,
                y,
                train_size=n_select,
                test_size=None,
                random_state=seed,
            )
            y_out = y_out.reshape(-1, 1)

        return x_out, y_out

    x_train, y_train = select_subset(
        df=traindf,
        n_select=n_train,
        LHD=LHD,
        seed=seed,
        label="training",
    )

    x_test, y_test = select_subset(
        df=testdf,
        n_select=n_test,
        LHD=LHD,
        seed=seed + 1,
        label="test",
    )

    print(f"x_train shape: {x_train.shape}")
    print(f"x_test shape:  {x_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape:  {y_test.shape}\n")

    return x_train, x_test, y_train, y_test


# Convenience wrapper

def load_and_split(
    dataset: str = "JAG",
    path_to_csv: Optional[str] = None,
    n_samples: int = 10000,
    random_rows: bool = True,
    seed: int = 42,
    LHD: bool = False,
    n_train: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function: load dataset, then split into train and test.

    Args:
        dataset: "JAG" or "borehole".
        path_to_csv: Optional explicit path, overrides default.
        n_samples: Number of samples to load from CSV.
        random_rows: Randomly choose rows or take first n_samples.
        seed: Random seed used for row sampling and splitting.
        LHD: Use LHD based train selection if True.
        n_train: Number of training samples.

    Returns:
        x_train, x_test, y_train, y_test
    """
    df = load_data(
        dataset=dataset,
        path_to_csv=path_to_csv,
        n_samples=n_samples,
        random=random_rows,
        seed=seed,
    )

    return split_data(df, LHD=LHD, n_train=n_train, seed=seed)