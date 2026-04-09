#!/usr/bin/env python3

"""
Bayesian optimization for HST drag neural-network hyperparameters.

This script treats neural-network hyperparameter tuning as a discrete Bayesian
optimization problem over a finite candidate set. It:

1. Loads the HST drag train/validation/test splits.
2. Standardizes inputs using training statistics only.
3. Evaluates an initial set of hyperparameter candidates.
4. Fits a GP surrogate on hyperparameter -> validation-score pairs.
5. Acquires new candidates with a chosen acquisition function.
6. Retrains the best configuration on train+validation and evaluates on test.

Usage examples:

poetry run python finalproject/bo_hst_nn.py
poetry run python finalproject/bo_hst_nn.py --num_epochs 200 --num_init 6 --num_iter 12
poetry run python finalproject/bo_hst_nn.py --acquisition UCB --kappa 2.5
poetry run python finalproject/bo_hst_nn.py --hidden_configs 64,64 64,64,64 128,128,64
"""

from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from surmod import bayesian_optimization as bo
from surmod import data_processing
from surmod import gaussian_process_regression as gp
from surmod import neural_network as nn


DEFAULT_HIDDEN_CONFIGS = [
    "64,64",
    "128,128",
    "64,64,64",
    "128,128,64",
    "128,64,32",
]


@dataclass(frozen=True)
class HyperparameterConfig:
    """Container for one NN hyperparameter setting."""

    hidden_sizes: tuple[int, ...]
    batch_size: int
    learning_rate: float


def parse_hidden_config(text: str) -> tuple[int, ...]:
    """Parse a comma-separated hidden-layer specification."""
    try:
        hidden_sizes = tuple(int(value) for value in text.split(",") if value.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid hidden-layer configuration: {text!r}"
        ) from exc

    if not hidden_sizes:
        raise argparse.ArgumentTypeError("Hidden-layer configuration cannot be empty.")
    if any(width <= 0 for width in hidden_sizes):
        raise argparse.ArgumentTypeError("Hidden-layer widths must be positive.")

    return hidden_sizes


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Bayesian optimization of HST drag NN hyperparameters.",
    )

    parser.add_argument(
        "-n",
        "--num_epochs",
        type=int,
        default=200,
        help="Number of training epochs per hyperparameter evaluation.",
    )
    parser.add_argument(
        "-in",
        "--num_init",
        type=int,
        default=5,
        help="Number of initial hyperparameter configurations to evaluate.",
    )
    parser.add_argument(
        "-it",
        "--num_iter",
        type=int,
        default=10,
        help="Number of BO acquisition iterations after initialization.",
    )
    parser.add_argument(
        "-acq",
        "--acquisition",
        type=str,
        choices=["EI", "PI", "UCB", "random"],
        default="EI",
        help="Acquisition function used to choose the next configuration.",
    )
    parser.add_argument(
        "-k",
        "--kernel",
        type=str,
        choices=["matern", "rbf", "matern_dot"],
        default="matern",
        help="Kernel used by the GP surrogate over hyperparameters.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed for candidate initialization and model training.",
    )
    parser.add_argument(
        "-xi",
        "--xi",
        type=float,
        default=0.01,
        help="Exploration parameter for EI/PI.",
    )
    parser.add_argument(
        "-kappa",
        "--kappa",
        type=float,
        default=2.0,
        help="Exploration parameter for UCB.",
    )
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=[32, 64, 128],
        help="Batch sizes included in the discrete search space.",
    )
    parser.add_argument(
        "--learning_rates",
        type=float,
        nargs="+",
        default=[3e-4, 1e-3, 3e-3],
        help="Learning rates included in the discrete search space.",
    )
    parser.add_argument(
        "--hidden_configs",
        type=parse_hidden_config,
        nargs="+",
        default=[parse_hidden_config(text) for text in DEFAULT_HIDDEN_CONFIGS],
        help="Hidden-layer configurations as comma-separated widths.",
    )

    return parser.parse_args()


def get_data_paths() -> tuple[Path, Path, Path]:
    """Return absolute paths to the HST drag split files."""
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data" / "HST-drag"
    return data_dir / "train.csv", data_dir / "validation.csv", data_dir / "test.csv"


def load_hst_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/validation/test HST drag dataframes."""
    train_path, validation_path, test_path = get_data_paths()
    traindf, testdf, validationdf = data_processing.load_data_from_file(
        dataset="HST",
        train_path=str(train_path),
        test_path=str(test_path),
        validation_path=str(validation_path),
    )
    return traindf, validationdf, testdf


def dataframe_to_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Convert a dataframe to feature and target arrays."""
    x = df.iloc[:, :-1].to_numpy(dtype=float)
    y = df.iloc[:, -1].to_numpy(dtype=float).reshape(-1, 1)
    return x, y


def to_float_tensors(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert arrays to float32 torch tensors."""
    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )


def build_candidates(args: argparse.Namespace) -> list[HyperparameterConfig]:
    """Build the discrete hyperparameter candidate set."""
    candidates = [
        HyperparameterConfig(
            hidden_sizes=hidden_sizes,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        for hidden_sizes, batch_size, learning_rate in itertools.product(
            args.hidden_configs, args.batch_sizes, args.learning_rates
        )
    ]
    return candidates


def encode_config(
    config: HyperparameterConfig,
    max_hidden_layers: int,
) -> np.ndarray:
    """Encode one hyperparameter configuration for the GP surrogate."""
    padded_hidden_sizes = list(config.hidden_sizes) + [0] * (
        max_hidden_layers - len(config.hidden_sizes)
    )
    return np.array(
        [
            len(config.hidden_sizes),
            *padded_hidden_sizes,
            np.log10(config.learning_rate),
            np.log2(config.batch_size),
        ],
        dtype=float,
    )


def evaluate_config(
    config: HyperparameterConfig,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    seed: int,
    num_epochs: int,
) -> dict[str, float]:
    """Train the NN for one configuration and return validation metrics."""
    start_time = time.perf_counter()
    _, train_losses, val_losses = nn.train_neural_net(
        x_train,
        y_train,
        x_val,
        y_val,
        list(config.hidden_sizes),
        num_epochs,
        config.learning_rate,
        config.batch_size,
        seed,
        initialize_weights_normal=True,
    )
    elapsed = time.perf_counter() - start_time

    best_val_mse = float(min(val_losses))
    final_val_mse = float(val_losses[-1])

    return {
        "best_val_mse": best_val_mse,
        "final_val_mse": final_val_mse,
        "best_val_objective": -best_val_mse,
        "final_val_objective": -final_val_mse,
        "final_train_mse": float(train_losses[-1]),
        "fit_time_s": elapsed,
    }


def fit_surrogate(
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    kernel: str,
    seed: int,
) -> tuple[GaussianProcessRegressor, StandardScaler]:
    """Fit the GP surrogate on observed hyperparameter evaluations."""
    hp_scaler = StandardScaler()
    x_obs_scaled = hp_scaler.fit_transform(x_obs)

    gp_model = GaussianProcessRegressor(
        kernel=gp.get_kernel(kernel, x_obs_scaled.shape[1], isotropic=False),
        n_restarts_optimizer=5,
        random_state=seed,
        normalize_y=True,
    )
    gp_model.fit(x_obs_scaled, y_obs)
    return gp_model, hp_scaler


def compute_acquisition_values(
    acquisition: str,
    x_remaining: np.ndarray,
    gp_model: GaussianProcessRegressor,
    y_best: float,
    xi: float,
    kappa: float,
) -> np.ndarray:
    """Compute acquisition values on the remaining candidate set."""
    if acquisition == "EI":
        return bo.expected_improvement(x_remaining, gp_model, y_best, xi=xi)
    if acquisition == "PI":
        return bo.probability_of_improvement(x_remaining, gp_model, y_best, xi=xi)
    if acquisition == "UCB":
        return bo.upper_confidence_bound(x_remaining, gp_model, kappa=kappa)
    raise ValueError(f"Unsupported acquisition: {acquisition}")


def format_config(config: HyperparameterConfig) -> str:
    """Render a hyperparameter configuration for logging."""
    hidden_sizes = ",".join(str(value) for value in config.hidden_sizes)
    return (
        f"hidden_sizes=[{hidden_sizes}], "
        f"batch_size={config.batch_size}, "
        f"learning_rate={config.learning_rate:.2e}"
    )


def run_final_training(
    best_config: HyperparameterConfig,
    traindf: pd.DataFrame,
    validationdf: pd.DataFrame,
    testdf: pd.DataFrame,
    num_epochs: int,
    seed: int,
) -> dict[str, float]:
    """Retrain the best configuration on train+validation and evaluate on test."""
    train_val_df = pd.concat([traindf, validationdf], ignore_index=True)

    x_train_val, y_train_val = dataframe_to_arrays(train_val_df)
    x_test, y_test = dataframe_to_arrays(testdf)

    x_scaler = StandardScaler()
    x_train_val = x_scaler.fit_transform(x_train_val)
    x_test = x_scaler.transform(x_test)

    x_train_val_t, y_train_val_t = to_float_tensors(x_train_val, y_train_val)
    x_test_t, y_test_t = to_float_tensors(x_test, y_test)

    model, train_losses, test_losses = nn.train_neural_net(
        x_train_val_t,
        y_train_val_t,
        x_test_t,
        y_test_t,
        list(best_config.hidden_sizes),
        num_epochs,
        best_config.learning_rate,
        best_config.batch_size,
        seed,
        initialize_weights_normal=True,
    )

    model.eval()
    with torch.no_grad():
        predictions = model(x_test_t).cpu().numpy().reshape(-1)

    y_test_np = y_test.reshape(-1)
    test_mse = mean_squared_error(y_test_np, predictions)
    test_rmse = float(np.sqrt(test_mse))
    test_mae = mean_absolute_error(y_test_np, predictions)

    return {
        "final_train_mse": float(train_losses[-1]),
        "final_test_mse_curve": float(test_losses[-1]),
        "test_mse": float(test_mse),
        "test_rmse": test_rmse,
        "test_mae": float(test_mae),
    }


def save_history(history: list[dict[str, object]]) -> Path:
    """Save BO evaluation history to CSV."""
    output_dir = Path(__file__).resolve().parent / "output_log"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    history_path = output_dir / f"bo_hst_nn_history_{timestamp}.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    return history_path


def main() -> None:
    """Run Bayesian optimization for HST drag NN hyperparameters."""
    args = parse_arguments()
    rng = np.random.RandomState(args.seed)

    candidates = build_candidates(args)
    if not candidates:
        raise ValueError("The hyperparameter search space is empty.")

    total_evaluations = args.num_init + args.num_iter
    if args.num_init < 2:
        raise ValueError("--num_init must be at least 2 for GP-based BO.")
    if total_evaluations > len(candidates):
        raise ValueError(
            f"Requested {total_evaluations} evaluations, but only "
            f"{len(candidates)} unique candidates are available."
        )

    traindf, validationdf, testdf = load_hst_splits()
    x_train, y_train = dataframe_to_arrays(traindf)
    x_val, y_val = dataframe_to_arrays(validationdf)

    x_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_train)
    x_val = x_scaler.transform(x_val)

    x_train_t, y_train_t = to_float_tensors(x_train, y_train)
    x_val_t, y_val_t = to_float_tensors(x_val, y_val)

    max_hidden_layers = max(len(config.hidden_sizes) for config in candidates)
    encoded_candidates = np.vstack(
        [encode_config(config, max_hidden_layers) for config in candidates]
    )

    print(f"Candidate configurations: {len(candidates)}")
    print(
        f"Evaluation budget: {args.num_init} initial + {args.num_iter} BO = "
        f"{total_evaluations}"
    )

    observed_indices: list[int] = []
    remaining_indices = list(range(len(candidates)))
    history: list[dict[str, object]] = []
    scores: dict[int, float] = {}
    metrics_by_index: dict[int, dict[str, float]] = {}

    initial_indices = rng.choice(len(candidates), size=args.num_init, replace=False)
    for step_number, candidate_index in enumerate(initial_indices, start=1):
        observed_indices.append(int(candidate_index))
        remaining_indices.remove(int(candidate_index))
        config = candidates[int(candidate_index)]

        print(f"\nInitial evaluation {step_number}/{args.num_init}")
        print(f"  {format_config(config)}")
        metrics = evaluate_config(
            config,
            x_train_t,
            y_train_t,
            x_val_t,
            y_val_t,
            args.seed,
            args.num_epochs,
        )
        scores[int(candidate_index)] = metrics["best_val_objective"]
        metrics_by_index[int(candidate_index)] = metrics
        history.append(
            {
                "phase": "init",
                "step": step_number,
                "candidate_index": int(candidate_index),
                "hidden_sizes": list(config.hidden_sizes),
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                **metrics,
            }
        )
        print(
            f"  best validation MSE={metrics['best_val_mse']:.6e}, "
            f"fit time={metrics['fit_time_s']:.2f}s"
        )

    for iteration in range(1, args.num_iter + 1):
        y_obs = np.array([scores[idx] for idx in observed_indices], dtype=float)

        if args.acquisition == "random":
            next_choice_position = int(rng.randint(len(remaining_indices)))
        else:
            x_obs = encoded_candidates[observed_indices]
            x_remaining = encoded_candidates[remaining_indices]
            gp_model, hp_scaler = fit_surrogate(x_obs, y_obs, args.kernel, args.seed)
            x_remaining_scaled = hp_scaler.transform(x_remaining)
            acquisition_values = compute_acquisition_values(
                args.acquisition,
                x_remaining_scaled,
                gp_model,
                y_best=float(np.max(y_obs)),
                xi=args.xi,
                kappa=args.kappa,
            )
            next_choice_position = int(np.argmax(acquisition_values))

        next_index = remaining_indices.pop(next_choice_position)
        observed_indices.append(next_index)
        config = candidates[next_index]

        print(f"\nBO iteration {iteration}/{args.num_iter}")
        print(f"  {format_config(config)}")
        metrics = evaluate_config(
            config,
            x_train_t,
            y_train_t,
            x_val_t,
            y_val_t,
            args.seed,
            args.num_epochs,
        )
        scores[next_index] = metrics["best_val_objective"]
        metrics_by_index[next_index] = metrics
        current_best = max(scores.values())
        history.append(
            {
                "phase": "bo",
                "step": args.num_init + iteration,
                "candidate_index": next_index,
                "hidden_sizes": list(config.hidden_sizes),
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "best_objective_so_far": current_best,
                **metrics,
            }
        )
        print(
            f"  best validation MSE={metrics['best_val_mse']:.6e}, "
            f"best seen MSE={-current_best:.6e}"
        )

    best_index = max(scores, key=scores.get)
    best_config = candidates[best_index]
    best_metrics = metrics_by_index[best_index]

    print("\nBest validation configuration:")
    print(f"  {format_config(best_config)}")
    print(f"  validation MSE={best_metrics['best_val_mse']:.6e}")

    final_metrics = run_final_training(
        best_config=best_config,
        traindf=traindf,
        validationdf=validationdf,
        testdf=testdf,
        num_epochs=args.num_epochs,
        seed=args.seed,
    )

    history_path = save_history(history)

    print("\nFinal train+validation -> test evaluation:")
    print(f"  test MSE={final_metrics['test_mse']:.6e}")
    print(f"  test RMSE={final_metrics['test_rmse']:.6e}")
    print(f"  test MAE={final_metrics['test_mae']:.6e}")
    print(f"  history saved to {history_path}")


if __name__ == "__main__":
    main()
