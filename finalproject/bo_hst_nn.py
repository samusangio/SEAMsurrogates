#!/usr/bin/env python3

"""
Bayesian optimization for HST drag neural-network hyperparameters.

This script treats neural-network hyperparameter tuning as a discrete Bayesian
optimization problem over a finite candidate set. It:

1. Loads the HST drag train/validation/test splits.
2. Standardizes inputs using training statistics only.
3. Evaluates an initial set of hyperparameter candidates.
4. Fits a GP surrogate on hyperparameter -> test-score pairs.
5. Acquires new candidates with a chosen acquisition function.
6. Retrains the best configuration on train+test and evaluates on validation.

Usage examples:

poetry run python finalproject/bo_hst_nn.py
poetry run python finalproject/bo_hst_nn.py --num_epochs 200 --num_init 6 --num_iter 12
poetry run python finalproject/bo_hst_nn.py --acquisition UCB --kappa 2.5
poetry run python finalproject/bo_hst_nn.py --layer_widths 32 64 128 --max_layers 3
poetry run python finalproject/bo_hst_nn.py --dataset_variant full
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

DEFAULT_LAYER_WIDTHS = [8 * n for n in range(1, 10)]
DEFAULT_MAX_LAYERS = 3
DATASET_VARIANT_DIRS = {
    "reduced": "HST-drag",
    "full": "HST-drag-full",
}


@dataclass(frozen=True)
class HyperparameterConfig:
    """Container for one NN hyperparameter setting."""

    hidden_sizes: tuple[int, ...]
    batch_size: int
    learning_rate: float


def generate_hidden_configs(
    layer_widths: list[int],
    max_layers: int,
) -> list[tuple[int, ...]]:
    """Generate all hidden layer configurations from layer widths.

    Creates architectures with 1 to max_layers layers, where each layer
    uses the same width (homogeneous architectures).

    Args:
        layer_widths: List of possible layer widths.
        max_layers: Maximum number of hidden layers.

    Returns:
        List of hidden layer configurations as tuples.
    """
    configs = []
    for width in layer_widths:
        for num_layers in range(1, max_layers + 1):
            configs.append(tuple([width] * num_layers))
    return configs


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
        default=10000,
        help="Number of training epochs per hyperparameter evaluation.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early-stopping patience for BO evaluation loss on the test split. Set to 0 to disable.",
    )
    parser.add_argument(
        "-in",
        "--num_init",
        type=int,
        default=3,
        help="Number of initial hyperparameter configurations to evaluate.",
    )
    parser.add_argument(
        "-it",
        "--num_iter",
        type=int,
        default=12,
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
        default=[16, 32, 64, 128, 256],
        help="Batch sizes included in the discrete search space.",
    )
    parser.add_argument(
        "--learning_rates",
        type=float,
        nargs="+",
        default=[1e-3, 5e-3, 1e-2, 5e-2],
        help="Learning rates included in the discrete search space.",
    )
    parser.add_argument(
        "--layer_widths",
        type=int,
        nargs="+",
        default=DEFAULT_LAYER_WIDTHS,
        help="Layer widths to include in the search space.",
    )
    parser.add_argument(
        "--max_layers",
        type=int,
        default=DEFAULT_MAX_LAYERS,
        help="Maximum number of hidden layers.",
    )
    parser.add_argument(
        "--dataset_variant",
        type=str,
        choices=sorted(DATASET_VARIANT_DIRS),
        default="reduced",
        help="Which prepared HST dataset split directory to load from data/.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help="Optional explicit directory containing train.csv, test.csv, and validation.csv.",
    )

    return parser.parse_args()


def get_data_paths(
    dataset_variant: str,
    data_dir: Path | None,
) -> tuple[Path, Path, Path]:
    """Return absolute paths to the HST drag split files."""
    repo_root = Path(__file__).resolve().parent.parent
    resolved_data_dir = (
        data_dir
        if data_dir is not None
        else repo_root / "data" / DATASET_VARIANT_DIRS[dataset_variant]
    )
    if not resolved_data_dir.is_absolute():
        resolved_data_dir = repo_root / resolved_data_dir
    return (
        resolved_data_dir / "train.csv",
        resolved_data_dir / "validation.csv",
        resolved_data_dir / "test.csv",
    )


def load_hst_splits(
    dataset_variant: str,
    data_dir: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/validation/test HST drag dataframes."""
    train_path, validation_path, test_path = get_data_paths(
        dataset_variant=dataset_variant,
        data_dir=data_dir,
    )
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
    hidden_configs = generate_hidden_configs(args.layer_widths, args.max_layers)
    candidates = [
        HyperparameterConfig(
            hidden_sizes=hidden_sizes,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        for hidden_sizes, batch_size, learning_rate in itertools.product(
            hidden_configs, args.batch_sizes, args.learning_rates
        )
    ]
    return candidates


def encode_config(
    config: HyperparameterConfig,
    max_hidden_layers: int,
) -> np.ndarray:
    """Encode one hyperparameter configuration for the GP surrogate."""
    log2_hidden_sizes = [float(np.log2(width)) for width in config.hidden_sizes]
    width_deltas = [
        current - previous
        for previous, current in zip(log2_hidden_sizes[:-1], log2_hidden_sizes[1:])
    ]
    padded_width_deltas = width_deltas + [0.0] * (
        max_hidden_layers - len(config.hidden_sizes)
    )

    return np.array(
        [
            len(config.hidden_sizes),
            log2_hidden_sizes[0],
            *padded_width_deltas,
            np.log10(config.learning_rate),
            np.log2(config.batch_size),
        ],
        dtype=float,
    )


def evaluate_config(
    config: HyperparameterConfig,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    seed: int,
    num_epochs: int,
    patience: int | None,
) -> dict[str, float]:
    """Train the NN for one configuration and return BO evaluation metrics."""
    start_time = time.perf_counter()
    _, train_losses, eval_losses = nn.train_neural_net(
        x_train,
        y_train,
        x_eval,
        y_eval,
        list(config.hidden_sizes),
        num_epochs,
        config.learning_rate,
        config.batch_size,
        seed,
        initialize_weights_normal=True,
        patience=patience,
    )
    elapsed = time.perf_counter() - start_time

    best_eval_mse = float(min(eval_losses))
    final_eval_mse = float(eval_losses[-1])

    return {
        "best_eval_mse": best_eval_mse,
        "final_eval_mse": final_eval_mse,
        "best_eval_objective": -best_eval_mse,
        "final_eval_objective": -final_eval_mse,
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


def get_encoded_feature_names(max_hidden_layers: int) -> list[str]:
    """Return the encoded BO feature names used by the GP surrogate."""
    return [
        "num_layers",
        "log2_hidden_size_1",
        *[
            f"delta_log2_hidden_size_{index}"
            for index in range(2, max_hidden_layers + 1)
        ],
        "log10_learning_rate",
        "log2_batch_size",
    ]


def extract_length_scales(gp_model: GaussianProcessRegressor) -> np.ndarray:
    """Extract GP kernel length scales across the supported kernel structures."""
    kernels_to_check = [
        gp_model.kernel_,
        getattr(gp_model.kernel_, "k1", None),
        getattr(getattr(gp_model.kernel_, "k1", None), "k2", None),
        getattr(getattr(getattr(gp_model.kernel_, "k1", None), "k2", None), "k1", None),
    ]

    for kernel in kernels_to_check:
        if kernel is not None and hasattr(kernel, "length_scale"):
            return np.atleast_1d(np.asarray(kernel.length_scale, dtype=float))

    raise ValueError(
        f"Could not extract length scales from fitted kernel: {gp_model.kernel_}"
    )


def print_length_scale_summary(
    gp_model: GaussianProcessRegressor,
    feature_names: Sequence[str],
) -> None:
    """Print the fitted GP length scales for each encoded BO feature."""
    length_scales = extract_length_scales(gp_model)

    print("\nFinal GP length scales:")
    if length_scales.size == 1:
        print(
            f"  shared across all BO features: {float(length_scales[0]):.6g}"
        )
        return

    if length_scales.size != len(feature_names):
        raise ValueError(
            "Length-scale dimension mismatch: "
            f"{length_scales.size} scales for {len(feature_names)} features."
        )

    for feature_name, length_scale in zip(feature_names, length_scales):
        print(f"  {feature_name}: {float(length_scale):.6g}")


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
    patience=None,
) -> dict[str, float]:
    """Retrain the best configuration on train+test and evaluate on validation."""
    train_test_df = pd.concat([traindf, testdf], ignore_index=True)

    x_train_final, y_train_final = dataframe_to_arrays(train_test_df)
    x_validation, y_validation = dataframe_to_arrays(validationdf)

    x_scaler = StandardScaler()
    x_train_final = x_scaler.fit_transform(x_train_final)
    x_validation = x_scaler.transform(x_validation)

    x_train_final_t, y_train_final_t = to_float_tensors(x_train_final, y_train_final)
    x_validation_t, y_validation_t = to_float_tensors(x_validation, y_validation)

    model, train_losses, validation_losses = nn.train_neural_net(
        x_train_final_t,
        y_train_final_t,
        x_validation_t,
        y_validation_t,
        list(best_config.hidden_sizes),
        num_epochs,
        best_config.learning_rate,
        best_config.batch_size,
        seed,
        initialize_weights_normal=True,
        patience=patience,
    )

    model.eval()
    with torch.no_grad():
        predictions = model(x_validation_t).cpu().numpy().reshape(-1)

    y_validation_np = y_validation.reshape(-1)
    validation_mse = mean_squared_error(y_validation_np, predictions)
    validation_rmse = float(np.sqrt(validation_mse))
    validation_mae = mean_absolute_error(y_validation_np, predictions)

    return {
        "final_train_mse": float(train_losses[-1]),
        "final_validation_mse_curve": float(validation_losses[-1]),
        "validation_mse": float(validation_mse),
        "validation_rmse": validation_rmse,
        "validation_mae": float(validation_mae),
    }


def find_fastest_config(
    candidates: list[HyperparameterConfig],
    batch_sizes: list[int],
    learning_rates: list[float],
) -> int:
    """Find the index of the fastest-to-compute configuration."""
    target_batch = max(batch_sizes)
    target_lr = max(learning_rates)

    # Find configs with 1 layer, largest batch, largest lr
    matching = [
        (i, config)
        for i, config in enumerate(candidates)
        if len(config.hidden_sizes) == 1
        and config.batch_size == target_batch
        and config.learning_rate == target_lr
    ]

    if not matching:
        raise ValueError(
            f"No configuration found with 1 layer, batch_size={target_batch}, "
            f"learning_rate={target_lr}"
        )

    # Among matching configs, pick the one with smallest layer width
    return min(matching, key=lambda x: x[1].hidden_sizes[0])[0]


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
    patience = None if args.patience <= 0 else args.patience

    candidates = build_candidates(args)
    if not candidates:
        raise ValueError("The hyperparameter search space is empty.")

    total_evaluations = args.num_init + args.num_iter
    # if args.num_init < 2:
    #     raise ValueError("--num_init must be at least 2 for GP-based BO.")
    if total_evaluations > len(candidates):
        raise ValueError(
            f"Requested {total_evaluations} evaluations, but only "
            f"{len(candidates)} unique candidates are available."
        )

    train_path, validation_path, test_path = get_data_paths(
        dataset_variant=args.dataset_variant,
        data_dir=args.data_dir,
    )
    print(f"Using HST split files from: {train_path.parent}")

    traindf, validationdf, testdf = load_hst_splits(
        dataset_variant=args.dataset_variant,
        data_dir=args.data_dir,
    )
    x_train, y_train = dataframe_to_arrays(traindf)
    x_eval, y_eval = dataframe_to_arrays(testdf)

    x_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_train)
    x_eval = x_scaler.transform(x_eval)

    x_train_t, y_train_t = to_float_tensors(x_train, y_train)
    x_eval_t, y_eval_t = to_float_tensors(x_eval, y_eval)

    max_hidden_layers = max(len(config.hidden_sizes) for config in candidates)
    encoded_candidates = np.vstack(
        [encode_config(config, max_hidden_layers) for config in candidates]
    )

    # Select first initialization: fastest configuration
    first_init_index = find_fastest_config(
        candidates, args.batch_sizes, args.learning_rates
    )

    print(f"Candidate configurations: {len(candidates)}")
    print(
        f"Evaluation budget: {args.num_init} initial + {args.num_iter} BO = "
        f"{total_evaluations}"
    )
    print(f"First init (fastest): {format_config(candidates[first_init_index])}")

    observed_indices: list[int] = []
    remaining_indices = list(range(len(candidates)))
    history: list[dict[str, object]] = []
    scores: dict[int, float] = {}
    metrics_by_index: dict[int, dict[str, float]] = {}

    initial_indices = [first_init_index]

    # Select remaining initializations randomly, excluding the first
    if args.num_init > 1:
        remaining_candidate_indices = [
            i for i in range(len(candidates)) if i != first_init_index
        ]
        additional_indices = rng.choice(
            remaining_candidate_indices,
            size=args.num_init - 1,
            replace=False,
        )
        initial_indices.extend(additional_indices)

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
            x_eval_t,
            y_eval_t,
            args.seed,
            args.num_epochs,
            patience,
        )
        scores[int(candidate_index)] = metrics["best_eval_objective"]
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
            f"  best test MSE={metrics['best_eval_mse']:.6e}, "
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
            x_eval_t,
            y_eval_t,
            args.seed,
            args.num_epochs,
            patience,
        )
        scores[next_index] = metrics["best_eval_objective"]
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
            f"  best test MSE={metrics['best_eval_mse']:.6e}, "
            f"best seen MSE={-current_best:.6e}"
        )

    best_index = max(scores, key=scores.get)
    best_config = candidates[best_index]
    best_metrics = metrics_by_index[best_index]
    x_obs = encoded_candidates[observed_indices]
    y_obs = np.array([scores[idx] for idx in observed_indices], dtype=float)
    final_gp_model, _ = fit_surrogate(x_obs, y_obs, args.kernel, args.seed)
    feature_names = get_encoded_feature_names(max_hidden_layers)

    print("\nBest BO configuration:")
    print(f"  {format_config(best_config)}")
    print(f"  test MSE={best_metrics['best_eval_mse']:.6e}")
    print_length_scale_summary(final_gp_model, feature_names)

    final_metrics = run_final_training(
        best_config=best_config,
        traindf=traindf,
        validationdf=validationdf,
        testdf=testdf,
        num_epochs=args.num_epochs,
        seed=args.seed,
        patience=args.patience,
    )

    history_path = save_history(history)

    print("\nFinal train+test -> validation evaluation:")
    print(f"  validation MSE={final_metrics['validation_mse']:.6e}")
    print(f"  validation RMSE={final_metrics['validation_rmse']:.6e}")
    print(f"  validation MAE={final_metrics['validation_mae']:.6e}")
    print(f"  history saved to {history_path}")


if __name__ == "__main__":
    main()
