"""Select polynomial Ridge models, assess a holdout, and predict both test sets.

Run: python ml_assignment_solution.py --data-dir path/to/BT2024103
The report source is generate_report.py. See README.md for the full workflow.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.linalg import svd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits


ROLL = "BT2024103"
PROBLEMS = {"var1": (6, 10), "var2": (3, 20)}
ALPHAS = np.logspace(-4, 4, 50)
SPLIT_SEED = 42
CV_SEED = 43
N_FOLDS = 5
SCRIPT_DIR = Path(__file__).resolve().parent


def load_csv(path, columns):
    """Check the personalized file format before converting it to arrays."""
    frame = pd.read_csv(path)
    if list(frame.columns) != columns:
        raise ValueError(f"{path.name}: expected columns {columns}, got {list(frame.columns)}")
    if len(frame) != 1000:
        raise ValueError(f"{path.name}: expected 1,000 rows, got {len(frame)}")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name}: missing or non-finite values")
    return values


def make_model(degree, alpha):
    # Ridge fits the intercept separately, so the polynomial expansion omits 1.
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha, fit_intercept=True, solver="svd")),
    ])


def ridge_path_predictions(X_fit, y_fit, X_valid, degree, alphas):
    """Evaluate all alphas after fitting preprocessing on this fold's rows.

    With centered Z = U diag(s) V.T, Ridge coefficients for each alpha are
    V diag(s / (s**2 + alpha)) U.T y. Reusing one SVD evaluates the same
    models as separate Ridge fits without repeating the decomposition.
    Return one prediction column per alpha for each row in fit and valid.
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    scaler = StandardScaler()
    Z_fit = scaler.fit_transform(poly.fit_transform(X_fit))
    Z_valid = scaler.transform(poly.transform(X_valid))
    offset = Z_fit.mean(axis=0)
    target_mean = y_fit.mean()
    Z_fit -= offset
    Z_valid -= offset
    U, singular_values, Vt = svd(Z_fit, full_matrices=False, check_finite=True)
    # Match sklearn's SVD solver threshold for zero singular values.
    keep = singular_values > 1e-15
    U, singular_values, Vt = U[:, keep], singular_values[keep], Vt[keep]
    projected_y = U.T @ (y_fit - target_mean)
    weights = projected_y[:, None] * singular_values[:, None] / (
        singular_values[:, None] ** 2 + alphas[None, :]
    )
    fit_predictions = U @ (singular_values[:, None] * weights) + target_mean
    valid_predictions = (Z_valid @ Vt.T) @ weights + target_mean
    if not np.isfinite(fit_predictions).all() or not np.isfinite(valid_predictions).all():
        raise ValueError(f"Non-finite predictions at degree {degree}")
    return fit_predictions, valid_predictions, Z_fit.shape[1]


def metrics_by_alpha(y, predictions):
    errors = y[:, None] - predictions
    squared_errors = np.sum(errors ** 2, axis=0)
    total_variation = np.sum((y - y.mean()) ** 2)
    if total_variation <= 0:
        raise ValueError("R2 requires a target with nonzero variation in every fold")
    return squared_errors / len(y), 1 - squared_errors / total_variation


def select_model(X, y, max_degree):
    """Joint degree/alpha search using the same five development folds."""
    folds = list(KFold(N_FOLDS, shuffle=True, random_state=CV_SEED).split(X))
    rows = []
    for degree in range(1, max_degree + 1):
        fit_mse, fit_r2, cv_mse, cv_r2 = [], [], [], []
        for fit_index, valid_index in folds:
            pred_fit, pred_valid, n_features = ridge_path_predictions(
                X[fit_index], y[fit_index], X[valid_index], degree, ALPHAS
            )
            a, b = metrics_by_alpha(y[fit_index], pred_fit)
            c, d = metrics_by_alpha(y[valid_index], pred_valid)
            fit_mse.append(a)
            fit_r2.append(b)
            cv_mse.append(c)
            cv_r2.append(d)
        for index, alpha in enumerate(ALPHAS):
            rows.append({
                "degree": degree,
                "alpha": float(alpha),
                "n_polynomial_features": n_features,
                "mean_train_mse": float(np.mean(fit_mse, axis=0)[index]),
                "mean_train_r2": float(np.mean(fit_r2, axis=0)[index]),
                "mean_cv_mse": float(np.mean(cv_mse, axis=0)[index]),
                "std_cv_mse": float(np.std(cv_mse, axis=0, ddof=0)[index]),
                "mean_cv_r2": float(np.mean(cv_r2, axis=0)[index]),
                "std_cv_r2": float(np.std(cv_r2, axis=0, ddof=0)[index]),
            })
    results = pd.DataFrame(rows)
    if not np.isfinite(results.to_numpy()).all():
        raise ValueError("Model selection produced a non-finite score")
    # Stable ordering resolves any exact ties by lower degree, then lower alpha.
    ranked = results.sort_values(["mean_cv_mse", "degree", "alpha"], kind="stable")
    selected = ranked.iloc[0]
    per_degree = ranked.groupby("degree", sort=True).first().reset_index()
    return results, per_degree, selected


def regression_metrics(y, prediction):
    return {"mse": float(mean_squared_error(y, prediction)),
            "r2": float(r2_score(y, prediction))}


def save_predictions(prediction, output_path):
    if prediction.shape != (1000,) or not np.isfinite(prediction).all():
        raise ValueError("Submission predictions must contain 1,000 finite values")
    pd.DataFrame({"y": prediction}).to_csv(output_path, index=False)


def save_figures(label, per_degree, selected, y, prediction, figures_dir):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 120})
    degrees = per_degree["degree"].to_numpy()
    means = per_degree["mean_cv_mse"].to_numpy()
    deviations = per_degree["std_cv_mse"].to_numpy()
    fig, ax = plt.subplots(figsize=(9.5, 4.2), layout="constrained")
    ax.plot(degrees, means, "o-", color="#215b87", markersize=4,
            label="Mean validation MSE at the best alpha for each degree")
    ax.fill_between(degrees, np.maximum(means - deviations, 1e-12),
                    means + deviations, color="#215b87", alpha=0.16,
                    label="Plus/minus one fold standard deviation")
    ax.scatter([selected["degree"]], [selected["mean_cv_mse"]], s=95,
               facecolors="white", edgecolors="#b74032", linewidths=2, zorder=4,
               label=f"Selected degree {int(selected['degree'])}")
    ax.set(xlabel="Total polynomial degree", ylabel="Validation MSE, logarithmic scale",
           title=f"{label.capitalize()}: five-fold selection on 800 development rows")
    ax.set_yscale("log")
    ax.set_xticks(degrees)
    ax.grid(axis="y", which="both", alpha=0.2)
    ax.legend(fontsize=8.5, loc="best")
    fig.savefig(figures_dir / f"{label}_degree_vs_error.png", dpi=180)
    plt.close(fig)

    residuals = y - prediction
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), layout="constrained")
    axes[0].scatter(prediction, residuals, s=10, alpha=0.45, color="#215b87")
    axes[0].axhline(0, color="#b74032", linewidth=1)
    axes[0].set(xlabel="Fitted target", ylabel="Residual, actual minus fitted",
                title="Residuals versus fitted values")
    axes[1].hist(residuals, bins=35, color="#215b87", alpha=0.8,
                 edgecolor="white", linewidth=0.5)
    axes[1].axvline(0, color="#b74032", linewidth=1)
    axes[1].set(xlabel="Residual, actual minus fitted", ylabel="Number of rows",
                title="Residual distribution")
    fig.suptitle(f"{label.capitalize()}: training diagnostics after fitting all 1,000 rows",
                 fontsize=12)
    fig.savefig(figures_dir / f"{label}_residuals.png", dpi=180)
    plt.close(fig)


def run_training(data_dir, output_dir, threads):
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    models_dir = output_dir / "models"
    for directory in (results_dir, figures_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)
    summary = {
        "roll_number": ROLL,
        "protocol": {"development_rows": 800, "holdout_rows": 200,
                     "folds": N_FOLDS, "split_seed": SPLIT_SEED, "cv_seed": CV_SEED,
                     "selection_metric": "mean validation MSE",
                     "alpha_grid": ALPHAS.tolist(), "threads": threads,
                     "std_ddof": 0, "intercept": "separate, unpenalized",
                     "holdout_used_for_selection": False},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__,
                     "scikit-learn": sklearn.__version__,
                     "matplotlib": matplotlib.__version__, "joblib": joblib.__version__},
        "datasets": {},
    }
    for label, (n_inputs, max_degree) in PROBLEMS.items():
        feature_names = [f"x{i}" for i in range(1, n_inputs + 1)]
        train_path = data_dir / f"{ROLL}_train_{label}.csv"
        test_path = data_dir / f"{ROLL}_test_{label}.csv"
        train = load_csv(train_path, feature_names + ["y"])
        X, y = train[:, :-1], train[:, -1]
        X_test = load_csv(test_path, feature_names)
        development, holdout = train_test_split(
            np.arange(len(y)), test_size=0.2, random_state=SPLIT_SEED
        )
        print(f"{label}: searching degrees 1-{max_degree}, 50 alphas, five folds...", flush=True)
        all_results, per_degree, selected = select_model(
            X[development], y[development], max_degree
        )
        degree, alpha = int(selected["degree"]), float(selected["alpha"])
        development_model = make_model(degree, alpha).fit(X[development], y[development])
        development_metrics = regression_metrics(y[development], development_model.predict(X[development]))
        holdout_metrics = regression_metrics(y[holdout], development_model.predict(X[holdout]))
        # The holdout score is recorded once. It does not change degree or alpha.
        final_model = make_model(degree, alpha).fit(X, y)
        train_prediction = final_model.predict(X)
        test_prediction = final_model.predict(X_test)
        final_metrics = regression_metrics(y, train_prediction)
        all_results.to_csv(results_dir / f"{label}_cv_grid.csv", index=False)
        per_degree.to_csv(results_dir / f"{label}_degree_selection.csv", index=False)
        save_predictions(test_prediction, output_dir / f"{ROLL}_pred_{label}.csv")
        joblib.dump({"model": final_model, "feature_names": feature_names,
                     "roll_number": ROLL, "dataset": label},
                    models_dir / f"{label}_model.joblib", compress=3)
        save_figures(label, per_degree, selected, y, train_prediction, figures_dir)
        summary["datasets"][label] = {
            "input_features": feature_names, "max_degree": max_degree,
            "train_rows": len(y), "test_rows": len(X_test),
            "selected_degree": degree, "selected_alpha": alpha,
            "n_polynomial_features": int(selected["n_polynomial_features"]),
            "selection_cv": {"mse": float(selected["mean_cv_mse"]),
                             "mse_std": float(selected["std_cv_mse"]),
                             "r2": float(selected["mean_cv_r2"])},
            "development_train": development_metrics, "holdout": holdout_metrics,
            "final_train": final_metrics,
            "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (train_path, test_path)},
        }
        print(f"{label}: degree={degree}, alpha={alpha:.8g}, "
              f"CV MSE={selected['mean_cv_mse']:.6f}, "
              f"holdout MSE={holdout_metrics['mse']:.6f}", flush=True)
    (results_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Saved predictions, models, figures, and results to {output_dir}", flush=True)


def run_prediction(data_dir, output_dir):
    for label in PROBLEMS:
        saved = joblib.load(output_dir / "models" / f"{label}_model.joblib")
        X_test = load_csv(data_dir / f"{ROLL}_test_{label}.csv", saved["feature_names"])
        save_predictions(saved["model"].predict(X_test),
                         output_dir / f"{ROLL}_pred_{label}.csv")
    print(f"Saved both prediction CSVs to {output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=SCRIPT_DIR / "data",
                        help="Folder containing the four BT2024103 input CSVs")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR,
                        help="Folder for predictions, models, figures, and results")
    parser.add_argument("--predict-only", action="store_true",
                        help="Use saved final models without repeating training")
    parser.add_argument("--threads", type=int, default=2,
                        help="BLAS threads, default 2 to limit CPU and memory use")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    data_dir, output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=args.threads):
        if args.predict_only:
            run_prediction(data_dir, output_dir)
        else:
            run_training(data_dir, output_dir, args.threads)


if __name__ == "__main__":
    main()
