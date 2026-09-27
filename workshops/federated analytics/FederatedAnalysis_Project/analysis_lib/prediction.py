"""Helper functions for the prediction notebook."""

# ====================================
# Imports
# ====================================
# Standard library

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ====================================
# Correlation between predictors
# ====================================
def plot_correlation_heatmap(df: pd.DataFrame, predictors: list[str], threshold: float = 0.7) -> None:
    """Plot the absolute pairwise correlations between predictors as a lower-triangle heatmap.

    Absolute values are used because a strong negative correlation means the same
    redundancy as a strong positive one. Each pair appears once (lower triangle), and
    cells above `threshold` are outlined, to spot redundant pairs at a glance.

    Args:
        df: Dataset with the predictors encoded as numbers.
        predictors: Names of the predictor columns to correlate.
        threshold: Absolute correlation above which a pair is highlighted.
    """
    # Drop the first row and last column, which would be empty in a lower triangle
    correlations = df[predictors].corr().abs().iloc[1:, :-1]
    n = len(correlations)

    # Hide the diagonal and upper triangle: each pair is shown once
    lower = np.tril(np.ones((n, n), dtype=bool))
    values = correlations.where(lower)

    fig, ax = plt.subplots(figsize=(0.5 * n + 2, 0.5 * n))
    image = ax.imshow(values, cmap="Reds", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, shrink=0.8, label="Absolute correlation")

    # Outline the cells above the threshold
    for i, j in zip(*np.nonzero(lower)):
        if correlations.iloc[i, j] > threshold:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", lw=2))

    ax.set_xticks(range(n), correlations.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(n), correlations.index, fontsize=8)
    ax.set_title(f"Correlation between predictors (outlined: |r| > {threshold})")
    plt.tight_layout()
    plt.show()


def strongest_correlations(df: pd.DataFrame, predictors: list[str], n: int = 5) -> pd.DataFrame:
    """Return the `n` predictor pairs with the highest absolute correlation, as in the heatmap.

    Args:
        df: Dataset with the predictors encoded as numbers.
        predictors: Names of the predictor columns to correlate.
        n: Number of pairs to return.

    Returns:
        One row per pair, indexed by (predictor, predictor), with its absolute correlation, sorted descending.
    """
    correlations = df[predictors].corr().abs()

    # Keep each pair once (lower triangle, without the diagonal)
    lower = np.tril(np.ones(correlations.shape, dtype=bool), k=-1)
    pairs = correlations.where(lower).stack()

    strongest = pairs.sort_values(ascending=False).iloc[:n]
    return strongest.rename_axis(["Predictor 1", "Predictor 2"]).to_frame("Absolute correlation").round(3)


# ====================================
# Model evaluation
# ====================================
def evaluate_model(model: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Compute the prediction metrics of a fitted binary classifier.

    Classification metrics use `model.predict` (threshold 0.5); AUC uses the predicted
    probability of the positive class.

    Args:
        model: Fitted classifier (or pipeline) with `predict` and `predict_proba`.
        X: Predictors of the patients to evaluate.
        y: True outcome of those patients (1 = event).

    Returns:
        Accuracy, Recall, Precision, F1 and AUC, indexed by metric name.
    """
    predicted = model.predict(X)
    probability = model.predict_proba(X)[:, 1]

    return pd.Series({
        "Accuracy": accuracy_score(y, predicted),
        "Recall": recall_score(y, predicted),
        # 0 instead of a warning if no patient is predicted as an event
        "Precision": precision_score(y, predicted, zero_division=0),
        "F1": f1_score(y, predicted),
        "AUC": roc_auc_score(y, probability),
    })


def plot_confusion_matrix(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    labels: tuple[str, str] = ("Alive", "Dead"),
    title: str | None = None,
) -> None:
    """Plot the confusion matrix of a fitted binary classifier (format of `plot_confusion_matrices`).

    Args:
        model: Fitted classifier (or pipeline) with `predict`.
        X: Predictors of the patients to evaluate.
        y: True outcome of those patients (0/1).
        labels: Display names of outcomes 0 and 1.
        title: Plot title.
    """
    counts = confusion_matrix(y, model.predict(X), labels=[0, 1])
    plot_confusion_matrices({title or "": counts}, labels=labels)


def plot_confusion_matrices(
    counts_by_title: dict[str, np.ndarray],
    labels: tuple[str, str] = ("Alive", "Dead"),
) -> None:
    """Plot one or more confusion matrices side by side, colored by row percentage.

    Each cell shows its count and, below it, its percentage of the patients with that
    true outcome (row), so the diagonal shows the share of each outcome classified correctly.
    Colors follow that percentage on a fixed 0-100 scale, so the large number of survivors
    doesn't dominate them and plots are comparable with each other.

    Args:
        counts_by_title: 2x2 count matrix (rows = true outcome) per plot title.
        labels: Display names of outcomes 0 and 1.
    """
    fig, axes = plt.subplots(1, len(counts_by_title), figsize=(4.5 * len(counts_by_title), 4.3), squeeze=False)
    for ax, (title, counts) in zip(axes[0], counts_by_title.items()):
        row_percentages = 100 * counts / counts.sum(axis=1, keepdims=True)
        matrix_display = ConfusionMatrixDisplay(row_percentages, display_labels=list(labels))
        matrix_display.plot(cmap="Blues", colorbar=False, ax=ax, im_kw={"vmin": 0, "vmax": 100})

        # Replace each cell's text with its count and row percentage
        for (i, j), text in np.ndenumerate(matrix_display.text_):
            text.set_text(f"{counts[i, j]:,}\n({row_percentages[i, j]:.1f}%)")

        ax.set_xlabel("Predicted outcome")
        ax.set_ylabel("True outcome")
        if title:
            ax.set_title(title)
    fig.text(
        0.5, 0.01, "Each cell: count (percentage of its true outcome), colored by the percentage",
        ha="center", fontsize=8,
    )
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.show()


# ====================================
# Model coefficients
# ====================================
def model_coefficients(model: Pipeline | LogisticRegression, predictors: list[str]) -> pd.Series:
    """Return the logistic regression coefficients of a fitted model, one per predictor.

    Args:
        model: Fitted `LogisticRegression`, or a pipeline whose last step, named "model", is one.
        predictors: Predictor names, in the order the model was fitted with.

    Returns:
        Coefficients indexed by predictor name.
    """
    if isinstance(model, Pipeline):
        model = model.named_steps["model"]
    return pd.Series(model.coef_[0], index=predictors)


# ====================================
# Federated scaling
# ====================================
def scaler_from_statistics(mean: pd.Series, std: pd.Series) -> StandardScaler:
    """Build a fitted `StandardScaler` from given per-predictor means and standard deviations.

    Args:
        mean: Mean of each predictor, indexed by predictor name.
        std: Population standard deviation of each predictor, same index as `mean`.

    Returns:
        Scaler that transforms data exactly as one fitted on data with these statistics.
    """
    scaler = StandardScaler()
    scaler.mean_ = mean.to_numpy()
    scaler.var_ = (std ** 2).to_numpy()
    scaler.scale_ = std.to_numpy()
    scaler.n_features_in_ = len(mean)
    # Column names, so transforming a DataFrame doesn't warn about missing feature names
    scaler.feature_names_in_ = mean.index.to_numpy(dtype=object)
    return scaler


# ====================================
# Federated evaluation
# ====================================
def site_evaluation(model: BaseEstimator, X: np.ndarray, y: pd.Series) -> dict:
    """Evaluate a model on one institution's data, returning only what it shares.

    Args:
        model: Fitted classifier with `predict` and `predict_proba`.
        X: The institution's predictors, already scaled.
        y: The institution's true outcome (0/1).

    Returns:
        Dict with the confusion-matrix counts ("counts", 2x2, rows = true outcome),
        the AUC ("auc") and the number of patients ("n").
    """
    return {
        "counts": confusion_matrix(y, model.predict(X), labels=[0, 1]),
        "auc": roc_auc_score(y, model.predict_proba(X)[:, 1]),
        "n": len(y),
    }


def combine_site_evaluations(evaluations: list[dict]) -> pd.Series:
    """Combine the institutions' evaluations into overall metrics.

    Accuracy, recall, precision and F1 come from the summed counts, so they're exact;
    AUC is the average of the institutions' AUCs weighted by their number of patients.

    Args:
        evaluations: One `site_evaluation` result per institution.

    Returns:
        Accuracy, Recall, Precision, F1 and AUC, in the same format as `evaluate_model`.
    """
    (true_negatives, false_positives), (false_negatives, true_positives) = sum(
        evaluation["counts"] for evaluation in evaluations
    )
    predicted_deaths = true_positives + false_positives
    recall = true_positives / (true_positives + false_negatives)
    # Same convention as `evaluate_model`: 0 when no patient is predicted as dead
    precision = true_positives / predicted_deaths if predicted_deaths else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = sum(evaluation["n"] for evaluation in evaluations)
    return pd.Series({
        "Accuracy": (true_positives + true_negatives) / total,
        "Recall": recall,
        "Precision": precision,
        "F1": f1,
        "AUC": sum(evaluation["n"] * evaluation["auc"] for evaluation in evaluations) / total,
    })