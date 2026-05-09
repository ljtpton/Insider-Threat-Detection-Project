"""Evaluation helpers for binary insider threat detection."""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .models import get_anomaly_scores, get_positive_scores, predict_anomaly_labels


def false_negative_rate(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Return FN / (FN + TP), or NaN if there are no positive cases."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    positive_total = fn + tp
    # For insider threats, false negatives are the scary misses
    return float(fn / positive_total) if positive_total else np.nan


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    model_name: str = "Model",
    model_type: str = "supervised",
) -> dict:
    """Compute project metrics for one model."""
    # Keep all metrics in a plain dict so it is easy to turn into a table later
    metrics = {
        "model": model_name,
        "type": model_type,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "false_negative_rate": false_negative_rate(y_true, y_pred),
    }

    if y_score is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    else:
        # ROC-AUC needs both classes. Tiny samples sometimes do not have that
        metrics["roc_auc"] = np.nan

    return metrics


def evaluate_supervised_models(models: Dict[str, object], X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Evaluate fitted supervised models."""
    rows = []
    for name, model in models.items():
        # Supervised models predict 0/1 directly
        y_pred = model.predict(X_test)
        y_score = get_positive_scores(model, X_test)
        rows.append(evaluate_predictions(y_test, y_pred, y_score, name, "supervised"))
    return pd.DataFrame(rows).sort_values(["recall", "f1"], ascending=False).reset_index(drop=True)


def evaluate_anomaly_models(models: Dict[str, object], X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Evaluate fitted anomaly models against known labels."""
    rows = []
    for name, model in models.items():
        # Anomaly models use -1/1 internally, so convert to the project labels
        y_pred = predict_anomaly_labels(model, X_test)
        y_score = get_anomaly_scores(model, X_test)
        rows.append(evaluate_predictions(y_test, y_pred, y_score, name, "anomaly"))
    return pd.DataFrame(rows).sort_values(["recall", "f1"], ascending=False).reset_index(drop=True)


def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, title: str = "Confusion Matrix") -> None:
    """Plot a confusion matrix with labels for normal and threat classes."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # A heatmap is easier to read in the notebook than a raw 2x2 array
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Normal", "Predicted Threat"],
        yticklabels=["Actual Normal", "Actual Threat"],
    )
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()


def plot_roc_curves(
    supervised_models: Dict[str, object],
    anomaly_models: Dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Plot ROC curves for supervised and anomaly models."""
    plt.figure(figsize=(8, 6))

    for name, model in supervised_models.items():
        # Higher scores should mean "more likely threat" for every curve
        scores = get_positive_scores(model, X_test)
        fpr, tpr, _ = roc_curve(y_test, scores)
        plt.plot(fpr, tpr, label=f"{name} (supervised)")

    for name, model in anomaly_models.items():
        # Dashed lines make anomaly models visually separate from supervised ones
        scores = get_anomaly_scores(model, X_test)
        fpr, tpr, _ = roc_curve(y_test, scores)
        plt.plot(fpr, tpr, linestyle="--", label=f"{name} (anomaly)")

    plt.plot([0, 1], [0, 1], color="gray", linestyle=":", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate / Recall")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
