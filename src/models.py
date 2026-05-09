"""Model definitions and training helpers."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC, OneClassSVM


def get_supervised_models(random_state: int = 42) -> Dict[str, object]:
    """Return the supervised models used in the project."""
    # The
    # class_weight setting matters because threat windows are rare
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            # Fewer trees and a max depth keep runtime reasonable on CERT samples
            n_estimators=100,
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "Support Vector Machine": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    # Linear SVM is used here because full RBF SVM is too slow
                    # for this laptop-sized CERT workflow
                    LinearSVC(
                        class_weight="balanced",
                        random_state=random_state,
                        max_iter=10000,
                        tol=1e-3,
                    ),
                ),
            ]
        ),
    }


def get_anomaly_models(contamination: float = 0.02, random_state: int = 42) -> Dict[str, object]:
    """Return unsupervised anomaly detection models."""
    # contamination is the expected anomaly rate. Clip it so the model does not
    # get an impossible value if the data is extremely imbalanced
    contamination = min(max(contamination, 0.001), 0.49)
    return {
        "Isolation Forest": IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        ),
        "One-Class SVM": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", OneClassSVM(kernel="linear", nu=contamination, max_iter=-1)),
            ]
        ),
    }


def fit_models(models: Dict[str, object], X_train: pd.DataFrame, y_train: Optional[pd.Series] = None) -> Dict[str, object]:
    """Fit a dictionary of models and return the fitted models."""
    fitted = {}
    for name, model in models.items():
        # same helper works for both supervised and unsupervised models
        if y_train is None:
            model.fit(X_train)
        else:
            model.fit(X_train, y_train)
        fitted[name] = model
    return fitted


def fit_anomaly_models(
    models: Dict[str, object],
    X_train: pd.DataFrame,
    y_train: Optional[pd.Series] = None,
    normal_label: int = 0,
) -> Dict[str, object]:
    """Fit anomaly models, using normal-only training rows when labels exist."""
    if y_train is not None and (y_train == normal_label).any():
        # Anomaly models should learn normal behavior first, then flag weird rows
        X_fit = X_train.loc[y_train == normal_label]
    else:
        X_fit = X_train
    return fit_models(models, X_fit, None)


def get_positive_scores(model: object, X: pd.DataFrame) -> np.ndarray:
    """Return scores where larger values mean higher threat probability/risk."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        # SScale them to 0-1
        # so ROC curves can compare models on the same plot
        raw_scores = model.decision_function(X)
        return MinMaxScaler().fit_transform(np.asarray(raw_scores).reshape(-1, 1)).ravel()
    return np.asarray(model.predict(X))


def predict_anomaly_labels(model: object, X: pd.DataFrame) -> np.ndarray:
    """Convert anomaly detector predictions to 1 = anomaly/threat and 0 = normal."""
    raw = model.predict(X)
    # scikit-learn anomaly detectors usually use -1 for anomaly and 1 for normal
    return np.where(raw == -1, 1, 0)


def get_anomaly_scores(model: object, X: pd.DataFrame) -> np.ndarray:
    """Return normalized anomaly risk scores where higher means more anomalous."""
    if hasattr(model, "decision_function"):
        # Lower anomaly decision scores are more suspicious, so flip the sign
        raw_scores = -np.asarray(model.decision_function(X)).reshape(-1, 1)
        return MinMaxScaler().fit_transform(raw_scores).ravel()
    return predict_anomaly_labels(model, X)
