"""Feature engineering for behavioral insider threat logs."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_EVENT_TYPES = ("logon", "file", "device", "email", "web")


def add_time_features(
    events: pd.DataFrame,
    business_start_hour: int = 8,
    business_end_hour: int = 18,
) -> pd.DataFrame:
    """Add time-based features used for behavioral analysis."""
    if "timestamp" not in events.columns:
        raise ValueError("events must contain a parsed 'timestamp' column.")

    # Work on a copy so EDA cells can still use the original events table
    out = events.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"])
    out["date"] = out["timestamp"].dt.date
    out["hour"] = out["timestamp"].dt.hour
    out["day_of_week"] = out["timestamp"].dt.dayofweek
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    # After-hours activity is one of the easiest signals to explain in a report
    out["is_after_hours"] = (
        (out["hour"] < business_start_hour) | (out["hour"] >= business_end_hour)
    ).astype(int)
    return out


def aggregate_user_time_window(
    events: pd.DataFrame,
    window: str = "1D",
    business_start_hour: int = 8,
    business_end_hour: int = 18,
) -> pd.DataFrame:
    """Aggregate event logs into one row per user per time window."""
    required = {"user", "timestamp", "event_type"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"events is missing required columns: {sorted(missing)}")

    events = add_time_features(events, business_start_hour, business_end_hour)
    events["event_type"] = events["event_type"].astype(str).str.lower().replace({"http": "web"})
    # A window is usually one user-day in the notebook. Changing window="1H"
    # would make it more sensitive but also noisier
    events["window_start"] = _floor_timestamp(events["timestamp"], window)

    grouped = events.groupby(["user", "window_start"], observed=True)
    # Start with general volume features, then add more specific behaviors
    features = grouped.size().rename("total_events").to_frame()
    features["after_hours_count"] = grouped["is_after_hours"].sum()
    features["after_hours_rate"] = features["after_hours_count"] / features["total_events"].clip(lower=1)
    features["weekend_count"] = grouped["is_weekend"].sum()
    features["weekend_rate"] = features["weekend_count"] / features["total_events"].clip(lower=1)

    if "pc" in events.columns:
        features["unique_pcs"] = grouped["pc"].nunique()
        features["most_common_pc"] = grouped["pc"].agg(_mode_or_unknown)
    else:
        features["unique_pcs"] = 0
        features["most_common_pc"] = "unknown"

    for event_type in DEFAULT_EVENT_TYPES:
        # These counts are the basic behavior profile for each user/window
        mask = events["event_type"].eq(event_type)
        count_series = mask.groupby([events["user"], events["window_start"]]).sum()
        features[f"{event_type}_count"] = count_series

    activity_text = events.get("activity", pd.Series("", index=events.index)).astype(str).str.lower()
    # CERT stores logon/logoff info in an activity text field, so use text checks
    is_logon = events["event_type"].eq("logon") & activity_text.str.contains("logon|login", regex=True)
    is_logoff = events["event_type"].eq("logon") & activity_text.str.contains("logoff|logout", regex=True)
    features["login_frequency"] = is_logon.groupby([events["user"], events["window_start"]]).sum()
    features["logoff_count"] = is_logoff.groupby([events["user"], events["window_start"]]).sum()
    features["after_hours_logon_count"] = (
        (is_logon & events["is_after_hours"].eq(1)).groupby([events["user"], events["window_start"]]).sum()
    )

    if "filename" in events.columns:
        features["unique_files"] = grouped["filename"].nunique()
    else:
        features["unique_files"] = 0

    if "is_decoy_file" in events.columns:
        # Decoy file access is a strong signal because those files are bait
        decoy_file_event = pd.to_numeric(events["is_decoy_file"], errors="coerce").fillna(0).astype(bool)
    else:
        decoy_file_event = pd.Series(False, index=events.index)
    features["decoy_file_count"] = decoy_file_event.groupby([events["user"], events["window_start"]]).sum()

    if "url" in events.columns:
        features["unique_urls"] = grouped["url"].nunique()
    else:
        features["unique_urls"] = 0

    if "attachments" in events.columns:
        # Attachments/size can catch unusual email exfiltration behavior
        attachments = pd.to_numeric(events["attachments"], errors="coerce").fillna(0)
        features["email_attachment_count"] = attachments.groupby([events["user"], events["window_start"]]).sum()
    else:
        features["email_attachment_count"] = 0

    if "size" in events.columns:
        email_size = pd.to_numeric(events["size"], errors="coerce").fillna(0)
        features["email_size_total"] = email_size.groupby([events["user"], events["window_start"]]).sum()
    else:
        features["email_size_total"] = 0

    if "label" in events.columns:
        # If any event in the window is malicious, label the whole window as 1
        labels = pd.to_numeric(events["label"], errors="coerce").fillna(0).astype(int)
        features["label"] = labels.groupby([events["user"], events["window_start"]]).max()

    features = features.fillna(0).reset_index()
    features = add_unusual_activity_spike_features(features)
    return features


def add_unusual_activity_spike_features(
    features: pd.DataFrame,
    metrics: Sequence[str] = ("total_events", "file_count", "device_count", "email_count", "web_count"),
    baseline_windows: int = 7,
    spike_threshold: float = 2.5,
) -> pd.DataFrame:
    """Add rolling z-score features that capture sudden activity spikes."""
    out = features.sort_values(["user", "window_start"]).copy()
    spike_cols = []

    for metric in metrics:
        if metric not in out.columns:
            continue
        spike_col = f"{metric}_spike_z"
        # Use each user's previous windows as their baseline. That keeps the
        # feature personal instead of comparing everyone to one global average
        out[spike_col] = out.groupby("user")[metric].transform(
            lambda series: _rolling_z_score(series, baseline_windows)
        )
        spike_cols.append(spike_col)

    if spike_cols:
        out["max_spike_z"] = out[spike_cols].max(axis=1)
        out["unusual_activity_spike"] = (out["max_spike_z"] >= spike_threshold).astype(int)
    else:
        out["max_spike_z"] = 0.0
        out["unusual_activity_spike"] = 0

    return out


def prepare_feature_matrix(
    feature_df: pd.DataFrame,
    label_col: str = "label",
    categorical_cols: Optional[Iterable[str]] = ("most_common_pc",),
    drop_cols: Iterable[str] = ("user", "window_start"),
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Return a model-ready X matrix and optional y labels."""
    out = feature_df.copy()
    y = None
    if label_col in out.columns:
        y = pd.to_numeric(out[label_col], errors="coerce").fillna(0).astype(int)
        out = out.drop(columns=[label_col])

    drop_existing = [col for col in drop_cols if col in out.columns]
    out = out.drop(columns=drop_existing)

    categorical_existing = [col for col in (categorical_cols or []) if col in out.columns]
    if categorical_existing:
        # One-hot encoding keeps this notebook simple and scikit-learn friendly
        out = pd.get_dummies(out, columns=categorical_existing, drop_first=False)

    for column in out.columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0)
    return out, y


def _floor_timestamp(timestamp: pd.Series, window: str) -> pd.Series:
    """Floor timestamps for fixed windows, with a period fallback for weekly/monthly windows."""
    try:
        return timestamp.dt.floor(window)
    except ValueError:
        return timestamp.dt.to_period(window).dt.start_time


def _rolling_z_score(series: pd.Series, baseline_windows: int) -> pd.Series:
    previous = series.shift(1)
    rolling_mean = previous.rolling(baseline_windows, min_periods=2).mean()
    rolling_std = previous.rolling(baseline_windows, min_periods=2).std().replace(0, np.nan)
    z_score = (series - rolling_mean) / rolling_std
    return z_score.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-10, 10)


def _mode_or_unknown(series: pd.Series) -> str:
    mode = series.dropna().mode()
    return str(mode.iloc[0]) if len(mode) else "unknown"
