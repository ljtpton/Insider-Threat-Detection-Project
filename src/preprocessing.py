"""Data loading and preprocessing helpers for CERT-style insider threat logs.

The functions in this module are intentionally lightweight and readable for an
undergraduate data science project. They support the common CERT Insider Threat
Dataset CSV names while also handling small sample CSVs with similar columns.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union
import tarfile
import warnings

import numpy as np
import pandas as pd


LOG_NAME_PATTERNS = {
    "logon": ("logon", "login", "logoff"),
    "device": ("device", "usb", "removable"),
    "decoy": ("decoy",),
    "file": ("file", "files"),
    "email": ("email", "mail"),
    "web": ("http", "web", "url", "browser"),
    "labels": ("answer", "label", "truth", "malicious", "insider"),
}

# Common column names vary a little between CERT versions and sample files
# Keeping these lists in one place makes the rest of the loader less brittle
USER_COLUMNS = ("user", "user_id", "userid", "username", "employee", "employee_id")
TIMESTAMP_COLUMNS = ("timestamp", "datetime", "date", "time", "event_time")
LABEL_COLUMNS = ("label", "target", "class", "malicious", "is_malicious", "insider_threat", "threat")
ANSWER_COLUMNS = ("event_type", "id", "date", "user", "pc")

# The full CERT CSVs can be huge, especially email/http. These column lists keep
# only the fields used later in the notebook so pandas does less work
EVENT_USE_COLUMNS = {
    "logon": {"id", "date", "timestamp", "user", "pc", "activity", "label"},
    "device": {"id", "date", "timestamp", "user", "pc", "file_tree", "activity", "label"},
    "file": {
        "id",
        "date",
        "timestamp",
        "user",
        "pc",
        "filename",
        "activity",
        "to_removable_media",
        "from_removable_media",
        "label",
    },
    "email": {
        "id",
        "date",
        "timestamp",
        "user",
        "pc",
        "to",
        "cc",
        "bcc",
        "from",
        "activity",
        "size",
        "attachments",
        "label",
    },
    "web": {"id", "date", "timestamp", "user", "pc", "url", "activity", "label"},
    "decoy": {"decoy_filename", "filename", "pc"},
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with simple snake_case column names."""
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return out


def infer_log_type(path: Path, df: Optional[pd.DataFrame] = None) -> str:
    """Infer the log type from a CSV file name and, if needed, its columns."""
    # Use only the local file/folder name here. The project folder itself has
    # "Insider" in the name, and using the full path would misclassify everything
    path_text = f"{path.parent.name}/{path.name}".lower()
    if any(pattern in path_text for pattern in LOG_NAME_PATTERNS["labels"]):
        return "labels"

    name = path.stem.lower()
    for log_type, patterns in LOG_NAME_PATTERNS.items():
        if any(pattern in name for pattern in patterns):
            return log_type

    if df is not None:
        columns = set(normalize_columns(df).columns)
        if "url" in columns or "domain" in columns:
            return "web"
        if "attachments" in columns or "to" in columns or "cc" in columns:
            return "email"
        if "filename" in columns or "file" in columns:
            return "file"
        if "activity" in columns and any("device" in str(value).lower() for value in df["activity"].head(100)):
            return "device"
        if "event_type" in columns:
            return "generic"

    return "unknown"


def read_csv_files(
    data_dir: Union[str, Path] = "data",
    sample_fraction: Optional[float] = None,
    max_rows_per_file: Optional[int] = None,
    answer_release: Optional[str] = None,
    random_state: int = 42,
    chunksize: int = 100_000,
) -> Dict[str, pd.DataFrame]:
    """Read all CSV files from data_dir and group them by inferred log type."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data folder '{data_path}' was not found. Create it and place CERT CSV files there, "
            "for example logon.csv, device.csv, file.csv, email.csv, http.csv, and answers.csv."
        )

    csv_paths = sorted(data_path.rglob("*.csv"))
    archive_csvs = read_answer_archives(data_path, answer_release=answer_release)
    label_users = collect_label_users([df for _, df in archive_csvs])
    if not csv_paths and not archive_csvs:
        raise FileNotFoundError(
            f"No CSV files were found in '{data_path}'. Add CERT-style CSV files or run the "
            "notebook demo mode to generate a small synthetic example."
        )

    grouped: Dict[str, list[pd.DataFrame]] = {}
    for path in csv_paths:
        log_type = infer_log_type(path)
        if log_type == "labels":
            # Some answer archives include multiple CERT releases. Filter to the
            # release used by the current logs, 
            if answer_release and answer_release.lower() not in str(path).lower():
                continue
            df = read_answer_csv(path)
            label_users.update(collect_label_users([df]))
        else:
            df = read_event_csv(
                path,
                log_type,
                sample_fraction=sample_fraction,
                max_rows_per_file=max_rows_per_file,
                label_users=label_users,
                random_state=random_state,
                chunksize=chunksize,
            )
        df["source_file"] = path.name
        grouped.setdefault(log_type, []).append(df)

    for source_path, df in archive_csvs:
        log_type = infer_log_type(source_path, df)
        df["source_file"] = str(source_path)
        grouped.setdefault(log_type, []).append(df)

    return {log_type: pd.concat(frames, ignore_index=True) for log_type, frames in grouped.items()}


def read_answer_archives(data_path: Path, answer_release: Optional[str] = None) -> List[Tuple[Path, pd.DataFrame]]:
    """Read CSV files stored inside answers.tar.bz2-style archives."""
    archive_frames: List[Tuple[Path, pd.DataFrame]] = []
    archive_patterns = ("*.tar.bz2", "*.tbz2", "*.tar.gz", "*.tgz")
    archive_paths = []
    for pattern in archive_patterns:
        archive_paths.extend(data_path.glob(pattern))

    for archive_path in sorted(set(archive_paths)):
        if "answer" not in archive_path.name.lower() and "label" not in archive_path.name.lower():
            continue
        try:
            with tarfile.open(archive_path) as archive:
                for member in archive.getmembers():
                    # answers.tar.bz2 is a folder-like archive, so each inner CSV
                    # has to be read separately
                    if not member.isfile() or not member.name.lower().endswith(".csv"):
                        continue
                    if answer_release and answer_release.lower() not in member.name.lower():
                        continue
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        continue
                    df = read_answer_csv(extracted)
                    source_path = Path(archive_path.name) / member.name
                    archive_frames.append((source_path, df))
        except (tarfile.TarError, OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
            warnings.warn(f"Could not read answer archive '{archive_path.name}': {exc}")

    return archive_frames


def read_answer_csv(path_or_buffer: Union[str, Path, object]) -> pd.DataFrame:
    """Read CERT answer-key CSV rows, which often do not include a header."""
    rows = read_variable_width_csv(path_or_buffer)
    if not rows:
        return pd.DataFrame(columns=ANSWER_COLUMNS)

    # CERT answer rows are not always the same length. A logon row is short,
    # while an email/file row can have many extra fields. Padding keeps it usable
    max_cols = max(len(row) for row in rows)
    padded_rows = [row + [None] * (max_cols - len(row)) for row in rows]
    first_row = [str(value).strip().lower() for value in padded_rows[0]]

    if "user" in first_row and ("date" in first_row or "timestamp" in first_row):
        columns = first_row + [f"answer_field_{idx}" for idx in range(len(first_row), max_cols)]
        return normalize_columns(pd.DataFrame(padded_rows[1:], columns=columns)).reset_index(drop=True)

    columns = list(ANSWER_COLUMNS)
    extra_columns = [f"answer_field_{idx}" for idx in range(len(columns), max_cols)]
    return normalize_columns(pd.DataFrame(padded_rows, columns=columns + extra_columns))


def read_variable_width_csv(path_or_buffer: Union[str, Path, object]) -> List[List[object]]:
    """Read CSV rows even when different lines have different field counts."""
    should_close = False
    if isinstance(path_or_buffer, (str, Path)):
        handle = open(path_or_buffer, "r", newline="", encoding="utf-8", errors="replace")
        should_close = True
    elif isinstance(path_or_buffer, io.TextIOBase):
        handle = path_or_buffer
    else:
        handle = io.TextIOWrapper(path_or_buffer, encoding="utf-8", errors="replace", newline="")
        should_close = True

    try:
        # csv.reader handles quoted commas better than splitting strings by comma
        return [row for row in csv.reader(handle)]
    finally:
        if should_close:
            handle.close()


def read_event_csv(
    path: Path,
    log_type: str,
    sample_fraction: Optional[float] = None,
    max_rows_per_file: Optional[int] = None,
    label_users: Optional[set] = None,
    random_state: int = 42,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """Read an event CSV, optionally sampling large files while preserving labeled users."""
    usecols = EVENT_USE_COLUMNS.get(log_type)
    usecols_filter = None
    if usecols:
        # usecols can take a function, which is handy after normalizing names
        usecols_filter = lambda column: str(column).strip().lower().replace("-", "_") in usecols

    if sample_fraction is None:
        # In the notebook, max_rows_per_file keeps the course project runnable on
        # a laptop while still using real CERT data
        df = pd.read_csv(path, usecols=usecols_filter, nrows=max_rows_per_file)
        return normalize_columns(df)

    sample_fraction = min(max(sample_fraction, 0.0), 1.0)
    label_users = set(label_users or [])
    frames = []
    for chunk_idx, chunk in enumerate(
        pd.read_csv(path, usecols=usecols_filter, chunksize=chunksize)
    ):
        chunk = normalize_columns(chunk)
        user_col = find_first_column(chunk, USER_COLUMNS)
        if user_col and label_users:
            # Keep all rows for known malicious users, then sample normal users
            # This avoids accidentally dropping the rare class
            user_values = chunk[user_col].astype(str).str.strip()
            labeled_mask = user_values.isin(label_users)
            labeled_rows = chunk.loc[labeled_mask]
            normal_rows = chunk.loc[~labeled_mask]
        else:
            labeled_rows = chunk.iloc[0:0]
            normal_rows = chunk

        if sample_fraction >= 1:
            sampled_rows = normal_rows
        elif sample_fraction > 0 and len(normal_rows):
            sampled_rows = normal_rows.sample(frac=sample_fraction, random_state=random_state + chunk_idx)
        else:
            sampled_rows = normal_rows.iloc[0:0]

        if len(labeled_rows) or len(sampled_rows):
            frames.append(pd.concat([labeled_rows, sampled_rows], ignore_index=True))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    if max_rows_per_file is not None and len(df) > max_rows_per_file:
        user_col = find_first_column(df, USER_COLUMNS)
        if user_col and label_users:
            # If the capped sample is still too big, shrink normal rows first
            labeled_mask = df[user_col].astype(str).str.strip().isin(label_users)
            labeled_rows = df.loc[labeled_mask]
            normal_rows = df.loc[~labeled_mask]
            normal_budget = max(max_rows_per_file - len(labeled_rows), 0)
            if len(normal_rows) > normal_budget:
                normal_rows = normal_rows.sample(n=normal_budget, random_state=random_state)
            df = pd.concat([labeled_rows, normal_rows], ignore_index=True)
        else:
            df = df.sample(n=max_rows_per_file, random_state=random_state)
    return df.reset_index(drop=True)


def collect_label_users(label_frames: Iterable[pd.DataFrame]) -> set:
    """Collect malicious user IDs from prepared answer/label frames."""
    users = set()
    for frame in label_frames:
        if frame is None or frame.empty:
            continue
        user_col = find_first_column(normalize_columns(frame), USER_COLUMNS)
        if user_col:
            users.update(frame[user_col].astype(str).str.strip().dropna().tolist())
    return users


def find_first_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Find the first candidate column that exists in df."""
    for column in candidates:
        if column in df.columns:
            return column
    return None


def parse_timestamp(df: pd.DataFrame, output_col: str = "timestamp") -> pd.DataFrame:
    """Parse a timestamp-like column and store it in output_col."""
    out = df.copy()
    timestamp_col = find_first_column(out, TIMESTAMP_COLUMNS)
    if timestamp_col is None:
        raise ValueError(f"No timestamp column found. Expected one of: {', '.join(TIMESTAMP_COLUMNS)}")

    out[output_col] = pd.to_datetime(out[timestamp_col], errors="coerce")
    missing_count = int(out[output_col].isna().sum())
    if missing_count:
        # Bad timestamps are not useful for time-window features, so drop them
        warnings.warn(f"{missing_count:,} rows had timestamps that could not be parsed and will be dropped.")
    out = out.dropna(subset=[output_col])
    return out


def standardize_event_log(df: pd.DataFrame, event_type: str) -> pd.DataFrame:
    """Convert one raw log table into a shared event format."""
    out = normalize_columns(df)
    user_col = find_first_column(out, USER_COLUMNS)
    if user_col is None:
        warnings.warn(f"Skipping {event_type} log because no user column was found.")
        return pd.DataFrame()

    try:
        out = parse_timestamp(out)
    except ValueError as exc:
        warnings.warn(f"Skipping {event_type} log: {exc}")
        return pd.DataFrame()

    out["user"] = out[user_col].astype(str).str.strip()
    out["event_type"] = "web" if event_type == "http" else event_type

    # Not every file has every column, so fill simple defaults for the common
    # fields expected 
    if "pc" not in out.columns:
        out["pc"] = "unknown"
    if "activity" not in out.columns:
        out["activity"] = out["event_type"]

    label_col = find_first_column(out, LABEL_COLUMNS)
    if label_col is not None:
        out["label"] = out[label_col].map(to_binary_label).fillna(0).astype(int)

    return out


def standardize_answer_events(label_frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Convert answer-key rows into malicious event rows.

    CERT answer files include the actual malicious log rows. Adding those rows
    to the sampled event table keeps positive examples available even when the
    large normal log files are only partially loaded.
    """
    frames = []
    for labels in label_frames:
        out = normalize_columns(labels)
        if not {"event_type", "date", "user"}.issubset(out.columns):
            continue

        # The answer key contains actual malicious events, so these rows become
        # labeled positive examples even if the giant raw log was sampled
        out = parse_timestamp(out)
        out["user"] = out["user"].astype(str).str.strip()
        out["event_type"] = out["event_type"].astype(str).str.lower().replace({"http": "web"})
        if "pc" not in out.columns:
            out["pc"] = "unknown"

        out["activity"] = _answer_activity(out)
        out["label"] = 1
        out["source_file"] = out.get("source_file", "answers")

        # Answer-key columns after pc differ by event type. Pull the useful bits
        # back into standard names when possible
        file_mask = out["event_type"].eq("file")
        if "filename" not in out.columns:
            out["filename"] = pd.Series(pd.NA, index=out.index, dtype="object")
        else:
            out["filename"] = out["filename"].astype("object")
        if "answer_field_5" in out.columns:
            out.loc[file_mask, "filename"] = out.loc[file_mask, "answer_field_5"]

        web_mask = out["event_type"].eq("web")
        if "url" not in out.columns:
            out["url"] = pd.Series(pd.NA, index=out.index, dtype="object")
        else:
            out["url"] = out["url"].astype("object")
        if "answer_field_5" in out.columns:
            out.loc[web_mask, "url"] = out.loc[web_mask, "answer_field_5"]

        if "size" not in out.columns:
            out["size"] = pd.Series(pd.NA, index=out.index, dtype="object")
        else:
            out["size"] = out["size"].astype("object")
        email_mask = out["event_type"].eq("email")
        if "answer_field_10" in out.columns:
            out.loc[email_mask, "size"] = out.loc[email_mask, "answer_field_10"]
        if "attachments" not in out.columns:
            out["attachments"] = pd.Series(pd.NA, index=out.index, dtype="object")
        else:
            out["attachments"] = out["attachments"].astype("object")
        if "answer_field_11" in out.columns:
            out.loc[email_mask, "attachments"] = out.loc[email_mask, "answer_field_11"]

        keep_cols = [
            "timestamp",
            "user",
            "pc",
            "event_type",
            "activity",
            "label",
            "filename",
            "url",
            "size",
            "attachments",
            "source_file",
        ]
        frames.append(out[keep_cols])

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _answer_activity(df: pd.DataFrame) -> pd.Series:
    """Choose a readable activity field from answer-key rows."""
    activity = pd.Series("answer_key_event", index=df.index, dtype="object")
    event_type = df["event_type"].astype(str).str.lower()
    field_map = {
        "logon": "answer_field_5",
        "device": "answer_field_6",
        "file": "answer_field_6",
        "web": "answer_field_6",
        "http": "answer_field_6",
        "email": "answer_field_9",
    }
    for kind, field in field_map.items():
        if field in df.columns:
            # Different event types put "activity" in different answer fields
            mask = event_type.eq(kind)
            activity.loc[mask] = df.loc[mask, field].fillna(activity.loc[mask])
    if "answer_field_6" in df.columns:
        activity = activity.fillna(df["answer_field_6"])
    if "activity" in df.columns:
        activity = df["activity"].fillna(activity)
    return activity.astype(str)


def to_binary_label(value: object) -> Optional[int]:
    """Map common label values to 0 or 1."""
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer, float, np.floating)):
        return int(value > 0)

    text = str(value).strip().lower()
    positive = {"1", "true", "yes", "y", "malicious", "insider", "threat", "anomaly", "bad"}
    negative = {"0", "false", "no", "n", "normal", "benign", "clean", "good"}
    if text in positive:
        return 1
    if text in negative:
        return 0
    return None


def prepare_label_table(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize a label or CERT answers table.

    CERT answer files often contain malicious user/time intervals rather than a
    label for every event. If no explicit label column exists, rows in an answer
    file are treated as malicious intervals.
    """
    labels = normalize_columns(df)
    user_col = find_first_column(labels, USER_COLUMNS)
    if user_col is None:
        warnings.warn("A labels file was found, but it has no recognizable user column.")
        return pd.DataFrame()

    labels["user"] = labels[user_col].astype(str).str.strip()

    label_col = find_first_column(labels, LABEL_COLUMNS)
    if label_col is None:
        labels["label"] = 1
    else:
        labels["label"] = labels[label_col].map(to_binary_label).fillna(0).astype(int)

    start_col = find_first_column(labels, ("start", "start_time", "begin", "from_date", "date", "timestamp"))
    end_col = find_first_column(labels, ("end", "end_time", "finish", "to_date"))

    # If only one date is given, treat it as a one-day malicious interval
    if start_col is not None:
        labels["start"] = pd.to_datetime(labels[start_col], errors="coerce")
    if end_col is not None:
        labels["end"] = pd.to_datetime(labels[end_col], errors="coerce")
    elif "start" in labels.columns:
        labels["end"] = labels["start"] + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    keep_cols = ["user", "label"]
    for col in ("start", "end"):
        if col in labels.columns:
            keep_cols.append(col)
    return labels[keep_cols].drop_duplicates().reset_index(drop=True)


def attach_interval_labels(events: pd.DataFrame, labels: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Attach binary malicious labels to standardized event rows."""
    out = events.copy()
    if "label" not in out.columns:
        out["label"] = 0
    else:
        out["label"] = out["label"].fillna(0).astype(int)

    if labels is None or labels.empty:
        return out

    label_table = prepare_label_table(labels)
    if label_table.empty:
        return out

    for _, row in label_table[label_table["label"] == 1].iterrows():
        # Mark matching user/time rows as positive. 
        mask = out["user"].eq(row["user"])
        if "start" in label_table.columns and pd.notna(row.get("start")):
            mask &= out["timestamp"].ge(row["start"])
        if "end" in label_table.columns and pd.notna(row.get("end")):
            mask &= out["timestamp"].le(row["end"])
        out.loc[mask, "label"] = 1

    return out


def load_behavioral_events(
    data_dir: Union[str, Path] = "data",
    demo_if_missing: bool = False,
    sample_fraction: Optional[float] = None,
    max_rows_per_file: Optional[int] = None,
    answer_release: Optional[str] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Load CERT-style logs and return standardized event rows.

    Returns
    -------
    events:
        One row per raw event with user, timestamp, event_type, activity, pc, and label.
    labels:
        Standardized labels or None if no label file was found.
    raw_tables:
        Raw normalized CSV tables grouped by inferred log type.
    """
    try:
        raw_tables = read_csv_files(
            data_dir,
            sample_fraction=sample_fraction,
            max_rows_per_file=max_rows_per_file,
            answer_release=answer_release,
            random_state=random_state,
        )
    except FileNotFoundError:
        if not demo_if_missing:
            raise
        events, labels = make_demo_events()
        return events, labels, {"demo": events.copy()}

    label_tables = []
    decoy_tables = []
    event_frames = []

    for log_type, df in raw_tables.items():
        if log_type == "labels":
            label_tables.append(df)
            continue
        if log_type == "decoy":
            decoy_tables.append(df)
            continue
        standardized = standardize_event_log(df, log_type)
        if not standardized.empty:
            event_frames.append(standardized)

    answer_events = standardize_answer_events(label_tables)
    if not answer_events.empty:
        # This makes sure the rare threat class is present even with sampled logs
        event_frames.append(answer_events)

    if not event_frames:
        if demo_if_missing:
            events, labels = make_demo_events()
            return events, labels, {"demo": events.copy()}
        raise ValueError("CSV files were found, but no behavioral event logs could be standardized.")

    events = pd.concat(event_frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    labels = pd.concat(label_tables, ignore_index=True) if label_tables else None
    # Decoy files are a lookup table, so flag matching file events after the
    # event tables are combined
    events = mark_decoy_file_access(events, decoy_tables)
    events = attach_interval_labels(events, labels)
    return events, labels, raw_tables


def mark_decoy_file_access(events: pd.DataFrame, decoy_tables: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Flag file events whose filename appears in a CERT decoy-file lookup table."""
    out = events.copy()
    out["is_decoy_file"] = 0
    if "filename" not in out.columns:
        return out

    decoy_filenames = set()
    for table in decoy_tables:
        table = normalize_columns(table)
        filename_col = find_first_column(table, ("decoy_filename", "filename", "file"))
        if filename_col:
            # Normalize case/spacing so path matches are not missed fs
            decoy_filenames.update(table[filename_col].astype(str).str.lower().str.strip().tolist())

    if decoy_filenames:
        file_values = out["filename"].astype(str).str.lower().str.strip()
        out["is_decoy_file"] = file_values.isin(decoy_filenames).astype(int)
    return out


def summarize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact missing-value summary."""
    missing = df.isna().sum()
    summary = pd.DataFrame(
        {
            "missing_count": missing,
            "missing_percent": (missing / len(df) * 100).round(2) if len(df) else 0,
        }
    )
    return summary[summary["missing_count"] > 0].sort_values("missing_count", ascending=False)


def make_demo_events(
    n_users: int = 30,
    n_days: int = 28,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a small CERT-like dataset for notebook demonstration.

    The synthetic data is not a substitute for the CERT dataset. It simply lets
    the notebook run when the real CSV files are not present.
    """
    rng = np.random.default_rng(random_state)
    users = [f"U{idx:03d}" for idx in range(1, n_users + 1)]
    pcs = [f"PC{idx:03d}" for idx in range(1, n_users + 1)]
    start_date = pd.Timestamp("2024-01-01")
    days = pd.date_range(start_date, periods=n_days, freq="D")
    threat_user = users[4]
    threat_days = set(days[-5:])
    rows = []

    for user_idx, user in enumerate(users):
        pc = pcs[user_idx]
        for day in days:
            # Normal users have a light weekend pattern and more weekday activity
            if day.weekday() >= 5:
                base_events = rng.poisson(2)
            else:
                base_events = rng.poisson(10)

            logon_hour = int(rng.integers(7, 10))
            logoff_hour = int(rng.integers(16, 19))
            rows.append(_demo_row(day, logon_hour, user, pc, "logon", "Logon", 0, rng))
            rows.append(_demo_row(day, logoff_hour, user, pc, "logon", "Logoff", 0, rng))

            for _ in range(base_events):
                event_type = rng.choice(["file", "email", "web"], p=[0.35, 0.30, 0.35])
                hour = int(rng.integers(8, 18))
                activity = {"file": "File Open", "email": "Email Sent", "web": "Visit"}[event_type]
                rows.append(_demo_row(day, hour, user, pc, event_type, activity, 0, rng))

            if rng.random() < 0.12:
                rows.append(_demo_row(day, int(rng.integers(9, 17)), user, pc, "device", "Connect", 0, rng))

            if user == threat_user and day in threat_days:
                # The demo threat is a late-night burst with files, USB, email,
                # and web events. It is just for running the notebook without CERT
                for _ in range(int(rng.integers(12, 22))):
                    event_type = rng.choice(["file", "device", "email", "web"], p=[0.45, 0.20, 0.20, 0.15])
                    hour = int(rng.integers(20, 24))
                    activity = {
                        "file": "Sensitive File Copy",
                        "device": "USB Connect",
                        "email": "External Email",
                        "web": "Job Site Visit",
                    }[event_type]
                    rows.append(_demo_row(day, hour, user, pc, event_type, activity, 1, rng))

    events = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    labels = pd.DataFrame(
        {
            "user": [threat_user],
            "label": [1],
            "start": [min(threat_days) + pd.Timedelta(hours=20)],
            "end": [max(threat_days) + pd.Timedelta(hours=23, minutes=59)],
        }
    )
    return events, labels


def _demo_row(
    day: pd.Timestamp,
    hour: int,
    user: str,
    pc: str,
    event_type: str,
    activity: str,
    label: int,
    rng: np.random.Generator,
) -> dict:
    minute = int(rng.integers(0, 60))
    timestamp = day + pd.Timedelta(hours=hour, minutes=minute)
    return {
        "timestamp": timestamp,
        "user": user,
        "pc": pc,
        "event_type": event_type,
        "activity": activity,
        "label": label,
        "filename": "confidential.docx" if event_type == "file" and label else np.nan,
        "url": "jobs.example.com" if event_type == "web" and label else np.nan,
        "size": 1500 if event_type == "email" else np.nan,
        "attachments": 1 if event_type == "email" and label else 0,
        "source_file": "synthetic_demo",
    }
