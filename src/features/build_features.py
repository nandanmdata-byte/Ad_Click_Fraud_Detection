"""
Build leakage-safe features for the TalkingData development dataset.

The development data must contain clicks from 2017-11-07 13:00:00 up to,
but not including, 15:00:00. Feature values are built on the complete,
chronologically ordered window before the training/validation split is applied.

Important: ``attributed_time`` is deliberately never loaded or used.

"""

from pathlib import Path
from time import perf_counter

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "development_clicks.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "development_features.parquet"
)

RAW_CATEGORICAL_COLUMNS = ["ip", "app", "device", "os", "channel"]
TARGET = "is_attributed"
CLICK_TIME = "click_time"

INPUT_COLUMNS = RAW_CATEGORICAL_COLUMNS + [CLICK_TIME, TARGET]

EXPECTED_ROWS = 6_397_036
EXPECTED_ATTRIBUTED_CLICKS = 15_775

WINDOW_START = pd.Timestamp("2017-11-07 13:00:00")
WINDOW_END = pd.Timestamp("2017-11-07 15:00:00")
VALIDATION_START = pd.Timestamp("2017-11-07 14:30:00")


def validate_input(df: pd.DataFrame) -> None:
    """Validate the project-specific development dataset."""

    if list(df.columns) != INPUT_COLUMNS:
        raise ValueError(
            f"Unexpected columns. Expected {INPUT_COLUMNS}, got {list(df.columns)}"
        )

    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} rows, got {len(df):,}")

    attributed_clicks = int(df[TARGET].sum())
    if attributed_clicks != EXPECTED_ATTRIBUTED_CLICKS:
        raise ValueError(
            "Expected "
            f"{EXPECTED_ATTRIBUTED_CLICKS:,} attributed clicks, "
            f"got {attributed_clicks:,}"
        )

    if df[INPUT_COLUMNS].isna().any().any():
        missing = df[INPUT_COLUMNS].isna().sum()
        raise ValueError(f"Unexpected missing values:\n{missing[missing > 0]}")

    if df[CLICK_TIME].min() < WINDOW_START:
        raise ValueError("Dataset contains clicks before the selected window")

    if df[CLICK_TIME].max() >= WINDOW_END:
        raise ValueError("Dataset contains clicks at or after the window end")


def add_time_features(df: pd.DataFrame) -> None:
    """Add compact calendar features derived only from click_time."""

    click_time = df[CLICK_TIME].dt

    df["click_day"] = click_time.day.astype("uint8")
    df["click_hour"] = click_time.hour.astype("uint8")
    df["click_minute"] = click_time.minute.astype("uint8")
    df["click_second"] = click_time.second.astype("uint8")
    df["click_day_of_week"] = click_time.dayofweek.astype("uint8")


def add_previous_count(
    df: pd.DataFrame,
    group_columns: list[str],
    feature_name: str,
) -> None:
    
    """Count earlier rows in each group; the current row is excluded."""

    df[feature_name] = (
        df.groupby(group_columns, sort=False, observed=True)
        .cumcount()
        .astype("uint32")
    )


def add_previous_click_gap(
    df: pd.DataFrame,
    group_columns: list[str],
    feature_name: str,
) -> None:
    
    """
    Add seconds since the preceding click in a group.

    A value of -1 identifies the first observed click for that group.
    """

    previous_time = df.groupby(
        group_columns,
        sort=False,
        observed=True,
    )[CLICK_TIME].shift(1)

    gap_seconds = (df[CLICK_TIME] - previous_time).dt.total_seconds()
    df[feature_name] = gap_seconds.fillna(-1).astype("float32")


def build_features(df: pd.DataFrame) -> pd.DataFrame:

    """Return raw, time, and strictly backward-looking behavioural features."""

    if not df[CLICK_TIME].is_monotonic_increasing:
        print("Input is not chronological; applying a stable time sort.")
        df = df.sort_values(CLICK_TIME, kind="stable").reset_index(drop=True)

    add_time_features(df)

    count_specs = [
        (["ip"], "previous_ip_clicks"),
        (["ip", "app"], "previous_ip_app_clicks"),
        (["ip", "channel"], "previous_ip_channel_clicks"),
    ]

    gap_specs = [
        (["ip"], "seconds_since_previous_ip_click"),
        (["ip", "app"], "seconds_since_previous_ip_app_click"),
        (["ip", "channel"], "seconds_since_previous_ip_channel_click"),
    ]

    for group_columns, feature_name in count_specs:
        add_previous_count(df, group_columns, feature_name)

    for group_columns, feature_name in gap_specs:
        add_previous_click_gap(df, group_columns, feature_name)

    return df


def validate_features(df: pd.DataFrame) -> None:

    """Run lightweight leakage and split-readiness checks."""

    feature_columns = [
        column
        for column in df.columns
        if column not in {CLICK_TIME, TARGET}
    ]

    if "attributed_time" in df.columns:
        raise ValueError("attributed_time must not appear in the feature data")

    if df[feature_columns].isna().any().any():
        raise ValueError("Feature data contains unexpected missing values")

    gap_columns = [
        column
        for column in df.columns
        if column.startswith("seconds_since_previous_")
    ]

    if (df[gap_columns] < -1).any().any():
        raise ValueError("Previous-click gaps contain invalid negative values")

    train_rows = int((df[CLICK_TIME] < VALIDATION_START).sum())
    validation_rows = len(df) - train_rows

    if train_rows == 0 or validation_rows == 0:
        raise ValueError("The chronological split produced an empty partition")

    print(f"Training rows:   {train_rows:>10,}")
    print(f"Validation rows: {validation_rows:>10,}")


def main() -> None:
    
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    started_at = perf_counter()

    print(f"Reading: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH, columns=INPUT_COLUMNS)
    validate_input(df)

    print("Building leakage-safe features...")
    featured_df = build_features(df)
    validate_features(featured_df)

    # Remove objects that are no longer needed before Parquet serialization.
    del df

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    featured_df.to_parquet(
        OUTPUT_PATH,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    elapsed_minutes = (perf_counter() - started_at) / 60
    memory_mb = featured_df.memory_usage(deep=True).sum() / 1024**2

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Shape: {featured_df.shape}")
    print(f"Memory: {memory_mb:,.2f} MB")
    print(f"Elapsed: {elapsed_minutes:,.2f} minutes")


if __name__ == "__main__":
    main()
