import json
import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data/raw/train.csv"
REPORT_PATH = PROJECT_ROOT / "reports/full_data_audit.json"

CHUNK_SIZE = 1_000_000

DTYPES = {
    "ip": "uint32",
    "app": "uint16",
    "device": "uint16",
    "os": "uint16",
    "channel": "uint16",
    "is_attributed": "uint8",
}

EXPECTED_COLUMNS = [
    "ip",
    "app",
    "device",
    "os",
    "channel",
    "click_time",
    "attributed_time",
    "is_attributed",
]

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )


# Read only the header first
actual_columns = pd.read_csv(
    DATA_PATH,
    nrows=0,
).columns.tolist()

if actual_columns != EXPECTED_COLUMNS:
    raise ValueError(
        f"Unexpected columns: {actual_columns}"
    )


# Running totals
total_rows = 0
target_0 = 0
target_1 = 0

missing_values = {
    column: 0
    for column in EXPECTED_COLUMNS
}

unique_values = {
    column: set()
    for column in [
        "ip",
        "app",
        "device",
        "os",
        "channel",
    ]
}

minimum_click_time = None
maximum_click_time = None
previous_last_time = None
chronologically_sorted = True

positive_without_time = 0
negative_with_time = 0
invalid_attribution_time = 0

start_time = time.perf_counter()


reader = pd.read_csv(
    DATA_PATH,
    dtype=DTYPES,
    parse_dates=[
        "click_time",
        "attributed_time",
    ],
    chunksize=CHUNK_SIZE,
)


for chunk_number, chunk in enumerate(reader, start=1):
    total_rows += len(chunk)

    # Missing values
    for column, count in chunk.isna().sum().items():
        missing_values[column] += int(count)

    # Target distribution
    target_0 += int(
        (chunk["is_attributed"] == 0).sum()
    )

    target_1 += int(
        (chunk["is_attributed"] == 1).sum()
    )

    # Unique identifier values
    for column in unique_values:
        unique_values[column].update(
            chunk[column].unique()
        )

    # Timestamp range
    chunk_minimum = chunk["click_time"].min()
    chunk_maximum = chunk["click_time"].max()

    if minimum_click_time is None:
        minimum_click_time = chunk_minimum
        maximum_click_time = chunk_maximum
    else:
        minimum_click_time = min(
            minimum_click_time,
            chunk_minimum,
        )

        maximum_click_time = max(
            maximum_click_time,
            chunk_maximum,
        )

    # Chronological-order checks
    if not chunk["click_time"].is_monotonic_increasing:
        chronologically_sorted = False

    if (
        previous_last_time is not None
        and chunk["click_time"].iloc[0]
        < previous_last_time
    ):
        chronologically_sorted = False

    previous_last_time = chunk["click_time"].iloc[-1]

    # Target and attributed_time consistency
    positive_without_time += int(
        (
            (chunk["is_attributed"] == 1)
            & chunk["attributed_time"].isna()
        ).sum()
    )

    negative_with_time += int(
        (
            (chunk["is_attributed"] == 0)
            & chunk["attributed_time"].notna()
        ).sum()
    )

    invalid_attribution_time += int(
        (
            chunk["attributed_time"].notna()
            & (
                chunk["attributed_time"]
                < chunk["click_time"]
            )
        ).sum()
    )

    if chunk_number == 1 or chunk_number % 10 == 0:
        print(f"Processed {total_rows:,} rows")


processing_seconds = round(
    time.perf_counter() - start_time,
    2,
)

audit = {
    "rows": total_rows,
    "columns": len(actual_columns),
    "schema_valid": actual_columns == EXPECTED_COLUMNS,
    "missing_values": missing_values,
    "target_counts": {
        "0": target_0,
        "1": target_1,
    },
    "positive_rate_percent": round(
        target_1 / total_rows * 100,
        6,
    ),
    "minimum_click_time": str(minimum_click_time),
    "maximum_click_time": str(maximum_click_time),
    "chronologically_sorted": chronologically_sorted,
    "positive_without_attributed_time": (
        positive_without_time
    ),
    "negative_with_attributed_time": (
        negative_with_time
    ),
    "attribution_before_click_time": (
        invalid_attribution_time
    ),
    "unique_identifier_counts": {
        column: len(values)
        for column, values in unique_values.items()
    },
    "processing_seconds": processing_seconds,
}


REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with REPORT_PATH.open("w", encoding="utf-8") as file:
    json.dump(audit, file, indent=4)


print("\nFull-data audit completed")

for key, value in audit.items():
    print(f"{key}: {value}")

print("\nReport saved to:", REPORT_PATH)