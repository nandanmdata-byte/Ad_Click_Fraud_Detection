# Libraries
import json

from pathlib import Path
import pandas as pd

# Locate the project and required files
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Path to full_data
SAMPLE_PATH = (
  PROJECT_ROOT
  / "data"
  / "raw"
  / "train_sample.csv"
)

# Path to audit report
REPORT_PATH = (
  PROJECT_ROOT
  / "reports"
  / "sample_data_audit.json"
)

# Smaller numeric types to reduce memory consumption
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

# 100,000-row sample load
df = pd.read_csv(
  SAMPLE_PATH,
  dtype=DTYPES,
  parse_dates=["click_time", "attributed_time"],
)

# Important validation results calculation
target_counts = df["is_attributed"].value_counts().sort_index()
target_percentages = (
  df["is_attributed"]
  .value_counts(normalize=True)
  .sort_index()
  .mul(100)
)

#---------------
positive_without_time = int(
  (
    (df["is_attributed"] == 1)
    & (df["attributed_time"].isna())
  ).sum()
)

negative_with_time = int(
  (
    (df["is_attributed"] == 0)
    & (df["attributed_time"].notna())
  ).sum()
)

attribution_before_click = int(
  (
    df["attributed_time"].notna()
    & (
      df["attributed_time"]
      < df["click_time"]
    )
  ).sum()
)

candidate_duplicate_records = (
  df[df.duplicated(keep=False)]
  .sort_values(
    [
      "click_time",
      "ip",
      "app",
      "device",
      "os",
      "channel",
    ]
  )
  .copy()
)
#---------------

# Audit report

audit = {
  "rows" : int(df.shape[0]),
  "columns" : int(df.shape[1]),
  "column_names": df.columns.tolist(),
  "schema_valid": df.columns.tolist() == EXPECTED_COLUMNS,
  "data_types": {
    column: str(dtype)
    for column, dtype in df.dtypes.items()
  },
  "memory_mb": round(
    df.memory_usage(deep=True).sum() / 1024**2,
    2,
  ),
  "missing_values": {
    column: int(count)
    for column, count in df.isna().sum().items()
  },
  "candidate_duplicate_rows": int(df.duplicated().sum()),
  "minimum_click_time": str(df["click_time"].min()),
  "maximum_click_time": str(df["click_time"].max()),
  "click_time_is_sorted": bool(
      df["click_time"].is_monotonic_increasing
  ),
  "target_counts": {
    str(label): int(count)
    for label, count in target_counts.items()
  },
  "target_percentages": {
    str(label): round(float(percentage), 6)
    for label, percentage in target_percentages.items()
  },
  "unique_values": {
    column: int(df[column].nunique())
    for column in [
      "ip",
      "app",
      "device",
      "os",
      "channel",
    ]
  },
  "outcome_consistency": {
    "positive_without_attributed_time": (
        positive_without_time
    ),
    "negative_with_attributed_time": (
        negative_with_time
    ),
    "attributed_time_before_click_time": (
        attribution_before_click
    ),
  },
}

# Present important results
print("Dataset shape:", df.shape)
print("Schema valid:", audit["schema_valid"])
print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum())

print("\nTarget counts:")
print(target_counts)

print("\nTarget percentages:")
print(target_percentages)

print(
  "\nClick-time range:",
  audit["minimum_click_time"],
  "to",
  audit["maximum_click_time"],
)

print(
  "\nCandidate duplicate rows:",
  audit["candidate_duplicate_rows"],
)

print("\nMemory usage:", audit["memory_mb"], "MB")

print(
  "\nClick time is chronologically sorted:",
  audit["click_time_is_sorted"],
)

print("\nOutcome consistency:")

print(
  "Positive clicks without attributed_time:",
  audit["outcome_consistency"][
      "positive_without_attributed_time"
  ],
)

print(
  "Negative clicks with attributed_time:",
  audit["outcome_consistency"][
      "negative_with_attributed_time"
  ],
)

print(
  "attributed_time earlier than click_time:",
  audit["outcome_consistency"][
      "attributed_time_before_click_time"
  ],
)

print("\nCandidate duplicate observations:")

if candidate_duplicate_records.empty:
    print("No candidate duplicates found.")
else:
  print(
      candidate_duplicate_records.to_string(
      index=False
      )
    )


# Save the small audit summary
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

with REPORT_PATH.open("w", encoding="utf-8") as file:
    json.dump(audit, file, indent=4)

print("\nAudit saved to:", REPORT_PATH)