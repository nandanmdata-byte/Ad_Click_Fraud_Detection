from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
  PROJECT_ROOT
  / "data"
  / "raw"
  / "train.csv"
)

OUTPUT_PATH = (
  PROJECT_ROOT
  / "data"
  / "interim"
  / "development_clicks.parquet"
)

START_TIME = pd.Timestamp("2017-11-07 13:00:00")
END_TIME = pd.Timestamp("2017-11-07 15:00:00")

CHUNK_SIZE = 1_000_000

EXPECTED_ROWS = 6_397_036
EXPECTED_COLUMNS = 8
EXPECTED_ATTRIBUTED = 15_775

DTYPES = {
  "ip": "uint32",
  "app": "uint16",
  "device": "uint16",
  "os": "uint16",
  "channel": "uint16",
  "is_attributed": "uint8",
}


def main():
  if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_PATH}"
      )

  selected_chunks = []

  reader = pd.read_csv(
    INPUT_PATH,
    dtype=DTYPES,
    parse_dates=[
      "click_time",
      "attributed_time",
    ],
    chunksize=CHUNK_SIZE,
  )

  for chunk_number, chunk in enumerate( reader, start=1, ):
    
    selected = chunk.loc[
      (chunk["click_time"] >= START_TIME)
      & (chunk["click_time"] < END_TIME)
    ]

    if not selected.empty:
      selected_chunks.append(selected)

    print(
      f"Checked chunk {chunk_number}",
      end="\r",
    )

    # The source file was confirmed to be chronologically
    # sorted, so no later rows can belong to this window.
    if chunk["click_time"].iloc[-1] >= END_TIME:
      break

  if not selected_chunks:
    raise RuntimeError(
      "No rows were found in the requested time window."
    )

  development_df = pd.concat(
    selected_chunks,
    ignore_index=True,
  )

  actual_rows = len(development_df)
  actual_columns = development_df.shape[1]
  actual_attributed = int(
      development_df["is_attributed"].sum()
  )
  actual_rate = (
      development_df["is_attributed"].mean() * 100
  )
  actual_start = development_df["click_time"].min()
  actual_end = development_df["click_time"].max()

  # Validate the extracted dataset before saving it.
  if actual_rows != EXPECTED_ROWS:
    raise ValueError(
      "Row-count validation failed: "
      f"expected {EXPECTED_ROWS:,}, "
      f"found {actual_rows:,}."
    )

  if actual_columns != EXPECTED_COLUMNS:
    raise ValueError(
      "Column-count validation failed: "
      f"expected {EXPECTED_COLUMNS}, "
      f"found {actual_columns}."
    )

  if actual_attributed != EXPECTED_ATTRIBUTED:
    raise ValueError(
      "Attributed-click validation failed: "
      f"expected {EXPECTED_ATTRIBUTED:,}, "
      f"found {actual_attributed:,}."
    )

  if actual_start < START_TIME:
    raise ValueError(
      "Start-boundary validation failed: "
      f"found {actual_start}."
    )

  if actual_end >= END_TIME:
    raise ValueError(
      "End-boundary validation failed: "
      f"found {actual_end}."
    )

  if not development_df["click_time"].is_monotonic_increasing:
    raise ValueError(
      "Chronological-order validation failed."
    )

  OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
  )

  development_df.to_parquet(
    OUTPUT_PATH,
    index=False,
  )

  memory_mb = (
    development_df.memory_usage(deep=True).sum()
    / 1024**2
  )

  print("\n")
  print("Development dataset created successfully.")
  print("Shape:", development_df.shape)

  print(
    "Time range:",
    actual_start,
    "to",
    actual_end,
  )

  print(
    "Attributed clicks:",
    f"{actual_attributed:,}",
  )

  print(
    "Attribution rate:",
    f"{actual_rate:.6f}%",
  )

  print(
    "Memory usage:",
    f"{memory_mb:.2f} MB",
  )

  print("All validations passed.")
  print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
  main()