from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
  PROJECT_ROOT
  / "data"
  / "raw"
  / "train.csv"
)

OUTPUT_PATH = (
  PROJECT_ROOT
  / "reports"
  / "hourly_target_profile.csv"
)

CHUNK_SIZE = 1_000_000

hourly_results = []


reader = pd.read_csv(
  DATA_PATH,
  usecols=[
    "click_time",
    "is_attributed",
  ],
  dtype={
    "is_attributed": "uint8",
  },
  parse_dates=[
    "click_time",
  ],
  chunksize=CHUNK_SIZE,
)


for chunk_number, chunk in enumerate(reader, start=1):
  chunk["click_hour"] = (
    chunk["click_time"].dt.floor("h")
  )

  hourly_chunk = (
    chunk.groupby("click_hour")
    .agg(
        total_clicks=(
          "is_attributed",
          "size",
        ),
        attributed_clicks=(
          "is_attributed",
          "sum",
        ),
      )
      .reset_index()
  )

  hourly_results.append(hourly_chunk)

  if chunk_number == 1 or chunk_number % 10 == 0:
    print(
      f"Processed chunk {chunk_number}"
    )


hourly_profile = (
  pd.concat(
    hourly_results,
    ignore_index=True,
  )
  .groupby(
    "click_hour",
    as_index=False,
  )
  .agg(
    total_clicks=(
      "total_clicks",
      "sum",
    ),
    attributed_clicks=(
      "attributed_clicks",
      "sum",
    ),
  )
)


hourly_profile["attributed_rate_percent"] = (
  hourly_profile["attributed_clicks"]
  / hourly_profile["total_clicks"]
  * 100
)


OUTPUT_PATH.parent.mkdir(
  parents=True,
  exist_ok=True,
)

hourly_profile.to_csv(
  OUTPUT_PATH,
  index=False,
)


print("\nHourly profile completed.")
print(hourly_profile.to_string(index=False))
print("\nOutput saved to:", OUTPUT_PATH)