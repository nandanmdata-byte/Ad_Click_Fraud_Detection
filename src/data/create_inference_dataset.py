"""Create a small unseen dataset for the inference stage."""

from pathlib import Path
from time import perf_counter

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "inference" / "new_clicks.parquet"

# This window starts after the development and calibration data ended.
START_TIME = pd.Timestamp("2017-11-07 15:15:00")
END_TIME = pd.Timestamp("2017-11-07 15:30:00")

CHUNK_SIZE = 1_000_000

# These are the only columns available when making a real prediction.
INPUT_COLUMNS = ["ip", "app", "device", "os", "channel", "click_time"]

DTYPES = {
    "ip": "uint32",
    "app": "uint16",
    "device": "uint16",
    "os": "uint16",
    "channel": "uint16",
}


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {INPUT_PATH}")

    started_at = perf_counter()
    selected_chunks = []

    print("Reading the raw dataset in chunks...")

    # Chunking prevents the complete 184-million-row CSV from entering memory.
    reader = pd.read_csv(
        INPUT_PATH,
        usecols=INPUT_COLUMNS,
        dtype=DTYPES,
        parse_dates=["click_time"],
        chunksize=CHUNK_SIZE,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        window_mask = (
            (chunk["click_time"] >= START_TIME)
            & (chunk["click_time"] < END_TIME)
        )

        if window_mask.any():
            selected_chunks.append(chunk.loc[window_mask].copy())

        # train.csv is chronological, so no later chunk can belong to the
        # selected window after this point.
        if chunk["click_time"].max() >= END_TIME:
            break

        if chunk_number % 10 == 0:
            print(f"Processed {chunk_number * CHUNK_SIZE:,} rows...")

    if not selected_chunks:
        raise ValueError("No clicks were found inside the selected time window")

    inference_clicks = pd.concat(selected_chunks, ignore_index=True)

    # Final checks before saving the inference data.
    if list(inference_clicks.columns) != INPUT_COLUMNS:
        raise ValueError("The inference dataset has unexpected columns")

    if inference_clicks[INPUT_COLUMNS].isna().any().any():
        raise ValueError("The inference dataset contains missing values")

    if not inference_clicks["click_time"].is_monotonic_increasing:
        raise ValueError("The inference clicks are not chronologically ordered")

    if inference_clicks["click_time"].min() < START_TIME:
        raise ValueError("The inference dataset starts before the selected window")

    if inference_clicks["click_time"].max() >= END_TIME:
        raise ValueError("The inference dataset extends beyond the selected window")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    inference_clicks.to_parquet(
        OUTPUT_PATH,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    elapsed_minutes = (perf_counter() - started_at) / 60

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(inference_clicks):,}")
    print(
        "Time range: "
        f"{inference_clicks['click_time'].min()} to "
        f"{inference_clicks['click_time'].max()}"
    )
    print(f"Elapsed: {elapsed_minutes:.2f} minutes")


if __name__ == "__main__":
    main()
