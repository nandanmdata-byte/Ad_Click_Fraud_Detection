# Data Irregularities

## IR-001: Sample data is not chronologically ordered

**Status:** Accepted limitation

The `train_sample.csv` file contains records covering
2017-11-06 16:00:00 through 2017-11-09 15:59:51, but its rows are
not chronologically ordered.

The file is a sampled development resource rather than a continuous
event stream. Sorting it would order the retained observations but
would not restore omitted click events.

**Decision:** Use the sample only for schema validation, debugging and
smoke tests. Create behavioural and temporal features from consecutive
records obtained from the full training data.

## IR-002: Two indistinguishable click records

**Status:** Retained

Two records contain identical values for all available columns:

- IP: 871
- App: 12
- Device: 1
- OS: 13
- Channel: 178
- Click time: 2017-11-08 10:00:05
- Attributed time: Missing
- Is attributed: 0

The dataset contains no unique click identifier, and timestamps are
recorded only to the second. The records cannot be proven to be
accidental duplicates.

**Decision:** Retain both records. Exact repetition may represent
genuine click frequency and could contain useful behavioural
information.