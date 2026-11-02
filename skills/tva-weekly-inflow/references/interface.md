# Supported commands

Run from the cloned repository. `python pipeline.py --help` lists the interface. If installed separately from the checkout, use `cd "$TVA_INFLOW_REPO"` first.

```bash
python pipeline.py inspect
python pipeline.py infer --inputs PATH.csv --metadata PATH.json --out outputs/NEW_NAME
python pipeline.py replay --split test --out outputs/NEW_NAME
python pipeline.py plot --predictions PATH.csv --sites ELF13 ELF39 --out outputs/NEW_NAME
```

`replay` and `infer` default to CPU and the frozen best checkpoint. They optionally accept `--checkpoint PATH` and `--device cuda` when an allocation supports it. `inspect` verifies each exported baseline file's SHA-256. It does not require a GPU.

## Inference contract

CSV required columns, in physical units:

`basin_id,start,precipitation_mm_week,temperature_mean_c,total_runoff_mm_week`

Each site needs at least 13 weekly rows, sorted internally. `start` is a Monday, and each row describes the following complete seven days. Duplicate site/date rows, unknown sites, or negative precipitation/runoff are rejected. Non-finite values and calendar gaps invalidate affected 13-week windows. No imputation occurs. The first 12 rows per site are context only.

The JSON metadata must declare:

```json
{
  "calendar": "Monday-Sunday",
  "precipitation_units": "mm/week",
  "temperature_units": "degC",
  "runoff_units": "mm/week",
  "runoff_definition": "surface_runoff_plus_baseflow",
  "source_kind": "historical"
}
```

For actual forecast forcings use `source_kind: "forecast"` and record issue time and lead definitions as additional provenance. That label documents the source; it does not establish forecast accuracy or automatically verify that no future observations were used. If runoff is surface runoff only or its definition is unknown, resolve that mismatch before inference. Do not relabel it to pass validation.

The stored static elevation and drainage area belong to each trained site's original boundary. New HEADWATER maps must be spatially aggregated to that same support. This package accepts basin CSVs and does not contain a raw NetCDF map adapter.

`infer` writes predictions.csv and provenance.json. Output depth is mean daily-equivalent mm/day; weekly m³ = depth × area_km² × 7000. It is already in physical target units, so do not apply another target inverse-standardization.

## Evaluation and plotting

`replay` writes predictions.csv, metrics_by_site.csv and summary.json. It uses `eligible AND split` exactly as saved and compares default-checkpoint validation/test results with the frozen predictions. Allow small CPU/ROCm numerical differences, not different masks or dates. Never improve agreement by dropping additional weeks.

`plot` requires paired `observed_volume_m3` and `predicted_volume_m3` with basin_id and start. It writes an all-site NSE figure and individual hydrographs, preserving masked calendar gaps. Future inference outputs have no reference observations, so they cannot produce accuracy metrics without an independently matched reference.

`--out` must be a new directory outside baseline. The pipeline exits with an error rather than replacing existing output.
