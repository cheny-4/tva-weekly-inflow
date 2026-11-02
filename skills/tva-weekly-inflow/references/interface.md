# Supported interface

From the repository checkout:

```bash
python pipeline.py inspect --model surface_runoff_static8
python pipeline.py preprocess --model surface_runoff_static8 --config CONFIG.json --out outputs/NEW_PREP
python pipeline.py infer --model surface_runoff_static8 --inputs outputs/NEW_PREP/weekly_inputs.csv --metadata outputs/NEW_PREP/forcing_metadata.json --out outputs/NEW_PREDICTIONS
python pipeline.py replay --model surface_runoff_static8 --split test --out outputs/NEW_REPLAY
python pipeline.py plot --model surface_runoff_static8 --predictions PAIRED.csv --sites ELF13 ELF39 --out outputs/NEW_FIGURES
```

Use `total_runoff` for the other checkpoint. Every command accepts --model; legacy default is total_runoff. Replay/infer default to CPU and the selected saved best checkpoint; optional --checkpoint and --device cuda are accepted when compatible. Inspect checks model-specific export checksums. Neither model includes bias correction.

## Basin CSV contract

Required columns:

- Total: basin_id,start,precipitation_mm_week,temperature_mean_c,total_runoff_mm_week.
- Surface: basin_id,start,precipitation_mm_week,temperature_mean_c,surface_runoff_mm_week.

Rows represent complete Monday–Sunday weeks in physical units. Metadata declares calendar Monday-Sunday; precipitation_units mm/week; temperature_units degC; runoff_units mm/week; source_kind historical/forecast; and runoff_definition surface_runoff_plus_baseflow or surface_runoff_only as appropriate. Include model_id explicitly in new data; preprocessing does so automatically. Surface examples are examples/weekly_inputs_surface_static8.csv and examples/forcing_metadata_surface_static8.json. Original total examples retain their filenames.

Require 13 weekly rows per site, preserving target week plus 12 preceding weeks. Unknown site heads, duplicate/non-Monday dates, negative P/runoff and incompatible metadata are rejected. Calendar gaps/non-finite values invalidate windows without filling. The first 12 rows are context only. Stored elevation and, in the surface model, eight additional attributes are loaded in their frozen feature order and normalized once with the selected checkpoint's training scalers.

Infer writes predictions.csv and provenance.json including model_id, checkpoint hash/epoch and bias_correction:false. Output is already mean daily-equivalent mm/day; m³/week = depth × target area_km2 × 7000. Do not inverse-standardize it again. Observed ELF is not an inference input.

## Map inputs

Preprocess is implemented in headwater_preprocess.py. Read the checkout's docs/headwater_preprocessing.md for the explicit config. It supports rectilinear geographic/projected NetCDF grids, fractional area weighting on the packaged regions, interval-unit conversion and complete weekly aggregation. It saves sparse grid weights, coverage/QC tables, source hashes and ready-to-infer CSV/metadata. Actual HEADWATER maps still need their real variable/CRS/unit/time declarations checked; synthetic examples do not validate an unknown future schema or forecast skill.

Process forecast members/issues separately, provide target-week forcings, and supply sufficient historical context through history_csv/history_metadata if needed. Do not relabel source kinds or runoff semantics to bypass errors. Do not disaggregate misaligned weekly/monthly totals or insert future observed forcings. The supported raw-grid input is georeferenced rectilinear NetCDF, not every possible raster format.

## Frozen evaluation

Replay uses eligible AND split exactly as saved:50,084 train,9,868 validation,15,025 test rows. Default validation/test replay checks identical keys/reference values and small CPU/ROCm numerical differences. It writes predictions, metrics and summary. Plot needs independently paired observed_volume_m3 and predicted_volume_m3; inference alone cannot produce accuracy metrics. Preserve missing calendar weeks in plots and complete-volume summaries.

All output paths must be new and outside baseline/, baseline_surface_runoff_static8/ and spatial/. No command implicitly trains or submits jobs.
