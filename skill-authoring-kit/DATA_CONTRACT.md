# Data contract

## Spatial support

Inputs must describe the same ELF-matched drainage area as the stored site metadata. At a headwater this generally covers the full contributing watershed. At downstream local/residual targets, it may cover an incremental area. Source-region allocation/proxy groups have additional target-construction qualifications. `baseline/data/node_metadata.csv` records those classes and review flags. The package does not supply original TVA GIS or assert that provisional boundaries are validated.

To connect future HEADWATER grids, intersect cell footprints with the matching drainage area and weight cell values by the area inside it. Use equal-area/geodesic area and fractional boundary cells. Recompute weights for a different grid. Do not take the total upstream watershed for every local ELF or sum grid depths as if they were basin-average depths. Raw-grid extraction remains a separate adapter task; the implemented interface begins with basin weekly CSVs.

## Features

The dynamic array is `(time=1816, sites=52, features=3)`, in this exact order:

1. `precipitation_mm_week`: weekly precipitation amount, basin spatial average, mm/week.
2. `temperature_mean_c`: weekly mean temperature, basin spatial average, °C. Historical temperature came from Daymet (Tmax+Tmin)/2.
3. `total_runoff_mm_week`: weekly NLDAS surface runoff + baseflow amount, basin spatial average, mm/week.

Static input: `elevation_mean_m`, shape `(52,1)`, basin mean elevation in metres. The LSTM concatenates the normalized static elevation to every time step. Drainage area is for target normalization/volume conversion and is not an additional learned feature.

For subweekly grids, integrate precipitation/runoff rates into amounts before summing. Average temperature over time. Use complete Monday–Sunday calendar weeks. For already-weekly maps, verify that amounts/means, weekly dates, units, and runoff definition match training. A weekly mean precipitation *rate* is not a weekly total amount until converted.

## Identity and normalization

Use the site order in `basin_ids` and `site_head_mapping.json`, ELF01–ELF52. Never reorder weights or assign a new location to an arbitrary head. Physical feature arrays enter normalization as float32, and the original train-only mean/std computations then return float32 normalized inputs. The saved checkpoint contains the scaler copy used for inference. Do not refit scalers on validation/test or new forecast data.

## Target and volume

ELF is the per-location reference quantity described by its target definition, not universally total upstream inflow. The trained output is mean daily-equivalent target depth in mm/day over a complete seven-day week. Weekly volume is:

`predicted_volume_m3 = predicted_mean_mm_day * area_km2 * 7000`

Output is already in target physical units. The per-site target training standard deviation is used in the loss denominator (`std + 0.1`), not as an output inverse transform.

## Dates and masks

The input window includes target week and 12 previous weeks. Splits: train1982–2004, validation2005–2009, test2010–October2016. Weeks crossing a split/year boundary or truncated archive edge may have split -1. Preserve `eligible`, `target_valid_all7`, historical window eligibility and QC bits as stored, including the inherited `graph_history13_valid` field name. That legacy field name does not mean this baseline uses a GNN.

Use `eligible AND split_code` for frozen evaluation. Counts are 50,084/9,868/15,025 for train/validation/test. Never infer validity from a finite/positive target alone. The archive has 52 sites, but 51 scoreable test sites. Individual valid-week counts differ by site. Gaps stay gaps in plots and volume summaries.

For new-input inference, no target mask or observed ELF is required. Require a complete, finite 13-week forcing window. Unknown IDs and inconsistent declared units/definitions fail validation. Missing weeks or input values flag the affected windows without gap filling.

## Forecast interpretation

Target-week forcings are inputs. A prospective forecast therefore needs target-week forecast precipitation, temperature and runoff. Thirteen weeks is history/context, not a guaranteed 13-week forecast horizon. Forecast issue/lead dates and member identities must be tracked by the upstream adapter. Changing `source_kind` in metadata does not prove that a product is a valid forecast.

Historical reconstruction used observed/retrospective weather and NLDAS runoff. The saved score does not establish HEADWATER/GEFS forecast accuracy. In particular, surface runoff alone does not match the trained surface-plus-baseflow feature. Confirm the future product's runoff definition before coupling.
