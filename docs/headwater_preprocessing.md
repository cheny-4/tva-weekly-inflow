# HEADWATER map preprocessing

`python pipeline.py preprocess --model MODEL --config CONFIG.json --out NEW_DIRECTORY` is a working NetCDF-to-basin adapter. Its output `weekly_inputs.csv` and `forcing_metadata.json` feed the same model's `infer` command directly. It adds no bias correction and does not train a model.

## Explicit configuration

Generate a fully runnable example with `python examples/make_headwater_demo.py --out outputs/demo_maps`. To configure real surface-runoff files:

```json
{
  "model_id": "surface_runoff_static8",
  "source_kind": "forecast",
  "time_basis": "UTC",
  "issue_time": "2027-03-01T00:00:00Z",
  "member_id": "control",
  "runoff_definition": "surface_runoff_only",
  "minimum_grid_coverage": 0.99,
  "minimum_valid_area_fraction": 0.99,
  "variables": {
    "precipitation": {
      "files": ["HEADWATER.nc"], "variable": "precipitation",
      "kind": "amount", "units": "mm", "crs": "EPSG:4326",
      "x": "lon", "y": "lat", "time": "time",
      "interval_hours": 168, "timestamp_position": "start"
    },
    "temperature": {
      "files": ["HEADWATER.nc"], "variable": "temperature",
      "kind": "mean", "units": "K", "crs": "EPSG:4326",
      "x": "lon", "y": "lat", "time": "time",
      "interval_hours": 168, "timestamp_position": "start"
    },
    "runoff": {
      "files": ["HEADWATER.nc"], "variable": "surface_runoff",
      "kind": "amount", "units": "mm", "crs": "EPSG:4326",
      "x": "lon", "y": "lat", "time": "time",
      "interval_hours": 168, "timestamp_position": "start"
    }
  }
}
```

Paths resolve relative to the config file. List multiple files explicitly, including one-map-per-file NetCDF files with a length-one time dimension. Each variable can use its own file list and rectilinear grid. `sites` can optionally restrict processing to trained IDs; the default is all 52. No new sites or outlet snapping are performed.

For `total_runoff`, change model_id to `total_runoff`, declare `surface_runoff_plus_baseflow`, and select a variable that actually contains total runoff. Surface runoff alone cannot be renamed or relabelled to satisfy that contract. For historical reconstructions use `source_kind: historical`; issue_time/member_id are then unnecessary. For forecasts each run represents one issue/member. Select explicit extra NetCDF dimensions using `isel`, e.g. `"isel": {"member": 0, "initialization": 0}`. Extra dimensions are otherwise rejected; no members are averaged automatically.

## Spatial processing

The packaged `spatial/forcing_regions.geojson` retains43 exact forcing regions; `site_region_mapping.csv` maps them to 52 site heads and records target areas, construction classes and review flags. Headwater regions generally cover their contributing basin; downstream local/residual sites use local/incremental support, and some sites share an estimation-source/proxy region. Those mappings are preserved rather than substituting full-upstream catchments everywhere. Target area used for ELF depth-to-volume conversion is deliberately distinct from forcing-region area.

The code intersects rectilinear cell footprints with polygons, including fractional edge cells and holes, then measures overlap in EPSG:5070. Weights are intersection areas, not equal cell counts. Cells outside a region contribute nothing. For valid cells the mean is `sum(value × overlap_area) / sum(valid_overlap_area)`. Grid coverage must meet minimum_grid_coverage, and valid data must cover minimum_valid_area_fraction of the grid-covered region. Both thresholds default to 0.99 and are recorded. Renormalization excludes the small allowed missing area; values are never filled. Insufficient valid area gives NaN, which invalidates its week and affected LSTM histories.

CF coordinate bounds are used when available, or specify x_bounds/y_bounds variable names. Otherwise edges are inferred from strictly monotonic one-dimensional cell centres; single-cell axes require explicit bounds. Longitude must be normalized to−180..180. Projected grid axes and their CRS must use metres. Curvilinear grids, rotated grids requiring two-dimensional coordinates, unstructured meshes, GeoTIFF-only inputs and antimeridian-wrapping grids need conversion to a supported georeferenced NetCDF representation first. No hidden nearest-neighbour resampling occurs.

Sparse weights and grid/geometry fingerprints are saved in `weights/`. Set `reuse_weights_from` to a previous weights directory to reuse matching weights. A changed grid or boundary gets a different key; cached checksums and coverage are checked. The method supports1km maps but does not assume that a field is1km simply because its filename says so.

## Time and units

Time coordinates must decode to Gregorian/standard UTC datetime64. Use CF time bounds, a time_bounds variable name in the config, or explicit interval_hours and timestamp_position (start/end/midpoint). Missing time support is rejected rather than guessed from neighbouring records. No interval may exceed seven days or cross a Monday00:00 boundary. Overlapping/duplicate intervals across files are rejected. A calendar week is valid only when nonoverlapping source intervals cover all 604800 seconds and all required basin values are valid. Incomplete weeks are retained as missing, never filled.

- Precipitation/runoff `kind: amount`: `mm`, `m`, or `kg m-2` interval depths, then sum complete weeks.
- Precipitation/runoff `kind: rate`: supported units include `mm/day`, `mm/hour`, `mm/s`, `kg m-2 s-1`, `m/day`, `m/s`. Rates must represent interval means; multiply by interval seconds before weekly summation.
- Temperature `kind: mean`: `K` or `degC`; convert to°C and duration-average. Temperature maxima/minima are not treated as mean temperature.

NetCDF units, when present, must agree with the declared conversion. Finite negative precipitation/runoff in intersected cells fail source QC. Cumulative-since-initialization fields must be converted to true nonoverlapping interval amounts upstream; declaring them as per-interval amounts would double-count. The adapter does not disaggregate monthly totals or distribute a Wednesday-to-Wednesday weekly total across Monday weeks. If HEADWATER uses another weekly calendar, agree on that interface or retrain/evaluate an appropriately matched model.

## Context and forecasts

Inference needs target week plus 12 previous weeks for each site. A35-day product alone cannot provide all prior context. Optional history_csv and history_metadata fields append existing physical-unit weekly historical basin data with the same selected model's unit/runoff contract. History must have unique Monday rows, end before the forecast issue time, and not overlap map weeks. It must not contain future observed inputs. Missing historical context still leads to insufficient or invalid windows; there is no climatology filling.

The adapter checks declared forecast issue time and refuses forecast intervals beginning earlier than it. This is a useful consistency check, not proof of the data's provenance. Preserve each member separately and record actual lead/issue definitions. Bias and input-distribution differences between historical Daymet/NLDAS and HEADWATER still require hindcast evaluation.

## Outputs and limits

Outputs: weekly_inputs.csv, forcing_metadata.json, weekly_quality_by_region.csv, site_region_mapping.csv, sparse weights/checksums, and preprocessing_audit.json. Metadata includes selected model/runoff definition and source kind; audit includes source file hashes, configuration hash, unit/time processing, spatial coverage and missing-row count. `infer` uses the model-specific stored statics and train-only scalers. It does not recompute elevation from forecast maps or replace source-native soil units.

Actual HEADWATER maps have not yet been supplied. The integration is ready for the supported explicit contract and tested using synthetic daily and weekly maps, fractional cells, rates/temperature, missing data, projected/geographic coordinates and the complete52-site mapping. It is not a claim that unknown future HEADWATER file layouts or forecast accuracy are validated. The USGS reconstruction/proxy qualifications and fixed2019 descriptor vintage remain unchanged.

Implementation references: [xarray NetCDF/CF decoding](https://docs.xarray.dev/en/stable/generated/xarray.open_dataset.html), [Shapely geometry operations](https://shapely.readthedocs.io/en/stable/manual.html), and [pyproj coordinate transforms](https://pyproj4.github.io/pyproj/stable/api/transformer.html).

Archived source-region tables may retain earlier experiment columns such as forest_fraction_2001. They are provenance only; the active static feature list is exactly models.json and the selected checkpoint scaler list, using2019 land-cover descriptors for surface_runoff_static8.
