# Data contract

## Two model identities

Use models.json and --model consistently:

| Model | Dynamic feature order | Static feature order |
|---|---|---|
| total_runoff | precipitation_mm_week,temperature_mean_c,total_runoff_mm_week | elevation_mean_m |
| surface_runoff_static8 | precipitation_mm_week,temperature_mean_c,surface_runoff_mm_week | elevation_mean_m,soil_permeability_native,soil_sand_percent,soil_clay_percent,depth_to_bedrock_native,depth_to_water_table_native,baseflow_index_percent,impervious_percent_2019_riparian100m,open_water_percent_2019 |

P/runoff are basin spatial mean weekly amounts in mm/week; T is basin mean weekly mean in°C, historically Daymet(Tmax+Tmin)/2. Runoff metadata must declare surface_runoff_plus_baseflow or surface_runoff_only respectively. Static values and scalers are stored per version and loaded automatically; no bias correction is used. Surface has eight **additional** statics plus elevation. Static BFI is not dynamic baseflow. Native permeability/depth units and 2019 land-cover/riparian definitions remain as documented in the static audit.

## Spatial support and map adapter

spatial/forcing_regions.geojson contains 43 exact frozen regions; site_region_mapping.csv maps all 52 head IDs to them and retains target areas, construction classes and review flags. Their manifest checksums are verified by preprocessing. Headwaters generally use full contributing basins; local downstream targets use incremental support; source/proxy groups sometimes share forcings. These are provisional USGS reconstructions, not original validated TVA GIS. Do not replace source regions with total-upstream catchments or change target areas to match forcing-region area.

pipeline.py preprocess implements rectilinear NetCDF ingestion, fractional polygon/cell intersection areas measured in EPSG:5070, finite-area coverage checks, unit conversions and complete Monday–Sunday aggregation. It preserves holes/edge fractions, saves reusable grid weights/fingerprints and emits ready-to-infer weekly CSV/metadata. Read ../docs/headwater_preprocessing.md for supported CRS, time intervals, dimensions, members/issues and limitations. An unknown actual HEADWATER layout is not automatically validated.

Amounts are summed over complete weeks, interval-mean P/runoff rates are integrated, and interval-mean temperatures are duration-averaged. Inputs must have explicit time support, unique nonoverlapping intervals inside their Monday week, and a consistent UTC calendar. Monthly/misaligned weekly totals and cumulative-since-init fields need an upstream interface change rather than guessed disaggregation. Spatial coverage defaults are 0.99 and recorded. Insufficient finite area/time stays missing; no values are filled.

## Identity, scaling and prediction

Preserve ELF01–ELF52 order and original head indices from site_head_mapping.json and checkpoint basin_ids. New sites need model adaptation. Both frozen NPZs have 1816×52×3 dynamics; statics are 52×1 or52×9. Inputs enter as float32 physical values and the original training-only scaler computation returns normalized float32. Do not fit scalers from forecast/validation/test data.

Each 13-week window includes the target week plus 12 preceding weeks. No observed ELF enters inference. Stored statics are concatenated at every step. Output is physical mean daily-equivalent ELF depth mm/day, not a standardized target. Weekly m³ = depth×stored target area_km2× 7000. Site training target std+.1 is used only in loss.

ELF is site-specific local/incremental/allocated reference flow, not universally total upstream reservoir inflow. Do not sum overlapping site volumes indiscriminately. Frozen output heads cannot represent arbitrary new sites.

## Splits and masks

Train1982–2004,validation2005–2009,test2010–October2016. Preserve eligible,target_valid_all7,all QC bits and inherited graph_history13_valid. Legacy graph naming does not imply a GNN. Use eligible AND split_code, retaining50,084/9,868/15,025 train/validation/test rows.52 sites train;51 have test scores. Invalid constant/small-value periods and incomplete weeks remain excluded. Gaps remain gaps in plots; volume scores require explicitly complete or matched valid periods.

For new inference, require known IDs, unique Monday rows and finite contiguous 13-week forcing windows. Reject incompatible declared units/definitions and negative P/runoff. Flag missing/non-finite windows without imputation. No reference target mask is required for inference, but no accuracy score is available without independent paired observations.

## Forecast context

Forecast inputs require the target week's forecast fields as well as past context. Thirteen weeks is not forecast lead. Process each member/issue separately; record issue_time/member_id and preserve actual forecast lead definitions. Optional historical context must match the model contract and end before issue time, without overlapping forecast-map weeks or using future observations. Metadata cannot prove the truth of the supplied source labels.

Historical Daymet/NLDAS reconstruction scores do not establish HEADWATER/GEFS forecast skill or remove distribution differences. User-requested fixed2019 surface descriptors are post-period information in the retrospective test. Test data have already been examined in many experiments; no tuning, calibration or checkpoint selection may be fitted to their labels.
