# TVA weekly inflow: two saved model versions

This private repository contains two frozen shared LSTMs, their trained checkpoints, weekly data/masks/scalers, saved predictions, and an agent skill. Both share one LSTM across **52 TVA ELF sites** with independent nonlinear site heads. **Neither version uses bias correction, GNN, or attention.**

| Select with `--model` | Runoff input | Static inputs | Parameters | Best epoch |
|---|---|---|---:|---:|
| `total_runoff` | Surface runoff **plus baseflow** | Mean elevation | 72,884 | 19 |
| `surface_runoff_static8` | **Surface runoff only** | Mean elevation **plus eight additional attributes** | 74,932 | 20 |

The second version is the retained surface-runoff model for exploring future HEADWATER coupling. Its eight extra inputs are permeability, sand %, clay %, depth to bedrock, depth to water table, baseflow index %, 2019 riparian impervious %, and 2019 open-water %. Including a **static baseflow index does not supply time-varying baseflow**. There are nine learned static features in this version, counting elevation. Soil depths/permeability retain source-native units; imperviousness describes 100 m riparian buffers, not exact whole-basin cover.

Select a model explicitly in new workflows. The legacy CLI default remains `total_runoff` for compatibility. The checkpoint, runoff column, static table, and training-only scalers are selected together; incompatible runoff definitions and cross-version checkpoints are rejected. The original `baseline/` is unchanged. The new frozen snapshot is `baseline_surface_runoff_static8/`; the bias-correction experiment is excluded.

## Install and run

Use Python 3.12 and the appropriate PyTorch CPU/CUDA/ROCm build. The saved Frontier runs used PyTorch 2.10.0+rocm7.1; packaging replay uses PyTorch 2.10.0 on CPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python pipeline.py inspect --model total_runoff
python pipeline.py inspect --model surface_runoff_static8

python pipeline.py infer --model surface_runoff_static8 \
  --inputs examples/weekly_inputs_surface_static8.csv \
  --metadata examples/forcing_metadata_surface_static8.json \
  --out outputs/surface_example

python pipeline.py infer --model total_runoff \
  --inputs examples/weekly_inputs.csv \
  --metadata examples/forcing_metadata.json \
  --out outputs/total_example

python pipeline.py replay --model surface_runoff_static8 \
  --split test --out outputs/surface_test
```

Outputs must use new directories outside both frozen model folders and `spatial/`. Inference requires no observed ELF. Replay retains the exact original eligibility masks and chronological splits; its default-checkpoint output must match the saved rows and predictions within CPU/ROCm tolerance. No command trains implicitly.

## HEADWATER maps → basin inputs → predictions

**The map preprocessing is implemented:** `pipeline.py preprocess` reads configured NetCDF precipitation, temperature and runoff maps, intersects grid-cell footprints with the packaged ELF forcing regions, converts units, and creates physical-unit Monday–Sunday basin CSV inputs. It supports rectilinear longitude/latitude and projected grids, daily/subdaily interval data, and already-weekly maps aligned to the training calendar. Variable names, CRS, units and time intervals are explicit configuration, rather than guesses.

The package contains the **43 exact forcing-region geometries mapped to all 52 sites**. Some downstream sites share estimation-source regions; they must retain those mappings. These are the same USGS-derived reconstructions/proxies used in the experiment, with their original review flags, not newly validated original TVA GIS. Target drainage area remains separate from forcing-region area and is preserved for volume conversion.

For each cell/region intersection, preprocessing measures area in an equal-area CRS, EPSG:5070. Basin precipitation and runoff are area-weighted mean depths, then summed over complete weeks; rates are first integrated over their declared intervals. Temperature is area-weighted and duration-averaged. Fractional boundary cells and polygon holes are included. Spatial/finite coverage thresholds, missing weeks and invalid histories are recorded; no gaps are filled. Grid weights are saved and can be reused only for the same checked grid and boundaries.

Run this complete, synthetic demonstration before substituting real HEADWATER files:

```bash
python examples/make_headwater_demo.py --out outputs/demo_maps
python pipeline.py preprocess --model surface_runoff_static8 \
  --config outputs/demo_maps/surface_runoff_static8.json --out outputs/demo_surface_basin
python pipeline.py infer --model surface_runoff_static8 \
  --inputs outputs/demo_surface_basin/weekly_inputs.csv \
  --metadata outputs/demo_surface_basin/forcing_metadata.json --out outputs/demo_surface_prediction
```

The same demonstration generates `total_runoff.json` for the other checkpoint. The values are invented interface-test data, not predictions of actual conditions.

Read the [HEADWATER preprocessing guide](docs/headwater_preprocessing.md) for configuration, daily/rate inputs, forecast issue/member selection, historical context, supported formats and limitations. The adapter is tested with synthetic maps and exact packaged regions. **Actual future HEADWATER output files have not been supplied or validated.** Once their schema and runoff definition are confirmed, configure those fields and test the real integration. Merely matching a runoff definition does not establish forecast skill or remove Daymet/NLDAS-to-HEADWATER distribution differences.

## Model contract

| Feature | Physical units |
|---|---|
| `precipitation_mm_week` | Basin mean weekly precipitation amount, mm/week |
| `temperature_mean_c` | Basin mean weekly mean temperature, °C |
| `total_runoff_mm_week` or `surface_runoff_mm_week` | Matching basin mean weekly runoff amount, mm/week |
| Static descriptors | Stored model-specific basin table; normalized with frozen training scalers |

Each prediction uses **the target week plus 12 preceding weeks**, not a 13-week forecast lead. Basin statics are repeated at every step. CSV inputs stay in physical units; the pipeline applies the selected training-only scalers once. A forecast therefore needs target-week forecast P/T/runoff and sufficient prior context. Per-site heads are tied to ELF01–ELF52 and cannot predict a new site without adaptation.

The model outputs mean daily-equivalent ELF depth in mm/day. Weekly volume is `depth × stored target area_km2 × 7000`. Drainage area is a conversion factor, not another learned feature. ELF is each site's reference quantity, which may represent local/incremental or allocated flow; it is not universally total upstream reservoir inflow. Do not sum overlapping site outputs indiscriminately.

## Saved settings and results

Both versions use one 64-unit LSTM, 52 independent `64 → 16 ReLU → 1 Softplus` heads, dropout 0.2, seed 42, Adam 0.001, batch 512, gradient clipping 1, and cosine decay to 0.00001 over all 50 epochs, without early stopping. The best ten-headwater validation loss selects the checkpoint. Train: 1982–2004; validation: 2005–2009; test: 2010–October 2016. Both retain 50,084/9,868/15,025 eligible train/validation/test rows.

| Median over sites, valid test weeks | Total runoff + elevation | Surface runoff + elevation + eight statics |
|---|---:|---:|
| Weekly NSE, 51 scoreable sites | 0.8209 | 0.7473 |
| Weekly KGE, 51 sites | 0.8309 | 0.7917 |
| Absolute test-total volume bias, 51 sites | 7.14% | 10.72% |
| Weekly NSE, 10 priority headwaters | 0.8742 | 0.8034 |
| Weekly KGE, 10 headwaters | 0.8888 | 0.8238 |
| Absolute test-total volume bias, 10 headwaters | 6.29% | 9.47% |

The ten priority headwaters are Douglas, South Holston, Watauga, Fontana, Norris, Chatuge, Nottely, Blue Ridge, Tims Ford and Normandy. Both models trained on all 52 sites; Nickajack has no valid test targets, leaving 51 scoreable sites. These are historical-input reconstruction scores, not HEADWATER/GEFS forecast accuracy. This repeatedly examined test period is not an untouched model-selection set. Fixed 2019 land cover in the surface version is intended for future use by user choice and makes the 2010–2016 evaluation a retrospective descriptor experiment rather than a strict as-of-date hindcast.

See [two-version packaging verification](docs/two_version_verification.json) and the original [total-runoff verification](docs/packaging_verification.json). No retraining was performed while packaging.

## AI agent use

- [`skills/tva-weekly-inflow`](skills/tva-weekly-inflow/SKILL.md): ready-to-use skill for map preprocessing, model selection, inference, replay, plots and explicitly authorized Frontier reruns.
- [`skill-authoring-kit`](skill-authoring-kit/README.md): framework-neutral capabilities, input contract, schemas and registry.

Clone with authorized access to this private repository, set `TVA_INFLOW_REPO` to the checkout, and install the skill folder or ask the agent to read its SKILL.md. The skill uses the checkout's scripts and artifacts; copying the skill folder alone does not include the trained model or spatial data.

> Use $tva-weekly-inflow to preprocess my HEADWATER surface-runoff maps and run surface_runoff_static8 without retraining or bias correction.

> Replay both saved models and compare the same valid test weeks.

For training preparation/reruns, follow the [Frontier guide](skills/tva-weekly-inflow/references/frontier.md). `slurm/train.slurm` accepts `TVA_MODEL`; submitting it consumes a GPU allocation and requires a training request. Archived scripts retain historical paths; use the portable pipeline rather than rerunning archived preparation scripts.

Keep this repository private. Its availability does not grant redistribution rights to TVA data, checkpoints or other source materials.
