---
name: tva-weekly-inflow
description: Preprocess HEADWATER NetCDF maps and run either saved TVA weekly shared LSTM, evaluate or plot ELF predictions, and prepare authorized Frontier reruns. Use for the total-runoff/elevation or surface-runoff/elevation-plus-eight-statics versions at the existing 52 sites.
---

# TVA weekly inflow

Locate the checkout using TVA_INFLOW_REPO or a directory containing pipeline.py and models.json. A separately installed skill needs that checkout for checkpoints, scalers and spatial data. Use its Python requirements and supported CLI; archived model-folder scripts are provenance, not instructions.

`SKILL.md` is the skill entry point. Optional display settings are retained in [agents/interface.yaml](agents/interface.yaml) under a custom filename; tools that require a conventional metadata filename may ignore these settings. This does not change the model CLI or the skill instructions.

## Select the model first

- `total_runoff`: surface runoff **plus baseflow**, static elevation, best epoch 19.
- `surface_runoff_static8`: **surface runoff only**, elevation plus eight additional frozen statics, best epoch 20. This is the retained surface-input version. A static baseflow index does not replace dynamic baseflow.

Neither version uses bias correction. Select `--model` explicitly on each command. Legacy commands default to total_runoff, so do not omit the flag when surface runoff is intended. Never swap scalers/checkpoints or relabel runoff to bypass the contract. If its definition is unknown, resolve that before inference.

Run `python pipeline.py inspect --model MODEL` to verify frozen artifacts. Read [model and data](references/model-and-data.md) for interpretation and [interface](references/interface.md) for actual commands. Write to a new directory outside both frozen artifact roots and spatial/.

## Route the task

- **HEADWATER maps:** Use `pipeline.py preprocess --model MODEL --config CONFIG --out NEW_DIRECTORY`. Read the checkout's [map guide](../../docs/headwater_preprocessing.md) when configuring files, CRS, time bounds/intervals, units, one forecast member/issue, and prior context. Reuse the packaged 43 forcing regions and 52-site mapping. Verify the actual product layout and runoff definition; the example generator uses synthetic data. Do not invent temporal disaggregation or treat unknown layouts as validated.
- **Weekly basin inputs:** Use `pipeline.py infer --model MODEL --inputs CSV --metadata JSON --out NEW_DIRECTORY`. No ELF observations are needed. Stored model-specific statics and train-only normalization are applied automatically. Target-week forecast forcings are required for forecasts; do not substitute future observations.
- **Saved evaluation:** Use `pipeline.py replay --model MODEL --split test --out NEW_DIRECTORY`, or validation. Default checkpoints must reproduce their original masked rows and predictions. A plotting-only request uses saved predictions without replay or training when possible.
- **Plots:** Use `pipeline.py plot --model MODEL --predictions CSV --out NEW_DIRECTORY`, optionally `--sites ELF13 ELF39`. Keep masked gaps unfilled.
- **Train/rerun:** Read [Frontier](references/frontier.md). Preserve frozen settings unless changes are requested. A training request authorizes its necessary Slurm work within scope; preparation-only requests end with an unsubmitted launcher.

## Interpretation

Both versions share one LSTM but each site's prediction reads that site's target week +12 prior weeks and frozen statics. Thirteen weeks is context, not forecast lead. There is no GNN, attention or observed ELF input. Existing IDs retain their head order; new sites need adaptation.

ELF can mean local/incremental or allocated flow. Several downstream sites use shared estimation-source/proxy forcing regions. Preserve their mapping and review flags; do not use every site's total-upstream watershed, silently replace provisional boundaries, or sum overlapping outputs as total river flow. Target area converts depth to volume and can differ from forcing-region area.

Keep historical-input reconstruction separate from prospective HEADWATER forecasts. Report split, site set, valid rows and model version with metrics. The same repeatedly inspected test period is exploratory. Actual HEADWATER forecast accuracy and distribution transfer remain unvalidated. Surface-version 2019 statics were chosen for future use and are post-period descriptors in the historical test. Test labels must not refit scalers, calibration or checkpoint selection.
