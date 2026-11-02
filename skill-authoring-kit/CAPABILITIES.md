# Capabilities and permissions

| Capability | Operation | Side effects and scope |
|---|---|---|
| Inspect | Verify snapshot SHA-256 and report model/data settings | Read-only, no training |
| Preprocess | Convert configured HEADWATER rectilinear NetCDF maps into weekly basin inputs | Write new forcings/QC/weights; CPU, no training |
| Infer | Use the saved checkpoint on weekly basin CSV inputs | Write a new local result directory, CPU by default |
| Replay | Reproduce validation/test predictions using the frozen original masks | Write predictions and accuracy metrics into a new directory |
| Plot | Create per-site hydrographs and all-site NSE from paired saved predictions | Write PNG figures and metrics, no model training |
| Train | Fresh 50-epoch reproduction with unchanged data/settings | Requires a Slurm GPU allocation and a user request that authorizes training |

Normal inference/evaluation requests permit their required local reads, computation and new outputs. They do not imply training, modifying model weights, making the repository public, emailing results, or sharing data externally. A direct user request to train/rerun authorizes the necessary job within that scope; do not turn this document into a redundant permission question. If the user only requests a prepared command or plan, leave the launcher unsubmitted.

Preserve `baseline/`, `baseline_surface_runoff_static8/` and `spatial/`. Use a unique output directory and stop on an existing path, failed data contract, checksum mismatch or incompatible runoff definition. Resolve the concrete issue before trying again. Do not replace invalid inputs with guessed values or refit normalization against test/future data.

No secret is required for local inference. Downloading additional NASA data, cloning this private repository or accessing Frontier may need the user's normal service authentication. Never place credentials in repository files, arguments, skill documents or run reports. Current repository access and user authorization govern sharing, not these capability descriptions.

Training settings reproduce an existing research baseline. The package offers no automatic hyperparameter search, scheduled jobs, external data download, operational forecast validation or automatic handling of unsupported raw map layouts. Site-specific heads have no defined weights for new locations.

Select --model explicitly: total_runoff or surface_runoff_static8. Neither uses bias correction. Read ../docs/headwater_preprocessing.md before configuring real maps; one forecast issue/member is processed at a time.
