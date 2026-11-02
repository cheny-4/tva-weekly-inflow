---
name: tva-weekly-inflow
description: Run the saved TVA weekly shared LSTM with site-specific nonlinear heads, evaluate or plot its ELF predictions, and prepare authorized Frontier reruns. Use for this 52-site baseline and its weekly basin inputs.
---

# TVA weekly inflow

Use the existing trained baseline unless the user requests training. Locate the repository from `TVA_INFLOW_REPO` or the current checkout containing `pipeline.py` and `baseline/export_manifest.json`. If the skill is installed separately, ask for the checkout path when it cannot be found. Run commands from that repository with a Python environment containing its requirements.

Start with `python pipeline.py inspect` to verify the frozen artifacts. Read [model and data](references/model-and-data.md) before interpreting predictions or preparing inputs. Choose a new output directory outside `baseline/`.

## Route the task

- **New input inference:** Read [interface](references/interface.md). Confirm physical-unit weekly basin inputs and the runoff definition. Use `pipeline.py infer`. No ELF observations are needed. Do not substitute the target week's unavailable forcing with future observed weather or runoff.
- **Evaluate the existing model:** Use `pipeline.py replay --split test --out outputs/UNIQUE_NAME` for original masks/splits, or `--split validation`. The default command checks the saved checkpoint against its original rows and predictions. Use saved predictions directly when the user only requests plots.
- **Plot saved results:** Use `pipeline.py plot --predictions PATH --out outputs/UNIQUE_NAME`, optionally `--sites ELF13 ELF39`. Do not retrain or fill masked periods for a figure request.
- **Train or rerun:** Read [Frontier operations](references/frontier.md). Preserve the baseline settings unless changes are requested. Submitting a job must be within the user's training request. A request for preparation only ends with a reviewable launcher, without submission.

## Interpretation

The model shares learned parameters across sites but each prediction reads only that site's 13-week basin inputs and elevation. It has 52 trained heads and no GNN or attention. Existing IDs must retain their original head indices. The input window is target week + 12 prior weeks, not a 13-week forecast lead.

The target is each site's ELF quantity; downstream targets may represent local/incremental contributions, not total upstream inflow. Drainage area converts the output to volume. Do not sum or route different sites' ELF volumes without checking definitions and overlap.

Keep historical-input reconstruction separate from operational forecasts. Report which split, valid weeks and site set a metric uses. The ten-headwater test median NSE is 0.8742, while the 51-site median is 0.8209. Validation selected the saved epoch-19 checkpoint after all 50 epochs ran. Do not use test results to refit scalers, calibrations, or checkpoint selection.
