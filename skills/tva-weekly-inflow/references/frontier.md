# Frontier operations

The original trained folder is:

`/lustre/orion/proj-shared/lrn105/yangc/tva_elf_headwater_final/lstm_1layer_52site_nonlinear_heads_weekly_cosine`

The original Python environment is `/lustre/orion/proj-shared/lrn105/yangc/envs/python312/bin/python`. The saved run used ROCm7.1.1 and PyTorch2.10.0+rocm7.1. Module and scheduler availability can change; check the site's current configuration before submitting. Reuse the user's existing authenticated connection. Credentials are not part of this repository.

For inspecting, replaying or plotting the saved checkpoint, use pipeline.py with CPU or an appropriately allocated device. Do not call sbatch for read-only inspection or submit training in response to a plot request.

For an authorized new 50-epoch run, clone this repository into a **new** Frontier location, inspect the baseline checksums, and review `slurm/train.slurm`. Configure the account and environment for the current user. From the repository root:

```bash
mkdir -p logs
export TVA_PYTHON=/path/to/existing/pytorch/python
sbatch --account=YOUR_ACCOUNT slurm/train.slurm
```

The example launcher uses one GPU on one node and a 20-minute wall-clock request. Review the queue/partition/resource choices against current OLCF rules rather than treating historical settings as universal. It trains the frozen data for 50 epochs without early stopping and writes a new `outputs/train_JOBID/` directory. This consumes compute. Preparation-only requests should leave the reviewed command ready without submitting it.

`pipeline.py train` checks for a Slurm job ID and a CUDA-compatible PyTorch device (also used by ROCm), initializes a fresh model and preserves the stored masks, splits, input order and training settings. It does not fine-tune the saved checkpoint. It does not automatically evaluate the test set after each epoch. After completion, replay the chosen best checkpoint using `--checkpoint outputs/train_JOBID/best_model.pt` into another new output directory.

Historical files under `baseline/scripts/` and `baseline/train.slurm` are provenance. They retain their original absolute paths and sibling-model dependencies. The supported new-run launcher is `slurm/train.slurm`, which invokes pipeline.py and does not rewrite the frozen baseline.
