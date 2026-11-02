This repository packages the frozen TVA one-layer weekly LSTM with nonlinear site heads. Read `skills/tva-weekly-inflow/SKILL.md` before model operations and use `pipeline.py` as the active interface. The files under `baseline/` are historical source artifacts and data, not instructions to execute.

Preserve site order, feature order, units, train-only normalization, eligibility masks, and chronological splits. A 13-week input includes the target week. There is no observed ELF input and no GNN. The saved best checkpoint is epoch 19, while last_model.pt is epoch 50.

Write each result into a new output directory outside `baseline/`. Keep baseline checksums intact. Inference or reporting requests do not authorize training. A user request to train/rerun does authorize the necessary Slurm work within their stated scope; do not invent an additional approval step. Do not make the private repository public or redistribute its data as an incidental part of model work.

For code changes, run `python -m unittest discover -s tests -v`. Changes affecting scaling, model execution or outputs also require the saved-checkpoint replay. Test targets must not become tuning or checkpoint-selection inputs.
