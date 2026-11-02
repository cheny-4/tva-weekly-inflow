TVA52 weekly LSTM: surface runoff plus eight RiTHyMs/StreamCat basin descriptors.
One LSTM layer64, 52 nonlinear site heads64->16->1, no GNN/attention/CAMELS.
Inputs: weekly P, T, NLDAS Qs; static elevation, permeability, sand%, clay%, bedrock depth, water-table depth, BFI%, 2019 riparian impervious%, 2019 open-water%.
Same masks, splits, 13-week history, seed42, 50 epochs, no early stopping and cosine LR as surface-only baseline.
2019 land cover explicitly requested for future use; retrospective test is not strict as-of-date. Statics are recomputed over the original forcing regions; processed RiTHyMs site values are not copied.
Source-native soil units retained without unverified conversions. Existing elevation/dynamic/target scaling exact; new statics standardized across sites with eligible training targets only.
Original initialization preserved; eight added static weight columns start at zero and are trainable.
Prepare data, run check_setup.py, submit once with scripts/submit_training.py. Reports compare surface-only and original total-runoff baseline on identical samples.
