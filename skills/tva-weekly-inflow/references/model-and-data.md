# Frozen baseline

Source: `tva_elf_headwater_final/lstm_1layer_52site_nonlinear_heads_weekly_cosine` on Frontier. Saved run: `weekly13_lstm1_nonlinear_cosine_s42_5751784`. `baseline/export_manifest.json` records original source paths and file checksums. Original scripts and their run snapshots remain unchanged.

One LSTM(4,64), 52 separate 64→16 ReLU→1 Softplus heads, dropout 0.2 before heads, 72,884 parameters. Inputs are the three dynamic features plus mean elevation repeated at each time step. No GNN, attention, CAMELS, observed-ELF lag or historical-median feature. All 52 sites have training examples; 51 have valid test targets.

The NPZ contains 1,816 calendar weeks × 52 sites × 3 dynamic features, one elevation per site, float32 training targets, float64 reference targets, original quality flags and eligibility. Site order comes from `basin_ids` and site_head_mapping.json. Keep all flags; in particular do not relax eligible merely because a target value is finite.

Chronological split: train 1982–2004, validation 2005–2009, test 2010–October 2016. Incomplete weeks crossing split boundaries have split -1. Counts: train 50,084, validation 9,868, test 15,025. Priority-headwater counts: 11,155, 2,297, 3,375. Some 2013 periods and long constant/invalid targets remain masked.

Dynamic inputs: weekly total precipitation, weekly mean temperature estimated historically from (Tmax+Tmin)/2, and weekly NLDAS surface runoff + baseflow. Feature units are mm/week, °C, mm/week. Training-only means and standard deviations normalize inputs and elevation. The original physical arrays are float32 before scaling. Model outputs are positive mean daily-equivalent target depth in mm/day, not standardized target values.

Loss is mean squared error of `(prediction - target)/(training site target std + 0.1)`. Target standard deviations control the loss only. Adam0.001, batch512, clip1, seed42, 50 epochs, cosine schedule T_max50/eta_min0.00001, no early stopping. Best validation loss over the ten priority headwaters selects epoch19. last_model.pt is epoch50 and should not silently replace best_model.pt.

Test medians from the saved results: all51 NSE0.8209257 and KGE0.8309070; headwater10 NSE0.8741982 and KGE0.8887924. These summarize valid weeks per site, followed by a median across sites. They are not percent accuracy, not medians over all calendar days, and not a fresh out-of-sample operational forecast assessment.

ELF drainage support depends on target definitions. Headwaters generally use their full contributing watershed. Downstream local/residual targets require matching incremental drainage support, and allocation/proxy groups have additional qualifications in node_metadata.csv. Do not treat all 52 outputs as total upstream flows or add their volumes indiscriminately.
