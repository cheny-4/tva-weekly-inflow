TVA-only shared one-layer LSTM, 52 nonlinear site heads, 13-week history.

Status: COMPLETED and independently verified. Job 5751784.
All50 epochs completed with no early stopping. Best validation epoch: 19.
Training+validation: 27.4 seconds.

RESULTS
/lustre/orion/lrn105/proj-shared/yangc/tva_elf_headwater_final/lstm_1layer_52site_nonlinear_heads_weekly_cosine/runs/weekly13_lstm1_nonlinear_cosine_s42_5751784
Headwater test site-weeks: 3375; improved NSE: 6/10.
Median NSE: 0.849501 -> 0.874198.
Median KGE: 0.858941 -> 0.888792.
Median absolute volume bias: 7.617% -> 6.295%.

SETTINGS
One LSTM layer, hidden64; per-site64->16 ReLU->1 Softplus; dropout0.2 before heads; 72,884 parameters.
Adam0.001 with fixed cosine decay over50 epochs to0.00001; no restarts or early stopping. Seed42, batch512, clip1.
13 weeks = target week plus12 preceding weeks. No ELF observations supplied as inputs. No GNN or CAMELS.

DATA
Original data, ELF quality masks, 13-week eligibility, scalers and date splits are copied byte-for-byte from the original TVA52 one-layer model. No gap filling. Train1982-2004; validation2005-2009; test2010-October2016.

OUTPUTS
The run folder contains REPORT.md, per-site comparisons, saved predictions, best_model.pt, last_model.pt(epoch50), verification.json and16 figures each in PNG/PDF.

INTERPRETATION
Checkpoint selection uses only ten-headwater validation loss; all original eligible test weeks match the baseline exactly. Head architecture, learning-rate schedule and stopping rule changed together, so individual effects are not isolated. Single-seed historical-input reconstruction, not operational forecast skill.
