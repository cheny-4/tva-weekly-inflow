"""Record completed results and Slurm resource use without changing the run."""
from pathlib import Path
import hashlib,json,shutil,subprocess
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
state=json.loads((ROOT/'STATUS.json').read_text());assert state['status']=='completed' and state['epochs_run']==50
run=Path(state['run']);verification=json.loads((run/'verification.json').read_text());assert verification['status']=='passed'
result=json.loads((run/'results_summary.json').read_text());summary=json.loads((run/'headwater10_comparison_summary.json').read_text())
config=json.loads((run/'run_config.json').read_text())
assert config['history']==13 and config['parameters']==72884 and config['early_stopping'] is False
assert summary['paired_test_weeks']==3375
r=subprocess.run(['sacct','-j',str(state['job_id']),'--format=JobID,JobName,Partition,State,Elapsed,AllocCPUS,AllocTRES,MaxRSS,ExitCode','--parsable2'],text=True,capture_output=True,check=True)
assert 'COMPLETED' in r.stdout and '0:0' in r.stdout
(ROOT/'slurm_accounting.txt').write_text(r.stdout)
for name in ['README.txt','deployment_sha256.json']:
    backup=ROOT/'logs'/('initial_'+name)
    if not backup.exists():shutil.copy2(ROOT/name,backup)
text='TVA-only shared one-layer LSTM, 52 nonlinear site heads, 13-week history.\n\n'
text+=f'Status: COMPLETED and independently verified. Job {state["job_id"]}.\nAll50 epochs completed with no early stopping. Best validation epoch: {state["best_epoch"]}.\n'
text+=f'Training+validation: {result["training_validation_seconds"]:.1f} seconds.\n\nRESULTS\n{run}\n'
text+=f'Headwater test site-weeks: {summary["paired_test_weeks"]}; improved NSE: {summary["improved_nse"]}/10.\n'
text+=f'Median NSE: {summary["median_nse_old"]:.6f} -> {summary["median_nse_new"]:.6f}.\nMedian KGE: {summary["median_kge_old"]:.6f} -> {summary["median_kge_new"]:.6f}.\nMedian absolute volume bias: {summary["median_abs_bias_old"]:.3f}% -> {summary["median_abs_bias_new"]:.3f}%.\n\n'
text+='SETTINGS\nOne LSTM layer, hidden64; per-site64->16 ReLU->1 Softplus; dropout0.2 before heads; 72,884 parameters.\nAdam0.001 with fixed cosine decay over50 epochs to0.00001; no restarts or early stopping. Seed42, batch512, clip1.\n13 weeks = target week plus12 preceding weeks. No ELF observations supplied as inputs. No GNN or CAMELS.\n\n'
text+='DATA\nOriginal data, ELF quality masks, 13-week eligibility, scalers and date splits are copied byte-for-byte from the original TVA52 one-layer model. No gap filling. Train1982-2004; validation2005-2009; test2010-October2016.\n\n'
text+='OUTPUTS\nThe run folder contains REPORT.md, per-site comparisons, saved predictions, best_model.pt, last_model.pt(epoch50), verification.json and16 figures each in PNG/PDF.\n\n'
text+='INTERPRETATION\nCheckpoint selection uses only ten-headwater validation loss; all original eligible test weeks match the baseline exactly. Head architecture, learning-rate schedule and stopping rule changed together, so individual effects are not isolated. Single-seed historical-input reconstruction, not operational forecast skill.\n'
(ROOT/'README.txt').write_text(text)
manifest=json.loads((ROOT/'deployment_sha256.json').read_text());manifest['README.txt']=hashlib.sha256((ROOT/'README.txt').read_bytes()).hexdigest()
(ROOT/'deployment_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
for name,digest in manifest.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
print('FINAL_RESULTS_RECORDED',json.dumps(summary),flush=True)
print(pd.read_csv(run/'test_headwater10_comparison.csv')[['basin_id','name','n','nse_old','nse_new','nse_change','pbias_percent_new']].to_string(index=False),flush=True)
print('EPOCH_AND_TIME',state['best_epoch'],result['training_validation_seconds'],flush=True)
print(r.stdout,flush=True)

