"""Submit this prepared experiment once; preserve the Slurm receipt."""
from pathlib import Path
import datetime,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
assert json.loads((ROOT/'STATUS.json').read_text())['status']=='prepared'
assert json.loads((ROOT/'preparation_audit.json').read_text())['status']=='passed'
assert not (ROOT/'job.json').exists(),'A submission receipt already exists'
r=subprocess.run(['sbatch','--parsable','train.slurm'],cwd=ROOT,text=True,capture_output=True,check=True)
job=r.stdout.strip().split(';')[0];assert job.isdigit()
receipt=dict(status='submitted',job_id=job,folder=str(ROOT),submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),history_weeks=13,nonlinear_heads=True,learning_rate_schedule="cosine",early_stopping=False,epochs=50)
(ROOT/'job.json').write_text(json.dumps(receipt,indent=2)+'\n')
(ROOT/'STATUS.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('TRAINING_SUBMITTED',json.dumps(receipt),flush=True)

