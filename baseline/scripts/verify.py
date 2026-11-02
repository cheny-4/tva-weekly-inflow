from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from model import SiteLSTM
from prepare_data import HEADWATERS
from utils import sha256,write_json,metrics

def verify(out,root):
    from train import load_data,evaluate
    out=Path(out);root=Path(root);config=json.loads((out/'run_config.json').read_text())
    for n,d in config['data_sha256'].items():assert sha256(root/'data'/n)==d,n
    for n,d in config['reference_sha256'].items():assert sha256(root/'references'/n)==d,n
    for n,d in config['code_sha256'].items():assert sha256(out/'code'/n)==d,n
    audit=json.loads((out/'preparation_audit.json').read_text())
    for p,d in audit['source_sha256'].items():assert sha256(p)==d,p
    hist=pd.read_csv(out/'training_history.csv');ck=torch.load(out/'best_model.pt',map_location='cpu',weights_only=True)
    assert ck['epoch']==int(hist.loc[hist.validation_loss.idxmin(),'epoch'])
    assert len(hist)==50 and hist.epoch.tolist()==list(range(1,51))
    assert config['history']==13 and config['early_stopping'] is False
    last=torch.load(out/'last_model.pt',map_location='cpu',weights_only=True);assert last['epoch']==50
    assert all('graph' not in k for k in ck['model_state'])
    model=SiteLSTM(config['layers']);model.load_state_dict(ck['model_state'])
    assert ck['model_state']['head_hidden_weight'].shape==(52,16,64)
    assert ck['model_state']['head_output_weight'].shape==(52,16)
    assert sum(p.numel() for p in model.parameters())==72884
    expected_used=.00001+.5*(.001-.00001)*(1+np.cos(np.pi*np.arange(50)/50))
    expected_next=.00001+.5*(.001-.00001)*(1+np.cos(np.pi*np.arange(1,51)/50))
    np.testing.assert_allclose(hist.learning_rate,expected_used,rtol=1e-10,atol=1e-14)
    np.testing.assert_allclose(hist.next_learning_rate,expected_next,rtol=1e-10,atol=1e-14)
    assert last['scheduler_state']['last_epoch']==50
    a,sc,tensors,samples=load_data(root/'data','cpu')
    with np.load(root/'data/weekly_dataset.npz') as z:original={k:z[k].copy() for k in z.files}
    history=np.zeros_like(original['eligible'])
    for i in range(12,len(history)):history[i]=np.isfinite(original['dynamic'][i-12:i+1]).all(axis=(0,2))
    np.testing.assert_array_equal(a['eligible'],original['eligible'])
    assert np.all(~a['eligible']|history) and np.all(~a['eligible']|a['target_valid_all7'])
    for split,code in [('train',0),('validation',1),('test',2)]:
        assert len(samples[split])==audit['counts'][split]
        assert (samples[split][:,0]>=12).all()
    # Independent CPU replay covers all saved test predictions and every output head.
    pred,_=evaluate(model,samples['test'],tensors);ti,bi=samples['test'].T
    replay=pd.DataFrame(dict(basin_id=a['basin_ids'][bi],start=a['starts'][ti],cpu_prediction=pred))
    saved=pd.read_csv(out/'test_paired_weekly_predictions.csv')
    joined=replay.merge(saved,on=['basin_id','start'],validate='one_to_one',how='outer',indicator=True)
    assert (joined['_merge']=='both').all()
    np.testing.assert_allclose(joined.cpu_prediction,joined.predicted_mean_mm_day,rtol=2e-4,atol=2e-5)
    scores=pd.read_csv(out/'metrics_by_basin.csv')
    for bid,g in saved.groupby('basin_id'):
        m=metrics(g.observed_volume_m3,g.predicted_volume_m3)
        row=scores[(scores.split=='test')&(scores.model=='lstm_nonlinear_heads_cosine')&(scores.basin_id==bid)].iloc[0]
        np.testing.assert_allclose([m['nse'],m['kge'],m['pbias_percent']],[row.nse,row.kge,row.pbias_percent],rtol=1e-10,atol=1e-10)
    # Recompute the headwater-only selection loss from the selected checkpoint.
    hix=np.array([a['basin_ids'].tolist().index(b) for b in HEADWATERS])
    val_samples=samples['validation'][np.isin(samples['validation'][:,1],hix)]
    _,val_loss=evaluate(model,val_samples,tensors)
    np.testing.assert_allclose(val_loss,ck['validation_loss'],rtol=2e-4,atol=2e-5)
    assert len(val_samples)==audit['counts']['validation_headwater10'] and len(saved)==audit['counts']['test']
    from PIL import Image
    pngs=list((out/'figures').glob('*.png'));pdfs=list((out/'figures').glob('*.pdf'))
    assert len(pngs)==16 and len(pdfs)==16
    for p in pngs:
        with Image.open(p) as im:im.verify()
    result=dict(status='passed',source_files_unchanged=True,original_input_target_masks_splits_and_scalers_byte_identical=True,history13_eligibility_unchanged=True,cosine_schedule_verified=True,epochs_completed=50,early_stopping=False,
        selected_validation_minimum=ck['epoch'],all_test_predictions_cpu_replayed=len(saved),
        max_replay_difference_mm_day=float(abs(joined.cpu_prediction-joined.predicted_mean_mm_day).max()),
        parameters=sum(p.numel() for p in model.parameters()),site_heads=52,gnn_layers=0,lstm_layers=config['layers'],metrics_recomputed=True,
        png_count=len(pngs),pdf_count=len(pdfs))
    write_json(out/'verification.json',result);print('VERIFICATION_PASSED '+json.dumps(result),flush=True)

