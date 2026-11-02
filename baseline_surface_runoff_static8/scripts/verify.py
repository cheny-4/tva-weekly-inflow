from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from model import SiteLSTM
from prepare_data import HEADWATERS,BASELINE,FIELDS
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
    assert sum(p.numel() for p in model.parameters())==74932
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
        row=scores[(scores.split=='test')&(scores.model=='lstm_surface_static8')&(scores.basin_id==bid)].iloc[0]
        np.testing.assert_allclose([m['nse'],m['kge'],m['pbias_percent']],[row.nse,row.kge,row.pbias_percent],rtol=1e-10,atol=1e-10)
    # Recompute the headwater-only selection loss from the selected checkpoint.
    hix=np.array([a['basin_ids'].tolist().index(b) for b in HEADWATERS])
    val_samples=samples['validation'][np.isin(samples['validation'][:,1],hix)]
    _,val_loss=evaluate(model,val_samples,tensors)
    np.testing.assert_allclose(val_loss,ck['validation_loss'],rtol=2e-4,atol=2e-5)
    assert len(val_samples)==audit['counts']['validation_headwater10'] and len(saved)==audit['counts']['test']
    # Every prior array stays exact; independently recompute the eight static aggregates.
    with np.load(BASELINE/'data/weekly_dataset.npz') as z:prior={k:z[k].copy() for k in z.files}
    for key in prior:
        np.testing.assert_array_equal(a[key][:,:1] if key=='static' else a[key],prior[key])
    assert a['static'].shape==(52,9) and model.lstm.input_size==12
    for name,digest in audit['static_preparation_sha256'].items():assert sha256(root/'static_preparation'/name)==digest,name
    assert sha256(root/'sources/streamcat_selected_catchments.parquet')==audit['source_subset_sha256']
    weights=pd.read_csv(root/'static_preparation/forest_catchment_intersection_weights.csv')
    raw=pd.read_parquet(root/'sources/streamcat_selected_catchments.parquet')
    mapping=pd.read_csv(root/'static_preparation/source_region_mapping_all52.csv').set_index('elf_id')
    merged=weights.merge(raw,on='comid',validate='many_to_one')
    assert len(merged)==len(weights)
    independently={}
    for region,g in merged.groupby('source_region_id'):
        values=[]
        for name,field in FIELDS.items():
            x=g[field].to_numpy(float);w=g.intersection_area_m2.to_numpy(float)
            good=np.isfinite(x)&(x>=0)
            if 'percent' in name:good &= x<=100
            assert w[good].sum()/w.sum()>.99
            values.append(np.average(x[good],weights=w[good]))
        independently[region]=values
    expected=np.array([independently[mapping.loc[bid,'source_region_id']] for bid in a['basin_ids']],dtype='float32')
    np.testing.assert_allclose(a['static'][:,1:],expected,rtol=1e-6,atol=1e-6)
    oldsc=json.loads((BASELINE/'data/train_only_scalers.json').read_text())
    for key in oldsc:
        if key in ['static_names','static_mean','static_std']:assert sc[key][:1]==oldsc[key]
        else:assert sc[key]==oldsc[key],key
    train_sites=(a['eligible']&(a['split'][:,None]==0)).any(0)
    values=a['static'][train_sites,1:].astype(float)
    np.testing.assert_allclose(sc['static_mean'][1:],values.mean(0),rtol=1e-12)
    np.testing.assert_allclose(sc['static_std'][1:],values.std(0),rtol=1e-12)
    assert sc['static_names'][1:]==list(FIELDS)
    assert audit['static_audit']['landcover_vintage']==2019
    # Every comparison uses exactly the original observed target and site/week pairs.
    roundtrip_differences={}
    for split in ['validation','test']:
        new=pd.read_csv(out/f'{split}_paired_weekly_predictions.csv')
        old=pd.read_csv(Path(audit['baseline_run'])/f'{split}_paired_weekly_predictions.csv')
        j=new.merge(old,on=['basin_id','start','end','days'],how='outer',validate='one_to_one',indicator=True,suffixes=('','_original'))
        assert (j['_merge']=='both').all()
        # CSV float64 parse/write round trips can differ by one floating-point ULP.
        # Frozen targets and reference files were checked exactly above.
        np.testing.assert_allclose(j.observed_volume_m3,j.observed_volume_m3_original,rtol=1e-12,atol=1e-5)
        np.testing.assert_allclose(j.history13_volume_m3,j.predicted_volume_m3_original,rtol=1e-12,atol=1e-5)
        total=pd.read_csv(root/'references'/f'total_{split}_paired_weekly_predictions.csv')
        t=new.merge(total,on=['basin_id','start','end','days'],validate='one_to_one',suffixes=('','_original'))
        assert len(t)==len(new)
        np.testing.assert_allclose(t.total_runoff_volume_m3,t.predicted_volume_m3_original,rtol=1e-12,atol=1e-5)
        roundtrip_differences[split]=dict(observed_max_abs_m3=float(abs(j.observed_volume_m3-j.observed_volume_m3_original).max()),baseline_max_abs_m3=float(abs(j.history13_volume_m3-j.predicted_volume_m3_original).max()))
    from PIL import Image
    pngs=list((out/'figures').glob('*.png'));pdfs=list((out/'figures').glob('*.pdf'))
    assert len(pngs)==22 and len(pdfs)==22
    for p in pngs:
        with Image.open(p) as im:im.verify()
    result=dict(status='passed',source_files_unchanged=True,original_dynamic_elevation_targets_masks_splits_and_eligibility_unchanged=True,eight_static_features_and_their_scalers_added=True,static_attributes_independently_recomputed=True,landcover_vintage=2019,history13_eligibility_unchanged=True,cosine_schedule_verified=True,epochs_completed=50,early_stopping=False,
        selected_validation_minimum=ck['epoch'],all_test_predictions_cpu_replayed=len(saved),
        max_replay_difference_mm_day=float(abs(joined.cpu_prediction-joined.predicted_mean_mm_day).max()),
        parameters=sum(p.numel() for p in model.parameters()),site_heads=52,gnn_layers=0,lstm_layers=config['layers'],metrics_recomputed=True,
        png_count=len(pngs),pdf_count=len(pdfs),csv_roundtrip_differences=roundtrip_differences)
    write_json(out/'verification.json',result);print('VERIFICATION_PASSED '+json.dumps(result),flush=True)

