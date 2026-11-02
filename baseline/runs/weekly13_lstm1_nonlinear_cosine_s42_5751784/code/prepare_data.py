"""Freeze the exact original 13-week TVA-only data, masks, splits and reference."""
from pathlib import Path
import json,shutil
import numpy as np
import pandas as pd
from utils import sha256,write_json
ROOT=Path(__file__).resolve().parents[1]
BASELINE=ROOT.parent/'lstm_1layer_52site_heads_weekly'
HEADWATERS=['ELF01','ELF03','ELF04','ELF10','ELF13','ELF17','ELF18','ELF21','ELF29','ELF39']
HISTORY=13

def main():
    state=json.loads((BASELINE/'STATUS.json').read_text());assert state['status']=='completed'
    oldrun=Path(state['run']);old=json.loads((oldrun/'run_config.json').read_text())
    assert old['layers']==1 and old['history']==13 and old['parameters']==21300
    assert json.loads((oldrun/'verification.json').read_text())['status']=='passed'
    data=ROOT/'data';data.mkdir();refs=ROOT/'references';refs.mkdir();sources={}
    def copy(source,dest):
        sources[str(source)]=sha256(source);shutil.copy2(source,dest)
        assert sha256(dest)==sources[str(source)]
    for name in ['weekly_dataset.npz','train_only_scalers.json','static_attributes.csv','node_metadata.csv','site_head_mapping.json']:
        p=BASELINE/'data'/name;assert sha256(p)==old['data_sha256'][name]
        copy(p,data/name)
    for name in ['run_config.json','training_history.csv','results_summary.json','metrics_by_basin.csv','validation_paired_weekly_predictions.csv','test_paired_weekly_predictions.csv']:
        copy(oldrun/name,refs/name)
    copy(oldrun/'code/model.py',refs/'baseline_model.py')
    with np.load(data/'weekly_dataset.npz') as z:a={k:z[k].copy() for k in z.files}
    ids=a['basin_ids'];assert len(ids)==52 and a['dynamic'].shape==(1816,52,3)
    history=np.zeros_like(a['eligible'],dtype=bool)
    for i in range(12,len(history)):history[i]=np.isfinite(a['dynamic'][i-12:i+1]).all(axis=(0,2))
    assert np.all(~a['eligible']|history) and np.all(~a['eligible']|a['target_valid_all7'])
    assert np.isfinite(a['static']).all()
    attrs=pd.read_csv(data/'static_attributes.csv').set_index('elf_id').loc[ids]
    head=np.isin(ids,HEADWATERS);counts={};rows=[]
    for split,code in [('train',0),('validation',1),('test',2)]:
        mask=a['eligible']&(a['split'][:,None]==code)
        counts[split]=int(mask.sum());counts[split+'_headwater10']=int(mask[:,head].sum())
        for j,bid in enumerate(ids):rows.append(dict(split=split,basin_id=bid,name=attrs.loc[bid,'name'],valid_weeks=int(mask[:,j].sum()),priority_headwater=bid in HEADWATERS))
        if code:
            ti,bi=np.where(mask)
            expected=pd.DataFrame(dict(basin_id=ids[bi],start=a['starts'][ti],end=a['ends'][ti],days=7,observed_volume_m3=a['target_mean_mm_day_float64'][ti,bi]*attrs.area_km2.to_numpy()[bi]*7000))
            reference=pd.read_csv(refs/f'{split}_paired_weekly_predictions.csv')
            paired=expected.merge(reference,on=['basin_id','start','end','days'],how='outer',validate='one_to_one',indicator=True,suffixes=('','_old'))
            assert (paired['_merge']=='both').all()
            np.testing.assert_allclose(paired.observed_volume_m3,paired.observed_volume_m3_old,rtol=1e-12,atol=1e-5)
    assert counts==dict(train=50084,train_headwater10=11155,validation=9868,validation_headwater10=2297,test=15025,test_headwater10=3375)
    pd.DataFrame(rows).to_csv(data/'valid_week_counts.csv',index=False)
    record=dict(status='passed',history_weeks=13,source_sha256=sources,
        data_sha256={p.name:sha256(p) for p in data.iterdir() if p.is_file()},reference_sha256={p.name:sha256(p) for p in refs.iterdir() if p.is_file()},counts=counts,
        weekly_inputs_targets_quality_masks_splits_scalers_and_eligibility_byte_identical=True,
        baseline_run=str(oldrun),headwaters=HEADWATERS)
    write_json(ROOT/'preparation_audit.json',record);write_json(ROOT/'STATUS.json',dict(status='prepared',counts=counts))
    print('PREPARATION_PASSED '+json.dumps(counts),flush=True)

if __name__=='__main__':main()

