"""Train all 52 sites; select checkpoints using the ten priority headwaters."""
from pathlib import Path
import argparse,json,os,random,shutil,sys,time
import numpy as np
import pandas as pd
import torch
from model import SiteLSTM,make_batch,check_models
from prepare_data import HEADWATERS,HISTORY
from utils import write_json,sha256,metrics

ROOT=Path(__file__).resolve().parents[1]

def load_data(data,device):
    sc=json.loads((data/'train_only_scalers.json').read_text())
    with np.load(data/'weekly_dataset.npz') as z:a={k:z[k].copy() for k in z.files}
    assert len(a['basin_ids'])==52
    x=((a['dynamic']-np.array(sc['dynamic_mean']))/np.array(sc['dynamic_std'])).astype('float32')
    s=((a['static']-np.array(sc['static_mean']))/np.array(sc['static_std'])).astype('float32')
    # Match the previous float32 std +0.1 calculation.
    scales=np.array(sc['target_train_std_by_basin'],dtype='float32')+.1
    tensors=tuple(torch.as_tensor(v,dtype=torch.float32,device=device) for v in (x,s,a['target_mean_mm_day'],scales))
    samples={n:np.argwhere(a['eligible']&(a['split'][:,None]==c)) for n,c in [('train',0),('validation',1),('test',2)]}
    return a,sc,tensors,samples

def evaluate(model,samples,tensors):
    model.eval();out=[]
    with torch.no_grad():
        for offset in range(0,len(samples),512):
            x,s,_,_,b=make_batch(samples[offset:offset+512],tensors)
            out.append(model(x,s,b).cpu().numpy())
    p=np.concatenate(out);t,b=samples.T
    assert np.isfinite(p).all() and (p>=0).all()
    y=tensors[2][t,b].cpu().numpy();scale=tensors[3][b].cpu().numpy()
    return p,float(np.mean(((p-y)/scale)**2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--layers',type=int,choices=[1],default=1);args=ap.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);(out/'code').mkdir()
    for f in (ROOT/'scripts').glob('*.py'):shutil.copy2(f,out/'code'/f.name)
    shutil.copy2(ROOT/'train.slurm',out/'code/train.slurm')
    data=ROOT/'data';audit=json.loads((ROOT/'preparation_audit.json').read_text())
    for p,digest in audit['source_sha256'].items():assert sha256(p)==digest,p
    torch.set_num_threads(4);check_models()
    assert torch.cuda.is_available(),'Submit through Slurm GPU allocation'
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    a,sc,tensors,samples=load_data(data,'cuda')
    model=SiteLSTM(args.layers).cuda()
    head_ix=np.array([a['basin_ids'].tolist().index(b) for b in HEADWATERS])
    head_validation=samples['validation'][np.isin(samples['validation'][:,1],head_ix)]
    assert len(head_validation)==audit['counts']['validation_headwater10']
    x,s,y,scale,bi=make_batch(samples['train'][:20],tensors)
    assert x.shape[1]==HISTORY
    assert torch.isfinite(x).all() and torch.isfinite(s).all() and torch.isfinite(y).all()
    old=json.loads((ROOT/'references/run_config.json').read_text())
    for name,digest in audit['data_sha256'].items():assert sha256(data/name)==digest
    for name,digest in audit['reference_sha256'].items():assert sha256(ROOT/'references'/name)==digest
    config=dict(architecture=f'Shared LSTM(12,64,{args.layers}), dropout 0.2 before 52 nonlinear site heads (64->16 ReLU->1 Softplus), no GNN',layers=args.layers,interlayer_dropout=0.0,
        parameters=sum(p.numel() for p in model.parameters()),lstm_parameters=19968,site_head_parameters=54964,head_hidden=16,head_activation='ReLU',output_activation='Softplus',history=HISTORY,history_unit='weeks',
        seed=42,hidden=64,dropout=.2,batch_size=512,epochs=50,patience=None,early_stopping=False,learning_rate=.001,gradient_clip=1.,
        scheduler='CosineAnnealingLR T_max50 eta_min0.00001',minimum_learning_rate=.00001,scheduler_T_max=50,loss=old['loss'],selection='Minimum normalized validation loss over ten priority headwaters only; test evaluated after checkpoint selection',selection_headwaters=HEADWATERS,
        dynamic_names=sc['dynamic_names'],static_names=sc['static_names'],basin_ids=a['basin_ids'].tolist(),
        train_samples=len(samples['train']),validation_samples=len(samples['validation']),test_samples=len(samples['test']),
        optimizer_steps_per_epoch=int(np.ceil(len(samples['train'])/512)),
        splits=dict(train='1982-2004',validation='2005-2009',test='2010-2016 (archive ends October 2016)'),
        target='Weekly mean daily-equivalent mm/day; volume = output * area_km2 * 7000',
        input_timing='Target week plus 12 preceding weeks; no observed ELF fed as input',
        initialization='Fresh seed42; original four input columns and nonlinear heads initialized identically to surface-only baseline; eight added static columns initialized to zero; no pretrained weights',
        differences='Eight RiTHyMs/StreamCat static descriptors added to the surface-only baseline; all original data, scalers, masks, splits and training settings preserved',
        interpretation='Historical-input reconstruction; single seed. Surface runoff excludes baseflow; this does not establish actual HEADWATER forecast skill',runoff_definition='NLDAS Qs surface runoff only; excludes Qsb',
        device=torch.cuda.get_device_name(),torch=str(torch.__version__),python=sys.version,slurm_job_id=os.getenv('SLURM_JOB_ID'),
        data_sha256={p.name:sha256(p) for p in data.iterdir() if p.is_file()},reference_sha256=audit['reference_sha256'],counts=audit['counts'],baseline_run=audit['baseline_run'],total_runoff_run=audit['total_runoff_run'],static_audit=audit['static_audit'],
        code_sha256={p.name:sha256(p) for p in (out/'code').iterdir() if p.is_file()})
    for k in ['seed','hidden','dropout','batch_size','epochs','learning_rate','gradient_clip','history','loss','selection']:
        assert config[k]==old[k],k
    write_json(out/'run_config.json',config)
    for n in ['train_only_scalers.json','static_attributes.csv','site_head_mapping.json']:shutil.copy2(data/n,out/n)
    shutil.copy2(ROOT/'preparation_audit.json',out/'preparation_audit.json')
    shutil.copy2(ROOT/'static_preparation/static_audit.json',out/'static_audit.json')
    write_json(ROOT/'STATUS.json',dict(status='training',run=str(out),job_id=os.getenv('SLURM_JOB_ID')))
    print('CONFIG '+json.dumps(config),flush=True)
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=50,eta_min=.00001)
    rng=np.random.default_rng(42);records=[];best=float('inf');best_epoch=0;reason='maximum_epochs';started=time.monotonic()
    for epoch in range(1,51):
        t0=time.monotonic();lr_used=opt.param_groups[0]['lr'];model.train();order=rng.permutation(len(samples['train']));total=0.;count=0
        for off in range(0,len(order),512):
            rows=samples['train'][order[off:off+512]];x,s,y,scale,bi=make_batch(rows,tensors)
            opt.zero_grad(set_to_none=True);pred=model(x,s,bi);loss=(((pred-y)/scale)**2).mean()
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            total+=float(loss.detach())*len(rows);count+=len(rows)
            if time.monotonic()-started>600:raise RuntimeError('Ten-minute training cap reached; not complete')
        all_pred,all_val=evaluate(model,samples['validation'],tensors)
        vti,vbi=samples['validation'].T; pick=np.isin(vbi,head_ix)
        vy=tensors[2][vti[pick],vbi[pick]].cpu().numpy();vs=tensors[3][vbi[pick]].cpu().numpy()
        val=float(np.mean(((all_pred[pick]-vy)/vs)**2))
        scheduler.step()
        row=dict(epoch=epoch,train_loss=total/count,validation_loss=val,validation_all_sites_loss=all_val,learning_rate=lr_used,next_learning_rate=opt.param_groups[0]['lr'],seconds=time.monotonic()-t0)
        records.append(row);pd.DataFrame(records).to_csv(out/'training_history.csv',index=False)
        checkpoint=dict(model_state=model.state_dict(),optimizer_state=opt.state_dict(),scheduler_state=scheduler.state_dict(),epoch=epoch,validation_loss=val,run_config=config,scalers=sc)
        if val<best:best=val;best_epoch=epoch;torch.save(checkpoint,out/'best_model.pt')
        torch.save(checkpoint,out/'last_model.pt');print('EPOCH '+json.dumps(dict(**row,best_epoch=best_epoch)),flush=True)
    assert len(records)==50 and records[-1]['epoch']==50
    seconds=time.monotonic()-started
    model.load_state_dict(torch.load(out/'best_model.pt',map_location='cuda',weights_only=True)['model_state'])
    attrs=pd.read_csv(data/'static_attributes.csv').set_index('elf_id');ids=a['basin_ids'];area=attrs.loc[ids,'area_km2'].to_numpy();scores=[]
    for split in ['validation','test']:
        p,_=evaluate(model,samples[split],tensors);ti,bi=samples[split].T
        frame=pd.DataFrame(dict(basin_id=ids[bi],start=a['starts'][ti],end=a['ends'][ti],days=7,
            observed_volume_m3=a['target_mean_mm_day_float64'][ti,bi]*area[bi]*7000,
            predicted_volume_m3=p.astype(float)*area[bi]*7000,predicted_mean_mm_day=p))
        base=pd.read_csv(ROOT/'references'/f'{split}_paired_weekly_predictions.csv')[['basin_id','start','end','days','observed_volume_m3','predicted_volume_m3','climatology_volume_m3']]
        base=base.rename(columns={'predicted_volume_m3':'history13_volume_m3'})
        frame=frame.merge(base,on=['basin_id','start','end','days'],how='left',validate='one_to_one',indicator=True,suffixes=('','_old'))
        assert (frame['_merge']=='both').all()
        np.testing.assert_allclose(frame.observed_volume_m3,frame.observed_volume_m3_old,rtol=1e-12,atol=1e-5)
        frame=frame.drop(columns=['_merge','observed_volume_m3_old']).sort_values(['basin_id','start'])
        total=pd.read_csv(ROOT/'references'/f'total_{split}_paired_weekly_predictions.csv')[['basin_id','start','end','days','predicted_volume_m3']].rename(columns={'predicted_volume_m3':'total_runoff_volume_m3'})
        frame=frame.merge(total,on=['basin_id','start','end','days'],validate='one_to_one',how='left');assert frame.total_runoff_volume_m3.notna().all()
        frame.to_csv(out/f'{split}_paired_weekly_predictions.csv',index=False)
        for bid,g in frame.groupby('basin_id'):
            for label,col in [('lstm_surface_static8','predicted_volume_m3'),('surface_runoff_baseline','history13_volume_m3'),('total_runoff_baseline','total_runoff_volume_m3'),('training_climatology','climatology_volume_m3')]:
                scores.append(dict(split=split,basin_id=bid,name=attrs.loc[bid,'name'],model=label,priority_headwater=bid in HEADWATERS,**metrics(g.observed_volume_m3,g[col])))
    scores=pd.DataFrame(scores);scores.to_csv(out/'metrics_by_basin.csv',index=False)
    summaries=[]
    for scope,subset in [('all_scoreable_sites',scores),('priority_headwater10',scores[scores.priority_headwater])]:
        for (split,label),g in subset.groupby(['split','model']):
            summaries.append(dict(scope=scope,split=split,model=label,locations=len(g),weeks=int(g.n.sum()),median_nse=float(g.nse.median()),median_kge=float(g.kge.median()),median_absolute_pbias_percent=float(g.pbias_percent.abs().median())))
    result=dict(status='completed',best_epoch=best_epoch,epochs_run=len(records),best_validation_loss=best,stop_reason=reason,training_validation_seconds=seconds,summary=summaries)
    write_json(out/'results_summary.json',result)
    from report import report
    report(out)
    from verify import verify
    verify(out,ROOT)
    write_json(ROOT/'STATUS.json',dict(status='completed',run=str(out),job_id=os.getenv('SLURM_JOB_ID'),best_epoch=best_epoch,layers=args.layers,history=HISTORY,early_stopping=False,epochs_run=len(records)))
    print('TRAINING_COMPLETE '+json.dumps(result),flush=True)

if __name__=='__main__':main()

