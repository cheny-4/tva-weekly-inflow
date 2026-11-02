"""Stable agent-facing interface for the frozen TVA weekly inflow baseline."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import random
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'baseline'
RUN = BASE / 'runs/weekly13_lstm1_nonlinear_cosine_s42_5751784'
FEATURES = ['precipitation_mm_week', 'temperature_mean_c', 'total_runoff_mm_week']
SPLITS = {'train': 0, 'validation': 1, 'test': 2}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_output(path):
    path = Path(path).resolve()
    if path == BASE.resolve() or BASE.resolve() in path.parents:
        raise ValueError('Choose an output directory outside the frozen baseline')
    path.mkdir(parents=True, exist_ok=False)
    return path


def load_model_module():
    spec = importlib.util.spec_from_file_location('tva_baseline_model', BASE / 'scripts/model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_arrays():
    with np.load(BASE / 'data/weekly_dataset.npz', allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def normalize(values, scalers):
    # Preserve the original float32 physical inputs followed by float64 scaling.
    x = np.asarray(values, dtype=np.float32)
    return ((x - np.asarray(scalers['dynamic_mean'])) /
            np.asarray(scalers['dynamic_std'])).astype(np.float32)


def normalize_static(values, scalers):
    x = np.asarray(values, dtype=np.float32)
    return ((x - np.asarray(scalers['static_mean'])) /
            np.asarray(scalers['static_std'])).astype(np.float32)


def checkpoint_model(checkpoint=None, device='cpu'):
    import torch
    torch.set_num_threads(4)
    checkpoint = Path(checkpoint) if checkpoint else RUN / 'best_model.pt'
    ck = torch.load(checkpoint, map_location=device, weights_only=True)
    config, sc = ck['run_config'], ck['scalers']
    if config['history'] != 13 or config['layers'] != 1 or config['parameters'] != 72884:
        raise ValueError('Checkpoint does not match this baseline interface')
    if sc['dynamic_names'] != FEATURES or sc['static_names'] != ['elevation_mean_m']:
        raise ValueError('Checkpoint feature order does not match the interface')
    model = load_model_module().SiteLSTM().to(device)
    model.load_state_dict(ck['model_state'], strict=True)
    model.eval()
    return model, ck, config, sc


def predict_batches(model, x, static, site, device='cpu'):
    import torch
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(site), 512):
            sl = slice(start, start + 512)
            pred = model(torch.as_tensor(x[sl], device=device),
                         torch.as_tensor(static[sl], device=device),
                         torch.as_tensor(site[sl], dtype=torch.long, device=device))
            predictions.append(pred.cpu().numpy())
    result = np.concatenate(predictions) if predictions else np.empty(0, np.float32)
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError('Model produced non-finite or negative predictions')
    return result


def replay(checkpoint=None, split='test', device='cpu'):
    a = load_arrays()
    model, ck, cfg, sc = checkpoint_model(checkpoint, device)
    if a['basin_ids'].tolist() != cfg['basin_ids']:
        raise ValueError('Dataset and checkpoint site orders differ')
    t, b = np.where(a['eligible'] & (a['split'][:, None] == SPLITS[split]))
    x = normalize(a['dynamic'], sc)
    s = normalize_static(a['static'], sc)
    windows = x[t[:, None] + np.arange(-12, 1), b[:, None]]
    if not np.isfinite(windows).all():
        raise ValueError('Non-finite input inside an eligible window')
    pred = predict_batches(model, windows, s[b], b, device)
    attrs = pd.read_csv(BASE / 'data/static_attributes.csv').set_index('elf_id')
    ids = a['basin_ids'][b]
    areas = attrs.loc[ids, 'area_km2'].to_numpy()
    frame = pd.DataFrame(dict(basin_id=ids, start=a['starts'][t], end=a['ends'][t], days=7,
        observed_volume_m3=a['target_mean_mm_day_float64'][t, b] * areas * 7000,
        predicted_mean_mm_day=pred, predicted_volume_m3=pred.astype(float) * areas * 7000))
    return frame.sort_values(['basin_id', 'start']).reset_index(drop=True), ck


def hydrologic_metrics(obs, pred):
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    good = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[good], pred[good]
    result = dict(n=int(len(obs)), nse=None, kge=None, pbias_percent=None, rmse_m3=None)
    if len(obs) < 2:
        return result
    den = np.sum((obs - obs.mean()) ** 2)
    if den > 0:
        result['nse'] = float(1 - np.sum((pred - obs) ** 2) / den)
    if obs.std() > 0 and pred.std() > 0 and obs.mean() != 0:
        r = np.corrcoef(obs, pred)[0, 1]
        result['kge'] = float(1 - np.sqrt((r - 1)**2 +
            (pred.std() / obs.std() - 1)**2 + (pred.mean() / obs.mean() - 1)**2))
    if obs.sum() != 0:
        result['pbias_percent'] = float(100 * (pred.sum() / obs.sum() - 1))
    result['rmse_m3'] = float(np.sqrt(np.mean((pred - obs)**2)))
    return result


def score_frame(frame):
    names = pd.read_csv(BASE / 'data/static_attributes.csv').set_index('elf_id')['name']
    priority = read_json(BASE / 'data/site_head_mapping.json')['priority_headwaters']
    return pd.DataFrame([dict(basin_id=bid, name=names.loc[bid], priority_headwater=bid in priority,
        **hydrologic_metrics(g.observed_volume_m3, g.predicted_volume_m3))
        for bid, g in frame.groupby('basin_id')])


def inspect_command(args):
    manifest = read_json(BASE / 'export_manifest.json')
    for name, record in manifest['files'].items():
        if sha256(BASE / name) != record['sha256']:
            raise ValueError('Frozen baseline checksum mismatch: ' + name)
    a = load_arrays()
    for split, code in SPLITS.items():
        mask = a['eligible'] & (a['split'][:, None] == code)
        if int(mask.sum()) != {'train': 50084, 'validation': 9868, 'test': 15025}[split]:
            raise ValueError('Frozen split/mask count differs: ' + split)
    if np.any(a['eligible'] & ~a['target_valid_all7']):
        raise ValueError('Eligible targets do not preserve the seven-day mask')
    print(json.dumps(dict(status='verified',frozen_files=len(manifest['files']),
        source=manifest['source'],config=read_json(RUN / 'run_config.json'),
        saved_results=read_json(RUN / 'results_summary.json')), indent=2))


def replay_command(args):
    out = new_output(args.out)
    frame, ck = replay(args.checkpoint, args.split, args.device)
    frame.to_csv(out / 'predictions.csv', index=False)
    scores = score_frame(frame)
    scores.to_csv(out / 'metrics_by_site.csv', index=False)
    summary = dict(split=args.split, selected_epoch=int(ck['epoch']), rows=len(frame),
        scoreable_sites=len(scores), median_nse=float(scores.nse.median()),
        headwater10_median_nse=float(scores.loc[scores.priority_headwater, 'nse'].median()),
        interpretation='Historical-input reconstruction, not operational forecast validation')
    if args.checkpoint is None and args.split in ['validation', 'test']:
        saved = pd.read_csv(RUN / f'{args.split}_paired_weekly_predictions.csv')
        paired = frame.merge(saved, on=['basin_id','start','end','days'], how='outer',
            validate='one_to_one', suffixes=('', '_saved'), indicator=True)
        if not (paired['_merge'] == 'both').all():
            raise ValueError('Replay rows differ from saved eligibility')
        saved_depth = paired.predicted_volume_m3_saved / (
            pd.read_csv(BASE / 'data/static_attributes.csv').set_index('elf_id')
            .loc[paired.basin_id, 'area_km2'].to_numpy() * 7000)
        diff = np.abs(paired.predicted_mean_mm_day - saved_depth)
        if diff.max() > 5e-5:
            raise ValueError('Replay exceeds cross-platform tolerance in mm/day')
        np.testing.assert_allclose(paired.observed_volume_m3, paired.observed_volume_m3_saved,
                                   rtol=1e-12, atol=1e-5)
        summary['max_saved_replay_difference_mm_day'] = float(diff.max())
    write_json(out / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


def validate_metadata(metadata):
    expected = dict(calendar='Monday-Sunday', precipitation_units='mm/week',
        temperature_units='degC', runoff_units='mm/week', runoff_definition='surface_runoff_plus_baseflow')
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f'Forcing metadata {key!r} must equal {value!r}')
    if metadata.get('source_kind') not in ['historical', 'forecast']:
        raise ValueError('Specify source_kind as historical or forecast')


def infer_frame(frame, metadata, checkpoint=None, device='cpu'):
    validate_metadata(metadata)
    required = ['basin_id', 'start', *FEATURES]
    if any(key not in frame for key in required):
        raise ValueError('Input columns must include: ' + ', '.join(required))
    frame = frame.copy()
    frame['start'] = pd.to_datetime(frame.start, format='%Y-%m-%d', errors='raise')
    if (frame.start.dt.day_of_week != 0).any() or frame.duplicated(['basin_id', 'start']).any():
        raise ValueError('Require unique Monday-start rows for each site')
    for name in FEATURES:
        frame[name] = pd.to_numeric(frame[name], errors='raise')
    if (frame[[FEATURES[0], FEATURES[2]]] < 0).any().any():
        raise ValueError('Precipitation and runoff totals must be non-negative')
    model, ck, cfg, sc = checkpoint_model(checkpoint, device)
    ids = cfg['basin_ids']
    if set(frame.basin_id) - set(ids):
        raise ValueError('No trained output heads for sites: ' + str(set(frame.basin_id) - set(ids)))
    attrs = pd.read_csv(BASE / 'data/static_attributes.csv').set_index('elf_id')
    rows, windows, static, sites, positions = [], [], [], [], []
    for bid, group in frame.sort_values(['basin_id', 'start']).groupby('basin_id'):
        if len(group) < 13:
            raise ValueError(f'{bid} needs at least 13 weekly rows')
        dates = group.start.to_numpy(dtype='datetime64[D]')
        values = normalize(group[FEATURES].to_numpy(), sc)
        for i in range(12, len(group)):
            dates13 = dates[i-12:i+1]
            valid = (np.diff(dates13) == np.timedelta64(7, 'D')).all() and np.isfinite(values[i-12:i+1]).all()
            row = dict(basin_id=bid, start=str(dates[i]),end=str(dates[i]+np.timedelta64(6,'D')),days=7,
                predicted_mean_mm_day=np.nan,predicted_volume_m3=np.nan,
                status='predicted' if valid else 'invalid_history_or_input')
            rows.append(row)
            if valid:
                positions.append(len(rows)-1);windows.append(values[i-12:i+1])
                static.append(normalize_static([attrs.loc[bid,'elevation_mean_m']], sc))
                sites.append(ids.index(bid))
    if windows:
        pred = predict_batches(model,np.asarray(windows),np.asarray(static),np.asarray(sites),device)
        for pos, depth in zip(positions, pred):
            rows[pos]['predicted_mean_mm_day'] = float(depth)
            rows[pos]['predicted_volume_m3'] = float(depth) * attrs.loc[rows[pos]['basin_id'],'area_km2'] * 7000
    return pd.DataFrame(rows), ck


def infer_command(args):
    frame, ck = infer_frame(pd.read_csv(args.inputs), read_json(args.metadata), args.checkpoint, args.device)
    out = new_output(args.out)
    frame.to_csv(out / 'predictions.csv', index=False)
    write_json(out / 'provenance.json',dict(input_sha256=sha256(args.inputs),metadata=read_json(args.metadata),
        checkpoint_sha256=sha256(args.checkpoint or RUN/'best_model.pt'),selected_epoch=int(ck['epoch']),
        output_rows=len(frame),invalid_windows=int((frame.status!='predicted').sum()),
        history='Target week plus 12 preceding weeks',
        limitation='Requires matching runoff definition and mapped basin inputs. HEADWATER forecast skill is unvalidated.'))
    print('PREDICTIONS_SAVED',out / 'predictions.csv',len(frame),'rows')


def plot_command(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame = pd.read_csv(args.predictions)
    required=['basin_id','start','observed_volume_m3','predicted_volume_m3']
    if any(k not in frame for k in required) or frame.duplicated(['basin_id','start']).any():
        raise ValueError('Plot requires unique paired predictions: '+', '.join(required))
    out = new_output(args.out)
    names = pd.read_csv(BASE / 'data/static_attributes.csv').set_index('elf_id')['name']
    scores = score_frame(frame)
    scores.to_csv(out / 'metrics_by_site.csv',index=False)
    fig, ax = plt.subplots(figsize=(14,5))
    ax.bar(scores.basin_id,scores.nse,color='#087E8B')
    ax.axhline(0,color='black',lw=.7);ax.tick_params(axis='x',rotation=90,labelsize=8)
    ax.set(ylabel='Weekly NSE',title='TVA weekly baseline: saved prediction accuracy by site')
    fig.tight_layout();fig.savefig(out/'weekly_nse_by_site.png',dpi=180);plt.close(fig)
    for bid,g in frame.groupby('basin_id'):
        if args.sites and bid not in args.sites: continue
        g=g.sort_values('start').copy();dates=pd.to_datetime(g.start)
        # Reindex explicitly, so plots do not bridge masked/missing calendar weeks.
        g.index=dates;g=g.reindex(pd.date_range(dates.min(),dates.max(),freq='7D'))
        fig,ax=plt.subplots(figsize=(12,3.7))
        ax.plot(g.index,g.observed_volume_m3/1e6,color='#2864A0',lw=1,label='ELF reference')
        ax.plot(g.index,g.predicted_volume_m3/1e6,color='#EA7626',lw=1,label='LSTM prediction')
        ax.set(title=f'{bid}  {names.loc[bid]}',ylabel='Weekly volume (million m³)')
        ax.legend();fig.tight_layout();fig.savefig(out/f'{bid}_weekly.png',dpi=180);plt.close(fig)
    print('FIGURES_SAVED',out)


def train_command(args):
    """Portable reproduction of the original 50-epoch loop; no automatic submission."""
    import torch
    if not torch.cuda.is_available() or not __import__('os').environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Training requires an allocated Frontier Slurm GPU job')
    inspect_command(args)
    out=new_output(args.out);a=load_arrays();cfg=read_json(RUN/'run_config.json')
    sc=read_json(BASE/'data/train_only_scalers.json');torch.set_num_threads(4)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    module=load_model_module();model=module.SiteLSTM().cuda()
    x=normalize(a['dynamic'],sc);s=normalize_static(a['static'],sc)
    scales=np.asarray(sc['target_train_std_by_basin'],np.float32)+.1
    tensors=tuple(torch.as_tensor(v,dtype=torch.float32,device='cuda')
        for v in (x,s,a['target_mean_mm_day'],scales))
    train=np.argwhere(a['eligible']&(a['split'][:,None]==0))
    val=np.argwhere(a['eligible']&(a['split'][:,None]==1))
    head_ix=np.array([a['basin_ids'].tolist().index(b) for b in cfg['selection_headwaters']])
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=50,eta_min=.00001)
    rng=np.random.default_rng(42);best=float('inf');records=[]
    config={**cfg,'source_checkpoint_used_for_initialization':False,
        'packaged_data_sha256':sha256(BASE/'data/weekly_dataset.npz'),
        'device':torch.cuda.get_device_name(),'torch':str(torch.__version__),'python':sys.version,
        'slurm_job_id':__import__('os').environ['SLURM_JOB_ID'],
        'pipeline_sha256':sha256(Path(__file__))}
    write_json(out/'run_config.json',config)
    for epoch in range(1,51):
        model.train();order=rng.permutation(len(train));total=0.;count=0;lr=opt.param_groups[0]['lr']
        for offset in range(0,len(order),512):
            batch=train[order[offset:offset+512]];xb,sb,y,scale,bi=module.make_batch(batch,tensors)
            opt.zero_grad(set_to_none=True);pred=model(xb,sb,bi);loss=(((pred-y)/scale)**2).mean()
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            total+=float(loss.detach())*len(batch);count+=len(batch)
        model.eval();errors=[]
        with torch.inference_mode():
            for offset in range(0,len(val),512):
                xb,sb,y,scale,bi=module.make_batch(val[offset:offset+512],tensors)
                pred=model(xb,sb,bi);pick=torch.as_tensor(np.isin(bi.cpu().numpy(),head_ix),device='cuda')
                errors.extend((((pred[pick]-y[pick])/scale[pick])**2).cpu().numpy())
        validation=float(np.mean(np.asarray(errors,dtype=np.float32)));scheduler.step()
        records.append(dict(epoch=epoch,train_loss=total/count,validation_loss=validation,
            learning_rate=lr,next_learning_rate=opt.param_groups[0]['lr']))
        ck=dict(model_state=model.state_dict(),optimizer_state=opt.state_dict(),scheduler_state=scheduler.state_dict(),
                epoch=epoch,validation_loss=validation,run_config=config,scalers=sc)
        if validation<best:best=validation;torch.save(ck,out/'best_model.pt')
        torch.save(ck,out/'last_model.pt');pd.DataFrame(records).to_csv(out/'training_history.csv',index=False)
        print(json.dumps(records[-1]),flush=True)
    write_json(out/'completion.json',dict(status='completed',epochs=50,early_stopping=False,best_validation_loss=best))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('inspect',help='Verify frozen checksums, masks and settings');p.set_defaults(func=inspect_command)
    p=sub.add_parser('replay',help='Replay an existing checkpoint on the frozen split')
    p.add_argument('--split',choices=list(SPLITS),default='test');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.set_defaults(func=replay_command)
    p=sub.add_parser('infer',help='Predict from physical-unit weekly basin CSV inputs')
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.set_defaults(func=infer_command)
    p=sub.add_parser('plot',help='Plot saved paired predictions without retraining')
    p.add_argument('--predictions',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--sites',nargs='+');p.set_defaults(func=plot_command)
    p=sub.add_parser('train',help='Explicit 50-epoch reproduction inside a Slurm GPU job')
    p.add_argument('--out',type=Path,required=True);p.set_defaults(func=train_command)
    args=parser.parse_args(argv)
    try:args.func(args)
    except (ValueError,RuntimeError,FileExistsError) as error:parser.exit(2,str(error)+'\n')


if __name__=='__main__':main()
