from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare_data import HEADWATERS
from utils import write_json

def report(out):
    out=Path(out);figs=out/'figures';figs.mkdir(exist_ok=True)
    config=json.loads((out/'run_config.json').read_text())
    scores=pd.read_csv(out/'metrics_by_basin.csv');test=scores[scores.split=='test']
    keys=['split','basin_id','name','n']
    def one(model,suffix):
        g=test[test.model==model].drop(columns=['model','priority_headwater'])
        return g.rename(columns={k:k+suffix for k in g if k not in keys})
    c=one('surface_runoff_baseline','_old').merge(one('lstm_surface_static8','_new'),on=keys,validate='one_to_one')
    c=c.merge(one('total_runoff_baseline','_total'),on=keys,validate='one_to_one').sort_values('basin_id')
    assert len(c)==51 and c.n.sum()==15025
    c['nse_change']=c.nse_new-c.nse_old;c['nse_change_vs_total']=c.nse_new-c.nse_total
    c.to_csv(out/'test_all51_comparison.csv',index=False)
    h=c[c.basin_id.isin(HEADWATERS)].copy();assert len(h)==10 and h.n.sum()==3375
    h.to_csv(out/'test_headwater10_comparison.csv',index=False)
    summaries={}
    for name,g in [('all51',c),('headwater10',h)]:
        s=dict(locations=len(g),paired_test_weeks=int(g.n.sum()),improved_nse_vs_surface=int((g.nse_change>0).sum()),improved_nse_vs_total=int((g.nse_change_vs_total>0).sum()))
        for label,suffix in [('surface','old'),('static8','new'),('total','total')]:
            s['median_nse_'+label]=float(g['nse_'+suffix].median())
            s['median_kge_'+label]=float(g['kge_'+suffix].median())
            s['median_abs_bias_'+label]=float(g['pbias_percent_'+suffix].abs().median())
        summaries[name]=s
    write_json(out/'comparison_summary.json',summaries)
    frame=pd.read_csv(out/'test_paired_weekly_predictions.csv',parse_dates=['start','end'])
    frame[frame.basin_id.isin(HEADWATERS)].to_csv(out/'test_headwater10_predictions.csv',index=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.16})
    def save(fig,name):
        for ext in ['png','pdf']:fig.savefig(figs/f'{name}.{ext}',dpi=170,bbox_inches='tight')
        plt.close(fig)
    for name,g,size in [('all51',c,(15,19)),('headwater10',h,(15,8))]:
        fig,axes=plt.subplots(1,2,figsize=size,layout='constrained');y=np.arange(len(g))
        for offset,col,label,color in [(-.25,'nse_total','Total runoff + elevation','#7b858b'),(0,'nse_old','Surface runoff + elevation','#3175aa'),(.25,'nse_new','Surface runoff + elevation + eight statics','#e4712a')]:
            axes[0].barh(y+offset,g[col],.23,label=label,color=color)
        axes[0].set_yticks(y,g.basin_id+' '+g['name']);axes[0].invert_yaxis();axes[0].set(xlabel='Weekly volume NSE');axes[0].legend(fontsize=8)
        axes[1].barh(y,g.nse_change,color=np.where(g.nse_change>=0,'#087E8B','#cf6549'))
        axes[1].set_yticks(y,g['name']);axes[1].invert_yaxis();axes[1].axvline(0,color='black',lw=.8)
        axes[1].set(xlabel='NSE change from adding eight statics to surface-runoff LSTM')
        s=summaries[name];fig.suptitle(f'{len(g)} locations | identical held-out weeks, 2010–2016\nSurface-runoff median NSE {s["median_nse_surface"]:.3f} to {s["median_nse_static8"]:.3f}; {s["improved_nse_vs_surface"]}/{len(g)} improve')
        save(fig,'weekly_nse_comparison_'+name)
    fig,ax=plt.subplots(figsize=(15,5),layout='constrained');x=np.arange(len(c))
    for offset,col,label,color in [(-.25,'pbias_percent_total','Total runoff','#7b858b'),(0,'pbias_percent_old','Surface runoff','#3175aa'),(.25,'pbias_percent_new','Surface runoff + eight statics','#e4712a')]:ax.bar(x+offset,c[col],.23,label=label,color=color)
    ax.set_xticks(x,c.basin_id,rotation=90,fontsize=8);ax.axhline(0,color='black',lw=.8)
    ax.set(ylabel='Volume bias (%)',title='Test-period volume bias over identical valid weeks');ax.legend(fontsize=8);save(fig,'test_volume_bias_comparison_all51')
    hist=pd.read_csv(out/'training_history.csv');best=int(hist.loc[hist.validation_loss.idxmin(),'epoch'])
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    for col,label in [('train_loss','Training: all52'),('validation_loss','Validation: headwater10, selection'),('validation_all_sites_loss','Validation: all scoreable sites')]:ax.plot(hist.epoch,hist[col],label=label)
    ax.axvline(best,color='gray',ls='--',label=f'Selected epoch {best}')
    ax.set(xlabel='Epoch',ylabel='Basin-normalized squared error',title='Surface runoff + eight statics: 50 epochs, no early stopping');ax.legend(fontsize=8);save(fig,'learning_curve')
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained');ax.plot(hist.epoch,hist.learning_rate,marker='.')
    ax.set(xlabel='Epoch',ylabel='Learning rate',title='Unchanged 50-epoch cosine learning-rate schedule');save(fig,'learning_rate_schedule')
    lookup=c.set_index('basin_id')
    def curve(ax,bid,year=None):
        start=pd.Timestamp('2010-01-01' if year is None else f'{year}-01-01');end=pd.Timestamp('2016-10-10' if year is None else f'{year}-12-31')
        g=frame[frame.basin_id==bid].set_index('end').reindex(pd.date_range(start,end,freq='W-SUN'))
        for col,color,label in [('observed_volume_m3','#263f4b','ELF reference'),('total_runoff_volume_m3','#9a9fa4','Total runoff + elevation'),('history13_volume_m3','#3175aa','Surface runoff + elevation'),('predicted_volume_m3','#e4712a','Surface runoff + eight statics')]:ax.plot(g.index,g[col]/1e6,lw=1,color=color,label=label)
        r=lookup.loc[bid];ax.set(xlim=(start,end),ylabel='Million m³/week',title=f'{bid} {r["name"]}: surface NSE {r.nse_old:.3f} to {r.nse_new:.3f}');ax.tick_params(labelsize=8)
    fig,axes=plt.subplots(2,1,figsize=(16,8),layout='constrained')
    for ax,bid in zip(axes,['ELF13','ELF39']):curve(ax,bid);ax.legend(fontsize=8,ncol=2)
    save(fig,'priority_norris_normandy_weekly')
    for year,label in [(None,'2010_2016'),(2013,'2013')]:
        fig,axes=plt.subplots(5,2,figsize=(18,16),layout='constrained')
        for ax,bid in zip(axes.flat,h.basin_id):curve(ax,bid,year)
        handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2)
        fig.suptitle('Headwater10: '+label.replace('_','–')+'; gaps unfilled; NSE uses full test period');save(fig,'test_weekly_hydrographs_'+label+'_headwater10')
    for bid in h.basin_id:
        fig,ax=plt.subplots(figsize=(15,4),layout='constrained');curve(ax,bid);ax.legend(fontsize=8,ncol=2);save(fig,bid+'_weekly_test_hydrograph')
    for page in range(4):
        selected=c.basin_id.iloc[page*13:(page+1)*13];fig,axes=plt.subplots(7,2,figsize=(18,21),layout='constrained')
        for ax,bid in zip(axes.flat,selected):curve(ax,bid,2013)
        for ax in list(axes.flat)[len(selected):]:ax.set_visible(False)
        handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2)
        fig.suptitle(f'2013 weekly hydrographs, page {page+1}/4; original masks retained');save(fig,f'test_hydrographs_2013_page{page+1}')
    result=json.loads((out/'results_summary.json').read_text())
    lines=['# Surface-runoff weekly LSTM with eight RiTHyMs static attributes','',
        'Controlled experiment: append eight static descriptors to the surface-runoff/elevation baseline. No GNN, attention or CAMELS. All original dynamic inputs, masks, targets, splits, normalization statistics and history remain unchanged. Only new static inputs and their training-site normalization are added.','',
        '## Comparison on identical valid test weeks','',
        '| Scope | Total-runoff median NSE | Surface-only median NSE | Surface + eight statics median NSE | Sites improved vs surface |','|---|---:|---:|---:|---:|']
    for scope,s in summaries.items():lines.append(f'| {scope} | {s["median_nse_total"]:.4f} | {s["median_nse_surface"]:.4f} | {s["median_nse_static8"]:.4f} | {s["improved_nse_vs_surface"]}/{s["locations"]} |')
    lines+=['','| Site | Valid weeks | Total NSE | Surface NSE | Surface + statics NSE | Change vs surface | New volume bias |','|---|---:|---:|---:|---:|---:|---:|']
    for r in c.itertuples():lines.append(f'| {r.basin_id} {r.name} | {r.n} | {r.nse_total:.4f} | {r.nse_old:.4f} | {r.nse_new:.4f} | {r.nse_change:+.4f} | {r.pbias_percent_new:+.2f}% |')
    lines+=['','## Inputs and processing','',
        'Dynamic: weekly P amount, weekly mean T, weekly NLDAS surface runoff Qs. Original static elevation plus eight descriptors: soil permeability, sand %, clay %, depth to bedrock, depth to water table, baseflow index %, 2019 riparian impervious %, 2019 open-water %. Soil depth/permeability source-native units retained without unverified conversion.',
        'Raw local-catchment attributes are aggregated using existing ELF forcing-region intersection areas (EPSG5070). Missing catchment values are excluded and valid area weights renormalized; no values are filled. The impervious metric describes 100m riparian buffers, not exact basin-wide impervious cover. Detailed field mapping, region coverage and provenance are in static_audit.json and the experiment static_preparation folder.',
        '2019 land cover is fixed by user request for future use. Retrospective 2010–2016 evaluation uses these post-period descriptors and is not a strict as-of-date hindcast. Static BFI is from external StreamCat, not calculated from ELF test targets.','',
        '## Training settings','',
        '52 sites trained; 51 scoreable in testing. Train1982–2004; validation2005–2009; test2010–October2016. Target week plus 12 preceding weeks, not a 13-week forecast lead. 50 epochs, no early stopping; validation loss on the ten priority headwaters selects the checkpoint.',
        'Shared one-layer LSTM hidden64; dropout0.2 before 52 independent nonlinear heads64→16 ReLU→1 Softplus. 74,932 parameters. Fresh seed42; original parameters initialized identically to surface-only baseline, new static input columns start at zero. Adam lr0.001, batch512, clip1, cosineT_max50 eta_min0.00001. Same normalized squared-error loss.',
        f'Selected epoch {result["best_epoch"]}; training/validation time {result["training_validation_seconds"]:.1f} seconds.','',
        '## Interpretation','',
        'This is a single-seed historical-input experiment. The extra statics can describe basin differences but do not provide time-varying baseflow. Actual HEADWATER forecast skill remains untested. Test outcomes do not select the checkpoint.','',
        'Full metric summary:','', '```json',json.dumps(summaries,indent=2),'```']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');print('STATIC8_REPORT_COMPLETE '+json.dumps(summaries),flush=True)
