from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare_data import HEADWATERS
from utils import metrics,write_json

def report(out):
    out=Path(out);figs=out/'figures';figs.mkdir(exist_ok=True)
    config=json.loads((out/'run_config.json').read_text());layers=config['layers']
    frame=pd.read_csv(out/'test_paired_weekly_predictions.csv',parse_dates=['start','end'])
    frame=frame[frame.basin_id.isin(HEADWATERS)].copy()
    frame.to_csv(out/'test_headwater10_predictions.csv',index=False)
    scores=pd.read_csv(out/'metrics_by_basin.csv');test=scores[(scores.split=='test')&scores.basin_id.isin(HEADWATERS)]
    old=test[test.model=='lstm_history13_baseline'].drop(columns=['model','priority_headwater'])
    new=test[test.model=='lstm_nonlinear_heads_cosine'].drop(columns=['model','priority_headwater'])
    c=old.merge(new,on=['split','basin_id','name','n'],suffixes=('_old','_new'),validate='one_to_one').sort_values('basin_id')
    assert len(c)==10 and c.n.sum()==config['counts']['test_headwater10']
    c['nse_change']=c.nse_new-c.nse_old;c.to_csv(out/'test_headwater10_comparison.csv',index=False)
    summary=dict(locations=10,paired_test_weeks=int(c.n.sum()),improved_nse=int((c.nse_change>0).sum()),
        median_nse_old=float(c.nse_old.median()),median_nse_new=float(c.nse_new.median()),
        median_kge_old=float(c.kge_old.median()),median_kge_new=float(c.kge_new.median()),
        median_abs_bias_old=float(c.pbias_percent_old.abs().median()),median_abs_bias_new=float(c.pbias_percent_new.abs().median()))
    write_json(out/'headwater10_comparison_summary.json',summary)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
    def save(fig,name):
        for ext in ['png','pdf']:fig.savefig(figs/f'{name}.{ext}',dpi=170,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(15,7),layout='constrained');y=np.arange(10)
    axes[0].barh(y-.18,c.nse_old,.34,color='#9aabb8',label='13-week linear heads (original settings)')
    axes[0].barh(y+.18,c.nse_new,.34,color='#e4712a',label='13-week nonlinear heads (cosine; 50 epochs)')
    axes[0].set_yticks(y,c.basin_id+' '+c['name']);axes[0].invert_yaxis();axes[0].set(xlabel='Weekly volume NSE',xlim=(min(0,c.nse_new.min()-.05),1))
    axes[1].barh(y,c.nse_change,color=np.where(c.nse_change>=0,'#267fba','#d8642c'))
    for i,r in enumerate(c.itertuples()):axes[1].annotate(f'{r.nse_change:+.3f}',(r.nse_change,i),xytext=(4 if r.nse_change>=0 else -4,0),textcoords='offset points',ha='left' if r.nse_change>=0 else 'right',va='center',fontsize=9)
    axes[1].margins(x=.3);axes[1].set_yticks(y,c['name']);axes[1].invert_yaxis();axes[1].axvline(0,color='black',lw=.7);axes[1].set(xlabel='NSE change: nonlinear minus original linear heads')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2)
    fig.suptitle(f'Ten headwaters | identical test weeks (2010–2016)\nMedian NSE {summary["median_nse_old"]:.3f} → {summary["median_nse_new"]:.3f}; {summary["improved_nse"]}/10 improve')
    save(fig,'weekly_nse_comparison_headwater10')
    hist=pd.read_csv(out/'training_history.csv');fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    ax.plot(hist.epoch,hist.train_loss,label='Training: all52 (dropout active)');ax.plot(hist.epoch,hist.validation_loss,label='Validation: priority10 (selection)')
    ax.plot(hist.epoch,hist.validation_all_sites_loss,ls=':',label='Validation: all scoreable sites')
    best=int(hist.loc[hist.validation_loss.idxmin(),'epoch']);ax.axvline(best,color='gray',ls='--',label=f'Chosen epoch {best}')
    ax.set(xlabel='Epoch',ylabel='Basin-normalized squared error',title='13-week LSTM + nonlinear site heads | cosine LR; no early stopping');ax.legend(fontsize=8);save(fig,'learning_curve')
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    ax.plot(hist.epoch,hist.learning_rate,marker='.',label='Learning rate used for epoch')
    ax.set(xlabel='Epoch',ylabel='Learning rate',title='Fixed cosine learning-rate schedule: 50 epochs')
    ax.legend();save(fig,'learning_rate_schedule')
    def curve(ax,bid,year=None):
        start='2010-01-01' if year is None else f'{year}-01-01';end='2016-10-10' if year is None else f'{year}-12-31'
        g=frame[frame.basin_id==bid].set_index('end').reindex(pd.date_range(start,end,freq='W-SUN'));r=c.set_index('basin_id').loc[bid]
        for col,color,label,lw in [('observed_volume_m3','#263f4b','ELF reference',1.2),('history13_volume_m3','#8aabc6','Original linear heads',1.0),('predicted_volume_m3','#e4712a','Nonlinear heads, cosine LR',1.1)]:
            ax.plot(g.index,g[col]/1e6,color=color,label=label,lw=lw,alpha=.9)
        ax.set(xlim=(pd.Timestamp(start),pd.Timestamp(end)),ylabel='Million m³/week',title=f'{bid} {r["name"]} | NSE {r.nse_old:.3f} → {r.nse_new:.3f}');ax.tick_params(labelsize=8)
    for year,label in [(None,'2010_2016'),(2013,'2013')]:
        fig,axes=plt.subplots(5,2,figsize=(18,16),layout='constrained')
        for ax,bid in zip(axes.flat,c.basin_id):curve(ax,bid,year)
        handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3)
        fig.suptitle(f'Ten headwater weekly hydrographs | {label.replace("_","–")}\nGaps unfilled; panel NSE uses the full matched test period');save(fig,f'test_weekly_hydrographs_{label}_headwater10')
    fig,axes=plt.subplots(2,1,figsize=(15,7),layout='constrained')
    for ax,bid in zip(axes,['ELF13','ELF39']):curve(ax,bid);ax.legend(fontsize=8)
    save(fig,'priority_norris_normandy_weekly')
    for bid in c.basin_id:
        fig,ax=plt.subplots(figsize=(14,4),layout='constrained');curve(ax,bid);ax.legend(fontsize=9);save(fig,f'{bid}_weekly_test_hydrograph')
    result=json.loads((out/'results_summary.json').read_text())
    lines=['# TVA-only one-layer LSTM: 13 weeks, nonlinear heads, cosine LR','',
        'Trained on the original TVA 52-site dataset. All 50 epochs completed; the minimum ten-headwater validation loss selects the checkpoint. The saved 13-week one-layer TVA-only model is the comparator.', '',
        '| Site | Matched test weeks | Original linear-head NSE | Nonlinear-head NSE | Change | New volume bias |', '|---|---:|---:|---:|---:|---:|']
    for r in c.itertuples():lines.append(f'| {r.basin_id} {r.name} | {r.n} | {r.nse_old:.4f} | {r.nse_new:.4f} | {r.nse_change:+.4f} | {r.pbias_percent_new:+.2f}% |')
    lines+=['',json.dumps(summary,indent=2),'',
        f'Best validation epoch {result["best_epoch"]}; {result["epochs_run"]} epochs; training/validation {result["training_validation_seconds"]:.1f} seconds.', '',
        '## Data and settings','',
        'Original weekly inputs, ELF target quality masks, date splits, static variables, scalers and 13-week eligibility are copied byte-for-byte. No calendar-gap filling or target-mask changes. Historical ELF is never an input. All original validation and test site-weeks are matched exactly.',
        'P=weekly sum, T=weekly mean, runoff=weekly sum of NLDAS surface runoff+baseflow, elevation static. The target week and preceding 12 weeks predict target-week ELF in both models. No GNN and no CAMELS.',
        'Train1982–2004; validation2005–2009; test2010–October2016. Validation/test comparisons match site and week exactly, and both models use the same observed volumes. Nickajack has no valid held-out target.',
        json.dumps(config['counts'],indent=2), '',
        'One LSTM layer, hidden64, dropout0.2 before independent 64->16 ReLU->1 Softplus heads. 72,884 parameters (17,920 shared LSTM + 54,964 site heads). Seed42, batch512, Adam0.001, gradient clip1, 50 epochs, CosineAnnealingLR T_max50 eta_min0.00001 stepped after each epoch. The first epoch uses0.001; the last uses approximately0.00001098; the schedule reaches0.00001 after epoch50. No warm restarts and no early stopping. The scheduler is independent of validation; checkpoint selection uses minimum normalized validation loss over the ten priority headwaters. best_model.pt is selected from all50 epochs; last_model.pt is epoch50.', '',
        '## Interpretation','',
        'This comparison changes output-head architecture, early stopping and learning-rate schedule together, so it cannot isolate their individual contributions. Input history, data, masks, eligibility, splits, scalers and all other training settings stay fixed. This is one seed and historical-input reconstruction, not operational HEADWATER forecast validation. The test period is not used for checkpoint selection.', '']
    (out/'REPORT.md').write_text('\n'.join(lines));print('HEADWATER_REPORT_COMPLETE '+json.dumps(summary),flush=True)

