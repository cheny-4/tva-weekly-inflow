"""Append eight basin descriptors to the frozen surface-runoff experiment."""
from pathlib import Path
import copy,json,shutil
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from utils import sha256,write_json

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT.parent
BASELINE=PARENT/'lstm_1layer_52site_nonlinear_heads_weekly_surface_runoff'
TOTAL=PARENT/'lstm_1layer_52site_nonlinear_heads_weekly_cosine'
PRIOR_STATIC=PARENT/'lstm_1layer_52site_nonlinear_heads_weekly_static4'
RAW=PARENT.parent/'rithyms/downloaded/inputs/nlcd/streamcat.parquet'
WEIGHTS=PRIOR_STATIC/'static_preparation/forest_catchment_intersection_weights.csv'
MAPPING=PRIOR_STATIC/'static_preparation/static_attributes_all52.csv'
HEADWATERS=['ELF01','ELF03','ELF04','ELF10','ELF13','ELF17','ELF18','ELF21','ELF29','ELF39']
HISTORY=13
FIELDS={
    'soil_permeability_native':'permcat',
    'soil_sand_percent':'sandcat',
    'soil_clay_percent':'claycat',
    'depth_to_bedrock_native':'rckdepcat',
    'depth_to_water_table_native':'wtdepcat',
    'baseflow_index_percent':'bficat',
    'impervious_percent_2019_riparian100m':'pctimp2019catrp100',
    'open_water_percent_2019':'pctow2019cat',
}

def aggregate_attributes(weights,table,mapping,ids):
    assert weights[['source_region_id','comid']].duplicated().sum()==0
    assert table.comid.is_unique and (weights.intersection_area_m2>0).all()
    joined=weights.merge(table,on='comid',how='left',validate='many_to_one',indicator=True)
    assert (joined['_merge']=='both').all()
    denominator=joined.intersection_area_m2.groupby(joined.source_region_id).sum()
    values=pd.DataFrame(index=denominator.index);coverage=pd.DataFrame(index=denominator.index)
    for name,field in FIELDS.items():
        v=pd.to_numeric(joined[field],errors='coerce')
        valid=np.isfinite(v)&(v>=0)
        if 'percent' in name:valid &= v<=100
        area=joined.intersection_area_m2.where(valid,0)
        numerator=(v.where(valid,0)*area).groupby(joined.source_region_id).sum()
        goodarea=area.groupby(joined.source_region_id).sum()
        coverage[name]=goodarea/denominator
        assert (coverage[name]>.99).all(),(name,coverage[name].min())
        values[name]=numerator/goodarea
    m=mapping.set_index('elf_id').loc[ids]
    result=m[['name','source_region_id','forest_polygon_coverage']].copy()
    for name in FIELDS:
        result[name]=m.source_region_id.map(values[name])
        result[name+'_valid_fraction']=m.source_region_id.map(coverage[name])
    assert np.isfinite(result[list(FIELDS)].to_numpy()).all()
    return result,coverage,joined

def main():
    assert not (ROOT/'preparation_audit.json').exists(),'Data already prepared'
    state=json.loads((BASELINE/'STATUS.json').read_text());assert state['status']=='completed'
    run=Path(state['run']);config=json.loads((run/'run_config.json').read_text())
    total_state=json.loads((TOTAL/'STATUS.json').read_text());assert total_state['status']=='completed'
    total_run=Path(total_state['run'])
    for r in [run,total_run]:assert json.loads((r/'verification.json').read_text())['status']=='passed'
    assert config['history']==13 and config['layers']==1 and config['parameters']==72884
    data=ROOT/'data';refs=ROOT/'references';st=ROOT/'static_preparation';src=ROOT/'sources'
    for p in [data,refs,st,src]:p.mkdir(exist_ok=True)
    sources={str(p):sha256(p) for p in [RAW,WEIGHTS,MAPPING]}
    def copy_file(source,dest):
        sources[str(source)]=sha256(source);shutil.copy2(source,dest)
        assert sha256(dest)==sources[str(source)]
    for name in ['weekly_dataset.npz','train_only_scalers.json','static_attributes.csv','node_metadata.csv','site_head_mapping.json','valid_week_counts.csv']:
        copy_file(BASELINE/'data'/name,data/name)
    for name in ['run_config.json','training_history.csv','results_summary.json','metrics_by_basin.csv','validation_paired_weekly_predictions.csv','test_paired_weekly_predictions.csv']:
        copy_file(run/name,refs/name)
    for name in ['validation_paired_weekly_predictions.csv','test_paired_weekly_predictions.csv']:
        copy_file(total_run/name,refs/('total_'+name))
    copy_file(run/'code/model.py',refs/'baseline_model.py')
    copy_file(WEIGHTS,st/WEIGHTS.name)
    copy_file(MAPPING,st/'source_region_mapping_all52.csv')
    copy_file(PRIOR_STATIC/'static_preparation/selected_forcing_regions.geojson',st/'selected_forcing_regions.geojson')
    with np.load(data/'weekly_dataset.npz') as z:a={k:z[k].copy() for k in z.files}
    ids=a['basin_ids'];assert a['dynamic'].shape==(1816,52,3) and a['static'].shape==(52,1)
    weights=pd.read_csv(WEIGHTS);mapping=pd.read_csv(MAPPING)
    catchment_ids=weights.comid.unique().astype(int).tolist()
    table=pq.read_table(RAW,columns=['comid']+list(FIELDS.values()),filters=[('comid','in',catchment_ids)]).to_pandas()
    table.to_parquet(src/'streamcat_selected_catchments.parquet',index=False)
    extra,coverage,joined=aggregate_attributes(weights,table,mapping,ids)
    extra.to_csv(st/'static_attributes_all52.csv');coverage.to_csv(st/'valid_area_fraction_by_region.csv')
    # Preserve original elevation, all dynamics, targets, eligibility and every QC mask.
    added=extra[list(FIELDS)].to_numpy(dtype='float32')
    a['static']=np.concatenate([a['static'],added],axis=1)
    np.savez_compressed(data/'weekly_dataset.npz',**a)
    with np.load(BASELINE/'data/weekly_dataset.npz') as z:
        for key in z.files:np.testing.assert_array_equal(a[key][:,:1] if key=='static' else a[key],z[key])
    sc=json.loads((data/'train_only_scalers.json').read_text());oldsc=copy.deepcopy(sc)
    assert sc['dynamic_names'][2]=='surface_runoff_mm_week'
    train_sites=(a['eligible']&(a['split'][:,None]==0)).any(axis=0)
    means=added[train_sites].astype(float).mean(0);stds=added[train_sites].astype(float).std(0)
    assert (stds>0).all()
    for name,values in [('static_names',list(FIELDS)),('static_mean',means.tolist()),('static_std',stds.tolist())]:sc[name]=list(sc[name])+values
    sc['new_static_scaler_note']='Unweighted site mean and population std across sites with eligible training targets; original surface-runoff dynamic, elevation and target scalers retained exactly.'
    for key in oldsc:
        if key.startswith('static_'):assert sc[key][:1]==oldsc[key]
        else:assert sc[key]==oldsc[key],key
    write_json(data/'train_only_scalers.json',sc)
    attrs=pd.read_csv(data/'static_attributes.csv').set_index('elf_id').loc[ids]
    attrs.join(extra[[c for c in extra if c not in attrs]]).to_csv(data/'static_attributes.csv')
    counts={};head=np.isin(ids,HEADWATERS)
    for split,code in [('train',0),('validation',1),('test',2)]:
        mask=a['eligible']&(a['split'][:,None]==code)
        counts[split]=int(mask.sum());counts[split+'_headwater10']=int(mask[:,head].sum())
    assert counts==config['counts']
    static_audit=dict(status='passed',fields=FIELDS,sites=52,source_regions=int(weights.source_region_id.nunique()),local_catchments=len(table),
        weighting='Original ELF forcing-region intersections with NHDPlusV2 local catchments; EPSG5070 intersection area. Finite nonnegative values averaged with valid intersection areas; no value imputation.',
        spatial_support='Catchment-mean soil, groundwater, BFI and open water; impervious metric is within 100m riparian buffers, averaged using catchment intersection areas. This is a basin descriptor, not an exact basin-wide impervious percentage.',
        source_polygon_coverage_min=float(extra.forest_polygon_coverage.min()),valid_area_fraction_min=coverage.min().to_dict(),
        landcover_vintage=2019,vintage_note='User explicitly requested fixed 2019 land cover for future application. Evaluation in 2010-2016 uses post-period descriptors and is not a strict as-of-date hindcast.',
        soil_units='Source-native units retained for permcat, rckdepcat and wtdepcat; names explicitly avoid unverified conversions. Standardization uses training sites.',
        bfi_note='External static StreamCat bficat raster summary; no ELF-derived hydrologic signature or test target used as input. Does not supply time-varying baseflow.',
        scaler_training_sites=int(train_sites.sum()),static_scaler_weighting='one vote per site with eligible training targets',source_url='https://www.epa.gov/national-aquatic-resource-surveys/streamcat-dataset')
    write_json(st/'static_audit.json',static_audit)
    for name in ['static_audit.json','static_attributes_all52.csv']:copy_file(st/name,refs/name)
    record=dict(status='passed',baseline_run=str(run),total_runoff_run=str(total_run),headwaters=HEADWATERS,history_weeks=13,counts=counts,source_sha256=sources,
        original_dynamic_elevation_targets_masks_splits_and_eligibility_unchanged=True,original_surface_runoff_input_and_all_scalers_unchanged=True,
        static_features_added=list(FIELDS),static_audit=static_audit,
        static_preparation_sha256={p.name:sha256(p) for p in st.iterdir() if p.is_file()},
        source_subset_sha256=sha256(src/'streamcat_selected_catchments.parquet'),
        data_sha256={p.name:sha256(p) for p in data.iterdir() if p.is_file()},reference_sha256={p.name:sha256(p) for p in refs.iterdir() if p.is_file()})
    write_json(ROOT/'preparation_audit.json',record);write_json(ROOT/'STATUS.json',dict(status='prepared',counts=counts,statics=9))
    print('STATIC8_PREPARATION_PASSED '+json.dumps(dict(counts=counts,static_audit=static_audit)),flush=True)

if __name__=='__main__':main()
