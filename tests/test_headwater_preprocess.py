from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box
from shapely import transform as geometry_transform
from pyproj import Transformer
import headwater_preprocess as h
import json
import pipeline as p

def transform(function, geometry):
    return geometry_transform(geometry, function, interleaved=False)


class MapPreprocessingTests(unittest.TestCase):
    def test_fractional_cell_area_weighting_is_not_equal_cell_mean(self):
        to_geo = Transformer.from_crs('EPSG:5070', 'EPSG:4326', always_xy=True).transform
        region = transform(to_geo, box(0, 0, 1500, 1000))
        weights, coverage = h.build_weights(np.array([[0,1000],[1000,2000]]), np.array([[0,1000]]), 'EPSG:5070', {'region': region}, .99)
        mean = np.asarray(weights @ np.array([10., 40.]))[0] / weights.sum()
        self.assertAlmostEqual(float(mean), 20., places=7)
        self.assertAlmostEqual(coverage.grid_area_fraction.iloc[0], 1., places=7)

    def test_incomplete_grid_fails_instead_of_scaling_small_patch(self):
        to_geo = Transformer.from_crs('EPSG:5070', 'EPSG:4326', always_xy=True).transform
        region = transform(to_geo, box(0, 0, 2000, 1000))
        with self.assertRaisesRegex(ValueError, 'Grid coverage'):
            h.build_weights(np.array([[0,1000]]), np.array([[0,1000]]), 'EPSG:5070', {'region': region}, .99)

    def test_unit_conversions_and_explicit_time_support(self):
        factor, offset = h.unit_conversion('K', 'mean', 'temperature')
        self.assertAlmostEqual(280 * factor + offset, 6.85)
        factor, _ = h.unit_conversion('kg m-2 s-1', 'rate', 'runoff')
        self.assertAlmostEqual(factor * .001 * 86400, 86.4)
        ds = xr.Dataset(coords={'time': pd.date_range('2010-01-04', periods=7, freq='D')})
        with self.assertRaisesRegex(ValueError, 'time bounds'):
            h.intervals(ds, {})
        starts, ends, seconds, weeks = h.intervals(ds, {'interval_hours':24, 'timestamp_position':'start'})
        self.assertEqual(sum(seconds), 604800)
        self.assertEqual(len(set(weeks)), 1)
        ds2 = xr.Dataset(coords={'time': [pd.Timestamp('2010-01-06')]})
        with self.assertRaisesRegex(ValueError, 'crosses a Monday'):
            h.intervals(ds2, {'interval_hours':168, 'timestamp_position':'start'})

    def test_daily_extraction_sums_amounts_averages_temperature_and_flags_missing(self):
        # Real geographic cells with one region, explicit time intervals, and a
        # missing day: compare physical amounts before any neural normalization.
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            times = pd.date_range('2010-01-04', periods=14, freq='D')
            v = np.full((14,2,2), 2.)
            v[10] = np.nan
            ds = xr.Dataset({'p':(('time','lat','lon'),v,{'units':'mm'}),
                             'q':(('time','lat','lon'),np.full((14,2,2),.001),{'units':'kg m-2 s-1'}),
                             't':(('time','lat','lon'),np.full((14,2,2),280.),{'units':'K'})},
                            coords={'time':times,'lat':[35.,36.],'lon':[-85.,-84.]})
            ds.to_netcdf(folder/'maps.nc')
            spec=dict(files=['maps.nc'],variable='p',kind='amount',units='mm',crs='EPSG:4326',interval_hours=24,timestamp_position='start')
            state=dict(weights_dir=folder,cache={},boundary_sha256='synthetic',min_grid_coverage=.99,min_valid_coverage=.99)
            regions={'test':box(-85.25,35.25,-84.75,35.75)}
            out,qc,_=h.extract_variable('precipitation',spec,folder,regions,state,'historical',None)
            self.assertAlmostEqual(out.value.iloc[0],14.)
            self.assertTrue(pd.isna(out.value.iloc[1]))
            self.assertFalse(qc.valid.iloc[1])
            spec.update(variable='t',kind='mean',units='K')
            temp,_,_=h.extract_variable('temperature',spec,folder,regions,state,'historical',None)
            np.testing.assert_allclose(temp.value,6.85)
            spec.update(variable='q',kind='rate',units='kg m-2 s-1')
            rate,_,_=h.extract_variable('runoff',spec,folder,regions,state,'historical',None)
            np.testing.assert_allclose(rate.value,604.8)
            spec.update(variable='t',kind='mean',units='K')
            with self.assertRaisesRegex(ValueError,'begin before issue_time'):
                h.extract_variable('temperature',spec,folder,regions,state,'forecast',pd.Timestamp('2010-01-05'))

    def test_cf_time_bounds_preserve_duration(self):
        starts=pd.date_range('2010-01-04',periods=7,freq='D')
        ds=xr.Dataset({'time_bounds':(('time','bound'),np.column_stack([starts.values,(starts+pd.Timedelta(days=1)).values]))},coords={'time':starts+pd.Timedelta(hours=12)})
        ds.time.attrs['bounds']='time_bounds'
        first,last,seconds,weeks=h.intervals(ds,{})
        self.assertEqual(first[0],pd.Timestamp('2010-01-04'))
        self.assertEqual(last[-1],pd.Timestamp('2010-01-11'))
        self.assertEqual(seconds.sum(),604800)

    def test_forecast_map_with_prior_context_runs_inference_and_blocks_future_history(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            ds=xr.Dataset({name:(('time','lat','lon'),np.full((1,2,2),value),{'units':unit}) for name,value,unit in [('p',20.,'mm'),('t',280.,'K'),('q',2.,'mm')]},coords={'time':[pd.Timestamp('2027-03-01')],'lat':[35.,38.],'lon':[-86.,-82.]})
            ds.to_netcdf(folder/'maps.nc')
            variables={role:dict(files=['maps.nc'],variable=name,kind='mean' if role=='temperature' else 'amount',units='K' if role=='temperature' else 'mm',crs='EPSG:4326',interval_hours=168,timestamp_position='start') for role,name in [('precipitation','p'),('temperature','t'),('runoff','q')]}
            cfg=dict(model_id='surface_runoff_static8',source_kind='forecast',runoff_definition='surface_runoff_only',time_basis='UTC',issue_time='2027-03-01T00:00:00Z',member_id='control',sites=['ELF13'],variables=variables,history_csv='history.csv',history_metadata='history.json')
            history=pd.DataFrame(dict(basin_id='ELF13',start=pd.date_range('2026-12-07',periods=12,freq='W-MON'),precipitation_mm_week=20.,temperature_mean_c=6.85,surface_runoff_mm_week=2.))
            history.to_csv(folder/'history.csv',index=False)
            meta=p.read_json(p.ROOT/'examples/forcing_metadata_surface_static8.json')
            (folder/'history.json').write_text(json.dumps(meta))
            (folder/'config.json').write_text(json.dumps(cfg))
            h.preprocess(folder/'config.json',folder/'prepared','surface_runoff_static8')
            weekly=pd.read_csv(folder/'prepared/weekly_inputs.csv')
            result,_=p.infer_frame(weekly,p.read_json(folder/'prepared/forcing_metadata.json'),model_id='surface_runoff_static8')
            self.assertEqual(len(result),1)
            self.assertEqual(result.status.iloc[0],'predicted')
            self.assertEqual(result.start.iloc[0],'2027-03-01')
            self.assertTrue(np.isfinite(result.predicted_volume_m3.iloc[0]))
            history.loc[11,'start']=pd.Timestamp('2027-03-01')
            history.to_csv(folder/'history.csv',index=False)
            with self.assertRaisesRegex(ValueError,'History must end before'):
                h.preprocess(folder/'config.json',folder/'bad_history','surface_runoff_static8')

    def test_overlapping_intervals_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            ds=xr.Dataset({'p':(('time','lat','lon'),np.ones((2,2,2)),{'units':'mm'})},coords={'time':pd.to_datetime(['2010-01-04','2010-01-04']),'lat':[35.,36.],'lon':[-85.,-84.]})
            ds.to_netcdf(folder/'maps.nc')
            spec=dict(files=['maps.nc'],variable='p',kind='amount',units='mm',crs='EPSG:4326',interval_hours=24,timestamp_position='start')
            state=dict(weights_dir=folder,cache={},boundary_sha256='synthetic',min_grid_coverage=.99,min_valid_coverage=.99)
            with self.assertRaisesRegex(ValueError,'Overlapping/duplicate'):
                h.extract_variable('precipitation',spec,folder,{'test':box(-85.25,35.25,-84.75,35.75)},state,'historical',None)

    def test_ensemble_dimension_requires_explicit_member_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            v=np.stack([np.full((7,2,2),2.),np.full((7,2,2),3.)])
            ds=xr.Dataset({'p':(('member','time','lat','lon'),v,{'units':'mm'})},coords={'member':[0,1],'time':pd.date_range('2010-01-04',periods=7,freq='D'),'lat':[35.,36.],'lon':[-85.,-84.]})
            ds.to_netcdf(folder/'maps.nc')
            spec=dict(files=['maps.nc'],variable='p',kind='amount',units='mm',crs='EPSG:4326',interval_hours=24,timestamp_position='start')
            state=dict(weights_dir=folder,cache={},boundary_sha256='synthetic',min_grid_coverage=.99,min_valid_coverage=.99)
            regions={'test':box(-85.25,35.25,-84.75,35.75)}
            with self.assertRaisesRegex(ValueError,'explicitly select one forecast member'):
                h.extract_variable('precipitation',spec,folder,regions,state,'historical',None)
            spec['isel']={'member':1}
            result,_,_=h.extract_variable('precipitation',spec,folder,regions,state,'historical',None)
            self.assertAlmostEqual(result.value.iloc[0],21.)

    def test_conflicting_netcdf_crs_is_rejected(self):
        from pyproj import CRS
        ds=xr.Dataset({'p':(('time','lat','lon'),np.ones((1,2,2)),{'units':'mm','grid_mapping':'crs'}),'crs':((),0,CRS.from_epsg(5070).to_cf())},coords={'time':[pd.Timestamp('2010-01-04')],'lat':[35.,36.],'lon':[-85.,-84.]})
        spec=dict(variable='p',crs='EPSG:4326')
        with self.assertRaisesRegex(ValueError,'CRS disagrees'):
            h.grid_weights(ds,spec,{}, {})


if __name__ == '__main__':
    unittest.main()
