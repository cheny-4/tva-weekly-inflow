"""Create small synthetic weekly HEADWATER-shaped maps for both model contracts.

Run from repository root: python examples/make_headwater_demo.py --out outputs/demo_maps
These are invented values for interface testing, not observed or forecast data.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
import xarray as xr
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline as p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    out = p.new_output(args.out)
    time = pd.date_range('2010-01-04', periods=16, freq='W-MON')
    lat, lon = np.arange(32., 39.01, .5), np.arange(-90., -79.99, .5)
    shape = (len(time), len(lat), len(lon))
    week = np.arange(len(time))[:, None, None]
    east = np.arange(len(lon))[None, None, :] * .05
    values = dict(precipitation=np.broadcast_to(30. + week + east, shape),
                  temperature=np.broadcast_to(278.15 + week * .4 + east, shape),
                  surface_runoff=np.broadcast_to(2. + week * .1 + east, shape))
    values['total_runoff'] = values['surface_runoff'] + 7.
    ds = xr.Dataset({name: (('time', 'lat', 'lon'), v, {'units': 'K' if name == 'temperature' else 'mm'}) for name, v in values.items()},
                    coords={'time': time, 'lat': lat, 'lon': lon})
    ds.to_netcdf(out / 'synthetic_maps.nc')
    for model_id, model in p.REGISTRY.items():
        variables = {}
        for role in ['precipitation', 'temperature', 'runoff']:
            name = role if role != 'runoff' else ('total_runoff' if model_id == 'total_runoff' else 'surface_runoff')
            variables[role] = dict(files=['synthetic_maps.nc'], variable=name, units='K' if role == 'temperature' else 'mm',
                kind='mean' if role == 'temperature' else 'amount', crs='EPSG:4326', x='lon', y='lat', time='time',
                interval_hours=168, timestamp_position='start')
        config = dict(model_id=model_id, source_kind='historical', time_basis='UTC',
                      runoff_definition=model['runoff_definition'], variables=variables,
                      example_note='Synthetic demonstration, not HEADWATER forecasts or hydrologic observations')
        (out / (model_id + '.json')).write_text(json.dumps(config, indent=2) + '\n')
    print('SYNTHETIC_MAPS_SAVED', out)


if __name__ == '__main__':
    main()
