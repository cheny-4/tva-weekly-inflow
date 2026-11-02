"""HEADWATER rectilinear NetCDF maps -> ELF-region weekly forcing CSV.

Only interval amounts/rates and interval-mean temperature are supported. Native
intervals must fit wholly inside Monday-Sunday weeks; no temporal disaggregation,
regridding, ensemble averaging, gap filling, or bias correction is performed.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def unit_key(unit):
    return str(unit).strip().lower().replace('**', '^').replace(' ', '').replace('°', 'deg')


def unit_conversion(unit, kind, role):
    key = unit_key(unit)
    if role == 'temperature':
        if kind != 'mean':
            raise ValueError('Temperature must be an interval mean')
        if key in ['k', 'kelvin']:
            return 1., -273.15
        if key in ['c', 'degc', 'celsius', 'degree_celsius', 'degrees_celsius']:
            return 1., 0.
    elif kind == 'amount':
        if key in ['mm', 'kgm-2', 'kgm^-2']:
            return 1., 0.
        if key == 'm':
            return 1000., 0.
    elif kind == 'rate':
        rates = {'mm/s': 1., 'mms-1': 1., 'mms^-1': 1., 'kgm-2s-1': 1., 'kgm^-2s^-1': 1.,
                 'mm/day': 1/86400., 'mmd-1': 1/86400., 'mm/hour': 1/3600., 'mm/h': 1/3600.,
                 'mm/week': 1/604800., 'm/s': 1000., 'ms-1': 1000., 'm/day': 1000/86400.}
        if key in rates:
            return rates[key], 0.
    raise ValueError(f'Unsupported {role} kind/units: {kind!r}, {unit!r}; declare interval amount/rate/mean explicitly')


def axis_bounds(ds, name, explicit=None):
    coord = ds[name]
    v = np.asarray(coord.values, float)
    if coord.ndim != 1 or len(v) == 0 or not np.isfinite(v).all():
        raise ValueError('Require finite one-dimensional rectilinear coordinates: ' + name)
    d = np.diff(v)
    if len(v) > 1 and not ((d > 0).all() or (d < 0).all()):
        raise ValueError('Coordinate must be strictly monotonic: ' + name)
    bound_name = explicit or coord.attrs.get('bounds') or coord.encoding.get('bounds')
    if bound_name:
        bounds = np.asarray(ds[bound_name].values, float)
        if bounds.shape != (len(v), 2):
            raise ValueError('Coordinate bounds must have shape (coordinate,2): ' + name)
    else:
        if len(v) < 2:
            raise ValueError('Single-cell axes require explicit coordinate bounds')
        edges = np.r_[v[0] - d[0]/2, (v[:-1] + v[1:])/2, v[-1] + d[-1]/2]
        bounds = np.column_stack([edges[:-1], edges[1:]])
    bounds = np.sort(bounds, axis=1)
    if not np.isfinite(bounds).all() or (bounds[:, 1] <= bounds[:, 0]).any():
        raise ValueError('Invalid cell bounds: ' + name)
    if ((v < bounds[:, 0]) | (v > bounds[:, 1])).any():
        raise ValueError('Coordinate centres must lie inside their cell bounds: ' + name)
    ordered = bounds[np.argsort(bounds[:, 0])]
    if (ordered[1:, 0] < ordered[:-1, 1] - 1e-10).any():
        raise ValueError('Overlapping grid cells: ' + name)
    return v, bounds


def intervals(ds, spec):
    time_name = spec.get('time', 'time')
    time = ds[time_name]
    if not np.issubdtype(time.dtype, np.datetime64):
        raise ValueError('Require Gregorian/standard datetime64 timestamps in UTC')
    name = spec.get('time_bounds') or time.attrs.get('bounds') or time.encoding.get('bounds')
    if name:
        values = np.asarray(ds[name].values)
        if values.shape != (len(time), 2) or not np.issubdtype(values.dtype, np.datetime64):
            raise ValueError('Require decoded time bounds with shape (time,2)')
        starts, ends = pd.DatetimeIndex(values[:, 0]), pd.DatetimeIndex(values[:, 1])
    else:
        if 'interval_hours' not in spec or spec.get('timestamp_position') not in ['start', 'end', 'midpoint']:
            raise ValueError('Provide CF time bounds or explicit interval_hours and timestamp_position; time intervals are never guessed')
        delta = pd.to_timedelta(float(spec['interval_hours']), unit='h')
        points = pd.DatetimeIndex(time.values)
        offset = {'start': 0., 'midpoint': .5, 'end': 1.}[spec['timestamp_position']]
        starts = points - offset * delta
        ends = starts + delta
    durations = (ends - starts).total_seconds().to_numpy()
    if starts.isna().any() or ends.isna().any() or (durations <= 0).any() or (durations > 604800).any():
        raise ValueError('Intervals must have positive duration no greater than seven days')
    points = pd.DatetimeIndex(time.values)
    if ((points < starts) | (points > ends)).any():
        raise ValueError('Timestamp lies outside its declared time bounds')
    weeks = starts.normalize() - pd.to_timedelta(starts.day_of_week, unit='D')
    if (ends > weeks + pd.Timedelta(days=7)).any():
        raise ValueError('An interval crosses a Monday boundary; supply aligned intervals, not disaggregated weekly/monthly totals')
    return starts, ends, durations, weeks


def build_weights(xbounds, ybounds, crs_input, regions, min_coverage):
    """Intersect fractional cells in grid CRS and measure areas in EPSG:5070."""
    from pyproj import CRS, Transformer
    from shapely.geometry import box
    from shapely import transform as geometry_transform
    from scipy.sparse import coo_matrix
    crs = CRS.from_user_input(crs_input)
    def transform(function, geometry):
        return geometry_transform(geometry, function, interleaved=False)
    to_grid = Transformer.from_crs('EPSG:4326', crs, always_xy=True).transform
    to_area = Transformer.from_crs(crs, 'EPSG:5070', always_xy=True).transform
    rows, cols, values, diagnostics = [], [], [], []
    nx = len(xbounds)
    for ri, (region_id, geo) in enumerate(regions.items()):
        region = transform(to_grid, geo)
        if not region.is_valid or region.is_empty or not np.isfinite(region.bounds).all():
            raise ValueError('Invalid transformed forcing region: ' + region_id)
        step = .01 if crs.is_geographic else 1000. / crs.axis_info[0].unit_conversion_factor
        full_area = transform(to_area, region.segmentize(step)).area
        lo_x, lo_y, hi_x, hi_y = region.bounds
        xx = np.flatnonzero((xbounds[:, 1] > lo_x) & (xbounds[:, 0] < hi_x))
        yy = np.flatnonzero((ybounds[:, 1] > lo_y) & (ybounds[:, 0] < hi_y))
        covered, cells = 0., 0
        for iy in yy:
            for ix in xx:
                cell = box(xbounds[ix, 0], ybounds[iy, 0], xbounds[ix, 1], ybounds[iy, 1])
                clipped = region.intersection(cell)
                if clipped.is_empty:
                    continue
                # Densify long cell-boundary segments before measuring projected
                # area; fractional edge cells remain fractional, including holes.
                clipped = clipped.segmentize(step)
                area = transform(to_area, clipped).area
                if area <= 0:
                    continue
                rows.append(ri); cols.append(int(iy * nx + ix)); values.append(area)
                covered += area; cells += 1
        fraction = covered / full_area
        if fraction > 1.0001 or fraction < min_coverage:
            raise ValueError(f'Grid coverage for {region_id} is {fraction:.6f}; required {min_coverage:.6f}. Extend grid or resolve CRS/bounds')
        diagnostics.append(dict(source_region_id=region_id, polygon_area_m2=full_area,
                                covered_area_m2=covered, grid_area_fraction=fraction, intersecting_cells=cells))
    weights = coo_matrix((values, (rows, cols)), shape=(len(regions), len(xbounds)*len(ybounds))).tocsr()
    return weights, pd.DataFrame(diagnostics)


def grid_weights(ds, spec, regions, state):
    from pyproj import CRS
    from scipy.sparse import load_npz, save_npz
    xname, yname = spec.get('x', 'lon'), spec.get('y', 'lat')
    xv, xb = axis_bounds(ds, xname, spec.get('x_bounds'))
    yv, yb = axis_bounds(ds, yname, spec.get('y_bounds'))
    if not spec.get('crs'):
        raise ValueError('Declare the grid CRS explicitly for every variable')
    crs = CRS.from_user_input(spec['crs'])
    variable = ds[spec['variable']]
    grid_mapping = variable.attrs.get('grid_mapping') or variable.encoding.get('grid_mapping')
    if grid_mapping and grid_mapping in ds:
        attributes = ds[grid_mapping].attrs
        wkt = attributes.get('spatial_ref') or attributes.get('crs_wkt')
        declared_crs = CRS.from_wkt(wkt) if wkt else CRS.from_cf(attributes)
        if not crs.equals(declared_crs, ignore_axis_order=True):
            raise ValueError('Configured CRS disagrees with NetCDF grid_mapping')
    if crs.is_geographic and ((xb < -180).any() or (xb > 180).any() or (yb < -90).any() or (yb > 90).any()):
        raise ValueError('Geographic grid must use longitude -180..180 and latitude -90..90; normalize coordinates first')
    if not crs.is_geographic and not crs.is_projected:
        raise ValueError('Only geographic or projected rectilinear grids are supported')
    for name in [xname, yname]:
        declared = unit_key(ds[name].attrs.get('units', ''))
        if crs.is_geographic and declared and declared not in ['degrees_east', 'degrees_north', 'degree_east', 'degree_north', 'degrees', 'degree']:
            raise ValueError('Geographic coordinate axes must use degrees')
        if crs.is_projected and declared and declared not in ['m', 'metre', 'metres', 'meter', 'meters']:
            raise ValueError('Projected coordinate values must use metres; convert axes explicitly first')
    if crs.is_projected and any(abs(a.unit_conversion_factor-1) > 1e-10 for a in crs.axis_info[:2]):
        raise ValueError('Projected CRS axes must use metres')
    key = hashlib.sha256(str((len(yv), len(xv))).encode() + xb.tobytes() + yb.tobytes() + crs.to_wkt().encode() +
                         state['boundary_sha256'].encode() + '\n'.join(regions).encode()).hexdigest()
    if key not in state['cache']:
        target = state['weights_dir'] / key
        reuse = state.get('reuse_weights')
        cached = Path(reuse) / key if reuse else None
        if cached and cached.with_suffix('.json').exists() and cached.with_suffix('.npz').exists():
            meta = json.loads(cached.with_suffix('.json').read_text())
            if meta['grid_sha256'] != key or meta['region_ids'] != list(regions):
                raise ValueError('Cached grid/region identity mismatch')
            if digest(cached.with_suffix('.npz')) != meta['weights_file_sha256']:
                raise ValueError('Cached weights checksum mismatch')
            weights = load_npz(cached.with_suffix('.npz'))
            if weights.shape != (len(regions), len(xv)*len(yv)):
                raise ValueError('Cached weights shape mismatch')
            diag = pd.DataFrame(meta['coverage'])
            if (diag.grid_area_fraction < state['min_grid_coverage']).any():
                raise ValueError('Cached weights fail requested spatial coverage')
        else:
            weights, diag = build_weights(xb, yb, crs, regions, state['min_grid_coverage'])
        save_npz(target.with_suffix('.npz'), weights)
        meta = dict(grid_sha256=key, boundary_sha256=state['boundary_sha256'], region_ids=list(regions),
                    crs=crs.to_string(), shape=[len(yv), len(xv)], coverage=diag.to_dict('records'),
                    weights_file_sha256=digest(target.with_suffix('.npz')))
        target.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n')
        state['cache'][key] = (weights, diag)
    return key, state['cache'][key], (yname, xname)


def extract_variable(role, spec, config_dir, regions, state, source_kind, issue_time):
    import xarray as xr
    kind, units = spec['kind'], spec['units']
    factor, offset = unit_conversion(units, kind, role)
    files = spec.get('files')
    if not isinstance(files, list) or not files:
        raise ValueError('Each variable requires a nonempty files list; glob expansion is not implicit')
    records, audits = [], []
    for filename in files:
        path = (config_dir / filename).resolve()
        with xr.open_dataset(path, decode_coords='all') as ds:
            selections = spec.get('isel', {})
            if any(k in selections for k in [spec.get('x', 'lon'), spec.get('y', 'lat'), spec.get('time', 'time')]):
                raise ValueError('isel is for explicit ensemble/initialization dimensions only, not time or grid subsetting')
            if selections:
                ds = ds.isel(selections)
            variable = ds[spec['variable']]
            if variable.attrs.get('units'):
                actual = unit_conversion(variable.attrs['units'], kind, role)
                if actual != (factor, offset):
                    raise ValueError('Configured units disagree with NetCDF variable units: ' + spec['variable'])
            key, (weights, diag), (yn, xn) = grid_weights(ds, spec, regions, state)
            time_name = spec.get('time', 'time')
            if set(variable.dims) != {time_name, yn, xn} or variable.ndim != 3:
                raise ValueError('Require dimensions (time,y,x); explicitly select one forecast member/initialization with isel')
            starts, ends, seconds, weeks = intervals(ds, spec)
            if source_kind == 'forecast' and (starts < issue_time).any():
                raise ValueError('Forecast intervals begin before issue_time; supply prior observations via history_csv instead')
            total_area = np.asarray(weights.sum(axis=1)).ravel()
            active_cells = np.unique(weights.indices)
            active_weights = weights[:, active_cells]
            ny, nx = variable.sizes[yn], variable.sizes[xn]
            # Read one map at a time: never load the complete gridded archive.
            for ti, (start, end, duration, week) in enumerate(zip(starts, ends, seconds, weeks)):
                values = np.asarray(variable.isel({time_name: ti}).transpose(yn, xn).values, float).reshape(ny*nx)[active_cells]
                finite = np.isfinite(values)
                if role != 'temperature' and (values[finite] < 0).any():
                    raise ValueError('Negative precipitation/runoff in intersected cells; resolve source QC before preprocessing')
                area = np.asarray(active_weights @ finite.astype(float)).ravel()
                coverage = area / total_area
                numerator = np.asarray(active_weights @ np.where(finite, values, 0.)).ravel()
                mean = np.divide(numerator, area, out=np.full(len(regions), np.nan), where=area > 0)
                mean[coverage < state['min_valid_coverage']] = np.nan
                mean = mean * factor + offset
                if kind == 'rate':
                    mean *= duration
                records.append(dict(start=start, end=end, week=week, seconds=duration, values=mean, coverage=coverage))
            audits.append(dict(role=role, file=str(path), sha256=digest(path), variable=spec['variable'],
                               grid_sha256=key, kind=kind, units=units, intervals=len(starts), isel=selections))
    records.sort(key=lambda r: r['start'])
    for a, b in zip(records, records[1:]):
        if b['start'] < a['end']:
            raise ValueError('Overlapping/duplicate time intervals across files for ' + role)
    groups = {}
    for r in records:
        groups.setdefault(r['week'], []).append(r)
    output, quality = [], []
    for week, group in sorted(groups.items()):
        duration = sum(r['seconds'] for r in group)
        full = abs(duration - 604800) < 1e-6
        values = np.stack([r['values'] for r in group])
        valid = np.isfinite(values).all(axis=0) & full
        result = np.sum(values, axis=0) if role != 'temperature' else np.sum(values * np.array([r['seconds'] for r in group])[:, None], axis=0) / duration
        result[~valid] = np.nan
        coverage = np.min(np.stack([r['coverage'] for r in group]), axis=0)
        for ri, region in enumerate(regions):
            output.append(dict(source_region_id=region, start=week, value=result[ri]))
            quality.append(dict(role=role, source_region_id=region, start=week, intervals=len(group),
                                time_coverage_fraction=duration/604800., minimum_valid_area_fraction=coverage[ri], valid=bool(valid[ri])))
    return pd.DataFrame(output), pd.DataFrame(quality), audits


def preprocess(config_path, output_path, model_id):
    from shapely.geometry import shape
    import pipeline as p
    config_path = Path(config_path).resolve()
    cfg = p.read_json(config_path)
    base, _, model = p.model_paths(model_id)
    if cfg.get('model_id') != model_id or cfg.get('runoff_definition') != model['runoff_definition']:
        raise ValueError('Map config model_id/runoff_definition must match the selected checkpoint; runoff is never relabelled')
    if cfg.get('time_basis') != 'UTC' or cfg.get('source_kind') not in ['historical', 'forecast']:
        raise ValueError('Specify time_basis UTC and source_kind historical/forecast')
    issue = None
    if cfg['source_kind'] == 'forecast':
        if not cfg.get('issue_time') or not cfg.get('member_id'):
            raise ValueError('Forecast maps require explicit UTC issue_time and member_id; process each member/issue separately')
        issue = pd.Timestamp(cfg['issue_time'])
        if issue.tz is None:
            raise ValueError('issue_time must include UTC offset or Z')
        issue = issue.tz_convert('UTC').tz_localize(None)
    spatial = p.ROOT / 'spatial'
    boundary_path = spatial / 'forcing_regions.geojson'
    manifest = p.read_json(spatial / 'manifest.json')
    for name, checksum in manifest['files'].items():
        if digest(spatial / name) != checksum:
            raise ValueError('Frozen spatial artifact checksum mismatch: ' + name)
    mapping = pd.read_csv(spatial / 'site_region_mapping.csv')
    ids = cfg.get('sites', mapping.basin_id.tolist())
    if len(ids) != len(set(ids)) or not ids or set(ids) - set(mapping.basin_id):
        raise ValueError('sites must be unique existing trained ELF IDs')
    mapping = mapping.set_index('basin_id').loc[ids].reset_index()
    geos = json.loads(boundary_path.read_text())
    available = {f['properties']['elf_id']: shape(f['geometry']) for f in geos['features']}
    regions = {name: available[name] for name in sorted(set(mapping.source_region_id))}
    if any(not g.is_valid or g.is_empty for g in regions.values()):
        raise ValueError('Invalid frozen forcing-region geometry')
    out = p.new_output(output_path)
    (out / 'weights').mkdir()
    state = dict(weights_dir=out / 'weights', cache={}, boundary_sha256=digest(boundary_path),
                 min_grid_coverage=float(cfg.get('minimum_grid_coverage', .99)),
                 min_valid_coverage=float(cfg.get('minimum_valid_area_fraction', .99)))
    if not all(0 < state[k] <= 1 for k in ['min_grid_coverage', 'min_valid_coverage']):
        raise ValueError('Spatial coverage thresholds must be in (0,1]')
    if cfg.get('reuse_weights_from'):
        state['reuse_weights'] = str((config_path.parent / cfg['reuse_weights_from']).resolve())
    region_frames, qcs, sources = [], [], []
    for role, feature in zip(['precipitation', 'temperature', 'runoff'], model['dynamic_names']):
        frame, qc, audit = extract_variable(role, cfg['variables'][role], config_path.parent, regions, state, cfg['source_kind'], issue)
        region_frames.append(frame.rename(columns={'value': feature}))
        qcs.append(qc); sources.extend(audit)
    combined = region_frames[0]
    for frame in region_frames[1:]:
        combined = combined.merge(frame, on=['source_region_id', 'start'], how='outer', validate='one_to_one')
    weeks = pd.date_range(combined.start.min(), combined.start.max(), freq='W-MON')
    index = pd.MultiIndex.from_product([list(regions), weeks], names=['source_region_id', 'start'])
    combined = combined.set_index(['source_region_id', 'start']).reindex(index).reset_index()
    weekly = mapping[['basin_id', 'source_region_id']].merge(combined, on='source_region_id', validate='many_to_many')
    weekly = weekly[['basin_id', 'start', *model['dynamic_names']]].sort_values(['basin_id', 'start'])
    if cfg.get('history_csv'):
        if not cfg.get('history_metadata'):
            raise ValueError('history_csv requires history_metadata with matching physical units and runoff definition')
        history_meta = p.read_json(config_path.parent / cfg['history_metadata'])
        p.validate_metadata(history_meta, model_id)
        if history_meta['source_kind'] != 'historical':
            raise ValueError('History context must be declared historical')
        history = pd.read_csv(config_path.parent / cfg['history_csv'], parse_dates=['start'])
        history = history[history.basin_id.isin(ids)][['basin_id', 'start', *model['dynamic_names']]]
        if history.duplicated(['basin_id', 'start']).any() or (history.start.dt.day_of_week != 0).any():
            raise ValueError('History requires unique Monday-start rows')
        if (history.start >= weeks.min()).any() or (issue is not None and (history.start + pd.Timedelta(days=7) > issue).any()):
            raise ValueError('History must end before forecast issue time and not overlap map weeks')
        if (history[[model['dynamic_names'][0], model['dynamic_names'][2]]] < 0).any().any():
            raise ValueError('Negative precipitation/runoff in history')
        weekly = pd.concat([history, weekly], ignore_index=True).sort_values(['basin_id', 'start'])
        sources.append(dict(role='history_context', file=str((config_path.parent / cfg['history_csv']).resolve()), sha256=digest(config_path.parent / cfg['history_csv'])))
    weekly['start'] = weekly.start.dt.strftime('%Y-%m-%d')
    weekly.to_csv(out / 'weekly_inputs.csv', index=False)
    pd.concat(qcs, ignore_index=True).to_csv(out / 'weekly_quality_by_region.csv', index=False)
    mapping.to_csv(out / 'site_region_mapping.csv', index=False)
    metadata = dict(model_id=model_id, calendar='Monday-Sunday', precipitation_units='mm/week',
                    temperature_units='degC', runoff_units='mm/week', runoff_definition=model['runoff_definition'],
                    source_kind=cfg['source_kind'], time_basis='UTC', boundary_sha256=digest(boundary_path))
    if issue is not None:
        metadata.update(issue_time=cfg['issue_time'], member_id=cfg['member_id'])
    p.write_json(out / 'forcing_metadata.json', metadata)
    p.write_json(out / 'preprocessing_audit.json', dict(status='completed', model_id=model_id, bias_correction=False,
       config_sha256=digest(config_path), spatial_manifest_sha256=digest(spatial / 'manifest.json'),
       source_files=sources, spatial_weighting='Fractional cell/ELF forcing-region overlap areas measured in EPSG:5070; finite-area renormalization above declared coverage threshold',
       minimum_grid_coverage=state['min_grid_coverage'], minimum_valid_area_fraction=state['min_valid_coverage'],
       region_count=len(regions), site_count=len(ids), output_rows=len(weekly),
       invalid_week_rows=int(weekly[model['dynamic_names']].isna().any(axis=1).sum()),
       temporal_weighting='Amounts summed; interval-mean rates integrated by seconds; interval-mean temperature duration-weighted; only complete calendar weeks',
       static_inputs='Use frozen model-specific basin statics at inference; no dynamic baseflow or new elevation maps inserted',
       limitation='Adapter tested with synthetic NetCDF, not actual future HEADWATER output. Rectilinear geographic/projected grids and explicit aligned intervals only; one issue/member per run. Current ELF boundary reconstructions/proxies retain their original qualifications.'))
    print('HEADWATER_PREPROCESSING_SAVED', out, len(weekly), 'rows', flush=True)
