import os
import json
import boto3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime, timezone, timedelta
from botocore import UNSIGNED
from botocore.client import Config
import cfgrib
from scipy.interpolate import griddata
plt.switch_backend('Agg')

# --- CONFIG ---
BUCKET_NAME = 'noaa-rtma-pds'
REGION = 'us-east-1'
DATA_DIR = 'site/data'
DATA_STRIDE = 4  # Downsample factor for interactive lookup binary files

def reproject_to_latlon(data, lats_2d, lons_2d, stride=DATA_STRIDE):
    """Reproject LCC-gridded data onto a Web Mercator-aligned grid.

    The RTMA NDFD grid uses Lambert Conformal Conic projection.  Mapbox GL JS
    displays raster image sources by bilinearly interpolating from the four
    corner coordinates in Web Mercator screen space.  An equirectangular
    (uniform-degree) target grid would still mis-align interior pixels because
    equal-degree latitude steps are NOT equal Mercator-Y steps — mid-CONUS
    rows can appear up to ~2.7° off from their true position.

    Fix: create a target grid whose rows are equally spaced in Web Mercator Y
    (metres).  Each image row then corresponds to an equal Mercator-Y step,
    which Mapbox interpolates correctly, giving exact alignment with the base
    map's coastlines and borders.

    Returns (reprojected [row 0 = southernmost], lat_min, lat_max, lon_min, lon_max).
    """
    R = 6378137.0  # WGS84 equatorial radius in metres

    data_sub = data[::stride, ::stride]
    lats_sub = lats_2d[::stride, ::stride]
    lons_sub = lons_2d[::stride, ::stride]
    nrows, ncols = data_sub.shape

    lat_min = float(lats_2d.min())
    lat_max = float(lats_2d.max())
    lon_min = float(lons_2d.min())
    lon_max = float(lons_2d.max())

    # Web Mercator Y bounds (metres)
    y_min = R * np.log(np.tan(np.pi / 4 + np.radians(lat_min) / 2))
    y_max = R * np.log(np.tan(np.pi / 4 + np.radians(lat_max) / 2))

    # Build a grid that is uniform in Mercator-Y (rows) and uniform in
    # longitude (columns), then convert back to geographic lat/lon so
    # griddata can query the LCC source points.
    y_grid = np.linspace(y_min, y_max, nrows)   # uniform Mercator-Y steps
    x_grid = np.linspace(np.radians(lon_min) * R, np.radians(lon_max) * R, ncols)
    xmesh, ymesh = np.meshgrid(x_grid, y_grid)

    lat_target = np.degrees(2 * np.arctan(np.exp(ymesh / R)) - np.pi / 2)
    lon_target = np.degrees(xmesh / R)

    values = data_sub.ravel()
    valid = np.isfinite(values)

    reprojected = griddata(
        np.column_stack([lats_sub.ravel()[valid], lons_sub.ravel()[valid]]),
        values[valid],
        (lat_target, lon_target),
        method='linear',
        fill_value=np.nan,
    )
    return reprojected, lat_min, lat_max, lon_min, lon_max


def compute_apparent_temp(temp_k, wind_ms, dpt_k):
    temp_c = temp_k - 273.15
    dpt_c = dpt_k - 273.15
    temp_f = temp_c * 9/5 + 32
    wind_mph = wind_ms * 2.23694

    # Calculate relative humidity
    e = 6.112 * np.exp(17.67 * dpt_c / (dpt_c + 243.5))
    es = 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))
    rh = 100 * (e / es)

    # Heat index (in F)
    hi = (-42.379 + 2.04901523 * temp_f + 10.14333127 * rh - 0.22475541 * temp_f * rh -
          0.00683783 * temp_f**2 - 0.05481717 * rh**2 + 0.00122874 * temp_f**2 * rh +
          0.00085282 * temp_f * rh**2 - 0.00000199 * temp_f**2 * rh**2)

    # Wind chill (in F)
    wc = (35.74 + 0.6215 * temp_f - 35.75 * wind_mph**0.16 +
          0.4275 * temp_f * wind_mph**0.16)

    # Apparent temperature logic
    apparent = np.where((temp_f > 80) & (rh > 40), hi,
                        np.where((temp_f < 50) & (wind_mph > 3), wc, temp_f))

    return apparent

def compute_relative_humidity(temp_k, dpt_k):
    temp_c = temp_k - 273.15
    dpt_c = dpt_k - 273.15
    e = 6.112 * np.exp(17.67 * dpt_c / (dpt_c + 243.5))
    es = 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))
    rh = np.clip(100 * (e / es), 0, 100)
    return rh

def save_image(data, name, cmap, vmin, vmax):
    os.makedirs(DATA_DIR, exist_ok=True)
    output_path = os.path.join(DATA_DIR, f"{name}.png")

    if data is None or np.isnan(data).all():
        print(f"!!! Skipping {name}: Data is null or all NaNs")
        return

    height, width = data.shape
    aspect = height / float(width)
    fig_width = 8
    fig_height = fig_width * aspect
    dpi = 150

    fig = plt.figure(figsize=(fig_width, fig_height), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    ax.imshow(data, cmap=cmap, norm=norm, origin='lower', aspect='auto')

    plt.savefig(output_path, transparent=True, pad_inches=0, format='png', dpi=dpi)
    plt.close(fig)
    print(f"✅ SAVED: {output_path} ({os.path.getsize(output_path)} bytes)")


def save_image_with_alpha(data, alpha_arr, name, cmap, vmin, vmax):
    """Save image with per-pixel alpha channel (0=transparent, 1=opaque)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    output_path = os.path.join(DATA_DIR, f"{name}.png")

    if data is None or np.isnan(data).all():
        print(f"!!! Skipping {name}: Data is null or all NaNs")
        return

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap(cmap)
    rgba = colormap(norm(np.ma.masked_invalid(data)))  # (H, W, 4), values 0-1
    # Out-of-domain pixels (NaN from reprojection) should be fully transparent
    rgba[:, :, 3] = np.where(np.isfinite(data), np.clip(alpha_arr, 0, 1), 0)

    height, width = data.shape
    aspect = height / float(width)
    fig_width = 8
    fig_height = fig_width * aspect
    dpi = 150

    fig = plt.figure(figsize=(fig_width, fig_height), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(rgba, origin='lower', aspect='auto')

    plt.savefig(output_path, transparent=True, pad_inches=0, format='png', dpi=dpi)
    plt.close(fig)
    print(f"✅ SAVED: {output_path} ({os.path.getsize(output_path)} bytes)")

def save_data(data, name, stride=DATA_STRIDE):
    """Save a downsampled float32 binary data file for interactive value lookups.

    Row 0 = southernmost (matches imshow origin='lower').
    Data is saved as little-endian float32, row-major.
    Returns (rows, cols) of the downsampled grid, or None on failure.
    """
    if data is None or np.isnan(data).all():
        return None
    os.makedirs(DATA_DIR, exist_ok=True)
    downsampled = data[::stride, ::stride].astype('<f4')  # little-endian float32
    output_path = os.path.join(DATA_DIR, f"{name}_data.bin")
    downsampled.flatten().tofile(output_path)
    rows, cols = downsampled.shape
    print(f"✅ SAVED data: {output_path} ({rows}x{cols}, {os.path.getsize(output_path)} bytes)")
    return (rows, cols)

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    s3 = boto3.client('s3', region_name=REGION, config=Config(signature_version=UNSIGNED))

    # Get latest available file (try last 3 days)
    current_day = datetime.now(timezone.utc)
    target_key = None
    data_timestamp = None
    for d in range(0, 3):
        day = (current_day - timedelta(days=d)).strftime('%Y%m%d')
        prefix = f"rtma2p5_ru.{day}/"
        resp = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)

        if 'Contents' in resp:
            keys = [obj['Key'] for obj in resp['Contents']
                    if '2dvaranl_ndfd.grb2' in obj['Key'] and not obj['Key'].endswith('.idx')]
            if keys:
                target_key = sorted(keys)[-1]
                parts = target_key.split('.')
                # RTMA-RU files: rtma2p5_ru.tHHMMz.2dvaranl_ndfd.grb2
                # parts[2] = 'tHHMMz', e.g. 't0815z' → hour='08', minute='15'
                time_part = parts[2]  # e.g. 't0815z'
                hour_str = time_part[1:3]
                minute_str = time_part[3:5]
                data_timestamp = datetime(int(day[0:4]), int(day[4:6]), int(day[6:8]),
                                          int(hour_str), int(minute_str), 0,
                                          tzinfo=timezone.utc).isoformat()
                break

    if target_key is None:
        print("No data found in the last 3 days.")
        return

    local_file = 'latest.grb2'
    print(f"Downloading {target_key}...")
    s3.download_file(BUCKET_NAME, target_key, local_file)

    print("Opening GRIB datasets...")
    datasets = cfgrib.open_datasets(local_file, backend_kwargs={'indexpath': ''})

    processed_vars = []
    vars_data = {}
    data_shape = None  # (rows, cols) of the reprojected lat/lon grid
    meta_bounds = None
    meta_corners = None
    lats_2d = None   # shared LCC lat/lon coordinate arrays for reprojection
    lons_2d = None

    for i, ds in enumerate(datasets):
        print(f"Dataset #{i}: {list(ds.data_vars)}")

        # Extract coordinate arrays from the first variable in this dataset that
        # carries them, so any variable processed below can use them even if
        # temperature hasn't been seen yet (cfgrib splits variables across
        # multiple Datasets in arbitrary order).
        if lats_2d is None:
            for _vname in ds.data_vars:
                try:
                    _var = ds[_vname]
                    if hasattr(_var, 'latitude') and hasattr(_var, 'longitude'):
                        lats_2d = _var.latitude.values
                        lons_2d = _var.longitude.values.copy()
                        if lons_2d.max() >= 180:
                            lons_2d = lons_2d - 360
                        print(f"  -> Coordinate arrays from '{_vname}'")
                        break
                except Exception:
                    pass

        # Temperature
        for t_key in ['t2m', '2t', 't']:
            if t_key in ds and 'temp' not in processed_vars:
                print(f"-> Temperature ({t_key})")
                temp_f = (ds[t_key] - 273.15) * 9/5 + 32
                vars_data['temp_k'] = ds[t_key].values

                # Refresh coordinate arrays from the temperature variable
                # (same grid, but ensures lats_2d/lons_2d match temp's shape)
                lats_2d = ds[t_key].latitude.values
                lons_2d = ds[t_key].longitude.values.copy()
                if lons_2d.max() >= 180:  # 0-360 → -180 to 180
                    lons_2d = lons_2d - 360

                # Reproject LCC → regular lat/lon to eliminate south-bias from
                # linear Mapbox interpolation between LCC corner coordinates.
                reproj, lat_min, lat_max, lon_min, lon_max = reproject_to_latlon(
                    temp_f.values, lats_2d, lons_2d)
                save_image(reproj, 'temp', 'turbo', -20, 110)
                shape = save_data(reproj, 'temp', stride=1)
                if shape and data_shape is None:
                    data_shape = shape
                processed_vars.append('temp')

                # Rectangular lat/lon bounds — correct for equirectangular image
                meta_corners = [
                    [lon_min, lat_max],  # topLeft  (NW)
                    [lon_max, lat_max],  # topRight (NE)
                    [lon_max, lat_min],  # bottomRight (SE)
                    [lon_min, lat_min],  # bottomLeft  (SW)
                ]
                meta_bounds = [[lat_min, lon_min], [lat_max, lon_max]]

        # Wind
        if 'u10' in ds and 'v10' in ds and 'wind' not in processed_vars:
            print("-> Wind (u10/v10)")
            wind_mph = np.sqrt(ds['u10']**2 + ds['v10']**2) * 2.23694
            vars_data['wind_ms'] = np.sqrt(ds['u10'].values**2 + ds['v10'].values**2)
            wind_alpha = np.clip(wind_mph.values / 10.0, 0, 1)
            if lats_2d is not None:
                reproj, *_ = reproject_to_latlon(wind_mph.values, lats_2d, lons_2d)
                reproj_wa, *_ = reproject_to_latlon(wind_alpha, lats_2d, lons_2d)
                reproj_wa = np.where(np.isfinite(reproj), np.clip(reproj_wa, 0, 1), 0)
                save_image_with_alpha(reproj, reproj_wa, 'wind', 'viridis', 0, 60)
                shape = save_data(reproj, 'wind', stride=1)
            else:
                save_image_with_alpha(wind_mph.values, wind_alpha, 'wind', 'viridis', 0, 60)
                shape = save_data(wind_mph.values, 'wind')
            if shape and data_shape is None:
                data_shape = shape
            processed_vars.append('wind')

        # Dew point for apparent temp / relative humidity
        for d_key in ['d2m', 'dpt', '2d']:
            if d_key in ds and 'dpt_k' not in vars_data:
                print(f"-> Dew Point ({d_key})")
                vars_data['dpt_k'] = ds[d_key].values

        # Visibility (RTMA stores in meters; convert to miles)
        for v_key in ['vis', 'vsby', 'visibility']:
            if v_key in ds and 'vis' not in processed_vars:
                print(f"-> Visibility ({v_key})")
                vis_miles = ds[v_key].values / 1609.34
                vis_alpha = np.clip((10.0 - vis_miles) / (10.0 - 7.0), 0, 1)
                vars_data['vis_miles'] = vis_miles
                if lats_2d is not None:
                    reproj_v, *_ = reproject_to_latlon(vis_miles, lats_2d, lons_2d)
                    reproj_a, *_ = reproject_to_latlon(vis_alpha, lats_2d, lons_2d)
                    reproj_a = np.where(np.isfinite(reproj_v), np.clip(reproj_a, 0, 1), 0)
                    save_image_with_alpha(reproj_v, reproj_a, 'vis', 'plasma', 0, 10)
                    shape = save_data(reproj_v, 'vis', stride=1)
                else:
                    save_image_with_alpha(vis_miles, vis_alpha, 'vis', 'plasma', 0, 10)
                    shape = save_data(vis_miles, 'vis')
                if shape and data_shape is None:
                    data_shape = shape
                processed_vars.append('vis')
                break

        # Cloud cover (total cloud cover in %)
        for c_key in ['tcc', 'cc', 'tcdc']:
            if c_key in ds and 'cloud' not in processed_vars:
                print(f"-> Cloud Cover ({c_key})")
                cloud_pct = ds[c_key].values
                cloud_alpha = np.clip(cloud_pct / 40.0, 0, 1)
                if lats_2d is not None:
                    reproj_c, *_ = reproject_to_latlon(cloud_pct, lats_2d, lons_2d)
                    reproj_ca, *_ = reproject_to_latlon(cloud_alpha, lats_2d, lons_2d)
                    reproj_ca = np.where(np.isfinite(reproj_c), np.clip(reproj_ca, 0, 1), 0)
                    save_image_with_alpha(reproj_c, reproj_ca, 'cloud', 'Blues', 0, 100)
                    shape = save_data(reproj_c, 'cloud', stride=1)
                else:
                    save_image_with_alpha(cloud_pct, cloud_alpha, 'cloud', 'Blues', 0, 100)
                    shape = save_data(cloud_pct, 'cloud')
                if shape and data_shape is None:
                    data_shape = shape
                processed_vars.append('cloud')
                break

    # Compute apparent temperature if all components available
    if 'temp_k' in vars_data and 'wind_ms' in vars_data and 'dpt_k' in vars_data:
        apparent_f = compute_apparent_temp(vars_data['temp_k'], vars_data['wind_ms'], vars_data['dpt_k'])
        if lats_2d is not None:
            reproj_app, *_ = reproject_to_latlon(apparent_f, lats_2d, lons_2d)
            save_image(reproj_app, 'apparent', 'turbo', -40, 140)
            save_data(reproj_app, 'apparent', stride=1)
        else:
            save_image(apparent_f, 'apparent', 'turbo', -40, 140)
            save_data(apparent_f, 'apparent')
        processed_vars.append('apparent')

    # Compute relative humidity if temp and dew point are available
    if 'temp_k' in vars_data and 'dpt_k' in vars_data and 'rh' not in processed_vars:
        rh = compute_relative_humidity(vars_data['temp_k'], vars_data['dpt_k'])
        if lats_2d is not None:
            reproj_rh, *_ = reproject_to_latlon(rh, lats_2d, lons_2d)
            save_image(reproj_rh, 'rh', 'YlGnBu', 0, 100)
            save_data(reproj_rh, 'rh', stride=1)
        else:
            save_image(rh, 'rh', 'YlGnBu', 0, 100)
            save_data(rh, 'rh')
        processed_vars.append('rh')

    # Fallback bounds/corners if no temperature variable was found
    if meta_bounds is None and datasets:
        ds = datasets[0]
        lats = ds.latitude.values
        lons = ds.longitude.values.copy()
        if lons.max() >= 180:
            lons = lons - 360
        lat_min, lat_max = float(lats.min()), float(lats.max())
        lon_min, lon_max = float(lons.min()), float(lons.max())
        meta_corners = [
            [lon_min, lat_max],
            [lon_max, lat_max],
            [lon_max, lat_min],
            [lon_min, lat_min],
        ]
        meta_bounds = [[lat_min, lon_min], [lat_max, lon_max]]

    # Write metadata
    meta = {
        "timestamp": data_timestamp,
        "bounds": meta_bounds,
        "corners": meta_corners,
        "data_rows": data_shape[0] if data_shape else None,
        "data_cols": data_shape[1] if data_shape else None,
        "data_stride": DATA_STRIDE
    }
    with open(os.path.join(DATA_DIR, 'metadata.json'), 'w') as f:
        json.dump(meta, f)

    print(f"\nFinished. Processed: {processed_vars}")
    print("Files in site/data:", os.listdir(DATA_DIR))

if __name__ == "__main__":
    main()
