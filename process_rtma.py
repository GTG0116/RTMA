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
plt.switch_backend('Agg')

# --- CONFIG ---
BUCKET_NAME = 'noaa-rtma-pds'
REGION = 'us-east-1'
DATA_DIR = 'site/data'
DATA_STRIDE = 8  # Downsample factor for interactive lookup binary files

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
    fig_width = 24  # Increased for higher resolution
    fig_height = fig_width * aspect
    dpi = 600

    fig = plt.figure(figsize=(fig_width, fig_height), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    ax.imshow(data, cmap=cmap, norm=norm, origin='lower', aspect='auto')

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
        prefix = f"rtma2p5.{day}/"
        resp = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)

        if 'Contents' in resp:
            keys = [obj['Key'] for obj in resp['Contents']
                    if '2dvaranl_ndfd.grb2' in obj['Key'] and not obj['Key'].endswith('.idx')]
            if keys:
                target_key = sorted(keys)[-1]
                parts = target_key.split('.')
                hour_str = parts[2][1:3]  # tHHHz — parts[2] is 't00z', e.g. 't06z'[1:3]='06'
                data_timestamp = datetime(int(day[0:4]), int(day[4:6]), int(day[6:8]),
                                          int(hour_str), 0, 0, tzinfo=timezone.utc).isoformat()
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
    data_shape = None  # (rows, cols) of downsampled data grid
    meta_bounds = None

    for i, ds in enumerate(datasets):
        print(f"Dataset #{i}: {list(ds.data_vars)}")

        # Temperature
        for t_key in ['t2m', '2t', 't']:
            if t_key in ds and 'temp' not in processed_vars:
                print(f"-> Temperature ({t_key})")
                temp_f = (ds[t_key] - 273.15) * 9/5 + 32
                save_image(temp_f.values, 'temp', 'turbo', -20, 110)
                shape = save_data(temp_f.values, 'temp')
                if shape and data_shape is None:
                    data_shape = shape
                processed_vars.append('temp')
                vars_data['temp_k'] = ds[t_key].values

                # Metadata with lon fix (0-360 → -180 to 180)
                min_lat = float(ds.latitude.min())
                max_lat = float(ds.latitude.max())
                min_lon = float(ds.longitude.min())
                max_lon = float(ds.longitude.max())
                if min_lon >= 180:  # 0-360 case
                    min_lon -= 360
                    max_lon -= 360
                meta_bounds = [[min_lat, min_lon], [max_lat, max_lon]]

        # Wind
        if 'u10' in ds and 'v10' in ds and 'wind' not in processed_vars:
            print("-> Wind (u10/v10)")
            wind_mph = np.sqrt(ds['u10']**2 + ds['v10']**2) * 2.23694
            save_image(wind_mph.values, 'wind', 'viridis', 0, 60)
            shape = save_data(wind_mph.values, 'wind')
            if shape and data_shape is None:
                data_shape = shape
            processed_vars.append('wind')
            vars_data['wind_ms'] = np.sqrt(ds['u10'].values**2 + ds['v10'].values**2)

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
                save_image(vis_miles, 'vis', 'plasma', 0, 10)
                shape = save_data(vis_miles, 'vis')
                if shape and data_shape is None:
                    data_shape = shape
                processed_vars.append('vis')
                vars_data['vis_miles'] = vis_miles
                break

        # Cloud cover (total cloud cover in %)
        for c_key in ['tcc', 'cc', 'tcdc']:
            if c_key in ds and 'cloud' not in processed_vars:
                print(f"-> Cloud Cover ({c_key})")
                cloud_pct = ds[c_key].values
                save_image(cloud_pct, 'cloud', 'Blues', 0, 100)
                shape = save_data(cloud_pct, 'cloud')
                if shape and data_shape is None:
                    data_shape = shape
                processed_vars.append('cloud')
                break

    # Compute apparent temperature if all components available
    if 'temp_k' in vars_data and 'wind_ms' in vars_data and 'dpt_k' in vars_data:
        apparent_f = compute_apparent_temp(vars_data['temp_k'], vars_data['wind_ms'], vars_data['dpt_k'])
        save_image(apparent_f, 'apparent', 'turbo', -40, 140)
        save_data(apparent_f, 'apparent')
        processed_vars.append('apparent')

    # Compute relative humidity if temp and dew point are available
    if 'temp_k' in vars_data and 'dpt_k' in vars_data and 'rh' not in processed_vars:
        rh = compute_relative_humidity(vars_data['temp_k'], vars_data['dpt_k'])
        save_image(rh, 'rh', 'YlGnBu', 0, 100)
        save_data(rh, 'rh')
        processed_vars.append('rh')

    # Fallback bounds if temp not found
    if meta_bounds is None and datasets:
        ds = datasets[0]
        min_lat = float(ds.latitude.min())
        max_lat = float(ds.latitude.max())
        min_lon = float(ds.longitude.min())
        max_lon = float(ds.longitude.max())
        if min_lon >= 180:
            min_lon -= 360
            max_lon -= 360
        meta_bounds = [[min_lat, min_lon], [max_lat, max_lon]]

    # Write metadata
    meta = {
        "timestamp": data_timestamp,
        "bounds": meta_bounds,
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
