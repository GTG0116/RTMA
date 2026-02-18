import os
import json
import boto3
import numpy as np
import xarray as xr
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
def save_image(data, name, cmap, vmin, vmax):
    os.makedirs(DATA_DIR, exist_ok=True)
    output_path = os.path.join(DATA_DIR, f"{name}.png")
   
    # Check for empty data
    if data is None or np.isnan(data).all():
        print(f"!!! Skipping {name}: Data is null or all NaNs")
        return
    fig = plt.figure(figsize=(12, 8), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    ax.imshow(data, cmap=cmap, norm=norm, origin='lower', aspect='auto')
   
    plt.savefig(output_path, transparent=True, pad_inches=0, format='png', dpi=150)
    plt.close(fig)
    print(f"✅ SAVED: {output_path} ({os.path.getsize(output_path)} bytes)")
def main():
    # 1. Setup
    os.makedirs(DATA_DIR, exist_ok=True)
    s3 = boto3.client('s3', region_name=REGION, config=Config(signature_version=UNSIGNED))
   
    # 2. Get latest key
    current_day = datetime.now(timezone.utc)
    target_key = None
    data_timestamp = None
    for d in range(0, 3):  # Try today, yesterday, day before
        day = (current_day - timedelta(days=d)).strftime('%Y%m%d')
        prefix = f"rtma2p5.{day}/"
        resp = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
   
        if 'Contents' in resp:
            keys = [obj['Key'] for obj in resp['Contents'] if '2dvaranl_ndfd.grb2' in obj['Key'] and not obj['Key'].endswith('.idx')]
            if keys:
                target_key = sorted(keys)[-1]
                # Extract timestamp from key
                parts = target_key.split('.')
                hour_str = parts[1][1:3]  # tHHHz -> HH
                data_timestamp = datetime(int(day[0:4]), int(day[4:6]), int(day[6:8]), int(hour_str), 0, 0, tzinfo=timezone.utc).isoformat()
                break
   
    if target_key is None:
        print("No data found in the last 3 days.")
        return
   
    # 3. Download
    local_file = 'latest.grb2'
    print(f"Downloading {target_key}...")
    s3.download_file(BUCKET_NAME, target_key, local_file)
   
    # 4. THE NUCLEAR LOAD
    # cfgrib.open_datasets loads ALL hypercubes in the GRIB file into a list
    print("Opening all GRIB datasets (this may take a minute)...")
    datasets = cfgrib.open_datasets(local_file, backend_kwargs={'indexpath': ''})
   
    processed_vars = []
    meta_saved = False
    for i, ds in enumerate(datasets):
        print(f"Checking Dataset #{i} with variables: {list(ds.data_vars)}")
       
        # TEMPERATURE (Look for 't2m', '2t', or 't')
        for t_key in ['t2m', '2t', 't']:
            if t_key in ds and 'temp' not in processed_vars:
                print(f"-> Found Temperature ({t_key})")
                temp_f = (ds[t_key] - 273.15) * 9/5 + 32
                save_image(temp_f.values, 'temp', 'turbo', -20, 110)
                processed_vars.append('temp')
               
                # Use this dataset for metadata (lat/lon)
                min_lat = float(ds.latitude.min())
                max_lat = float(ds.latitude.max())
                min_lon = float(ds.longitude.min())
                max_lon = float(ds.longitude.max())
                if min_lon > 180:
                    min_lon -= 360
                    max_lon -= 360
                meta = {
                    "timestamp": data_timestamp,
                    "bounds": [[min_lat, min_lon],
                               [max_lat, max_lon]]
                }
                with open(os.path.join(DATA_DIR, 'metadata.json'), 'w') as f:
                    json.dump(meta, f)
                meta_saved = True
   
        # WIND (Look for u/v components)
        if 'u10' in ds and 'v10' in ds and 'wind' not in processed_vars:
            print("-> Found Wind Components (u10/v10)")
            wind_mph = np.sqrt(ds['u10']**2 + ds['v10']**2) * 2.23694
            save_image(wind_mph.values, 'wind', 'viridis', 0, 60)
            processed_vars.append('wind')
   
        # GUSTS
        if 'gust' in ds and 'gust' not in processed_vars:
            print("-> Found Gusts")
            save_image(ds['gust'].values * 2.23694, 'gust', 'plasma', 0, 80)
            processed_vars.append('gust')
   
    # If meta not saved but other vars processed, save meta from first ds
    if not meta_saved and datasets:
        ds = datasets[0]
        min_lat = float(ds.latitude.min())
        max_lat = float(ds.latitude.max())
        min_lon = float(ds.longitude.min())
        max_lon = float(ds.longitude.max())
        if min_lon > 180:
            min_lon -= 360
            max_lon -= 360
        meta = {
            "timestamp": data_timestamp,
            "bounds": [[min_lat, min_lon],
                       [max_lat, max_lon]]
        }
        with open(os.path.join(DATA_DIR, 'metadata.json'), 'w') as f:
            json.dump(meta, f)
   
    print(f"\nFinished. Processed: {processed_vars}")
    print("Files in site/data:", os.listdir(DATA_DIR))
if __name__ == "__main__":
    main()
