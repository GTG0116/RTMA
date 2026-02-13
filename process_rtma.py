import os
import json
import boto3
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime, timezone
from botocore import UNSIGNED
from botocore.client import Config
import sys

# Ensure Matplotlib doesn't try to open a window (important for GitHub Actions)
plt.switch_backend('Agg')

# --- CONFIGURATION ---
BUCKET_NAME = 'noaa-rtma-pds'
REGION = 'us-east-1'
SITE_DIR = 'site'
DATA_DIR = os.path.join(SITE_DIR, 'data')

def setup_environment():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
    # Proof of life
    with open(os.path.join(DATA_DIR, 'permission_test.txt'), 'w') as f:
        f.write(f"Last run: {datetime.now(timezone.utc)}")

def get_latest_rtma_key():
    s3 = boto3.client('s3', region_name=REGION, config=Config(signature_version=UNSIGNED))
    today = datetime.now(timezone.utc)
    # Try today, then yesterday if today is empty (early UTC runs)
    for i in range(2):
        date_str = today.strftime('%Y%m%d')
        prefix = f"rtma2p5.{date_str}/"
        print(f"Checking S3 prefix: {prefix}")
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
        if 'Contents' in response:
            files = [obj['Key'] for obj in response['Contents'] 
                     if '2dvaranl_ndfd.grb2' in obj['Key'] and not obj['Key'].endswith('.idx')]
            if files:
                return sorted(files)[-1]
    return None

def save_image(data, name, cmap, vmin, vmax):
    output_path = os.path.join(DATA_DIR, f"{name}.png")
    try:
        # Check if data is empty or all NaNs
        if np.isnan(data).all():
            print(f" ! Skipping {name}: Data contains only NaNs")
            return

        fig = plt.figure(figsize=(12, 8), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        # Use 'lower' because GRIB arrays usually start from the bottom-left
        ax.imshow(data, cmap=cmap, norm=norm, origin='lower', aspect='auto')
        
        plt.savefig(output_path, transparent=True, pad_inches=0, format='png', dpi=150)
        plt.close(fig)
        print(f" -> SUCCESS: Saved {name}.png ({os.path.getsize(output_path)} bytes)")
    except Exception as e:
        print(f" -> ERROR saving {name}: {e}")

def main():
    setup_environment()
    key = get_latest_rtma_key()
    if not key:
        print("No RTMA data found.")
        return

    local_file = 'latest.grb2'
    s3 = boto3.client('s3', region_name=REGION, config=Config(signature_version=UNSIGNED))
    s3.download_file(BUCKET_NAME, key, local_file)
    print(f"Downloaded {local_file} ({os.path.getsize(local_file)} bytes)")

    # --- THE SMART SCAN ---
    # We open the dataset without filters first to see what keys exist
    print("Scanning GRIB variables...")
    try:
        # Load temperature-level data (2m)
        ds = xr.open_dataset(local_file, engine='cfgrib', 
                             backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}, 'indexpath': ''})
        
        print(f"Available variables at 2m: {list(ds.data_vars)}")

        # Temperature: could be 't2m', 't', or '2t'
        t_var = next((v for v in ['t2m', 't', '2t'] if v in ds), None)
        if t_var:
            print(f"Processing Temperature using key: {t_var}")
            temp_f = (ds[t_var] - 273.15) * 9/5 + 32
            save_image(temp_f.values, 'temp', 'turbo', -20, 110)
            
            # Save Metadata based on this successful load
            lats, lons = ds.latitude.values, ds.longitude.values
            meta = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "bounds": [[float(lats.min()), float(lons.min())], [float(lats.max()), float(lons.max())]]
            }
            with open(os.path.join(DATA_DIR, 'metadata.json'), 'w') as f:
                json.dump(meta, f)
        
        # Apparent Temp (Wind Chill/Heat Index)
        apt_var = next((v for v in ['aptmp', 'at'] if v in ds), None)
        if apt_var:
            app_f = (ds[apt_var] - 273.15) * 9/5 + 32
            save_image(app_f.values, 'feel', 'RdYlBu_r', -40, 120)

    except Exception as e:
        print(f"2m Level Error: {e}")

    try:
        # Load wind-level data (10m)
        ds_wind = xr.open_dataset(local_file, engine='cfgrib', 
                                  backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}, 'indexpath': ''})
        
        print(f"Available variables at 10m: {list(ds_wind.data_vars)}")

        if 'u10' in ds_wind and 'v10' in ds_wind:
            speed = np.sqrt(ds_wind['u10']**2 + ds_wind['v10']**2) * 2.23694
            save_image(speed.values, 'wind', 'viridis', 0, 60)
        
        # Gusts
        g_var = next((v for v in ['gust', 'si10'] if v in ds_wind), None)
        if g_var:
            gust_mph = ds_wind[g_var] * 2.23694
            save_image(gust_mph.values, 'gust', 'plasma', 0, 80)

    except Exception as e:
        print(f"10m Level Error: {e}")

    print("\n--- FINAL DATA DIRECTORY LISTING ---")
    if os.path.exists(DATA_DIR):
        print(os.listdir(DATA_DIR))
    else:
        print("ERROR: DATA_DIR does not exist!")

if __name__ == "__main__":
    main()
