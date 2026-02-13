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

# --- CONFIGURATION ---
BUCKET_NAME = 'noaa-rtma-pds'
REGION = 'us-east-1'

# 1. HARDCODED PATHS relative to where the command is run (Repo Root)
# We do not use __file__ anymore to avoid confusion.
SITE_DIR = 'site'
DATA_DIR = os.path.join(SITE_DIR, 'data')

def setup_environment():
    print(f"Current Working Directory: {os.getcwd()}")
    
    # 2. Force Create Directories
    if not os.path.exists(DATA_DIR):
        print(f"Creating directory: {DATA_DIR}")
        os.makedirs(DATA_DIR, exist_ok=True)
    else:
        print(f"Directory exists: {DATA_DIR}")

    # 3. PERMISSION TEST: Write a dummy file to prove we can save things
    test_file = os.path.join(DATA_DIR, 'permission_test.txt')
    try:
        with open(test_file, 'w') as f:
            f.write("Write access confirmed.")
        print(f"SUCCESS: Created test file at {test_file}")
    except Exception as e:
        print(f"CRITICAL ERROR: Cannot write to {DATA_DIR}. Error: {e}")
        sys.exit(1)

def get_latest_rtma_key():
    s3 = boto3.client('s3', region_name=REGION, config=Config(signature_version=UNSIGNED))
    today = datetime.now(timezone.utc)
    date_str = today.strftime('%Y%m%d')
    prefix = f"rtma2p5.{date_str}/"
    
    print(f"Scanning S3: {BUCKET_NAME}/{prefix}")
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
    
    if 'Contents' not in response:
        print("No S3 contents found.")
        return None

    files = [obj['Key'] for obj in response['Contents'] 
             if '2dvaranl_ndfd.grb2' in obj['Key'] and not obj['Key'].endswith('.idx')]
    
    return sorted(files)[-1] if files else None

def save_image(data, name, cmap, vmin, vmax):
    output_path = os.path.join(DATA_DIR, f"{name}.png")
    print(f"Generating {output_path}...")
    
    try:
        fig = plt.figure(figsize=(10, 10), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        ax.imshow(data, cmap=cmap, norm=norm, origin='upper', aspect='auto')
        
        plt.savefig(output_path, transparent=True, pad_inches=0, format='png')
        plt.close()
        
        if os.path.exists(output_path):
            print(f" -> Saved {name}.png ({os.path.getsize(output_path)} bytes)")
        else:
            print(f" -> ERROR: File {name}.png was NOT created.")
            
    except Exception as e:
        print(f" -> Failed to generate image: {e}")

def main():
    setup_environment()
    
    key = get_latest_rtma_key()
    if not key:
        print("No RTMA file found.")
        sys.exit(1)

    print(f"Downloading {key}...")
    local_file = 'latest.grb2'
    
    s3 = boto3.client('s3', region_name=REGION, config=Config(signature_version=UNSIGNED))
    s3.download_file(BUCKET_NAME, key, local_file)

    # --- TEMP ---
    try:
        print("Opening GRIB (Temp)...")
        # backend_kwargs={'indexpath': ''} is CRITICAL for read-only environments
        ds = xr.open_dataset(local_file, engine='cfgrib', 
                             backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}, 'indexpath': ''})
        
        if 't2m' in ds:
            temp_f = (ds['t2m'] - 273.15) * 9/5 + 32
            save_image(temp_f.values, 'temp', 'turbo', -20, 110)
            
            # Save Metadata
            lats, lons = ds.latitude.values, ds.longitude.values
            meta = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "bounds": [[float(lats.min()), float(lons.min())], [float(lats.max()), float(lons.max())]]
            }
            with open(os.path.join(DATA_DIR, 'metadata.json'), 'w') as f:
                json.dump(meta, f)
                print(" -> Saved metadata.json")
    except Exception as e:
        print(f"Error processing Temp: {e}")

    # --- WIND ---
    try:
        print("Opening GRIB (Wind)...")
        ds_wind = xr.open_dataset(local_file, engine='cfgrib', 
                                  backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}, 'indexpath': ''})
        
        if 'u10' in ds_wind:
            wind_speed = np.sqrt(ds_wind['u10']**2 + ds_wind['v10']**2) * 2.23694
            save_image(wind_speed.values, 'wind', 'viridis', 0, 60)
            
    except Exception as e:
        print(f"Error processing Wind: {e}")

    # --- FINAL CHECK ---
    print("\n--- CONTENT OF SITE/DATA ---")
    print(os.listdir(DATA_DIR))

if __name__ == "__main__":
    main()
