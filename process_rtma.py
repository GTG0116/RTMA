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
from pathlib import Path

# --- CONFIGURATION ---
BUCKET_NAME = 'noaa-rtma-pds'
REGION = 'us-east-1'

# Strict Path Definitions
BASE_DIR = Path(__file__).parent.resolve()
SITE_DIR = BASE_DIR / "site"
DATA_DIR = SITE_DIR / "data"

def setup_directories():
    """Force create the directories."""
    if not DATA_DIR.exists():
        print(f"Creating directory: {DATA_DIR}")
        DATA_DIR.mkdir(parents=True, exist_ok=True)

def get_latest_rtma_key():
    s3 = boto3.client('s3', region_name=REGION, config=Config(signature_version=UNSIGNED))
    today = datetime.now(timezone.utc)
    date_str = today.strftime('%Y%m%d')
    prefix = f"rtma2p5.{date_str}/"
    
    print(f"Searching S3 bucket: {BUCKET_NAME} prefix: {prefix}")
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
    
    if 'Contents' not in response:
        print("No contents found in S3.")
        return None

    # Filter strictly for GRIB2 files (no indices)
    files = [obj['Key'] for obj in response['Contents'] 
             if '2dvaranl_ndfd.grb2' in obj['Key'] and not obj['Key'].endswith('.idx')]
    
    return sorted(files)[-1] if files else None

def generate_layer(ds, variable, output_name, cmap, vmin, vmax):
    if variable not in ds:
        print(f"Skipping {output_name}: Variable {variable} not found.")
        return

    print(f"Generating image for {output_name}...")
    data = ds[variable].values
    
    fig = plt.figure(figsize=(10, 10), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    # RTMA is usually upside down relative to Web Mercator, verify 'origin'
    ax.imshow(data, cmap=cmap, norm=norm, origin='upper', aspect='auto')
    
    save_path = DATA_DIR / f"{output_name}.png"
    plt.savefig(save_path, transparent=True, pad_inches=0, format='png')
    plt.close()
    
    # Verify file was created
    if save_path.exists():
        print(f" -> Saved: {save_path} ({save_path.stat().st_size} bytes)")
    else:
        print(f" -> ERROR: Failed to save {save_path}")

def main():
    setup_directories()
    
    key = get_latest_rtma_key()
    if not key:
        print("Critical: No RTMA file found on S3.")
        return

    print(f"Downloading: {key}")
    local_file = BASE_DIR / "latest.grb2"
    
    s3 = boto3.client('s3', region_name=REGION, config=Config(signature_version=UNSIGNED))
    s3.download_file(BUCKET_NAME, key, str(local_file))

    # --- PROCESS 2M TEMP ---
    try:
        # We perform separate opens to avoid GRIB index conflicts
        ds_temp = xr.open_dataset(local_file, engine='cfgrib', 
                                  backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}, 'indexpath': ''})
        
        ds_temp['t2m_f'] = (ds_temp['t2m'] - 273.15) * 9/5 + 32
        generate_layer(ds_temp, 't2m_f', 'temp', 'turbo', -20, 110)
        
        # Save Metadata
        lats, lons = ds_temp.latitude.values, ds_temp.longitude.values
        metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bounds": [[float(lats.min()), float(lons.min())], [float(lats.max()), float(lons.max())]]
        }
        with open(DATA_DIR / "metadata.json", "w") as f:
            json.dump(metadata, f)
            print(" -> Saved: metadata.json")
            
    except Exception as e:
        print(f"Error processing Temp: {e}")

    # --- PROCESS WIND (10M) ---
    try:
        ds_wind = xr.open_dataset(local_file, engine='cfgrib', 
                                  backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}, 'indexpath': ''})
        
        # Calculate speed from U/V components if explicit speed missing
        u = ds_wind['u10'] if 'u10' in ds_wind else None
        v = ds_wind['v10'] if 'v10' in ds_wind else None
        
        if u is not None and v is not None:
            ds_wind['wind_mph'] = np.sqrt(u**2 + v**2) * 2.23694
            generate_layer(ds_wind, 'wind_mph', 'wind', 'viridis', 0, 60)
            
        if 'gust' in ds_wind:
             ds_wind['gust_mph'] = ds_wind['gust'] * 2.23694
             generate_layer(ds_wind, 'gust_mph', 'gust', 'plasma', 0, 80)
             
    except Exception as e:
        print(f"Error processing Wind: {e}")

    # Final Verification
    print("\n--- FINAL CONTENT OF DATA DIR ---")
    print(list(DATA_DIR.glob('*')))

if __name__ == "__main__":
    main()
