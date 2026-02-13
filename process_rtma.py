import os
import boto3
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import matplotlib.colors as mcolors
from botocore import UNSIGNED
from botocore.client import Config
import json

# --- CONFIGURATION ---
BUCKET_NAME = 'noaa-rtma-pds'
REGION = 'us-east-1'
# Use absolute pathing for the runner
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'site', 'data')

def get_latest_rtma_key():
    s3 = boto3.client('s3', region_name=REGION, config=Config(signature_version=UNSIGNED))
    today = datetime.now(timezone.utc)
    date_str = today.strftime('%Y%m%d')
    prefix = f"rtma2p5.{date_str}/"
    
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
    if 'Contents' not in response:
        return None

    files = [obj['Key'] for obj in response['Contents'] 
             if '2dvaranl_ndfd.grb2' in obj['Key'] and not obj['Key'].endswith('.idx')]
    
    return sorted(files)[-1] if files else None

def generate_layer(ds, variable, output_name, cmap, vmin, vmax):
    if variable not in ds:
        print(f"Variable {variable} not found in dataset.")
        return

    data = ds[variable].values
    
    # Ensure directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fig = plt.figure(figsize=(12, 8), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    # Use 'lower' or 'upper' based on how RTMA grid is stored
    ax.imshow(data, cmap=cmap, norm=norm, origin='lower', aspect='auto')
    
    save_path = os.path.join(OUTPUT_DIR, f"{output_name}.png")
    plt.savefig(save_path, transparent=True, pad_inches=0, format='png', dpi=150)
    plt.close()
    print(f"Successfully saved: {save_path}")

def main():
    key = get_latest_rtma_key()
    if not key:
        print("No data found.")
        return

    # Create local temp file
    local_file = os.path.join(BASE_DIR, "latest.grb2")
    s3 = boto3.client('s3', region_name=REGION, config=Config(signature_version=UNSIGNED))
    s3.download_file(BUCKET_NAME, key, local_file)

    # Open with cfgrib (ignoring index files for speed/permissions)
    ds = xr.open_dataset(local_file, engine='cfgrib', 
                         backend_kwargs={'indexpath': '', 'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}})

    # Temperature
    ds['t2m_f'] = (ds['t2m'] - 273.15) * 9/5 + 32
    generate_layer(ds, 't2m_f', 'temp', 'turbo', 0, 100)

    # Wind (Switching to 10m level)
    ds_wind = xr.open_dataset(local_file, engine='cfgrib', 
                              backend_kwargs={'indexpath': '', 'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}})
    
    if 'u10' in ds_wind:
        ds_wind['wind_mph'] = np.sqrt(ds_wind['u10']**2 + ds_wind['v10']**2) * 2.23694
        generate_layer(ds_wind, 'wind_mph', 'wind', 'viridis', 0, 50)

    # Metadata for Leaflet
    lats, lons = ds.latitude.values, ds.longitude.values
    metadata = {
        "bounds": [[float(lats.min()), float(lons.min())], [float(lats.max()), float(lons.max())]],
        "updated": datetime.now(timezone.utc).isoformat()
    }
    
    with open(os.path.join(OUTPUT_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
    
    print("Files in data directory:", os.listdir(OUTPUT_DIR))

if __name__ == "__main__":
    main()
