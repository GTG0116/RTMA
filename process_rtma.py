import os
import boto3
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import matplotlib.colors as mcolors
from botocore import UNSIGNED
from botocore.client import Config

# --- CONFIGURATION ---
BUCKET_NAME = 'noaa-rtma-pds'
REGION = 'us-east-1'
OUTPUT_DIR = 'site/data'

def get_latest_rtma_key():
    """Finds the most recent available RTMA file on S3."""
    s3 = boto3.client('s3', region_name=REGION, config=Config(signature_version=UNSIGNED))
    
    # Check today's date folders
    today = datetime.now(timezone.utc)
    date_str = today.strftime('%Y%m%d')
    prefix = f"rtma2p5.{date_str}/"
    
    print(f"Searching bucket {BUCKET_NAME} with prefix {prefix}...")
    
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
    if 'Contents' not in response:
        print("No data found for today yet.")
        return None

    # --- CRITICAL FIX: Filter out .idx files ---
    files = [
        obj['Key'] for obj in response['Contents'] 
        if '2dvaranl_ndfd.grb2' in obj['Key'] 
        and not obj['Key'].endswith('.idx')  # <--- FIX HERE
    ]
    
    if not files:
        print("No valid GRIB2 files found.")
        return None
        
    # Return the latest one based on sorting (timestamps in filename)
    return sorted(files)[-1] 

def download_file(key):
    """Downloads file from S3 to local disk."""
    s3 = boto3.client('s3', region_name=REGION, config=Config(signature_version=UNSIGNED))
    local_name = "latest.grb2"
    
    # Remove previous file if exists to prevent stale data
    if os.path.exists(local_name):
        os.remove(local_name)

    print(f"Downloading {key}...")
    s3.download_file(BUCKET_NAME, key, local_name)
    
    # Verify file size
    file_size = os.path.getsize(local_name)
    print(f"Downloaded {file_size} bytes.")
    
    if file_size < 1000: # GRIB files are large; if it's tiny, it's an error/xml
        raise ValueError("File is too small. Likely an error XML or empty file.")
        
    return local_name

def generate_layer(ds, variable, output_name, cmap, vmin, vmax, label):
    """Converts a GRIB data array into a transparent PNG."""
    print(f"Processing {label}...")
    
    try:
        data = ds[variable].values
        
        fig = plt.figure(figsize=(10, 10), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        ax.imshow(data, cmap=cmap, norm=norm, origin='upper', aspect='auto')
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(f"{OUTPUT_DIR}/{output_name}.png", transparent=True, pad_inches=0, format='png')
        plt.close()
        print(f"Saved {output_name}.png")
    except Exception as e:
        print(f"Failed to generate {label}: {e}")

def main():
    try:
        key = get_latest_rtma_key()
        if not key:
            print("Could not find suitable RTMA data.")
            return

        local_file = download_file(key)

        # Force cfgrib to ignore the creation of an index file if it can't write it
        # and explicitly ask for specific variables to speed up read
        backend_args = {'indexpath': ''} 

        print("Opening GRIB file...")
        
        # 1. Temperature (Level: 2m above ground)
        try:
            ds_temp = xr.open_dataset(
                local_file, 
                engine='cfgrib', 
                backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}, 'indexpath': ''}
            )
            
            # Process Temperature
            if 't2m' in ds_temp:
                ds_temp['t2m_f'] = (ds_temp['t2m'] - 273.15) * 9/5 + 32
                generate_layer(ds_temp, 't2m_f', 'temp', 'turbo', -20, 110, "Temperature")
            
            # Process Heat Index/Wind Chill (Apparent Temp)
            if 'aptmp' in ds_temp:
                ds_temp['app_f'] = (ds_temp['aptmp'] - 273.15) * 9/5 + 32
                generate_layer(ds_temp, 'app_f', 'feel', 'RdYlBu_r', -30, 120, "Feels Like")

        except Exception as e:
            print(f"Error reading Temperature block: {e}")

        # 2. Wind (Level: 10m above ground)
        try:
            ds_wind = xr.open_dataset(
                local_file, 
                engine='cfgrib', 
                backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}, 'indexpath': ''}
            )
            
            # Process Wind Speed
            if 'u10' in ds_wind and 'v10' in ds_wind:
                ds_wind['wind_mph'] = np.sqrt(ds_wind['u10']**2 + ds_wind['v10']**2) * 2.23694
                generate_layer(ds_wind, 'wind_mph', 'wind', 'viridis', 0, 60, "Wind Speed")
            
            # Process Gusts
            if 'gust' in ds_wind:
                ds_wind['gust_mph'] = ds_wind['gust'] * 2.23694
                generate_layer(ds_wind, 'gust_mph', 'gust', 'plasma', 0, 80, "Wind Gusts")
                
        except Exception as e:
            print(f"Error reading Wind block: {e}")

        # Metadata
        if 'ds_temp' in locals() and ds_temp:
            lats = ds_temp.latitude.values
            lons = ds_temp.longitude.values
            min_lat, max_lat = lats.min(), lats.max()
            min_lon, max_lon = lons.min(), lons.max()
            
            import json
            metadata = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "bounds": [[float(min_lat), float(min_lon)], [float(max_lat), float(max_lon)]],
                "source": key
            }
            with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
                json.dump(metadata, f)
                print("Metadata saved.")

    except Exception as e:
        print(f"Critical Failure: {e}")
        # Re-raise to fail the GitHub Action
        raise e

if __name__ == "__main__":
    main()
