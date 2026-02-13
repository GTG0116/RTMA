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
# We focus on the CONUS (Continental US) 2.5km grid
# Standard RTMA file pattern: rtma2p5.t{HH}z.2dvaranl_ndfd.grb2

def get_latest_rtma_key():
    """Finds the most recent available RTMA file on S3."""
    s3 = boto3.client('s3', region_name=REGION, config=Config(signature_version=UNSIGNED))
    
    # Check today's date folders
    today = datetime.now(timezone.utc)
    date_str = today.strftime('%Y%m%d')
    prefix = f"rtma2p5.{date_str}/"
    
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
    if 'Contents' not in response:
        print("No data found for today yet.")
        return None

    # Filter for the standard 2dvaranl_ndfd file
    files = [obj['Key'] for obj in response['Contents'] if '2dvaranl_ndfd.grb2' in obj['Key']]
    if not files:
        return None
        
    return sorted(files)[-1] # Return the latest one

def download_file(key):
    """Downloads file from S3 to local disk."""
    s3 = boto3.client('s3', region_name=REGION, config=Config(signature_version=UNSIGNED))
    local_name = "latest.grb2"
    print(f"Downloading {key}...")
    s3.download_file(BUCKET_NAME, key, local_name)
    return local_name

def generate_layer(ds, variable, output_name, cmap, vmin, vmax, label):
    """Converts a GRIB data array into a transparent PNG."""
    print(f"Processing {label}...")
    
    # Extract data values
    data = ds[variable].values
    
    # Create figure without borders/axes
    fig = plt.figure(figsize=(10, 10), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    # Normalize color scale
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    # Plot using imshow (fastest for raster)
    # We flip standard image origin because matplotlib and maps differ in y-axis direction
    ax.imshow(data, cmap=cmap, norm=norm, origin='upper', aspect='auto')
    
    # Save as transparent PNG
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(f"{OUTPUT_DIR}/{output_name}.png", transparent=True, pad_inches=0, format='png')
    plt.close()

def main():
    key = get_latest_rtma_key()
    if not key:
        print("Could not find suitable RTMA data.")
        return

    local_file = download_file(key)

    # Open GRIB2 file
    # We filter by 'typeOfLevel' to avoid conflicting variable messages
    # 2m Temp, 10m Wind, etc.
    
    try:
        # Load datasets (sometimes variables are in different "hypercubes" in GRIB)
        # 1. Temperature & Apparent Temp (2m above ground)
        ds_temp = xr.open_dataset(local_file, engine='cfgrib', 
                                  backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}})
        
        # 2. Wind (10m above ground)
        ds_wind = xr.open_dataset(local_file, engine='cfgrib', 
                                  backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}})
        
        # 3. Gusts often live in surface or specialized levels, but standard RTMA puts them at 10m too usually
        # If specific gust variable isn't found, we skip it to prevent crash
    except Exception as e:
        print(f"Error opening GRIB file: {e}")
        return

    # --- PROCESSING VARIABLES ---
    
    # 1. Temperature (t2m) - Convert Kelvin to Fahrenheit
    # RTMA Variable: t2m
    temp_data = (ds_temp['t2m'] - 273.15) * 9/5 + 32
    ds_temp['t2m_f'] = temp_data
    generate_layer(ds_temp, 't2m_f', 'temp', 'turbo', -20, 110, "Temperature")

    # 2. Wind Speed (10m) - Convert m/s to mph
    # RTMA Variable: si10 (Speed index) or similar. Often calculated from u10/v10 if not explicit.
    # We use magnitude of u and v vectors
    wind_speed = np.sqrt(ds_wind['u10']**2 + ds_wind['v10']**2) * 2.23694
    ds_wind['wind_mph'] = wind_speed
    generate_layer(ds_wind, 'wind_mph', 'wind', 'viridis', 0, 60, "Wind Speed")

    # 3. Wind Gust
    # Sometimes gusts are separate. If 'gust' exists in ds_wind:
    if 'gust' in ds_wind:
        gust_speed = ds_wind['gust'] * 2.23694
        ds_wind['gust_mph'] = gust_speed
        generate_layer(ds_wind, 'gust_mph', 'gust', 'plasma', 0, 80, "Wind Gusts")

    # 4. Apparent Temperature (Heat Index / Wind Chill)
    # RTMA has 'aptmp' (Apparent Temp) in the 2m dataset usually
    if 'aptmp' in ds_temp:
        app_temp = (ds_temp['aptmp'] - 273.15) * 9/5 + 32
        ds_temp['app_f'] = app_temp
        generate_layer(ds_temp, 'app_f', 'feel', 'RdYlBu_r', -30, 120, "Feels Like")

    # --- METADATA GENERATION ---
    # We need the bounds (lat/lon) for Leaflet to know where to place the image
    lats = ds_temp.latitude.values
    lons = ds_temp.longitude.values
    
    # Get bounds
    min_lat, max_lat = lats.min(), lats.max()
    min_lon, max_lon = lons.min(), lons.max()
    
    # Save a small JSON manifest with bounds
    import json
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bounds": [[min_lat, min_lon], [max_lat, max_lon]],
        "source": key
    }
    with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f)

if __name__ == "__main__":
    main()
