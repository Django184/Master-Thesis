import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
import matplotlib.pyplot as plt

# Define functions for T_max and T_min
def t_max(ndvi):
    # Placeholder coefficients for T_max(NDVI) = a * NDVI + b
    a = 40
    b = 300
    return a * ndvi + b

def t_min(ndvi):
    # Placeholder coefficients for T_min(NDVI) = c * NDVI + d
    c = 20
    d = 250
    return c * ndvi + d

# Calculate TVDI
def calculate_tvdi(temperature_raster, ndvi_raster):
    # Read temperature raster
    with rasterio.open(temperature_raster) as temp_src:
        temperature = temp_src.read(1)
        temp_profile = temp_src.profile

    # Read NDVI raster
    with rasterio.open(ndvi_raster) as ndvi_src:
        ndvi = ndvi_src.read(1)
        ndvi_profile = ndvi_src.profile

    # Resample NDVI to match temperature raster dimensions
    ndvi_resampled = np.zeros_like(temperature)
    reproject(
        ndvi, ndvi_resampled,
        src_transform=ndvi_src.transform,
        src_crs=ndvi_src.crs,
        dst_transform=temp_profile['transform'],
        dst_crs=temp_profile['crs'],
        resampling=Resampling.nearest,
        dst_resolution=(temp_profile['transform'][0], -temp_profile['transform'][4]))

    # Calculate T_max and T_min
    t_max_values = t_max(ndvi_resampled)
    t_min_values = t_min(ndvi_resampled)

    # Calculate TVDI
    tvdi = (temperature - t_min_values) / (t_max_values - t_min_values)

    # Set nodata value to -9999
    tvdi[np.isnan(tvdi)] = -9999

    # Adjusting TVDI range to 0-255 for storing as unsigned 8-bit integer
    tvdi_adjusted = ((tvdi - tvdi.min()) / (tvdi.max() - tvdi.min()) * 255).astype(np.uint8)

    # Write TVDI to a new raster
    tvdi_profile = temp_profile.copy()
    tvdi_profile.update(dtype=rasterio.uint8, count=1, nodata=np.uint8(-9999))

    with rasterio.open(r"D:\Coding\Master-Thesis\tvdii.tif", 'w', **tvdi_profile) as dst:
        dst.write(tvdi_adjusted, 1)

    # Plot TVDI
    plt.imshow(tvdi_adjusted, cmap='jet', vmin=0, vmax=255)
    plt.colorbar(label='TVDI')
    plt.title('Temperature Vegetation Dryness Index (TVDI)')
    plt.show()

temperature_raster = "D:/Cours bioingé/BIR M2/Mémoire/Data/thermal/MR20240205_georeferenced_thermal_cali.tif"
ndvi_raster = "D:/Cours bioingé/BIR M2/Mémoire/Data/multispectral/NDVI/MR20230719_georeferenced_multi_ndvi.tif"
calculate_tvdi(temperature_raster, ndvi_raster)