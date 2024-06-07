import numpy as np
import rasterio
import matplotlib.pyplot as plt

def calculate_tvdi(temperature_raster, ndvi_raster):
    # Read temperature raster
    with rasterio.open(temperature_raster) as temp_src:
        temperature = temp_src.read(1)
        temp_profile = temp_src.profile

    # Read NDVI raster
    with rasterio.open(ndvi_raster) as ndvi_src:
        ndvi = ndvi_src.read(1)
        ndvi_profile = ndvi_src.profile

    temperature = temperature[:15233, :14564]
    ndvi = ndvi[:15233, :14564]

    # Calculate TVDI
    numerator = (temperature - 273.15) - ndvi
    denominator = (temperature - 273.15) + ndvi
    tvdi = numerator / denominator

    # Adjusting TVDI range to 0-255 for storing as unsigned 8-bit integer
    tvdi_adjusted = ((tvdi - tvdi.min()) / (tvdi.max() - tvdi.min()) * 255).astype(np.uint8)

    # Write TVDI to a new raster
    tvdi_profile = temp_profile.copy()
    tvdi_profile.update(dtype=rasterio.uint8, count=1)

    with rasterio.open(r"D:\Coding\Master-Thesis\tvdii.tif", 'w', **tvdi_profile) as dst:
        dst.write(tvdi_adjusted, 1)

    print("TVDI calculation completed and saved as tvdi.tif")

# Example usage
temperature_raster = "D:/Cours bioingé/BIR M2/Mémoire/Data/thermal/MR20240205_georeferenced_thermal_cali.tif"
ndvi_raster = "D:/Cours bioingé/BIR M2/Mémoire/Data/multispectral/NDVI/MR20230719_georeferenced_multi_ndvi.tif"
calculate_tvdi(temperature_raster, ndvi_raster)
calculate_tvdi(temperature_raster, ndvi_raster)

# Plot the TVDI
plt.imshow(tvdi_adjusted, cmap='gray')
plt.colorbar()
plt.title('TVDI')
plt.show()

print("TVDI calculation completed and plotted")
