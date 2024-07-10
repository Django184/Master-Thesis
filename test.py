import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt

class MultispecAnalysis:
    TEMPERATURE_RASTER = glob.glob("Data/Multispectral/thermal/*.tif")
    NDVI_RASTER = glob.glob("Data/Multispectral/NDVI/*.tif")

    def __init__(self, temperature_raster=TEMPERATURE_RASTER, ndvi_raster=NDVI_RASTER, sample_number=0):
        self.temperature_raster = temperature_raster
        self.ndvi_raster = ndvi_raster
        self.sample_number = sample_number

    def import_rasters(self):
        with rasterio.open(self.temperature_raster[self.sample_number]) as temp_src:
            temperature = temp_src.read(1)
            temp_profile = temp_src.profile

        with rasterio.open(self.ndvi_raster[self.sample_number]) as ndvi_src:
            ndvi = ndvi_src.read(1)
            ndvi_profile = ndvi_src.profile
        
        return temperature, ndvi, temp_profile, ndvi_profile, temp_src, ndvi_src

    def calculate_tvdi(self):
        temperature, ndvi, temp_profile, ndvi_profile, temp_src, ndvi_src = self.import_rasters()

        # Resample NDVI raster to match temperature raster's dimensions
        ndvi_resampled = np.zeros_like(temperature)
        reproject(
            ndvi, ndvi_resampled,
            src_transform=ndvi_src.transform, src_crs=ndvi_src.crs,
            dst_transform=temp_src.transform, dst_crs=temp_src.crs,
            resampling=Resampling.nearest
        )

        # Calculate the dry edge parameters (a and b)
        a, b = self.calculate_dry_edge(ndvi_resampled, temperature)
        # Calculate the minimum temperature for the given NDVI value
        t_min_values = self.calculate_wet_edge(ndvi_resampled, temperature)

        # Calculate TVDI
        tvdi = (temperature - t_min_values) / (a - b * ndvi_resampled + t_min_values)

        # Set the nodata value to -9999
        tvdi[np.isnan(tvdi)] = -9999

        # Adjust the TVDI range to 0-255 for storage as an unsigned 8-bit integer
        tvdi_adjusted = ((tvdi - tvdi.min()) / (tvdi.max() - tvdi.min()) * 255).astype(np.uint8)

        # Plot the TVDI
        plt.imshow(tvdi_adjusted, cmap="jet")
        plt.colorbar(label="TVDI")
        plt.title("Temperature Vegetation Dryness Index (TVDI)")
        plt.show()

    def calculate_dry_edge(self, ndvi, temperature):
        # Create NDVI-Ts scatter plot and determine dry edge parameters
        scatter_points = np.vstack((ndvi.ravel(), temperature.ravel())).T
        scatter_points = scatter_points[~np.isnan(scatter_points).any(axis=1)]

        # Calculate the dry edge using linear regression on the maximum temperatures
        bins = np.linspace(ndvi.min(), ndvi.max(), 100)
        bin_indices = np.digitize(ndvi.ravel(), bins)
        max_temps = [np.max(temperature.ravel()[bin_indices == i]) for i in range(1, len(bins))]

        valid_bins = ~np.isnan(max_temps)
        bins = bins[:-1][valid_bins]
        max_temps = np.array(max_temps)[valid_bins]

        # Perform linear regression to find the slope (a) and intercept (b)
        a, b = np.polyfit(bins, max_temps, 1)
        return a, b

    def calculate_wet_edge(self, ndvi, temperature):
        # Calculate the wet edge (minimum temperature for a given NDVI value)
        bins = np.linspace(ndvi.min(), ndvi.max(), 100)
        bin_indices = np.digitize(ndvi.ravel(), bins)
        min_temps = [np.min(temperature.ravel()[bin_indices == i]) for i in range(1, len(bins))]

        valid_bins = ~np.isnan(min_temps)
        bins = bins[:-1][valid_bins]
        min_temps = np.array(min_temps)[valid_bins]

        # Interpolate minimum temperatures to the full NDVI range
        t_min_values = np.interp(ndvi, bins, min_temps)
        return t_min_values

test = MultispecAnalysis()
test.calculate_tvdi()  
