from mpl_toolkits.axes_grid1 import make_axes_locatable
from pykrige.ok import OrdinaryKriging
from rasterio.warp import reproject, Resampling
from skgstat import models
from sklearn.preprocessing import QuantileTransformer
import gstatsim as gs
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pyproj  # for reprojection
import rasterio
import skgstat as skg
import glob


class GprAnalysis:
    """Visualisation of the GPR field data"""

    FIELD_A_PATHS = glob.glob("D:/Cours bioingé/BIR M2/Mémoire/Data/Drone GPR/Field A/*.txt")
    FIELD_B_PATHS = glob.glob("D:/Cours bioingé/BIR M2/Mémoire/Data/Drone GPR/Field B/*.txt")

    def __init__(self, field_paths=FIELD_A_PATHS, sample_number=0):
        """Initialisation of the GPR field data"""
        self.field_paths = field_paths
        self.sample_number = sample_number

        if field_paths == self.FIELD_A_PATHS:
            self.field_letter = "A"
        elif field_paths == self.FIELD_B_PATHS:
            self.field_letter = "B"
        else:
            raise ValueError("field_paths must be either FIELD_A_PATHS or FIELD_B_PATHS")

    def import_data(self, show=False):
        """Importation of the GPR field A data"""
        gpr_data_table = []
        for gpr_path in self.field_paths:
            data_frame = pd.read_csv(gpr_path, sep="  ", engine="python")  # read csv file
            data_frame.columns = ["y", "x", "vwc"]  # rename columns
            gpr_data_table.append(data_frame)

        if show:
            print(gpr_data_table)

        return gpr_data_table

    def extract_dates(self, show=False):
        """Dates extraction from files names"""
        dates = []
        for gpr_path in self.field_paths:
            file_name = os.path.basename(gpr_path)
            file_name_without_extension = os.path.splitext(file_name)[0]
            date = (
                file_name_without_extension[4:6]
                + "/"
                + file_name_without_extension[2:4]
                + "/"
                + "20"
                + file_name_without_extension[:2]
            )
            dates.append(date)

        if show:
            print(dates)

        return dates

    def plot_raw_sample(self, plot=True):
        """GPR raw data plot"""
        studied_field = self.import_data()[self.sample_number]

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            studied_field["x"], studied_field["y"], c=studied_field["vwc"], cmap="viridis", label="Sampling points"
        )
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")

        if plot:
            plt.title(f"Field {self.field_letter} GPR sampling {self.extract_dates()[self.sample_number]}")
            cb = plt.colorbar(scatter)
            cb.set_label("Volumetric Water Content [/]")
            plt.grid(False)
            plt.legend()
            plt.show()

    def plot_mean_median(self, plot=True):
        """GPR mean and median data plot"""
        studied_field = self.import_data()

        mean_evolution = []
        for gpr_data_table in studied_field:
            mean_evolution.append(gpr_data_table["vwc"].mean())

        median_evolution = []
        for gpr_data_table in studied_field:
            median_evolution.append(gpr_data_table["vwc"].median())

        dates = pd.to_datetime(self.extract_dates(), format="%d/%m/%Y")  # Convert dates to datetime objects

        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(dates, median_evolution, marker="o", label="Median")
            plt.plot(dates, mean_evolution, marker="o", label="Mean")
            plt.xlabel("Date")
            plt.ylabel("VWC [/]")
            plt.title("Evolution of Median and Mean Volumetric Water Content on field A")
            plt.xticks(rotation=45)
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(12))
            plt.ylim(0.2, 0.5)
            plt.grid(True)
            plt.legend()
            plt.show()

    def kriging(self, x_grid_step=0.00001, y_grid_step=0.00001, plot=True):
        """Kriging interpolation"""

        studied_field = self.import_data()[self.sample_number]

        # Define your prediction grid
        x_min, x_max = min(studied_field["x"]), max(studied_field["x"])
        y_min, y_max = min(studied_field["y"]), max(studied_field["y"])

        grid_x = np.arange(x_min, x_max, x_grid_step)  # Adjust the step size as needed
        grid_y = np.arange(y_min, y_max, y_grid_step)  # Adjust the step size as needed

        ordinary_kriging = OrdinaryKriging(
            studied_field["x"],
            studied_field["y"],
            studied_field["vwc"],
            variogram_model="exponential",
            verbose=False,
            enable_plotting=False,
        )

        z, ss = ordinary_kriging.execute("grid", grid_x, grid_y)  # Execute the interpolation

        z_grid = np.transpose(z)  # Transpose the result to match the grid shape

        if plot:
            plt.imshow(
                z_grid, extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()], origin="lower", cmap="viridis"
            )
            plt.colorbar(label="Volumetric Water Content [/]")
            plt.xlabel("X [m]")
            plt.ylabel("Y [m]")
            plt.xticks(rotation=45)
            plt.title(f"Field {self.field_letter} - Interpolated Surface {self.extract_dates()[self.sample_number]}")
            plt.grid(False)

            plt.tight_layout()
            plt.show()


class Variogram:
    """Variogram creation and fitting"""

    def __init__(self, resolution=0.00002, field_paths=GprAnalysis
    .FIELD_A_PATHS, sample_number=0):
        '''Initialisation of the GPR field data'''
        self.resolution = resolution
        self.field_paths = field_paths
        self.sample_number = sample_number

    def determ_experimental_vario(self, maxlag=10, n_lags=100, solo_plot=True):
        """Determine the experimental variogram model"""
        # grid data to ? m resolution
        df_grid, grid_matrix, rows, cols = gs.Gridding.grid_data(
            GprAnalysis.import_data(self)[self.sample_number], "x", "y", "vwc", self.resolution
        )

        df_grid = df_grid[df_grid["Z"].isnull() == False]  # remove nans

        # normal score transformation
        data = df_grid["Z"].values.reshape(-1, 1)
        nst_trans = QuantileTransformer(n_quantiles=500, output_distribution="normal").fit(data)
        df_grid["Nbed"] = nst_trans.transform(data)

        # compute experimental (isotropic) variogram
        coords = df_grid[["X", "Y"]].values
        values = df_grid["Nbed"]

        maxlag = 10  # maximum range distance
        n_lags = 100  # num of bins

        # compute variogram
        v1 = skg.Variogram(coords, values, bin_func="even", n_lags=n_lags, maxlag=maxlag, normalize=False)

        # extract variogram values
        xdata = v1.bins
        ydata = v1.experimental

        if solo_plot:
            plt.figure(figsize=(6, 4))
            plt.scatter(xdata, ydata, s=12, c="g")
            plt.title("Isotropic Experimental Variogram")
            plt.xlabel("Lag (m)")
            plt.ylabel("Semivariance")
            plt.show()

        return v1, xdata, ydata

    def fit_models(self, maxlag=10, n_lags=100, solo_plot=False, multi_plot=True, multi_zoom_plot=True):
        """
        Fits variogram models to the experimental variogram.

        Parameters:
        - maxlag: int, the maximum lag distance for the variogram
        - n_lags: int, the number of lag distances to consider
        - solo_plot: bool, whether to plot each model individually
        - multi_plot: bool, whether to plot all models together
        - multi_zoom_plot: bool, whether to plot all models with zoomed-in x-axis

        Returns:
        None
        """
        # extract experimental variogram values
        v1, xdata, ydata = self.experimental_vario(maxlag, n_lags, solo_plot)

        # use exponential variogram model
        v1.model = "exponential"
        v1.parameters

        # use Gaussian model
        v2 = v1
        v2.model = "gaussian"
        v2.parameters

        # use spherical model
        v3 = v1
        v3.model = "spherical"
        v3.parameters

        # evaluate models
        xi = np.linspace(0, xdata[-1], 100)

        y_exp = [models.exponential(h, v1.parameters[0], v1.parameters[1], v1.parameters[2]) for h in xi]
        y_gauss = [models.gaussian(h, v2.parameters[0], v2.parameters[1], v2.parameters[2]) for h in xi]
        y_sph = [models.spherical(h, v3.parameters[0], v3.parameters[1], v3.parameters[2]) for h in xi]

        # plot variogram models
        if multi_plot:
            plt.figure(figsize=(6, 4))
            plt.plot(xdata / 1000, ydata, "og", label="Experimental variogram")
            plt.plot(xi / 1000, y_gauss, "b--", label="Gaussian variogram")
            plt.plot(xi / 1000, y_exp, "r-", label="Exponential variogram")
            plt.plot(xi / 1000, y_sph, "m*-", label="Spherical variogram")
            plt.title("Isotropic variogram")
            plt.xlabel("Lag [km]")
            plt.ylabel("Semivariance")
            plt.legend(loc="lower right")
            plt.show()

        # plot zoom in models
        if multi_zoom_plot:
            plt.figure(figsize=(6, 4))
            plt.plot(xdata / 1000, ydata, "og", label="Experimental variogram")
            plt.plot(xi / 1000, y_gauss, "b--", label="Gaussian variogram")
            plt.plot(xi / 1000, y_exp, "r-", label="Exponential variogram")
            plt.plot(xi / 1000, y_sph, "m*-", label="Spherical variogram")
            plt.title("Isotropic variogram")
            plt.xlim(0, 0.0000003)
            plt.xlabel("Lag [km]")
            plt.ylabel("Semivariance")
            plt.legend(loc="lower right")
            plt.show()




class MultispecAnalysis:
    TEMPERATURE_RASTER = "D:/Cours bioingé/BIR M2/Mémoire/Data/thermal/MR20240205_georeferenced_thermal_cali.tif"
    NDVI_RASTER = "D:/Cours bioingé/BIR M2/Mémoire/Data/multispectral/NDVI/MR20230719_georeferenced_multi_ndvi.tif"

    def __init__(self, temperature_raster=TEMPERATURE_RASTER, ndvi_raster=NDVI_RASTER):
        self.temperature_raster = temperature_raster
        self.ndvi_raster = ndvi_raster

    def calculate_tvdi(self):
        """Calculate TVDI"""
        # Read temperature raster
        with rasterio.open(self.temperature_raster) as temp_src:
            temperature = temp_src.read(1)
            temp_profile = temp_src.profile

        # Read NDVI raster
        with rasterio.open(self.ndvi_raster) as ndvi_src:
            ndvi = ndvi_src.read(1)
            ndvi_profile = ndvi_src.profile

        # Resample NDVI to match temperature raster dimensions
        ndvi_resampled = np.zeros_like(temperature)
        reproject(
            ndvi,
            ndvi_resampled,
            src_transform=ndvi_src.transform,
            src_crs=ndvi_src.crs,
            dst_transform=temp_profile["transform"],
            dst_crs=temp_profile["crs"],
            resampling=Resampling.nearest,
            dst_resolution=(temp_profile["transform"][0], -temp_profile["transform"][4]),
        )

        # Calculate T_max and T_min
        t_max_values = MultispecAnalysis.t_max(ndvi_resampled)
        t_min_values = MultispecAnalysis.t_min(ndvi_resampled)

        # Calculate TVDI
        tvdi = (temperature - t_min_values) / (t_max_values - t_min_values)

        # Set nodata value to -9999
        tvdi[np.isnan(tvdi)] = -9999

        # Adjusting TVDI range to 0-255 for storing as unsigned 8-bit integer
        tvdi_adjusted = ((tvdi - tvdi.min()) / (tvdi.max() - tvdi.min()) * 255).astype(np.uint8)

        # Plot TVDI
        plt.imshow(tvdi_adjusted, cmap="jet", vmin=0, vmax=255)
        plt.colorbar(label="TVDI")
        plt.title("Temperature Vegetation Dryness Index (TVDI)")
        plt.show()

    def t_max(ndvi):
        '''Placeholder coefficients for T_max(NDVI) = a * NDVI + b'''
        a = 40
        b = 300
        return a * ndvi + b


    def t_min(ndvi):
        '''Placeholder coefficients for T_min(NDVI) = c * NDVI + d'''
        c = 20
        d = 250
        return c * ndvi + d
        

test1 = MultispecAnalysis()
test1.calculate_tvdi()
