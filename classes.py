from mpl_toolkits.axes_grid1 import make_axes_locatable
from pykrige.ok import OrdinaryKriging
from rasterio.warp import reproject, Resampling
from skgstat import models
from sklearn.preprocessing import QuantileTransformer
import gstatsim as gs
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import numpy as np
import os
import pandas as pd
import pyproj  # for reprojection
import rasterio
import skgstat as skg
import glob
from shapely.geometry import Polygon

FIELD_A_PATHS = glob.glob("Data/Drone GPR/Field A/*.txt")
FIELD_B_PATHS = glob.glob("Data/Drone GPR/Field B/*.txt")


class GprAnalysis:
    """Visualisation of the GPR field data"""

    def __init__(self, field_letter="A", sample_number=0):
        """Initialisation of the GPR field data"""
        self.field_letter = field_letter
        self.sample_number = sample_number

        if self.field_letter == "A":
            self.field_paths = FIELD_A_PATHS
        elif self.field_letter == "B":
            self.field_paths = FIELD_B_PATHS
        else:
            raise ValueError("field_letter must be either A or B")

    def import_data(self, show=False):
        """Importation of the GPR field A data"""
        gpr_data_table = []
        for gpr_path in self.field_paths:
            data_frame = pd.read_csv(
                gpr_path, sep="  ", engine="python"
            )  # read csv file
            data_frame.columns = ["y", "x", "vwc"]  # rename columns
            gpr_data_table.append(data_frame)

        if show:
            print(gpr_data_table)

        return gpr_data_table

    def extract_dates(self):
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

        return dates

    def plot_raw_data(self):
        """Plot the raw GPR data"""
        # Read csv file
        studied_field = self.import_data()[self.sample_number]

        # Convert latitude and longitude to UTM coordinates
        utm_x, utm_y = self.convert_to_utm(
            studied_field["x"].values, studied_field["y"].values
        )

        # Plot the raw data
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            utm_x, utm_y, c=studied_field["vwc"], cmap="viridis_r", label="Raw data"
        )
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.title(
            f"GPR sampling - Field {self.field_letter} ({self.extract_dates()[self.sample_number]})"
        )
        cb = plt.colorbar(scatter)
        cb.set_label("Volumetric Water Content [/]")
        plt.grid(False)
        plt.legend()
        plt.show()

    def convert_to_utm(self, latitudes, longitudes):
        """Convert latitude and longitude to UTM coordinates"""
        # Define the WGS84 and UTM coordinate systems
        crs_wgs84 = pyproj.CRS("EPSG:4326")
        crs_utm = pyproj.CRS("EPSG:32632")  # UTM Zone 32

        # Create the transformer from WGS84 to UTM
        transformer = pyproj.Transformer.from_crs(crs_wgs84, crs_utm)

        # Convert latitude and longitude to UTM coordinates
        utm_x, utm_y = transformer.transform(longitudes, latitudes)

        # Shift the UTM coordinates to start at 0m
        utm_x -= utm_x.min()
        utm_y -= utm_y.min()

        return utm_x, utm_y

    def plot_mean_median(self, plot=True):
        """GPR mean and median data plot"""
        studied_field = self.import_data()

        mean_evolution = []
        for gpr_data_table in studied_field:
            mean_evolution.append(gpr_data_table["vwc"].mean())

        median_evolution = []
        for gpr_data_table in studied_field:
            median_evolution.append(gpr_data_table["vwc"].median())

        dates = pd.to_datetime(
            self.extract_dates(), format="%d/%m/%Y"
        )  # Convert dates to datetime objects

        if plot:
            plt.figure(figsize=(8, 6))
            plt.plot(dates, median_evolution, marker="o", label="Median")
            plt.plot(dates, mean_evolution, marker="o", label="Mean")
            plt.xlabel("Date")
            plt.ylabel("VWC [/]")
            plt.title(
                f"Evolution of GPR derived Volumetric Water Content - (Field {self.field_letter})"
            )
            plt.xticks(rotation=45)
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(12))
            plt.ylim(0.2, 0.5)
            plt.grid(True)
            plt.legend()
            plt.show()

    def kriging(self, plot=True):
        """
        Ordinary Kriging interpolation
        """
        studied_field = self.import_data()[self.sample_number]

        # Convert latitude and longitude to UTM coordinates
        utm_x, utm_y = self.convert_to_utm(
            studied_field["x"].values, studied_field["y"].values
        )

        # Define your prediction grid
        x_min, x_max = min(utm_x), max(utm_x)
        y_min, y_max = min(utm_y), max(utm_y)

        grid_x = np.arange(x_min, x_max, 1)
        # Adjust the step size as needed
        grid_y = np.arange(y_min, y_max, 1)
        # Adjust the step size as needed

        # Define the mask polygon coordinates
        polygon_coords = np.array(
            [[75, 0], [190, 110], [120, 210], [60, 225], [0, 175], [75, 0]]
        )
        xlim, ylim = 200, 250
        if self.field_letter == "B":
            polygon_coords = np.array(
                [[65, 0], [150, 75], [90, 175], [0, 125], [65, 0]]
            )
            xlim, ylim = 150, 175

        # Create a mask for the polygon
        polygon = path.Path(polygon_coords)
        mask = []
        for y in grid_y:
            mask.append([])
            for x in grid_x:
                if not polygon.contains_point((x, y)):
                    mask[int(y)].append(True)
                else:
                    mask[int(y)].append(False)
        mask = np.array(mask)
        # Execute the interpolation
        ordinary_kriging = OrdinaryKriging(
            utm_x,
            utm_y,
            studied_field["vwc"],
            variogram_model="exponential",
            verbose=False,
            enable_plotting=False,
        )

        z, ss = ordinary_kriging.execute(
            "masked", grid_x, grid_y, mask=mask
        )  # Execute the interpolation

        if plot:
            plt.figure(figsize=(10, 6))
            plt.imshow(
                z,
                extent=(x_min, x_max, y_min, y_max),
                origin="lower",
                cmap="viridis_r",
                aspect="auto",
            )
            plt.xlim(-5, xlim)
            plt.ylim(-5, ylim)
            plt.colorbar()
            plt.xlabel("X [m]")
            plt.ylabel("Y [m]")
            plt.title(
                f"Kriging Interpolation - Field {self.field_letter} ({self.extract_dates()[self.sample_number]})"
            )
            plt.grid(False)
            plt.show()


class Variogram:
    """Variogram creation and fitting"""

    def __init__(self, resolution=0.00002, field_letter="A", sample_number=0):
        """Initialisation of the GPR field data"""
        self.resolution = resolution
        self.field_letter = field_letter
        self.sample_number = sample_number

        if field_letter == "A":
            self.field_paths = FIELD_A_PATHS
        elif field_letter == "B":
            self.field_paths = FIELD_B_PATHS
        else:
            raise ValueError("field_letter must be either A or B")

        self.gpr_analysis = GprAnalysis(field_letter, sample_number)

    def determ_experimental_vario(self, maxlag=30, n_lags=200, solo_plot=True):
        """
        Determine the experimental variogram model
        Parameters:
        - maxlag: int, the maximum range distance
        - n_lags: int, the number of bins

        """
        # Read csv file
        studied_field = self.gpr_analysis.import_data()[self.sample_number]

        # Convert latitude and longitude to UTM coordinates
        utm_x, utm_y = self.gpr_analysis.convert_to_utm(
            studied_field["x"].values, studied_field["y"].values
        )
        # Create a new DataFrame with UTM coordinates
        df_grid = pd.DataFrame({"X": utm_x, "Y": utm_y, "Z": studied_field["vwc"]})

        # Remove NaN values
        df_grid = df_grid[df_grid["Z"].isnull() == False]

        # Normal score transformation
        data = df_grid["Z"].values.reshape(-1, 1)
        nst_trans = QuantileTransformer(
            n_quantiles=500, output_distribution="normal"
        ).fit(data)
        df_grid["Nbed"] = nst_trans.transform(data)

        # Compute experimental (isotropic) variogram
        coords = df_grid[["X", "Y"]].values
        values = df_grid["Nbed"]

        # Compute variogram
        v1 = skg.Variogram(
            coords,
            values,
            bin_func="even",
            n_lags=n_lags,
            maxlag=maxlag,
            normalize=False,
        )

        # Extract variogram values
        xdata = v1.bins
        ydata = v1.experimental

        if solo_plot:
            plt.figure(figsize=(8, 6))
            plt.scatter(xdata, ydata, s=12, c="g")
            plt.title(
                f"Isotropic experimental model - Field {self.field_letter} ({GprAnalysis.extract_dates(self)[self.sample_number]})"
            )
            plt.xlabel("Lag (m)")
            plt.ylabel("Semivariance")
            plt.show()

        return v1, xdata, ydata

    def fit_models(
        self,
        maxlag=30,
        n_lags=200,
        solo_plot=False,
        multi_plot=True,
        multi_zoom_plot=True,
        sample_number=0,
    ):
        """Fits variogram models to the experimental variogram."""
        # extract experimental variogram values
        v1, xdata, ydata = self.determ_experimental_vario(maxlag, n_lags, solo_plot)

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

        y_exp = [
            models.exponential(h, v1.parameters[0], v1.parameters[1], v1.parameters[2])
            for h in xi
        ]
        y_gauss = [
            models.gaussian(h, v2.parameters[0], v2.parameters[1], v2.parameters[2])
            for h in xi
        ]
        y_sph = [
            models.spherical(h, v3.parameters[0], v3.parameters[1], v3.parameters[2])
            for h in xi
        ]

        # plot variogram models
        if multi_plot:
            plt.figure(figsize=(8, 6))
            plt.plot(xdata, ydata, "og", label="Experimental variogram")
            plt.plot(xi, y_gauss, "b--", label="Gaussian variogram")
            plt.plot(xi, y_exp, "r-", label="Exponential variogram")
            plt.plot(xi, y_sph, "m*-", label="Spherical variogram")
            plt.title(
                f"Isotropic variogram models comparison - Field {self.field_letter} ({GprAnalysis.extract_dates(self)[self.sample_number]})"
            )
            plt.xlabel("Lag [m]")
            plt.ylabel("Semivariance")
            plt.legend(loc="lower right")
            plt.show()

        # plot zoom in models
        if multi_zoom_plot:
            plt.figure(figsize=(8, 6))
            plt.plot(xdata, ydata, "og", label="Experimental variogram")
            plt.plot(xi, y_gauss, "b--", label="Gaussian variogram")
            plt.plot(xi, y_exp, "r-", label="Exponential variogram")
            plt.plot(xi, y_sph, "m*-", label="Spherical variogram")
            plt.title(
                f"Isotropic variogram models comparison (zoom in) - Field {self.field_letter} ({GprAnalysis.extract_dates(self)[self.sample_number]})"
            )
            plt.xlim(0, 5)
            plt.xlabel("Lag [m]")
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
            dst_resolution=(
                temp_profile["transform"][0],
                -temp_profile["transform"][4],
            ),
        )

        # Calculate T_max and T_min
        t_max_values = self.t_max(ndvi_resampled)
        t_min_values = self.t_min(ndvi_resampled)

        # Calculate TVDI
        tvdi = (temperature - t_min_values) / (t_max_values - t_min_values)

        # Set nodata value to -9999
        tvdi[np.isnan(tvdi)] = -9999

        # Adjusting TVDI range to 0-255 for storing as unsigned 8-bit integer
        tvdi_adjusted = ((tvdi - tvdi.min()) / (tvdi.max() - tvdi.min()) * 255).astype(
            np.uint8
        )

        # Plot TVDI
        plt.imshow(tvdi_adjusted, cmap="jet", vmin=200, vmax=300)
        plt.colorbar(label="TVDI")
        plt.title("Temperature Vegetation Dryness Index (TVDI)")
        plt.show()

    def t_max(self, ndvi):
        """Placeholder coefficients for T_max(NDVI) = a * NDVI + b"""
        a = 40
        b = 300
        return a * ndvi + b

    def t_min(self, ndvi):
        """Placeholder coefficients for T_min(NDVI) = c * NDVI + d"""
        c = 20
        d = 250
        return c * ndvi + d


class TdrAnalysis:
    FIELD_PATHS = glob.glob(
        "D:/Cours bioingé/BIR M2/Mémoire/Data/VWC verification/*.xlsx"
    )

    def __init__(self, field_paths=FIELD_PATHS, sample_number=0):
        """Initialisation of the TDR field data"""
        self.field_paths = field_paths
        self.sample_number = sample_number

    def import_data(self):
        """Importation of the TDR field data"""
        tdr_data_table = []
        for tdr_path in self.field_paths:
            data_frame = pd.read_excel(tdr_path)  # read excel file
            tdr_data_table.append(data_frame)

        return tdr_data_table

    def extract_dates(self):
        """Dates extraction from files names"""
        dates = []
        for tdr_path in self.field_paths:
            file_name = os.path.basename(tdr_path)
            file_name_without_extension = os.path.splitext(file_name)[0]
            date = (
                file_name_without_extension[12:14]
                + "/"
                + file_name_without_extension[9:11]
                + "/"
                + "20"
                + file_name_without_extension[6:8]
            )
            dates.append(date)

        return dates

    def plot_tdr_evolution(self, plot=True):
        """TDR median data plot"""
        studied_field = self.import_data()

        # Separate data for fields A and B based on latitude
        field_a_median = []
        field_b_median = []
        field_a_sd_median = []
        field_b_sd_median = []
        for table in studied_field:
            field_a_data = []
            field_a_sd = []
            field_b_data = []
            field_b_sd = []
            for index, lat in enumerate(table["Lat"].values):
                if lat < 50.496773:
                    field_a_data.append(table["VWC"].values[index])
                    field_a_sd.append(table["sd"].values[index])
                else:
                    field_b_data.append(table["VWC"].values[index])
                    field_b_sd.append(table["sd"].values[index])
            field_a_median.append(np.median(field_a_data))
            field_a_sd_median.append(np.median(field_a_sd))
            field_b_median.append(np.median(field_b_data))
            field_b_sd_median.append(np.median(field_b_sd))
        # Create an instance of TdrAnalysis
        tdr_analysis = TdrAnalysis()

        # Plot for Field A
        tdr_analysis.plot_data(field_a_median, field_a_sd_median, "Field A")

        # Plot for Field B
        tdr_analysis.plot_data(field_b_median, field_b_sd_median, "Field B")

    def plot_data(self, medians, sds, field_name):
        """TDR median data plot"""
        variance_upper = [median + sds[i] for i, median in enumerate(medians)]
        variance_lower = [median - sds[i] for i, median in enumerate(medians)]

        dates = pd.to_datetime(self.extract_dates(), format="%d/%m/%Y")

        plt.figure(figsize=(8, 6))
        plt.plot(dates, medians, marker="o", label="Mean")
        plt.fill_between(
            dates,
            variance_lower,
            variance_upper,
            color="gray",
            alpha=0.5,
            label="Variance",
        )
        plt.xlabel("Date")
        plt.ylabel("VWC [/]")
        plt.title(f"TDR derived Volumetric Water Content - {field_name}")
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(12))
        plt.ylim(0.45, 0.95)
        plt.grid(True)
        plt.legend()
        plt.show()


class Rainfall:
    PATHS = glob.glob(r"D:\Cours bioingé\BIR M2\Mémoire\Data\Météo\*.xlsx")

    def __init__(self, paths=PATHS):
        """Initialisation of the TDR field data"""
        self.paths = paths

    def import_excel(self, show=False):
        """Importation of the raifall field data"""
        rf_data = []
        for rf_path in self.paths:
            data_frame = pd.read_excel(rf_path)  # read excel file
            rf_data.append(data_frame)

        if show:
            print(rf_data)

        return rf_data

    def plot_data(self, paths=PATHS, plot=True):
        """Rainfall data plot"""

        field = self.import_excel()

        precipitations = []
        dates = []
        for rf_data in field:
            precipitations.extend(rf_data["prcp"].tolist())
            dates.extend(rf_data["date"].tolist())

        f_dates = pd.to_datetime(pd.Series(dates), format="%Y-%m-%d")
        f_precipitations = pd.Series(precipitations, index=f_dates)

        # Add a smooth curve of the evolution of precipitation
        rolling_window = 10  # Adjust this value to control the smoothing
        smoothed_precipitations = f_precipitations.rolling(
            window=rolling_window,
            min_periods=1,
            center=True,
        ).mean()

        if plot:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(
                f_dates,
                f_precipitations.values,
                align="center",
                alpha=0.5,
                color="cornflowerblue",
            )
            ax.plot(
                smoothed_precipitations.index,
                smoothed_precipitations.values,
                color="firebrick",
                linewidth=2,
            )
            ax.set_xlabel("Date")
            ax.set_ylabel("Precipitation [mm]")
            ax.set_title(f"Rainfall Precipitations Mont Rigi 2023-2024")
            ax.xaxis.set_major_locator(plt.MaxNLocator(12))
            ax.set_ylim(0, 40)
            ax.grid(True)
            plt.xticks(rotation=45)
            ax.legend()
            fig.tight_layout()
            plt.show()


tdr_test = TdrAnalysis()
tdr_test.plot_tdr_evolution()
