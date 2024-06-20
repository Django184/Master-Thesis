from mpl_toolkits.axes_grid1 import make_axes_locatable
from pykrige.ok import OrdinaryKriging
from rasterio.warp import reproject, Resampling
from skgstat import models
from sklearn.preprocessing import QuantileTransformer
import gstatsim as gs
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pyproj  # for reprojection
import rasterio
import skgstat as skg
import glob


class GprSample:
    """Visualisation of the GPR field data"""

    FIELD_A_PATHS = glob.glob("D:/Cours bioingé/BIR M2/Mémoire/Data/Drone GPR/Field A/*.txt")
    FIELD_B_PATHS = glob.glob("D:/Cours bioingé/BIR M2/Mémoire/Data/Drone GPR/Field B/*.txt")

    def __init__(self, field_paths=FIELD_A_PATHS, sample_number=0):
        """Initialisation of the GPR field data"""
        self.field_paths = field_paths
        self.sample_number = sample_number

    def import_data(self):
        """Importation of the GPR field A data"""
        gpr_data_table = []
        for gpr_path in self.field_paths:
            data_frame = pd.read_csv(gpr_path, sep="  ", engine="python")  # read csv file
            data_frame.columns = ["y", "x", "vwc"]  # rename columns
            gpr_data_table.append(data_frame)
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

    def raw_sample_plot(self):
        """GPR raw data plot"""
        studied_field = self.import_data()[self.sample_number]
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            studied_field["x"], studied_field["y"], c=studied_field["vwc"], cmap="viridis", label="Sampling points"
        )
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        if self.field_paths == GprSample.FIELD_A_PATHS:
            field_letter = "A"
        elif self.field_paths == GprSample.FIELD_B_PATHS:
            field_letter = "B"
        else:
            raise ValueError("field_paths must be either FIELD_A_PATHS or FIELD_B_PATHS")
        plt.title(f"Field {field_letter} GPR sampling {self.extract_dates()[self.sample_number]}")
        cb = plt.colorbar(scatter)
        cb.set_label("Volumetric Water Content [/]")
        plt.grid(False)
        plt.legend()
        plt.show()

    def mean_median_plot(self):
        """GPR mean and median data plot"""
        studied_field = self.import_data()

        mean_evolution = []
        for gpr_data_table in studied_field:
            mean_evolution.append(gpr_data_table["vwc"].mean())

        median_evolution = []
        for gpr_data_table in studied_field:
            median_evolution.append(gpr_data_table["vwc"].median())

        dates = pd.to_datetime(self.extract_dates(), format="%d/%m/%Y")  # Convert dates to datetime objects

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


class Variogram:
    """Variogram creation and fitting"""

    def __init__(self, resolution=0.00002, field_paths=GprSample.FIELD_A_PATHS, sample_number=0):
        self.resolution = resolution
        self.field_paths = field_paths
        self.sample_number = sample_number

    def model_fitting(self):
        # grid data to ? m resolution
        df_grid, grid_matrix, rows, cols = gs.Gridding.grid_data(
            GprSample.import_data(self)[self.sample_number], "x", "y", "vwc", self.resolution
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

        plt.figure(figsize=(6, 4))
        plt.scatter(xdata, ydata, s=12, c="g")
        plt.title("Isotropic Experimental Variogram")
        plt.xlabel("Lag (m)")
        plt.ylabel("Semivariance")
        plt.show()

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

        # plot variogram model
        fig = plt.figure()
        plt.plot(xdata / 1000, ydata, "og", label="Experimental variogram")
        plt.plot(xi / 1000, y_gauss, "b--", label="Gaussian variogram")
        plt.plot(xi / 1000, y_exp, "r-", label="Exponential variogram")
        plt.plot(xi / 1000, y_sph, "m*-", label="Spherical variogram")
        plt.title("Isotropic variogram")
        plt.xlabel("Lag [km]")
        plt.ylabel("Semivariance")
        plt.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1.0)  # adjust the plot size
        plt.legend(loc="lower right")
        plt.show()

        fig = plt.figure()
        plt.plot(xdata / 1000, ydata, "og", label="Experimental variogram")
        plt.plot(xi / 1000, y_gauss, "b--", label="Gaussian variogram")
        plt.plot(xi / 1000, y_exp, "r-", label="Exponential variogram")
        plt.plot(xi / 1000, y_sph, "m*-", label="Spherical variogram")
        plt.title("Isotropic variogram")
        plt.xlim(0, 0.0000003)
        plt.xlabel("Lag [km]")
        plt.ylabel("Semivariance")
        plt.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1.0)  # adjust the plot size
        plt.legend(loc="lower right")
        plt.show()


test1 = Variogram()

test1.model_fitting()
