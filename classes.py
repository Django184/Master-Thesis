import glob
import os
import re
from datetime import datetime
from pathlib import Path

import contextily as cx
import gstatsim as gs
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj  # for reprojection
import rasterio
import scipy.stats
import seaborn as sns
import skgstat as skg
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pykrige.ok import OrdinaryKriging
from rasterio.warp import Resampling
from rasterio.warp import reproject
from scipy.spatial import distance as dist
from shapely.geometry import Polygon
from skgstat import models
from sklearn.metrics import r2_score
from sklearn.preprocessing import QuantileTransformer

GPR_A_PATHS = sorted(glob.glob("Data/Drone GPR/Field A/*.txt"))
GPR_B_PATHS = sorted(glob.glob("Data/Drone GPR/Field B/*.txt"))

# ---- RASTER FILE LISTS (define before any code uses/prints them) ----
# Thermal
TEMP_RASTER = sorted(
    glob.glob(os.path.join("Data", "thermal", "**", "*.tif"), recursive=True)
    + glob.glob(os.path.join("Data", "thermal", "**", "*.TIF"), recursive=True)
    + glob.glob(os.path.join("Data", "thermal", "**", "*.tiff"), recursive=True)
    + glob.glob(os.path.join("Data", "thermal", "**", "*.TIFF"), recursive=True)
)

# Multispectral
NDVI_RASTER = sorted(glob.glob(os.path.join("Data", "Multispectral", "NDVI", "**", "*.[tT][iI][fF]"), recursive=True))
BLUE_RASTER = sorted(glob.glob(os.path.join("Data", "Multispectral", "Blue", "**", "*.[tT][iI][fF]"), recursive=True))
GREEN_RASTER = sorted(glob.glob(os.path.join("Data", "Multispectral", "Green", "**", "*.[tT][iI][fF]"), recursive=True))
RED_RASTER = sorted(glob.glob(os.path.join("Data", "Multispectral", "Red", "**", "*.[tT][iI][fF]"), recursive=True))
NIR_RASTER = sorted(glob.glob(os.path.join("Data", "Multispectral", "NIR", "**", "*.[tT][iI][fF]"), recursive=True))
RED_EDGE_RASTER = sorted(
    glob.glob(os.path.join("Data", "Multispectral", "Rededge", "**", "*.[tT][iI][fF]"), recursive=True)
)

print("[DEBUG] NDVI_RASTER count:", len(NDVI_RASTER))
if NDVI_RASTER:
    print("  e.g.", NDVI_RASTER[0])

print("[DEBUG] TEMP_RASTER count:", len(TEMP_RASTER))
if TEMP_RASTER:
    print("  e.g.", TEMP_RASTER[0])


class GprAnalysis:
    """Visualisation of the GPR field data"""

    def __init__(self, field_letter="A", sample_number=0):
        """Initialisation of the GPR field data"""

        self.field_letter = field_letter
        self.sample_number = sample_number

        if self.field_letter == "A":
            self.field_paths = GPR_A_PATHS
        elif self.field_letter == "B":
            self.field_paths = GPR_B_PATHS
        else:
            raise ValueError("field_letter must be either A or B")

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

    def extract_dates(self):
        """Dates extraction from files names"""
        dates = []
        for gpr_path in self.field_paths:
            file_name = os.path.basename(gpr_path)
            file_name_without_extension = os.path.splitext(file_name)[0]
            date = (
                file_name_without_extension[8:10]
                + "/"
                + file_name_without_extension[6:8]
                + "/"
                + "20"
                + file_name_without_extension[4:6]
            )
            dates.append(date)

        return dates

    def plot_raw_data(self):
        """Plot the raw GPR data"""
        # Read csv file
        studied_field = self.import_data()[self.sample_number]

        # Convert latitude and longitude to UTM coordinates
        utm_x, utm_y = self.convert_to_utm(studied_field["x"].values, studied_field["y"].values)

        # Plot the raw data
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(utm_x, utm_y, c=studied_field["vwc"], cmap="viridis_r", label="Raw data")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.title(f"GPR sampling - Field {self.field_letter} ({self.extract_dates()[self.sample_number]})")
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

        dates = pd.to_datetime(self.extract_dates(), format="%d/%m/%Y")  # Convert dates to datetime objects

        if plot:
            plt.figure(figsize=(8, 6))
            plt.plot(dates, median_evolution, marker="o", label="Median")
            plt.plot(dates, mean_evolution, marker="o", label="Mean")
            plt.xlabel("Date")
            plt.ylabel("VWC [/]")
            plt.title(
                f"Evolution of GPR derived Volumetric Water Content - Field {self.field_letter} (May 2023 - Feb 2024)"
            )
            plt.xticks(rotation=45)
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(12))
            # plt.ylim(0.2, 0.5)
            plt.grid(True)
            plt.legend()
            plt.show()

        return mean_evolution, median_evolution

    def zonal_check(self):
        if self.field_letter == "A":
            polygon_coords = np.array([[0, 50], [150, 200], [75, 250], [0, 200], [0, 50]])
        else:
            polygon_coords = np.array([[30, 50], [140, 125], [140, 200], [0, 200], [0, 125], [30, 50]])
        polygon = path.Path(polygon_coords)
        self.plot_raw_data_by_zone(polygon)

        upper_evolution = []
        lower_evolution = []
        for data in self.import_data():
            upper_zone = []
            lower_zone = []
            utm_x, utm_y = self.convert_to_utm(data["x"].values, data["y"].values)
            for x, y, vwc in zip(utm_x, utm_y, data["vwc"]):
                if polygon.contains_point((x, y)):
                    upper_zone.append(vwc)
                else:
                    lower_zone.append(vwc)

            upper_evolution.append(np.median(upper_zone))
            lower_evolution.append(np.median(lower_zone))

        dates = pd.to_datetime(self.extract_dates(), format="%d/%m/%Y")  # Convert dates to datetime objects
        plt.figure(figsize=(8, 6))
        plt.plot(dates, upper_evolution, marker="o", label="Zone 2")
        plt.plot(dates, lower_evolution, marker="o", label="Zone 1")
        plt.xlabel("Date")
        plt.ylabel("VWC [/]")
        plt.title(f"Evolution of GPR derived VWC by zone - Field {self.field_letter}")
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(12))
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_raw_data_by_zone(self, zone_1):
        """Plot the raw GPR data"""
        # Read csv file
        studied_field = self.import_data()[self.sample_number]

        # Convert latitude and longitude to UTM coordinates
        utm_x, utm_y = self.convert_to_utm(studied_field["x"].values, studied_field["y"].values)
        zone_1_x = []
        zone_1_y = []
        zone_1_vwc = []
        zone_2_x = []
        zone_2_y = []
        zone_2_vwc = []
        for x, y, i in zip(utm_x, utm_y, range(len(utm_x))):
            if zone_1.contains_point((x, y)):
                zone_1_x.append(x)
                zone_1_y.append(y)
                zone_1_vwc.append(studied_field["vwc"].values[i])
            else:
                zone_2_x.append(x)
                zone_2_y.append(y)
                zone_2_vwc.append(studied_field["vwc"].values[i])
        # Plot the raw data
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(zone_1_x, zone_1_y, c=zone_1_vwc, cmap="viridis_r", label="Raw data")
        scatter2 = plt.scatter(zone_2_x, zone_2_y, c=zone_2_vwc, cmap="BrBG_r", label="Raw data")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.title(f"GPR sampling by zone - Field {self.field_letter} ({self.extract_dates()[self.sample_number]})")
        cb = plt.colorbar(scatter)
        cb.set_label("Zone 2 Volumetric Water Content [/]")
        cb = plt.colorbar(scatter2)
        cb.set_label("Zone 1 Volumetric Water Content [/]")
        plt.grid(False)
        plt.legend()
        plt.show()

    def kriging(self, plot=True):
        """
        Ordinary Kriging interpolation
        """
        studied_field = self.import_data()[self.sample_number]

        # Convert latitude and longitude to UTM coordinates
        utm_x, utm_y = self.convert_to_utm(studied_field["x"].values, studied_field["y"].values)
        resolution = 5
        # Define your prediction grid
        x_min, x_max = 0, 250.0
        y_min, y_max = 0, 250.0
        step_size = 1
        # Adjust the step size as needed
        grid_x = np.arange(x_min, x_max, step_size)
        # Adjust the step size as needed
        grid_y = np.arange(y_min, y_max, step_size)
        # Adjust the step size as needed

        # Define the mask polygon coordinates
        polygon_coords = np.array([[75, 0], [190, 110], [120, 210], [60, 225], [0, 175], [75, 0]])
        xlim, ylim = 200, 250
        if self.field_letter == "B":
            polygon_coords = np.array([[70, 0], [150, 75], [90, 175], [10, 130], [70, 0]])
            xlim, ylim = 150, 175

        # Create a mask for the polygon
        polygon = path.Path(polygon_coords)
        mask = []
        for y in grid_y:
            mask.append([])
            for x in grid_x:
                if not polygon.contains_point((x, y)):
                    mask[int(y / step_size)].append(True)
                else:
                    mask[int(y / step_size)].append(False)
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

        z, ss = ordinary_kriging.execute("masked", grid_x, grid_y, mask=mask)  # Execute the interpolation

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
            plt.title(f"Kriging Interpolation - Field {self.field_letter} ({self.extract_dates()[self.sample_number]})")
            plt.grid(False)
            plt.show()

    def import_tdr_data(self):
        tdr_data_AB = pd.read_excel(TDR_PATHS[self.sample_number - 3])
        tdr_data = []
        if self.field_letter == "A":
            tdr_data = tdr_data_AB[tdr_data_AB["Lat"].values < 50.496773]
        else:
            tdr_data = tdr_data_AB[tdr_data_AB["Lat"].values >= 50.496773]
        tdr_data.columns = ["y", "x", "vwc", "sd"]
        return tdr_data

    def tdr_verification(self, verification_radius=10):
        """
        Performs TDR verification based on the given sample number (3-9)
        """
        gpr_data = self.import_data()[self.sample_number]
        tdr_data = self.import_tdr_data()

        tdr_xs, tdr_ys = self.convert_to_utm(tdr_data["x"].values, tdr_data["y"].values)
        gpr_xs, gpr_ys = self.convert_to_utm(gpr_data["x"].values, gpr_data["y"].values)
        gpr_vwc_median = []
        for tdr_x, tdr_y in zip(tdr_xs, tdr_ys):
            gpr_vwcs = []
            for gpr_x, gpr_y, gpr_vwc in zip(gpr_xs, gpr_ys, gpr_data["vwc"].values):
                if dist.euclidean((gpr_x, gpr_y), (tdr_x, tdr_y)) < verification_radius:
                    gpr_vwcs.append(gpr_vwc)
            if len(gpr_vwcs) > 0:
                gpr_vwc_median.append(np.median(gpr_vwcs))
            else:
                gpr_vwc_median.append(0)

        tdr_vwcs = list(tdr_data["vwc"].values)
        tdr_sds = list(tdr_data["sd"].values)
        for i in range(len(gpr_vwc_median) - 1, -1, -1):
            if gpr_vwc_median[i] == 0:
                tdr_vwcs[i], tdr_vwcs[-1] = tdr_vwcs[-1], tdr_vwcs[i]
                gpr_vwc_median[i], gpr_vwc_median[-1] = (
                    gpr_vwc_median[-1],
                    gpr_vwc_median[i],
                )
                tdr_sds[i], tdr_sds[-1] = tdr_sds[-1], tdr_sds[i]
                gpr_vwc_median.pop()
                tdr_vwcs.pop()
                tdr_sds.pop()

        date = self.extract_dates()[self.sample_number]

        plt.figure(figsize=(10, 6))

        x = np.arange(len(gpr_vwc_median))  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 0

        for attribute, measurement, sd in zip(["GPR", "TDR"], [gpr_vwc_median, tdr_vwcs], [0, tdr_sds]):
            offset = width * multiplier
            plt.bar(x + offset, measurement, width, label=attribute, yerr=sd)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        plt.ylabel("VWC [/]")
        plt.xlabel("TDR verification points")
        plt.title(
            f"GPR and TDR derived VWC zonal comparison (area = {verification_radius}m) - Field {self.field_letter} ({date})"
        )
        plt.xticks(rotation=45)
        plt.legend(loc="upper left", ncols=2)
        plt.ylim(0, 1.15)
        plt.show()
        # Plot the raw data
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(tdr_xs, tdr_ys, c="red", label="TDR", marker="s")
        scatter2 = plt.scatter(gpr_xs, gpr_ys, c=gpr_data["vwc"], cmap="viridis_r", label="GPR")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.title(f"GPR sampling - Field {self.field_letter} ({self.extract_dates()[self.sample_number]})")
        # cb = plt.colorbar(scatter)
        # cb.set_label("GBR Volumetric Water Content [/]")
        cb = plt.colorbar(scatter2)
        cb.set_label("GBR Volumetric Water Content [/]")
        plt.grid(False)
        plt.legend()
        plt.show()

    def correlate_gpr_terros(self):
        """
        Plot the correlation between GPR derived Volumetric Water Content
        mean and median values with Teros derived Volumetric Water Content
        values.
        """
        gpr_mean, gpr_median = self.plot_mean_median(plot=False)
        dates = pd.to_datetime(self.extract_dates(), format="%d/%m/%Y")  # Convert dates to datetime objects

        terros_data = pd.read_csv("Data/Teros Piezo/teros_piezo.csv")
        terros_data["Dates (hours)"] = pd.to_datetime(terros_data["Dates (hours)"], format="%Y-%m-%d %H:%M:%S")
        terros_data["Date"] = terros_data["Dates (hours)"].dt.date

        # Group by day and calculate median VWC values
        terros_median = (
            terros_data.groupby("Date")[
                ["T_LS1A", "T_LS1B", "T_LS2A", "T_LS2B", "T_LS3A", "T_LS3B", "T_LS4A", "T_LS4B", "T_LS5A", "T_LS5B"]
            ]
            .median()
            .mean(axis=1)
        )

        # Select only the dates that correspond to the GPR data dates
        terros_median = terros_median[terros_median.index.isin(dates.date)]

        # Convert gpr_median to numpy array
        gpr_median = np.array(gpr_median)

        # Use numpy.polyfit to fit a linear regression line
        coefficients = np.polyfit(gpr_median, terros_median.values, 1)
        slope, intercept = coefficients

        print(f"y = {slope}x + {intercept}")

        # Calculate R-squared
        r2 = r2_score(terros_median.values, intercept + slope * gpr_median)

        # Plot scatter plot and linear regression
        plt.figure(figsize=(10, 5))
        plt.scatter(gpr_median, terros_median.values, color="blue", label="Data points")
        plt.plot(
            gpr_median,
            intercept + slope * gpr_median,
            color="red",
            label=f"Linear Fit: y = {slope:.2f}x + {intercept:.2f}",
        )
        plt.xlabel("GPR Median VWC", labelpad=10)
        plt.ylabel("Teros Median VWC")
        plt.title(f"Correlation between GPR and Terros derived VWC - Field {self.field_letter}")
        plt.grid(True)
        plt.legend()

        # Add R-squared text annotation
        plt.text(0.05, 0.95, f"R$^2$ = {r2:.2f}", transform=plt.gca().transAxes)

        plt.show()

        # Plot GPR and Terros VWC over time
        plt.figure(figsize=(10, 5))
        plt.plot(dates, gpr_median, "s-", color="green", label="GPR Median VWC")
        plt.plot(dates, terros_median.values, "o-", color="purple", label="Teros Median VWC")
        plt.xlabel("Date")
        plt.ylabel("VWC")
        plt.title(f"Evolution of GPR and Terros derived VWC - Field {self.field_letter} (May 2023 - Feb 2024)")
        plt.legend()
        plt.grid(True)
        plt.gcf().autofmt_xdate()
        plt.show()

    def extract_tdr_dates(self):
        """Dates extraction from TDR files names"""
        dates = []
        for tdr_path in TDR_PATHS:
            file_name = os.path.basename(tdr_path)
            file_name_without_extension = os.path.splitext(file_name)[0]
            date = (
                file_name_without_extension[-2:]
                + "/"
                + file_name_without_extension[-5:-3]
                + "/"
                + "20"
                + file_name_without_extension[-8:-6]
            )
            dates.append(date)
        return dates

    def correlate_gpr_tdr(self):
        """
        Plot the correlation between GPR derived Volumetric Water Content
        mean and median values with TDR derived Volumetric Water Content
        values.
        """
        gpr_mean, gpr_median = self.plot_mean_median(plot=False)
        dates = pd.to_datetime(self.extract_dates(), format="%d/%m/%Y")  # Convert dates to datetime objects

        # Extract TDR data
        tdr_data = []
        for tdr_path in TDR_PATHS:
            tdr_data.append(pd.read_excel(tdr_path))

        tdr_dates = self.extract_tdr_dates()
        tdr_dates = pd.to_datetime(tdr_dates, format="%d/%m/%Y")  # Convert dates to datetime objects

        # Calculate median VWC values for each date
        tdr_medians = []
        for data in tdr_data:
            if self.field_letter == "A":
                tdr_medians.append(data[data["Lat"] < 50.496773]["VWC"].median())
            else:
                tdr_medians.append(data[data["Lat"] >= 50.496773]["VWC"].median())

        # Convert gpr_median to numpy array
        gpr_median = np.array(gpr_median)
        tdr_medians = np.array(tdr_medians)

        gpr_median_tdr_dates = gpr_median[dates.isin(tdr_dates)]

        # Use numpy.polyfit to fit a linear regression line
        coefficients = np.polyfit(gpr_median_tdr_dates, tdr_medians, 1)
        slope, intercept = coefficients

        print(f"y = {slope}x + {intercept}")

        # Calculate R-squared
        r2 = r2_score(tdr_medians, intercept + slope * gpr_median_tdr_dates)

        # Plot scatter plot and linear regression
        plt.figure(figsize=(10, 5))
        plt.scatter(gpr_median_tdr_dates, tdr_medians, color="blue", label="Data points")
        plt.plot(
            gpr_median_tdr_dates,
            intercept + slope * gpr_median_tdr_dates,
            color="red",
            label=f"Linear Fit: y = {slope:.2f}x + {intercept:.2f}",
        )
        plt.xlabel("GPR Median VWC", labelpad=10)
        plt.ylabel("TDR Median VWC")
        plt.title(f"Correlation between GPR and TDR VWC - Field {self.field_letter} (May 2023 - Feb 2024)")
        plt.grid(True)
        plt.legend()

        # Add R-squared text annotation
        plt.text(0.05, 0.95, f"R$^2$ = {r2:.2f}", transform=plt.gca().transAxes)

        plt.show()

        # Plot GPR and TDR VWC over time
        plt.figure(figsize=(10, 5))
        plt.plot(dates[: len(gpr_median_tdr_dates)], gpr_median_tdr_dates, "s-", color="green", label="GPR Median VWC")
        plt.plot(dates[: len(tdr_medians)], tdr_medians, "o-", color="purple", label="TDR Median VWC")
        plt.xlabel("Date")
        plt.ylabel("VWC")
        plt.title(f"Evolution of GPR and TDR derived VWC - Field {self.field_letter} (May 2023 - Feb 2024)")
        plt.legend()
        plt.grid(True)
        plt.gcf().autofmt_xdate()
        plt.show()


class Variogram:
    """Variogram creation and fitting"""

    def __init__(self, resolution=0.00002, field_letter="A", sample_number=0):
        """Initialisation of the GPR field data"""
        self.resolution = resolution
        self.field_letter = field_letter
        self.sample_number = sample_number

        if field_letter == "A":
            self.field_paths = GPR_A_PATHS
        elif field_letter == "B":
            self.field_paths = GPR_B_PATHS
        else:
            raise ValueError("field_letter must be either A or B")

        self.gpr_analysis = GprAnalysis(field_letter, sample_number)

    def determ_experimental_vario(self, maxlag=30, n_lags=50, solo_plot=True):
        """
        Determine the experimental variogram model
        Parameters:
        - maxlag: int, the maximum range distance
        - n_lags: int, the number of bins

        """
        # Read csv file
        studied_field = self.gpr_analysis.import_data()[self.sample_number]

        # Convert latitude and longitude to UTM coordinates
        utm_x, utm_y = self.gpr_analysis.convert_to_utm(studied_field["x"].values, studied_field["y"].values)
        # Create a new DataFrame with UTM coordinates
        df_grid = pd.DataFrame({"X": utm_x, "Y": utm_y, "Z": studied_field["vwc"]})

        # Remove NaN values
        df_grid = df_grid[df_grid["Z"].isnull() == False]

        # Normal score transformation
        data = df_grid["Z"].values.reshape(-1, 1)
        nst_trans = QuantileTransformer(n_quantiles=500, output_distribution="normal").fit(data)
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
        n_lags=50,
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

        y_exp = [models.exponential(h, v1.parameters[0], v1.parameters[1], v1.parameters[2]) for h in xi]
        y_gauss = [models.gaussian(h, v2.parameters[0], v2.parameters[1], v2.parameters[2]) for h in xi]
        y_sph = [models.spherical(h, v3.parameters[0], v3.parameters[1], v3.parameters[2]) for h in xi]

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


TDR_PATHS = sorted(glob.glob("data/VWC verification/*.xlsx"))


class Rainfall:
    PATHS = glob.glob("Data/Météo/*.xlsx")

    def __init__(self, paths=PATHS):
        """Initialisation of the raifall field data"""
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
            ax.set_title(f"Rainfall Precipitations - Mont Rigi (May 2023 - Feb 2024)")
            ax.xaxis.set_major_locator(plt.MaxNLocator(12))
            ax.set_ylim(0, 40)
            # ax.grid(True)
            plt.xticks(rotation=45)
            ax.legend()
            fig.tight_layout()
            plt.show()


class Teros:
    COORD_PATH = "Data/Teros Piezo/coordonnees.xlsx"
    DATA_PATH = "Data/Teros Piezo/teros_piezo.csv"

    def __init__(self, paths=[COORD_PATH, DATA_PATH]):
        """Initialization of the Teros Piezo field data"""
        self.paths = paths
        self.sampler_coords = self.import_coordinates()
        self.data = self.import_vwc_values()

    def import_coordinates(self):
        """Importation of the coordinates of the Teros Piezo"""
        coord = pd.read_excel(self.COORD_PATH)
        sampler_coords = coord.set_index("Sampler")[["North", "East"]]
        return sampler_coords

    def import_vwc_values(self):
        """Importation of the Teros Piezo field data"""
        data = pd.read_csv(self.DATA_PATH, parse_dates=["Dates (hours)"])
        return data

    def plot_vwc_evolution(self, plot=True):
        # Ensure the 'Dates (hours)' column is set as the index
        """
        Plot the median evolution of Volumetric Water Content (VWC) over time for
        each of the Teros probes.

        The 'Dates (hours)' column is used as the index and the median VWC is
        calculated for each day.

        The resulting plot shows the median VWC evolution over time for each probe.
        """
        self.data.set_index("Dates (hours)", inplace=True)

        # Select the columns of interest
        vwc_columns = [col for col in self.data.columns if col.startswith("T_")]

        # Resample the data by day and calculate the median VWC
        vwc_daily_median = self.data[vwc_columns].resample("D").median()

        # Plot the median evolution of VWC over time
        if plot:
            plt.figure(figsize=(12, 8))
            for col in vwc_columns:
                plt.plot(vwc_daily_median.index, vwc_daily_median[col], label=col)

            plt.xlabel("Time (days)")
            plt.ylabel("VWC")
            plt.legend(loc="upper left")
            plt.title("Teros - VWC")
            plt.show()

        return vwc_daily_median

    def plot_piezo_sampler_locations(self):
        """Plot the locations of the different samplers with more distinctive colors"""
        fig = plt.figure(figsize=(10, 6))  # Create a figure and axis
        ax = fig.add_subplot(111)  # Create an axis

        # Filter samplers for A or B field data
        field_samplers = self.sampler_coords[self.sampler_coords.index.str.contains("[AB]$")]

        # Assign more distinctive colors based on sampler names using the 'tab20' colormap
        colors = plt.cm.tab20(np.linspace(0, 1, len(field_samplers)))

        for i, (sampler_name, sampler_data) in enumerate(field_samplers.iterrows()):
            plt.scatter(
                sampler_data["East"],
                sampler_data["North"],
                color=colors[i],
                marker="^",
                label=sampler_name,
            )

        plt.xlabel("East Coordinate")
        plt.ylabel("North Coordinate")
        plt.title("Piezo Samplers Locations")
        plt.legend()
        plt.grid(True)
        plt.show()

        # # Add background map with WGS84 CRS
        # cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.OpenStreetMap.Mapnik)


class WaterTable:
    def __init__(
        self,
        path="Data/Water Table/profondeur nappe-final.xlsx",
        coord_path="Data/Teros Piezo/coordonnees.xlsx",
    ):
        self.path = path
        self.coord_path = coord_path
        self.data, self.coord = self.import_data()
        self.sampler_coords = self.import_coordinates()

    def import_data(self):
        wt_data = pd.read_excel(self.path)
        wt_coord = pd.read_excel(self.coord_path)

        return wt_data, wt_coord

    def import_coordinates(self):
        """Importation of the coordinates of the Teros Piezo"""
        coord = pd.read_excel(self.coord_path)
        sampler_coords = coord.set_index("Sampler")[["North", "East"]]
        return sampler_coords

    def plot_wt_evolution(self):
        # Ensure the 'Time' column is set as the index
        self.data.set_index("Time", inplace=True)

        # Resample the data by day and calculate the median water table
        wt_daily_median = self.data.resample("D").median()

        # Plot the median evolution of water table over time
        plt.figure(figsize=(12, 8))
        for col in wt_daily_median.columns:
            plt.plot(wt_daily_median.index, wt_daily_median[col], label=col)

        plt.xlabel("Time (days)")
        plt.ylabel("Water table [cm]")
        plt.legend(loc="upper left")
        plt.title("Depth Water Table Evolution")
        plt.show()

    def plot_wt_sampler_locations(self):
        """Plot the locations of the LS1, LS2, LS3, LS4, and LS5 samplers with more distinctive colors"""
        plt.figure(figsize=(10, 6))

        # Filter samplers for LS1, LS2, LS3, LS4, and LS5
        ls_samplers = self.sampler_coords[
            self.sampler_coords.index.str.contains("LS[1-5]") & ~self.sampler_coords.index.str.contains("[AB]$")
        ]

        # Assign more distinctive colors based on sampler names using the 'tab20' colormap
        colors = plt.cm.tab20(np.linspace(0, 1, len(ls_samplers)))

        for i, (sampler_name, sampler_data) in enumerate(ls_samplers.iterrows()):
            plt.scatter(
                sampler_data["East"],
                sampler_data["North"],
                color=colors[i],
                marker="^",
                label=sampler_name,
            )

        plt.xlabel("East Coordinate")
        plt.ylabel("North Coordinate")
        plt.title("Water Table Samplers Locations")
        plt.legend()
        plt.grid(True)
        plt.show()


class Thermal:
    TEMPERATURE_RASTER = glob.glob("Data/Thermal/*.tif")
    pass


NDVI_RASTER = sorted(glob.glob(os.path.join("Data", "Multispectral", "NDVI", "*.[tT][iI][fF]")))
BLUE_RASTER = glob.glob("Data/Multispectral/Blue/*.tif")
GREEN_RASTER = glob.glob("Data/Multispectral/Green/*.tif")
RED_RASTER = glob.glob("Data/Multispectral/Red/*.tif")
NIR_RASTER = glob.glob("Data/Multispectral/NIR/*.tif")
RED_EDGE_RASTER = glob.glob("Data/Multispectral/Rededge/*.tif")


class MultispecAnalysis:
    def __init__(self, raster=NDVI_RASTER, sample_number=0, field_letter="A"):
        self.raster = raster
        self.sample_number = sample_number
        self.field_letter = field_letter
        self.out_dir = Path("Data/Outputs")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.raster_paths = {
            "ndvi": NDVI_RASTER,
            "red": RED_RASTER,
            "green": GREEN_RASTER,
            "blue": BLUE_RASTER,
            "nir": NIR_RASTER,
            "red_edge": RED_EDGE_RASTER,
            "temp": TEMP_RASTER,
        }

    # ---------- NEW SAFE HELPERS (used only by TVDI) ----------
    @staticmethod
    def _parse_date_from_name(path_str):
        """Find YYYYMMDD in filename; return date or None."""
        m = re.search(r"(\d{8})", os.path.basename(path_str))
        if not m:
            return None
        try:
            return datetime.strptime(m.group(1), "%Y%m%d").date()
        except Exception:
            return None

    @staticmethod
    def _nearest_by_date(target_date, candidates, max_delta_days=2):
        """Pick path with date nearest to target_date. Returns (path, date, within_window: bool)."""
        if target_date is None or not candidates:
            return None, None, False
        best = None
        best_dt = None
        best_abs = 10**9
        for p in candidates:
            dt = MultispecAnalysis._parse_date_from_name(p)
            if dt is None:
                continue
            diff = abs((dt - target_date).days)
            if diff < best_abs:
                best, best_dt, best_abs = p, dt, diff
        return best, best_dt, (best_abs <= max_delta_days)

    def import_raster_array(self, raster_path):
        """Safe read that returns (array, profile) only — no open handle."""
        with rasterio.open(raster_path) as src:
            arr = src.read(1).astype(np.float32)
            profile = src.profile.copy()
        return arr, profile

    def _import_pair_for_tvdi(self, sample_number):
        """
        Load NDVI at index and thermal for the EXACT same date.
        Returns: temp_C, ndvi_resampled, temp_profile, ndvi_profile, temp_path, ndvi_path
        """
        ndvi_path = self.raster_paths["ndvi"][sample_number]
        ndvi_date = self._parse_date_from_name(ndvi_path)

        if ndvi_date is None:
            raise ValueError(f"Could not parse date from NDVI file: {ndvi_path}")

        print("[DEBUG] Matching NDVI:", ndvi_path)
        print("[DEBUG] Thermal candidates:", len(self.raster_paths["temp"]))

        # Find thermal raster with exactly same date
        exact_matches = [p for p in self.raster_paths["temp"] if self._parse_date_from_name(p) == ndvi_date]

        if not exact_matches:
            print(f"[TVDI] Skipping NDVI date {ndvi_date} — no matching thermal raster found.")  # <<< CHANGE
            return None, None, None, None, None, ndvi_path  # <<< CHANGE

        temp_path = exact_matches[0]

        ndvi, ndvi_prof = self.import_raster_array(ndvi_path)
        temp, temp_prof = self.import_raster_array(temp_path)

        ndvi[(ndvi < -1.0) | (ndvi > 1.0)] = np.nan

        if np.isfinite(temp).any() and np.nanmedian(temp) > 150:
            temp = temp - 273.15
        temp[(temp < -100) | (temp > 80)] = np.nan

        ndvi_resampled = np.full_like(temp, np.nan, dtype=np.float32)
        reproject(
            source=ndvi,
            destination=ndvi_resampled,
            src_transform=ndvi_prof["transform"],
            src_crs=ndvi_prof["crs"],
            dst_transform=temp_prof["transform"],
            dst_crs=temp_prof["crs"],
            resampling=Resampling.nearest,
        )
        return temp, ndvi_resampled, temp_prof, ndvi_prof, temp_path, ndvi_path

    def import_raster(self, raster_path):
        # Open the temperature raster for the specified sample number
        with rasterio.open(raster_path) as raster_src:
            # Read the raster values
            raster_values = raster_src.read(1)  # In rasterio, band numbers are 1-indexed
            # Read the raster profile (metadata)
            raster_profile = raster_src.profile

        return raster_src, raster_values, raster_profile

    def extract_dates(self, raster_type):
        """Dates extraction from files names"""
        dates = []
        for temp_path in self.raster_paths[raster_type]:
            file_name = os.path.basename(temp_path)
            file_name_without_extension = os.path.splitext(file_name)[0]
            date = (
                file_name_without_extension[8:10]
                + "/"
                + file_name_without_extension[6:8]
                + "/"
                + "20"
                + file_name_without_extension[4:6]
            )
            dates.append(date)

        return dates

    def plot_rasters(
        self, sample_number=0, ndvi=False, red=False, green=False, blue=False, nir=False, red_edge=False, ndwi=False
    ):
        if ndvi:
            raster_path = self.raster_paths["ndvi"][sample_number]
            dates = self.extract_dates("ndvi")
            ndvi_src, ndvi_values, ndvi_profile = self.import_raster(raster_path)
            ndvi_values[ndvi_values < -1] = np.nan
            plt.imshow(ndvi_values, cmap="viridis")
            plt.colorbar(label="NDVI")
            plt.title(
                f"Normalized Difference Vegetation Index (NDVI) - Field {self.field_letter} ({dates[sample_number]})"
            )
            plt.show()

        if red:
            raster_path = self.raster_paths["red"][sample_number]
            dates = self.extract_dates("red")
            red_src, red_values, red_profile = self.import_raster(raster_path)
            red_values[red_values < 0] = np.nan
            plt.imshow(red_values, cmap="viridis")
            plt.colorbar(label="Red Band")
            plt.title(f"Red Band - Field {self.field_letter} ({dates[sample_number]})")
            plt.show()

        if green:
            raster_path = self.raster_paths["green"][sample_number]
            dates = self.extract_dates("green")
            green_src, green_values, green_profile = self.import_raster(raster_path)
            green_values[green_values < 0] = np.nan
            plt.imshow(green_values, cmap="viridis")
            plt.colorbar(label="Green Band")
            plt.title(f"Green Band - Field {self.field_letter} ({dates[sample_number]})")
            plt.show()

        if blue:
            raster_path = self.raster_paths["blue"][sample_number]
            dates = self.extract_dates("blue")
            blue_src, blue_values, blue_profile = self.import_raster(raster_path)
            blue_values[blue_values < 0] = np.nan
            plt.imshow(blue_values, cmap="viridis")
            plt.colorbar(label="Blue Band")
            plt.title(f"Blue Band - Field {self.field_letter} ({dates[sample_number]})")
            plt.show()

        if nir:
            raster_path = self.raster_paths["nir"][sample_number]
            dates = self.extract_dates("nir")
            nir_src, nir_values, nir_profile = self.import_raster(raster_path)
            nir_values[nir_values < 0] = np.nan
            plt.imshow(nir_values, cmap="viridis")
            plt.colorbar(label="NIR Band")
            plt.title(f"NIR Band - Field {self.field_letter} ({dates[sample_number]})")
            plt.show()

        if red_edge:
            raster_path = self.raster_paths["red_edge"][sample_number]
            dates = self.extract_dates("red_edge")
            red_edge_src, red_edge_values, red_edge_profile = self.import_raster(raster_path)
            red_edge_values[red_edge_values < 0] = np.nan
            plt.imshow(red_edge_values, cmap="viridis")
            plt.colorbar(label="Red Edge Band")
            plt.title(f"Red Edge Band - Field {self.field_letter} ({dates[sample_number]})")
            plt.show()

        if ndwi:
            ndwi_values = self.calculate_ndwi(sample_number)
            dates = self.extract_dates("green")
            plt.imshow(ndwi_values, cmap="viridis")
            plt.colorbar(label="NDWI")
            plt.title(f"Normalized Water Difference Index (NDWI)  - Field {self.field_letter} ({dates[sample_number]})")
            plt.show()

    def calculate_ndwi(self, sample_number):
        green_raster_path = self.raster_paths["green"][sample_number]
        nir_raster_path = self.raster_paths["nir"][sample_number]
        green_src, green_values, green_profile = self.import_raster(green_raster_path)
        nir_src, nir_values, nir_profile = self.import_raster(nir_raster_path)

        # Calculate NDWI values
        ndwi_values = (green_values - nir_values) / (green_values + nir_values)

        ndwi_values[(ndwi_values < -1) | (ndwi_values == 0)] = np.nan

        return ndwi_values

    def calculate_tvdi(
        self,
        sample_number=None,
        save=True,
        plot=True,
        ndvi_bin_width=0.01,
        pct_wet=5,
        pct_dry=95,
        min_pixels_per_bin=50,
    ):
        if sample_number is None:
            sample_number = self.sample_number

        result = self._import_pair_for_tvdi(sample_number)
        if result[0] is None:
            return None, None

        temp, ndvi_r, temp_prof, ndvi_prof, tpath, npath = result

        mask = np.isfinite(temp) & np.isfinite(ndvi_r)
        if mask.sum() < 1000:
            raise RuntimeError("Too few valid pixels to build TVDI.")
        ts = temp[mask]
        vi = ndvi_r[mask]

        # Bin NDVI and compute robust envelopes
        vi_min, vi_max = np.nanmin(vi), np.nanmax(vi)
        edges = np.arange(
            np.floor(vi_min / ndvi_bin_width) * ndvi_bin_width,
            np.ceil(vi_max / ndvi_bin_width) * ndvi_bin_width + ndvi_bin_width,
            ndvi_bin_width,
        )
        centers, wet_vals, dry_vals, counts = [], [], [], []
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            sel = (vi >= lo) & (vi < hi)
            if sel.sum() >= min_pixels_per_bin:
                centers.append((lo + hi) / 2.0)
                counts.append(sel.sum())
                wet_vals.append(np.nanpercentile(ts[sel], pct_wet))
                dry_vals.append(np.nanpercentile(ts[sel], pct_dry))
        centers = np.asarray(centers, np.float32)
        wet_vals = np.asarray(wet_vals, np.float32)
        dry_vals = np.asarray(dry_vals, np.float32)
        counts = np.asarray(counts, np.float32)
        if centers.size < 3:
            raise RuntimeError("Not enough NDVI bins with data to fit edges.")

        # Weighted linear fits for edges
        W = np.sqrt(counts / (counts.max() + 1e-9))
        a_dry, b_dry = np.polyfit(centers, dry_vals, 1, w=W)  # Tsmax(NDVI)
        a_wet, b_wet = np.polyfit(centers, wet_vals, 1, w=W)  # Tsmin(NDVI)

        # TVDI
        tsmax = a_dry * ndvi_r + b_dry
        tsmin = a_wet * ndvi_r + b_wet
        denom = tsmax - tsmin
        tvdi = (temp - tsmin) / denom
        tvdi[~np.isfinite(tvdi)] = np.nan
        tvdi = np.clip(tvdi, 0, 1).astype(np.float32)

        # Save float32 GeoTIFF
        date_obj = self._parse_date_from_name(npath)
        date_str = date_obj.strftime("%Y%m%d") if date_obj else "unknown"
        out_path = self.out_dir / f"TVDI_Field{self.field_letter}_{date_str}.tif"
        if save:
            prof = temp_prof.copy()
            prof.update(dtype="float32", count=1, nodata=np.nan, compress="lzw")
            with rasterio.open(out_path, "w", **prof) as dst:
                dst.write(tvdi, 1)

        # Plots (keeps your pixel‑poly mask style)
        if plot:
            # --- Scatter triangle ---
            plt.figure(figsize=(5, 4), dpi=300)
            idx = np.random.choice(ts.size, size=min(5000, ts.size), replace=False)
            plt.scatter(vi[idx], ts[idx], s=2, alpha=0.3)
            x_line = np.linspace(np.nanmin(centers), np.nanmax(centers), 100)
            plt.plot(x_line, a_dry * x_line + b_dry, lw=2, label="Dry edge (95th)")
            plt.plot(x_line, a_wet * x_line + b_wet, lw=2, label="Wet edge (5th)")
            ttl_date = date_obj.strftime("%d/%m/%Y") if date_obj else "unknown date"
            plt.xlabel("NDVI [-]")
            plt.ylabel("LST [°C]")
            plt.title(f"NDVI–LST Triangle • Field {self.field_letter} • {ttl_date}")
            plt.legend()
            plt.tight_layout()
            plt.show()

            # --- TVDI map ---
            masked_tvdi = self.apply_field_mask(tvdi)  # <<< ADDED
            plt.figure(figsize=(5, 5), dpi=300)
            plt.imshow(masked_tvdi, cmap="viridis", vmin=0, vmax=1)
            cbar = plt.colorbar()
            cbar.set_label("TVDI [-]")
            plt.title(f"TVDI • Field {self.field_letter} • {ttl_date}")
            plt.tight_layout()
            plt.show()

        return tvdi, (str(out_path) if save else None)

    def apply_field_mask(self, array):
        if self.field_letter == "A":
            polygon_coords = np.array([[3100, 4800], [4550, 3400], [3550, 2350], [2530, 3400], [3100, 4800]])
            x_offset, y_offset = 650, -10

        else:  # Field B
            polygon_coords = np.array(
                [[2530, 3400], [3550, 2350], [3350, 2150], [3550, 1950], [3000, 1000], [1700, 1300], [2530, 3400]]
            )
            x_offset, y_offset = 650, -150

        # Apply offset
        adjusted_coords = polygon_coords.copy()
        adjusted_coords[:, 0] += x_offset  # Shift X
        adjusted_coords[:, 1] += y_offset  # Shift Y

        # Create mask
        poly = path.Path(adjusted_coords)
        ny, nx = array.shape
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
        points = np.vstack((X.ravel(), Y.ravel())).T
        mask = poly.contains_points(points).reshape(array.shape)

        masked_array = array.copy()
        masked_array[~mask] = np.nan
        return masked_array


# test = MultispecAnalysis()
# test.plot_rasters(sample_number=7, ndvi=True, red=True, green=True, blue=True, nir=True, red_edge=True, ndwi=True)

msa = MultispecAnalysis(sample_number=1, field_letter="A")
tvdi, out_path = msa.calculate_tvdi(plot=True, save=True)  # one date
# msa.calculate_tvdi_all(plot=False)  # batch
