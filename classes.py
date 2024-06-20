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


# Fields paths
field_a_paths = glob.glob("D:/Cours bioingé/BIR M2/Mémoires/Data/Drone GPR/Field A/*.txt")
field_b_paths = glob.glob("D:/Cours bioingé/BIR M2/Mémoires/Data/Drone GPR/Field B/*.txt")


class GprData:

    def __init__(self, field_path):
        """Import the data"""
        if field_path == field_a_paths:
            self.path = field_a_paths
        elif field_path == field_b_paths:
            self.path = field_b_paths
        else:
            raise ValueError("Invalid field path")

    def extract_dates(self):
        """Extract dates from the file names"""
        dates = []
        for file_path in self.path:
            file_name = os.path.basename(file_path)
            file_name_without_extension = os.path.splitext(file_name)[0]
            date = (
                file_name_without_extension[4:6]
                + "/"
                + file_name_without_extension[2:4]
                + "/20"
                + file_name_without_extension[:2]
            )
            dates.append(date)
        return dates

    def plot_gpr(self, field_letter, sample_number):
        """Plot the GPR data"""
        if field_letter == "A":
            field_path = field_a_paths
        elif field_letter == "B":
            field_path = field_b_paths
        else:
            raise ValueError("Invalid field letter")

        studied_field = import_data(field_path)[sample_number]
        utm_x, utm_y = transformer.transform(studied_field["x"].values, studied_field["y"].values)

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(utm_x, utm_y, c=studied_field["vwc"], cmap="viridis", label="Sampling points")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        title = f"Field {field_letter} GPR sampling {self.extract_dates()[sample_number]}"
        plt.title(title)
        cb = plt.colorbar(scatter)
        cb.set_label("Volumetric Water Content [/]")
        plt.grid(False)
        plt.legend()
        plt.show()


test1 = GprData(field_a_paths)
test1.plot_gpr("A", 0)
