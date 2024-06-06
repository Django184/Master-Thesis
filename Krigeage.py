
# Import libraries
import pandas as pd
import glob 
import numpy as np
import matplotlib.pyplot as plt
import os
import skgstat as skg
from skgstat import models
import gstatsim as gs
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import QuantileTransformer 
import pyproj # for reprojection

# Fields paths
field_a_paths = glob.glob("D:/Cours bioingé/BIR M2/Mémoires/Data/Drone GPR/Field A/*.txt") # return all file paths that match a specific pattern
field_b_paths = glob.glob("D:/Cours bioingé/BIR M2/Mémoires/Data/Drone GPR/Field B/*.txt")

sample_number = 0           # [0-10]
field_path = field_a_paths  # field_a_paths or field_b_paths

# Define the transformer for WGS84 to UTM (UTM zone 32N)
transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:32632", always_xy=True)

# Import data function
def import_data(file_paths=glob.glob("D:/Cours bioingé/BIR M2/Mémoires/Data/Drone GPR/Field A/*.txt")):
    gpr_data_tables = []
    for file_path in file_paths:
        data_frame = pd.read_csv(file_path, sep = "  ", engine="python") # read csv file
        data_frame.columns = ['y', 'x', 'vwc'] # rename columns
        gpr_data_tables.append(data_frame)

    return gpr_data_tables

# Extract dates function
dates = []
def extract_dates(file_paths=glob.glob("D:/Cours bioingé/BIR M2/Mémoires/Data/Drone GPR/Field A/*.txt")):
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        file_name_without_extension = os.path.splitext(file_name)[0]
        date = file_name_without_extension[4:6] + "/" + file_name_without_extension[2:4] + "/" + "20" + file_name_without_extension[:2]
        dates.append(date)
        
    return dates

# Letter of the field
if field_path == field_a_paths:
    field_letter = "A"
else:
    field_letter = "B"

# Date of the files
extract_dates(field_path)

# Read csv file
Studied_field = import_data(field_path)# grid data to 0,5 m resolution and remove coordinates with NaNs
res = 0.5
df_grid, grid_matrix, rows, cols = gs.Gridding.grid_data(Studied_field, 'x', 'y', 'vwc', res)
df_grid = df_grid[df_grid["Z"].isnull() == False]
df_grid = df_grid.rename(columns = {"Z": "vwc"})

# normal score transformation
data = df_grid['vwc'].values.reshape(-1,1)
nst_trans = QuantileTransformer(n_quantiles=500, output_distribution="normal").fit(data)
df_grid['nvwc'] = nst_trans.transform(data) 

# compute experimental (isotropic) variogram
coords = df_grid[['x','y']].values.reshape(-1, 2)
values = df_grid['nvwc']

maxlag = 50000             # maximum range distance
n_lags = 70                # num of bins

V1 = skg.Variogram(coords, values, bin_func='even', n_lags=n_lags, 
                   maxlag=maxlag, normalize=False)

# use exponential variogram model
V1.model = 'exponential'
V1.parameters# grid data to 0,5 m resolution and remove coordinates with NaNs
res = 0.5
df_grid, grid_matrix, rows, cols = gs.Gridding.grid_data(Studied_field, 'x', 'y', 'vwc', res)
df_grid = df_grid[df_grid["Z"].isnull() == False]
df_grid = df_grid.rename(columns = {"Z": "vwc"})

# normal score transformation
data = df_grid['vwc'].values.reshape(-1,1)
nst_trans = QuantileTransformer(n_quantiles=500, output_distribution="normal").fit(data)
df_grid['nvwc'] = nst_trans.transform(data) 

# compute experimental (isotropic) variogram
coords = df_grid[['x','y']].values.reshape(-1, 2)
values = df_grid['nvwc']

maxlag = 50000             # maximum range distance
n_lags = 70                # num of bins

V1 = skg.Variogram(coords, values, bin_func='even', n_lags=n_lags, 
                   maxlag=maxlag, normalize=False)

# use exponential variogram model
V1.model = 'exponential'
V1.parameters# grid data to 100 m resolution and remove coordinates with NaNs
res = 1000
df_grid, grid_matrix, rows, cols = gs.Gridding.grid_data(Studied_field, 'x', 'y', 'vwc', res)
df_grid = df_grid[df_grid["Z"].isnull() == False]

# Convert latitude and longitude to UTM coordinates
utm_x, utm_y = transformer.transform(Studied_field['x'].values, Studied_field['y'].values)

# Plot the sampling points
plt.figure(figsize=(10, 6))
scatter = plt.scatter(utm_x, utm_y, c=Studied_field['vwc'], cmap='viridis', label='Sampling points')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.title(f'Field {field_letter} GPR sampling {dates[sample_number]}')
cb = plt.colorbar(scatter)
cb.set_label('Volumetric Water Content [/]')
plt.grid(False)
plt.legend()
plt.show()

# grid data to 100 m resolution and remove coordinates with NaNs
res = 0.5
df_grid, grid_matrix, rows, cols = gs.Gridding.grid_data(Studied_field, 'x', 'y', 'vwc', res)
df_grid = df_grid[df_grid["vwc"].isnull() == False]

# normal score transformation
data = df_grid['X'].values.reshape(-1,1)
nst_trans = QuantileTransformer(n_quantiles=500, output_distribution="normal").fit(data)
df_grid['nvwc'] = nst_trans.transform(data) 

# compute experimental (isotropic) variogram
coords = df_grid[['X','Y']].values
values = df_grid['vwc']

maxlag = 50000             # maximum range distance
n_lags = 70                # num of bins

V1 = skg.Variogram(coords, values, bin_func='even', n_lags=n_lags, 
                   maxlag=maxlag, normalize=False)

# use exponential variogram model
V1.model = 'exponential'
V1.parameters


