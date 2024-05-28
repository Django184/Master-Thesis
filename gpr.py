import pandas as pd
import glob 

file_paths = glob.glob("D:/Cours bioingé/BIR M2/Mémoire/Data/Drone GPR/*.txt") # return all file paths that match a specific pattern

gpr_data_tables = []

for file_path in file_paths:
    gpr_data_table = pd.read_csv(file_path, sep = "  ", engine="python")
    gpr_data_tables.append(gpr_data_tables)


print(gpr_data_tables[0].head())
print(gpr_data_tables[0].describe())

import matplotlib.pyplot as plt

# Plot x and y coordinates
plt.figure(figsize=(10, 10))
plt.scatter(gpr_data_tables[0].iloc[:, 0], gpr_data_tables[0].iloc[:, 1])
plt.xlabel('Longitude (WGS84)')
plt.ylabel('Latitude (WGS84)')
plt.title('Locations of GPR measurements')
plt.show()

# Plot volumetric water content
plt.figure(figsize=(10, 10))
plt.scatter(gpr_data_tables[0].iloc[:, 0], gpr_data_tables[0].iloc[:, 1], c=gpr_data_tables[0].iloc[:, 2])
plt.xlabel('Longitude (WGS84)')
plt.ylabel('Latitude (WGS84)')
plt.title('Volumetric water content of the soil')
plt.colorbar()
plt.show()










#gpr_data = pd.read_csv(file_path, sep = "  ", engine="python")






