import pandas as pd

# Read the original Excel file
input_file = "Data/Teros Piezo/data-final.csv"
data = pd.read_csv(input_file)

# Define the columns for the new DataFrame
columns = [
    "Dates (hours)",
    "T_LS1A",
    "T_LS1B",
    "T_LS2A",
    "T_LS2B",
    "T_LS3A",
    "T_LS3B",
    "T_LS4A",
    "T_LS4B",
    "T_LS5A",
    "T_LS5B",
    "P_LS1",
    "P_LS2",
    "P_LS3",
    "P_LS4",
    "P_LS5",
]

# Create a new DataFrame with the specified columns
output_data = pd.DataFrame(columns=columns)

# Extract the required columns and assign them to the new DataFrame
output_data["Dates (hours)"] = data["Time"]
output_data["T_LS1A"] = data[["LS1A10.VWC", "LS1A30.VWC", "LS1A80.VWC"]].median(axis=1)
output_data["T_LS1B"] = data[["LS1B10.VWC", "LS1B30.VWC", "LS1B80.VWC"]].median(axis=1)
output_data["T_LS2A"] = data[["LS2A10.VWC", "LS2A30.VWC", "LS2A50.VWC"]].median(axis=1)
output_data["T_LS2B"] = data[["LS2B10.VWC", "LS2B30.VWC", "LS2B50.VWC"]].median(axis=1)
output_data["T_LS3A"] = data[["LS3A10.VWC", "LS3A30.VWC", "LS3A90.VWC"]].median(axis=1)
output_data["T_LS3B"] = data[["LS3B10.VWC", "LS3B30.VWC", "LS3B90.VWC"]].median(axis=1)
output_data["T_LS4A"] = data[["LS4A10.VWC", "LS4A30.VWC", "LS4A55.VWC"]].median(axis=1)
output_data["T_LS4B"] = data[["LS4B10.VWC", "LS4B30.VWC", "LS4B55.VWC"]].median(axis=1)
output_data["T_LS5A"] = data[["LS5A10.VWC", "LS5A30.VWC", "LS5A60.VWC"]].median(axis=1)
output_data["T_LS5B"] = data[["LS5B10.VWC", "LS5B30.VWC", "LS5B60.VWC"]].median(axis=1)
output_data["P_LS1"] = data["LS1.Temp"]
output_data["P_LS2"] = data["LS2.Temp"]
output_data["P_LS3"] = data["LS3.Temp"]
output_data["P_LS4"] = data["LS4.Temp"]
output_data["P_LS5"] = data["LS5.Temp"]

# Save the new DataFrame to an Excel file
output_file = "teros_piezo.csv"
output_data.to_csv(output_file, index=False)

print(f"Data has been successfully extracted to {output_file}")
