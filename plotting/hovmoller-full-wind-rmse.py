import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.dates import YearLocator, DateFormatter
import utils
import os
import config
from scipy.interpolate import interp1d

FAA = "SLC"
WMO = utils.lookupWMO(FAA)
Station_Name = utils.lookupStationName(FAA)
CO = utils.lookupCountry(FAA)
lat,lon,el = utils.lookupCoordinate(FAA)
print(WMO, FAA, Station_Name, lat,lon,el)


path = "Pictures/Data-Hovmoller/"
# Load the CSV files
era5_file = path + FAA + "-2023-era5.csv"
radiosonde_file = path + FAA + "-2023-radiosonde.csv"

era5_df = pd.read_csv(era5_file)
radiosonde_df = pd.read_csv(radiosonde_file)

# Convert time columns to datetime format
era5_df.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
radiosonde_df.rename(columns={'Unnamed: 0': 'time'}, inplace=True)

era5_df['time'] = pd.to_datetime(era5_df['time'])
radiosonde_df['time'] = pd.to_datetime(radiosonde_df['time'])

# Set time as index for easier alignment
era5_df.set_index('time', inplace=True)
radiosonde_df.set_index('time', inplace=True)

# Convert column names (altitudes) to floats
era5_df.columns = era5_df.columns.astype(float)
radiosonde_df.columns = radiosonde_df.columns.astype(float)

#SBBV 2023
#radiosonde_df.loc[pd.to_datetime("2023-02-04 12:00:00")] = np.nan
#radiosonde_df.loc[pd.to_datetime("2023-06-26 12:00:00")] = np.nan

#SCSN
'''
radiosonde_df.loc[pd.to_datetime("2023-04-03 12:00:00")] = np.nan
radiosonde_df.loc[pd.to_datetime("2023-04-07 12:00:00")] = np.nan
radiosonde_df.loc[pd.to_datetime("2023-09-28 12:00:00")] = np.nan
radiosonde_df.loc[pd.to_datetime("2023-09-30 12:00:00")] = np.nan
radiosonde_df.loc[pd.to_datetime("2023-12-19 12:00:00")] = np.nan
radiosonde_df.loc[pd.to_datetime("2023-12-21 12:00:00")] = np.nan
'''


# Step 1: Regrid ERA5 data to match radiosonde altitude levels
era5_regridded = pd.DataFrame(index=era5_df.index, columns=radiosonde_df.columns)



for time in era5_df.index:
    interp_func = interp1d(era5_df.columns, era5_df.loc[time], bounds_error=False, fill_value=np.nan)
    era5_regridded.loc[time] = interp_func(radiosonde_df.columns)

#Ensure both DataFrames have exactly the same time index
era5_regridded = era5_regridded.reindex(radiosonde_df.index)

print(era5_regridded)
print(radiosonde_df)

angular_difference = np.abs(era5_regridded- radiosonde_df)
angular_difference = np.minimum(angular_difference, 360 - angular_difference)


print(angular_difference)
#dfgdfg


fig, ax = plt.subplots(1, 1 , figsize=(18,3))
opposing_wind_probability = angular_difference.to_numpy()
# Convert the DataFrame to a numeric array, coercing errors to NaN
opposing_wind_probability = np.array(opposing_wind_probability, dtype=float)

plt.title("Angular Difference",
              fontsize=12)



im = ax.contourf(angular_difference.index, angular_difference.columns, opposing_wind_probability.T, levels=np.linspace(0, 180., 19), cmap="magma")

#plt.title("Salt Lake City Utah (40$^\circ$N)" +
#          "\nWind Directionality for Station #" + str(WMO).zfill(5) +  " in " + str(config.start_year), fontsize=13)

#plt.title(Station_Name + "- " + CO +
#          "\nWind Directionality for Station #" + str(WMO).zfill(5) +  " in " + str(config.start_year), fontsize=13)
plt.ylabel('Altitude (m)')
plt.xlabel('Date')



divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "1%", pad="1%")
im.set_clim(0.,180.)
cbar = fig.colorbar(im, cax=cax, boundaries=np.linspace(0, 180, 13))

cbar.set_label("Angular Difference (degrees)", labelpad=10)



ax.xaxis.set_minor_locator(YearLocator(1))
ax.xaxis.set_minor_formatter(DateFormatter('%Y'))
for tick in ax.xaxis.get_minor_ticks():
    tick.tick1line.set_markersize(0)
    tick.tick2line.set_markersize(0)
    tick.label1.set_horizontalalignment('center')

for tick in ax.yaxis.get_major_ticks()[::2]:
    tick.set_visible(False)




fig.tight_layout()

path = "Pictures/Hovmoller-DIFF/"
isExist = os.path.exists(path)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(path)
#plt.savefig("Pictures/Hovmoller/" +  str(FAA), bbox_inches='tight')
plt.savefig(path +  str(FAA) + "-" + str(config.start_year) + "-NO-TITLE", bbox_inches='tight')
fig.tight_layout()
plt.show()
dfgdfg






#=====================



# Align time indices
common_times = era5_df.index.intersection(radiosonde_interp.index)
era5_aligned = era5_df.loc[common_times]
radiosonde_aligned = radiosonde_interp.loc[common_times]

# Compute angular difference, ensuring values are within [0, 180] degrees
angular_difference = np.abs(era5_aligned - radiosonde_aligned)
angular_difference = np.minimum(angular_difference, 360 - angular_difference)

# Ensure correct shape by transposing angular difference data
angular_difference_corrected = angular_difference.T

# Create correctly shaped time-altitude meshgrid
time_mesh, altitude_mesh = np.meshgrid(common_times, era5_aligned.columns, indexing="ij")

# Ensure that angular_difference_corrected matches the shape of time_mesh and altitude_mesh
print(f"Time mesh shape: {time_mesh.shape}, Altitude mesh shape: {altitude_mesh.shape}, Angular difference shape: {angular_difference_corrected.shape}")

# If needed, reshape the angular_difference_corrected matrix
if angular_difference_corrected.shape != time_mesh.shape:
    angular_difference_corrected = angular_difference_corrected.T

# Confirm shapes before plotting
print(f"Fixed Angular Difference Shape: {angular_difference_corrected.shape}")

# Plot the Hovmöller diagram
fig, ax = plt.subplots(figsize=(18, 3))
im = ax.contourf(time_mesh, altitude_mesh, angular_difference_corrected, levels=np.linspace(0, 180, 19), cmap="coolwarm")

# Titles and labels
#plt.title("Angular Difference in Wind Direction (ERA5 vs Radiosonde)\nFairbanks, Alaska (65°N)", fontsize=13)

plt.title("Salt Lake City, Utah USA (40$^\circ$N)" +
              "\nWind Directionality Difference (ERA5 vs Radiosonde) for Station #" + str(WMO).zfill(5) +  " in " + str(config.start_year), fontsize=13)


plt.ylabel("Altitude (m)")
plt.xlabel("Date")

# Colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", "1%", pad="3%")
cbar = fig.colorbar(im, cax=cax, boundaries=np.linspace(0, 180, 13))
cbar.set_label("Angular Difference (degrees)", labelpad=10)

# Format time axis
ax.xaxis.set_minor_locator(YearLocator(1))
ax.xaxis.set_minor_formatter(DateFormatter('%Y'))
for tick in ax.xaxis.get_minor_ticks():
    tick.tick1line.set_markersize(0)
    tick.tick2line.set_markersize(0)
    tick.label1.set_horizontalalignment('center')

fig.tight_layout()

path = "Pictures/Hovmoller-DIFF/"
isExist = os.path.exists(path)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(path)
#plt.savefig("Pictures/Hovmoller/" +  str(FAA), bbox_inches='tight')
plt.savefig(path +  str(FAA) + "-" + str(config.start_year), bbox_inches='tight')
plt.show()