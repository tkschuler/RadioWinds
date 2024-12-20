import numpy as np
import os

# **************** DOWNLOAD AND ANALYSIS ************************

type = "ALT"                    # ALT or PRES
mode = "radiosonde"             # or radiosonde or era5
continent = "North_America"     #
mapping_mode = mode             # or "diff"

# Multithreading can be finicky and run out of memory on Windows (it also seems slower)
# I have no memory issues on WSL or Ubuntu.
parallelize = False             # It's recommended to change logging to False if parallelize is True.
logging = True                  # Displays extra debugging and status text in the Terminal

start_year = 2022
end_year = 2022

monthly_export_color = False
annual_export_color = True
dfi_mode = "chrome"  # Default is "chrome," "selenium has worked in the past. Matplotlib doesn not support custom css

alt_step = 500                  # m
min_alt = 15000                 # m
max_alt = 28000 + alt_step-1    # m  The +alt_step -1 is to include all data points above the max - the next step size.
n_sectors = 16                  # m
speed_threshold = 4             # knots for Radiosonde,  m/s for ERA5

# This pressure range is similar in altitude to 15.5 - 26 km
# for radiosonde, add an extra 1/3 to the next level?
# otherwise it will only include data that is right on 125 hpa,  which varies from radiosonde to radiosonde +/- about 5
min_pressure = 20  - 3
max_pressure = 125 + 13

# ******************** DIRECTORY SETUP **************************

base_directory = os.getcwd() + '/'  # The default is the RadioWinds directory
parent_folder = base_directory + 'SOUNDINGS_DATA/'

# Best to Change the analysis folders depending on which type of analsi you're doing
#analysis_folder = base_directory + mode + '_ANALYSIS_' + type + '-CALM' + '/'
#analysis_folder = base_directory + mode + '_ANALYSIS_' + type + '-FULL' + '/'
#analysis_folder = base_directory + mode + '_ANALYSIS_' + type + '-BURST' + '/'
analysis_folder = base_directory + mode + '_ANALYSIS_' + type + '/'
maps_folder = base_directory + 'MAPS/'

# ****************** OTHER STUFF *********************************

# Default is blowing to for path planning
blowing_to = True  # False (typical wind rose); True (direction balloon will drift in, opposite)
g = 9.80665


# ************************ ERA5 **********************************
combined = False
era_file = "forecasts/" + "western_hemisphere-2022-NORTH.nc"

# Mandatory pressure levels downloaded from ERA5  (~9.5km - 31km?)
era5_pressure_levels = np.asarray([300, 250, 225, 200, 175, 150, 125, 100, 70, 50,  30,  20, 10])

if mode == "era5":
    speed_threshold = speed_threshold / 2.  # to roughly convert from knots to m/s since forecasts aren't in decimals
