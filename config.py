import numpy as np
import os

# **************** DOWNLOAD AND ANALYSIS ************************

type = "ALT"                    # ALT or PRES
mode = "radiosonde"             # or radiosonde or era5
continent = "North_America"     #
mapping_mode = mode             # or "diff"

parallelize = False             # It's recommended to change logging to False if parallelize is True.
logging = True                  # Displays extra debugging and status text in the Terminal

start_year = 2012
end_year = 2012

monthly_export_color = False
annual_export_color = True

# ******************** DIRECTORY SETUP **************************
base_directory = os.getcwd() + '/'  # The default is the RadioWinds directory
parent_folder = base_directory + 'SOUNDINGS_DATA/'
analysis_folder = base_directory + mode + '_ANALYSIS_' + type + '/'
maps_folder = base_directory + 'MAPS/'


# ****************** OTHER STUFF ***************************

# Default is blowing to for path planning
blowing_to = True  # False (typical wind rose); True (direction balloon will drift in, opposite)
g = 9.80665


# ******************* ERA5 ***************************

era_file = "forecasts/" + "western_hemisphere-2012-SOUTH.nc"

# Mandatory pressure levels downloaded from ERA5  (~9.5km - 31km?)
era5_pressure_levels = np.asarray([300, 250, 225, 200, 175, 150, 125, 100, 70, 50,  30,  20, 10])


alt_step = 500                  # m
min_alt = 15000                 # m
max_alt = 28000 + alt_step-1    # m  The +alt_step -1 is to include all datapoints above the max - the next step size.
n_sectors = 16                  # m
speed_threshold = 4             # knots for Radiosonde,  m/s for ERA5
if mode == "era5":
    speed_threshold = speed_threshold / 2.  # to roughly convert from knots to m/s since forecasts aren't in decimals


# This is similar in range to 15.5 - 26 km
#for radiosonde, add an extra 1/3 to the next level?
#other wise it will only inclde data that is right on 125,  which varries from radiosonde to radiosonde +/- about 5
min_pressure = 20  - 3
max_pressure = 125 + 13


