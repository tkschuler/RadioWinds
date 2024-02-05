import numpy as np

type = "ALT" #ALT or PRES"
mode = "radiosonde" # or radiosonde or era5
continent = "North_America"
mapping_mode = mode #or "diff"

base_directory = '/mnt/d/RadioWinds/' #Windows 11
#base_directory = '/home/schuler/RadioWinds/' #Linux

parent_folder =  base_directory + 'SOUNDINGS_DATA/' # add in a slash here
#analysis_folder = '/home/schuler/RadioWinds/ERA5_ANALYSIS3/'
analysis_folder = base_directory + mode + '_ANALYSIS_' + type + '/'
maps_folder =  base_directory + 'MAPS/' # add in a slash here

#Default is blowing to for path planning
blowing_to = True # -1 FOR (typical wind rose),  1 for TO (where balloon will drift to)

g = 9.80665


#########FOR BATCH ANALYSIS ####################
era_file = "forecasts/" + "western_hemisphere-2012-SOUTH.nc"
#For downloading Radiosonde datasets in parralel or sequence
parallelize = False
#by_pressure = False # True or False
logging = False

start_year = 2012
end_year = 2012

############################################


min_alt = 14500 #15000     # m
max_alt = 27500     # m
alt_step = 500      # m
n_sectors = 16      # m
speed_threshold = 4 # knots for Radiosonde,  m/s for ERA5
if mode == "era5":
    speed_threshold = speed_threshold / 2.,  #to roughly convert from knots to m/s since forecasts aren't in decimals

#Mandatory pressure levels downloaded from ERA5  (~9.5km - 31km?)
era5_pressure_levels = np.asarray([300, 250, 225, 200, 175, 150, 125, 100, 70, 50,  30,  20, 10])


# This is similar in range to 15.5 - 26 km
#for radiosonde, add an extra 1/3 to the next level?
#other wise it will only inclde data that is right on 125,  which varries from radiosonde to radiosonde +/- about 5
min_pressure = 20  - 3
max_pressure = 125 + 13


