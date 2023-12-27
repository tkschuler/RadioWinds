import numpy as np

parent_folder =  '/home/schuler/RadioWinds/SOUNDINGS_DATA3/' # add in a slash here
#parent_folder =  '/home/schuler/RadioWinds/ERA5_Analysis/' # add in a slash here

#Default is blowing to for path planning
blowing_to = True # -1 FOR (typical wind rose),  1 for TO (where balloon will drift to)

g = 9.80665

#"""
#For batch analysis and opposing winds analysis:
#altitude_type = "alt" # or 'pressure"


min_alt = 14500 #15000     # m
max_alt = 27500     # m
alt_step = 500      # m
n_sectors = 16      # m
speed_threshold = 1 # knots for Radiosonde,  m/s for ERA5
#"""

"""
#For batch analysis and opposing winds analysis:
altitude_type = "pres" # or 'pressure"
min_alt = 0 #15000     # m
max_alt = 12     # m #leave off 10 hpa
alt_step = 1      # m
n_sectors = 16      # m
speed_threshold = 4 # knots
"""

#Mandatory pressure levels downloaded from ERA5  (~9.5km - 31km?)
era5_pressure_levels = np.asarray([300, 250, 225, 200, 175, 150, 125, 100, 70, 50,  30,  20, 10])
mode = "radiosonde" # or radiosonde or era5
by_pressure = True # True or False


# This is similar in range to 15.5 - 26 km
#for radiosonde, add an extra 1/3 to the next level?
#other wise it will only inclde data that is right on 125,  which varries from radiosonde to radiosonde +/- about 5
min_pressure = 20  - 3
max_pressure = 125 + 13
