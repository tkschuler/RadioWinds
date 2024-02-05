import pandas as pd
import config

def convert_stations_coords(stations_df):
    '''Converts the station lat/long coordinates for mapping'''

    stations_df['lat_era5'] = stations_df.apply(lambda x: (-1 * x['  LAT'] if x['N'] == 'S' else 1 * x['  LAT']),
                                                axis=1)
    #This works with 360-x or -1*x???
    stations_df['lon_era5'] = stations_df.apply(lambda x: (360 - x[' LONG'] if x['E'] == 'W' else 1 * x[' LONG']),
                                                axis=1)

    return stations_df

def get_analysis_folder(FAA, WMO, year):
    return config.analysis_folder + str(FAA) + " - " + str(WMO) + "/" + str(year) + "_analysis/"
