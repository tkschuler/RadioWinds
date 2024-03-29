import cartopy.crs as ccrs
import cartopy.io
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cartopy.feature as cfeature
from scipy.interpolate import griddata
import calendar
import sys
sys.path.insert(0, sys.path[0] + '/../') #add config from 1 directory up.
import os
import config
import utils

#Example stuff ---------------------
#GRID DATA EXAMPLES
# https://scitools.org.uk/cartopy/docs/v0.13/matplotlib/advanced_plotting.html
# https://climate-cms.org/posts/2020-09-22-wrapping-pcolormesh.html
# https://pbett.wordpress.com/datafun/plotting-maps/
# https://xarray.pydata.org/en/v0.7.0/plotting.html


#Configuration stuff
#--------------------------------------
font = {'size'   : 22}
plt.rc('font', **font)


#MAP CONFIGURATION STUFF:
method = 'nearest'
year = config.start_year
prefix = "Western_Hemisphere"  #title of the maps that are exported to the MAPS folder

type = 'std'  # 'mean' or 'std'

#These are the values download from Copernicus for 2022 in degrees

#Western Hemisphere
min_lat = -65
max_lat = 75
min_lon = 360-175
max_lon = 360-20

'''
#CONUS
min_lat = -65
max_lat = 75
min_lon = 360-175
max_lon = 360-20
'''

res = 1 # degrees

lons = np.arange(min_lon,max_lon,res)
lats = np.arange(min_lat,max_lat,res)

grid_x, grid_y = np.meshgrid(lons, lats)

#--------------------------
continent = "North_America"
stations_df = pd.read_csv('Radiosonde_Stations_Info/CLEANED/' + continent + ".csv", index_col=1)
#stations_df = stations_df.loc[stations_df["CO"] == "US"]


continent2 = "South_America"
stations_df2 = pd.read_csv('Radiosonde_Stations_Info/CLEANED/' + continent2 + ".csv", index_col=1)

stations_df = pd.concat([stations_df, stations_df2])

#--------------------------------------

#Generate a new dataframe of montly probaibilties for each station to add to the stations_df. Take the max probability (per alt/pres)
df_probabilities = pd.DataFrame(columns=[i for i in range(1,12)])

for row in stations_df.itertuples(index = 'WMO'):
    WMO = row.Index
    FAA = row.FAA
    Name = row.Station_Name

    analysis_folder = config.analysis_folder
    file_name = analysis_folder + str(FAA) + " - " + str(WMO) + '/analysis-wind_probabilities-DECADAL-STATISTICS.csv'

    df = pd.read_csv(file_name, index_col=0 )
    df = df.T

    if type == 'mean':
        df = df.drop("std")
        #df = df.apply(['max'])
        df = df.rename(index={'mean': WMO})
    elif type == 'std':
        df = df.drop("mean")
        #df = df.apply(['max'])
        df = df.rename(index={'std': WMO})

    df.index.set_names('WMO', level=None, inplace=True)
    df_probabilities = pd.concat([df_probabilities, df], ignore_index=False)

stations_df = stations_df.join(df_probabilities)

#Convert Station Coordinates for mapping
stations_df = utils.convert_stations_coords(stations_df)


#Drop any stations that collected no data for the entire year.
stations_df.dropna(subset=df.columns[-12:], how = 'all', inplace = True)


# Make a new plot for each month
for month in range (1,12+1):
    #stations_df.dropna(subset=[month], inplace = True)
    values = stations_df.loc[:,month].to_numpy()
    lonlat = stations_df[['lon_era5','lat_era5']]
    points = lonlat.to_numpy()

    zi = griddata(points,values,(grid_x, grid_y),method=method)


    # North America
    #extent = [-125 , -70, 20, 50]

    # Western Hemisphere
    extent = [-170, -20, -25, 40]


    central_lon = np.mean(extent[:2])
    central_lat = np.mean(extent[2:])

    fig = plt.figure(figsize=(12, 12))

    # ax = plt.axes(projection=ccrs.AlbersEqualArea(central_lon, central_lat))
    # ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=stn_lon))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent)

    if type == 'mean':
        D = ax.pcolormesh(lons, lats, zi, transform=ccrs.PlateCarree(), cmap='RdYlGn', alpha=.9, vmin=0.0, vmax=1)
        fig.colorbar(D, ax=ax, shrink=.5, pad=.01)
    elif type == 'std':
        D = ax.pcolormesh(lons, lats, zi, transform=ccrs.PlateCarree(), cmap='cool', alpha=.9, vmin=0.0, vmax=0.3)
        fig.colorbar(D, ax=ax, shrink=.5, pad=.01, extend='max')


    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=2)
    ax.add_feature(cfeature.STATES.with_scale('50m'))

    # Plot Radiosonde Locations
    ax.scatter(stations_df['lon_era5'], stations_df['lat_era5'], marker = ".", c = "blue", s= 55, transform = ccrs.Geodetic(), zorder=200)
    ax.scatter(stations_df['lon_era5'], stations_df['lat_era5'], marker = ".", c = "cyan", s= 4, transform = ccrs.Geodetic(), zorder=200)

    ax.add_feature(cfeature.OCEAN, facecolor = 'gray', alpha = 1, zorder = 150)

    Months = ['', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    ax.set_title(Months[month], fontsize = 30)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    print("generating map for " + str(month))

    path = config.maps_folder
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)

    if type == 'mean':
        plt.savefig(path +"/" +  "DECADAL-MEAN-NEW-"+ str(month), bbox_inches='tight')
    elif type == 'std':
        plt.savefig(path + "/" + "DECADAL-MEAN-NEW-" + str(month), bbox_inches='tight')

