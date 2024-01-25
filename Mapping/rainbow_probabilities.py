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

#Example stuff ---------------------
#GRID DATA EXAMPLES
# https://scitools.org.uk/cartopy/docs/v0.13/matplotlib/advanced_plotting.html
# https://climate-cms.org/posts/2020-09-22-wrapping-pcolormesh.html
# https://pbett.wordpress.com/datafun/plotting-maps/
# https://xarray.pydata.org/en/v0.7.0/plotting.html


'''

def func(x, y):
    return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2


min_lat = 0



min_lat = 10
max_lat = 75
min_lon = 360-185
max_lon = 360-20


#min_lat = -58
#max_lat = 85
#min_lon = 360-180
#max_lon = 360-30
res = 1

lons = np.arange(min_lon,max_lon,res)
lats = np.arange(min_lat,max_lat,res)

grid_x, grid_y = np.meshgrid(lons,
                             lats)



rng = np.random.default_rng()
points = rng.random((1000, 2))

print(points)

#sdfs
values = func(points[:,0], points[:,1])

print(np.arange(min_lon,max_lon,1))
print(grid_x.shape)
print(grid_y.shape)
print(points.shape)
print(values.shape)

grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')

#print(values)
print(grid_z0.shape)

#asfa
'''


#--------------------------------------
#DOWNLOAD THE DATA

#MAP CONFIGURATION STUFF:
method = 'nearest'
year = config.start_year
prefix = "Western_Hemisphere"  #title of the maps that are exported to the MAPS folder

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



continent = "South_America"
stations_df = pd.read_csv('Radisonde_Stations_Info/CLEANED/' + continent + ".csv", index_col=1)
#stations_df = stations_df.loc[stations_df["CO"] == "US"]


continent2 = "South_America"
stations_df2 = pd.read_csv('Radisonde_Stations_Info/CLEANED/' + continent2 + ".csv", index_col=1)

stations_df = pd.concat([stations_df, stations_df2])



#Generate a new dataframe of montly probaibilties for each station to add to the stations_df. Take the max probability (per alt/pres)
df_probabilities = pd.DataFrame(columns=[i for i in range(1,12)])

for row in stations_df.itertuples(index = 'WMO'):
    WMO = row.Index
    FAA = row.FAA
    Name = row.Station_Name

    analysis_folder = config.analysis_folder

    #file_name = analysis_folder[:-14]  + "analysis_" + str(year) + '-wind_probabilities-TOTAL.csv'
    file_name = analysis_folder + str(FAA) + " - " + str(WMO) + "/analysis_" + str(year) + '-wind_probabilities-TOTAL.csv'

    df = pd.read_csv(file_name, index_col=0 )
    df = df.T
    df = df.apply(['max'])
    #df.index.max = 'WMO'
    df = df.rename(index={'max': WMO})
    df.index.set_names('WMO', level=None, inplace=True)

    df_probabilities = pd.concat([df_probabilities, df], ignore_index=False)

stations_df = stations_df.join(df_probabilities)
print(stations_df)




#Convert Ranges of Coordinates from stations list for Cartopy
stations_df[' LONG'] = stations_df.apply(lambda x: (360-x[' LONG'] if x['E'] == 'W' else 1*x[' LONG']), axis = 1)
stations_df['  LAT'] = stations_df.apply(lambda x: (-1*x['  LAT'] if x['N'] == 'S' else 1*x['  LAT']), axis = 1)


#Drop any stations that collected no data for the entire year.
stations_df.dropna(subset=df.columns[-12:], how = 'all', inplace = True)





# Make a new plot for each month
for month in range (1,12+1):
    #stations_df.dropna(subset=[month], inplace = True)
    values = stations_df.loc[:,month].to_numpy()
    lonlat = stations_df[[' LONG','  LAT']]
    points = lonlat.to_numpy()

    values = stations_df.loc[:,month].to_numpy()
    lonlat = stations_df[[' LONG','  LAT']]
    points = lonlat.to_numpy()

    '''
    print("OK HERE IS THE NEW STUFF)")
    print()
    print(grid_x.shape)
    print(grid_y.shape)
    print(points.shape)
    print(values.shape)
    print(lons.shape)
    print(lats.shape)
    print()
    '''

    #zi = griddata(points,values,(grid_x, grid_y),method='linear', fill_value=0)
    zi = griddata(points,values,(grid_x, grid_y),method=method)

    #print(zi.shape)
    #print(zi)


    #North America
    stn_lat = 40
    stn_lon = -100
    #extent = [-125 , -70, 20, 50]



    extent = [-180, 0, -5, 35] # Western Hemisphere



    #extent = [min_lon-10, max_lon + 10, min_lat-10, max_lat +10]
    #extent = [(min_lon -360)-20, (max_lon -360)-15, min_lat - 1, max_lat]



    central_lon = np.mean(extent[:2])
    central_lat = np.mean(extent[2:])

    fig = plt.figure(figsize=(12, 12))
    #ax = plt.axes(projection=ccrs.AlbersEqualArea(central_lon, central_lat))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=stn_lon))
    ax.set_extent(extent)


    #ax = plt.axes(projection=ccrs.PlateCarree())
    D = ax.pcolormesh(lons, lats, zi, transform=ccrs.PlateCarree(), cmap='RdYlGn', alpha=.8, vmin=0, vmax=1)
    fig.colorbar(D, ax=ax, shrink=.5, pad=.01)
    #fig.colorbar.set_ylim(0, 1)
    #plt.clim(0,1)
    #ax.scatter(stn_lon, stn_lat, transform=ccrs.PlateCarree(), marker='+', s=100, c='k', linewidth=3)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=2)
    ax.add_feature(cfeature.STATES.with_scale('50m'))

    # Plot Radiosonde Locations
    ax.scatter(stations_df[' LONG'], stations_df['  LAT'], marker = ".", c = "blue", s= 55, transform = ccrs.Geodetic(), zorder=200)
    ax.scatter(stations_df[' LONG'], stations_df['  LAT'], marker = ".", c = "cyan", s= 4, transform = ccrs.Geodetic(), zorder=200)

    ax.add_feature(cfeature.OCEAN, facecolor = 'gray', alpha = 1, zorder = 150)

    ax.set_title(prefix + " " + config.mode + "_" + config.type+ "_" + "\n Opposing Winds Probabilities\n Alt:{15-25 km} in " + calendar.month_name[month] + " " + str(year), fontsize=24)
    plt.tight_layout()

    print("generating map for " + prefix + "_" + config.type+ "_" + config.mode +  "-" + str(year) + '-' + str(month))

    path = config.maps_folder + "/" + str(year)
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)

    plt.savefig(path +"/" +  prefix + "_" + config.type+ "_" + config.mode + "-" + str(year) + '-' + str(month))
    #plt.show()
