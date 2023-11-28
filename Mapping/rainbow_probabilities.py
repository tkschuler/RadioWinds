import cartopy.crs as ccrs
import cartopy.io
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cartopy.feature as cfeature
from scipy.interpolate import griddata
#Example stuff ---------------------

def func(x, y):
    return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2


min_lat = 0



#GRID DATA EXAMPLES
# https://scitools.org.uk/cartopy/docs/v0.13/matplotlib/advanced_plotting.html
# https://climate-cms.org/posts/2020-09-22-wrapping-pcolormesh.html
# https://pbett.wordpress.com/datafun/plotting-maps/





max_lat = 55
min_lon = 360-125
max_lon = 360-60


min_lat = 0
max_lat = 77
min_lon = 360-180
max_lon = 360-5
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


#--------------------------------------
#Get teh data

continent = "North_America"
stations_df = pd.read_csv('Radisonde_String_Parsing/CLEANED/' + continent + ".csv", index_col=1)
#stations_df = stations_df.loc[stations_df["CO"] == "US"]

print(stations_df)

df_probabilities = pd.DataFrame(columns=[i for i in range(1,12)])
#print(df_probabilities)

#sdfs

for row in stations_df.itertuples(index = 'WMO'):
    #print(row.Index)
    #print("hey")
    WMO = row.Index
    FAA = row.FAA
    Name = row.Station_Name

    year = 2012

    #data_folder = 'SOUNDINGS_DATA2/' + str(FAA) + " - " + str(WMO) + "/"
    analysis_folder = 'SOUNDINGS_DATA2/' + str(FAA) + " - " + str(WMO) + "/" + str(year) + "_analysis/"

    file_name = analysis_folder[:-14]  + "analysis_" + str(year) + '-wind_probabilities-TOTAL.csv'


    df = pd.read_csv(file_name, index_col=0 )
    df = df.T
    df = df.apply(['max'])
    #df.index.max = 'WMO'
    df = df.rename(index={'max': WMO})
    df.index.set_names('WMO', level=None, inplace=True)

    df_probabilities = pd.concat([df_probabilities, df], ignore_index=False)

#print(df_probabilities)

stations_df = stations_df.join(df_probabilities)

month = 8
method = 'linear'

#stations_df.dropna(subset=[stations_df.loc[:,1],stations_df.loc[:,4], stations_df.loc[:,7], stations_df.loc[:,9], stations_df.loc[:,12] ], how = 'all', inplace = True)
print(stations_df)
# Drop any stations where there is missing data
stations_df.dropna(subset=[month], inplace = True)
print(stations_df)
#print(stations_df.loc[:,1])


#----------------------------------------

stations_df[' LONG'] = 360 - stations_df[' LONG']

values = stations_df.loc[:,month].to_numpy()
lonlat = stations_df[[' LONG','  LAT']]
points = lonlat.to_numpy()

print("OK HERE IS THE NEW STUFF)")
print()
print(grid_x.shape)
print(grid_y.shape)
print(points.shape)
print(values.shape)
print(lons.shape)
print(lats.shape)
print()

#aafss

#zi = griddata(points,values,(grid_x, grid_y),method='linear', fill_value=0)
zi = griddata(points,values,(grid_x, grid_y),method=method)

#print(zi)
print(zi.shape)

print(zi)


stn_lat = 40
stn_lon = -100


extent = [-180, 0, 0, 35]
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

ax.scatter(stations_df[' LONG'], stations_df['  LAT'], marker = ".", c = "blue", s= 30, transform = ccrs.Geodetic(), zorder=100)


plt.show()