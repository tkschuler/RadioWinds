import cartopy.crs as ccrs
import cartopy.io
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import iris
import xarray


import xarray as xr

'''
fname = cartopy.io.shapereader.natural_earth(resolution='10m',
                                               category='cultural',
                                               name='populated_places_simple')
'''

#https://www.weather.gov/gjt/education_corner_balloon

fname = iris.sample_data_path('rotated_pole.nc')
temperature = iris.load_cube(fname)

print(temperature)

temperature.coord('grid_latitude').guess_bounds()
temperature.coord('grid_longitude').guess_bounds()

gridlons = temperature.coord('grid_longitude').contiguous_bounds()
gridlats = temperature.coord('grid_latitude').contiguous_bounds()
temperature = temperature.data


continent = "North_America"
stations_df = pd.read_csv('Radisonde_String_Parsing/CLEANED/' + continent + ".csv")

print(stations_df)
print(stations_df.columns)

#ds=xr.open_dataset(fname)
#print(ds)

'''
plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.Robinson())

ax.set_title('Populated places of the world.')
ax.coastlines()

points = list(cartopy.io.shapereader.Reader(fname).geometries())
'''

central_lat = 37.5
central_lon = -96
extent = [-180, -20, 0, 45]
central_lon = np.mean(extent[:2])
central_lat = np.mean(extent[2:])

plt.figure(figsize=(12, 12))
#ax = plt.axes(projection=ccrs.AlbersEqualArea(central_lon, central_lat))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=central_lon))
ax.set_extent(extent)

ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=0.5)
ax.add_feature(cartopy.feature.LAND, edgecolor='black')
ax.add_feature(cartopy.feature.LAKES, edgecolor='black')
ax.add_feature(cartopy.feature.RIVERS)
ax.gridlines()


da = xarray.DataArray(np.random.random((100,100)),
                     coords=[
                         ('lat', np.linspace(-90,90,num=100, endpoint=False)+18/2),
                         ('lon', np.linspace(0,360,num=100, endpoint=False)),
                     ])

#ax.pcolormesh(da.lon, da.lat, da, transform=ccrs.PlateCarree())

ax.scatter(360 - stations_df[' LONG'], stations_df['  LAT'], c = "red", transform = ccrs.Geodetic())

rotated_pole = ccrs.RotatedPole(pole_longitude=177.5, pole_latitude=37.5)
plt.pcolormesh(gridlons, gridlats, temperature, alpha = .5,  transform=ccrs.PlateCarree(central_longitude=central_lon))

plt.show()