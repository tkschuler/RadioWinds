import datetime as dt
import numpy as np
from netCDF4 import Dataset  # http://unidata.github.io/netcdf4-python/

import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, WeekdayLocator, DateFormatter
import matplotlib.ticker as ticker

import config

#Examples:
#https://github.com/royalosyin/Python-Practical-Application-on-Climate-Variability-Studies/blob/master/ex13-Apply%20Hovmoller%20Diagram-The%20seasonal%20migration%20of%20rainfall%20in%20Africa.ipynb
# https://www.climate.gov/news-features/understanding-climate/hovm%C3%B6ller-diagram-climate-scientist%E2%80%99s-best-friend
# https://unidata.github.io/python-gallery/examples/Hovmoller_Diagram.html

#Download netcdf file
infile = config.era_file
nc  = Dataset(infile)
lon = nc['longitude'][:]
lat = nc['latitude'][:]

#Pull out relevant data
u  = nc.variables['u'][:,0,:,:]#[:,0,lon,:]
u_mean = np.mean(u, axis=2)

print(u_mean)
print(u_mean.shape)
print(type(u_mean))


#Create timestamps for plotting, that match dataset
base = dt.datetime(2012, 1, 1)
dates = [base + dt.timedelta(hours=x*12) for x in range(0, 732)]
daysx, latsx = np.meshgrid(dates,lat)

print(daysx.shape, type(daysx))
print(latsx.shape, type(latsx))
print(u_mean.T.shape, type(u_mean))
#sdfsd

#Plotting
fig, ax = plt.subplots(1, 1 , figsize=(15,5))
im = ax.pcolormesh(daysx, latsx, u_mean.T, cmap='YlGnBu', vmin=-60.0, vmax=60)
fig.colorbar(im)

plt.title('Daily Zonal Winds at surface level for South America in 2012', fontsize=12)
plt.ylabel('Latitude')
plt.xlabel('Date')

# make labels centered
ax.xaxis.set_major_locator(MonthLocator())
ax.xaxis.set_minor_locator(MonthLocator(bymonthday=15))

ax.xaxis.set_major_formatter(ticker.NullFormatter())
ax.xaxis.set_minor_formatter(DateFormatter('%b'))

for tick in ax.xaxis.get_minor_ticks():
    tick.tick1line.set_markersize(0)
    tick.tick2line.set_markersize(0)
    tick.label1.set_horizontalalignment('center')

plt.show()


