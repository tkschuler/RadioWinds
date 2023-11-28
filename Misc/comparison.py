# Copyright (c) 2017 Siphon Contributors.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""
Wyoming Upper Air Data Request
==============================

This example shows how to use siphon's `simplewebswervice` support to create a query to
the Wyoming upper air archive.
"""

from datetime import datetime

from metpy.units import units

from siphon.simplewebservice.wyoming import WyomingUpperAir

from scipy.interpolate import CubicSpline, interp1d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

####################################################
# Create a datetime object for the sounding and string of the station identifier.
date = datetime(2023, 5, 16, 12)
station = 'FGZ'

####################################################
# Make the request (a pandas dataframe is returned).
for i in range(0,5):
    try:
        df = WyomingUpperAir.request_data(date, station)
        print(i)
        break
    except:
        print("Server busy, trying again. Attempt", i+1)

print(df)

####################################################
# Inspect data columns in the dataframe.
print(df.columns)

####################################################
# Pull out a specific column of data.
print(df['pressure'])

####################################################
# Units are stored in a dictionary with the variable name as the key in the `units` attribute
# of the dataframe.
print(df.units)

####################################################
print(df.units['pressure'])

####################################################
# Units can then be attached to the values from the dataframe.
height = df['height'].values * units(df.units['height'])
pressure = df['pressure'].values * units(df.units['pressure'])
temperature = df['temperature'].values * units(df.units['temperature'])
dewpoint = df['dewpoint'].values * units(df.units['dewpoint'])
u_wind = df['u_wind'].values * units(df.units['u_wind'])
v_wind = df['v_wind'].values * units(df.units['v_wind'])

def polar_to_cartesian(angles, magnitudes):
    # Convert angles from degrees to radians
    angles = np.radians(angles) #+ np.pi #(multiply by )

    # Calculate x and y components
    x_components = magnitudes * np.cos(angles)
    y_components = magnitudes * np.sin(angles)

    return x_components, y_components

u2, v2 = polar_to_cartesian(df['direction'].values * units(df.units['direction']), df['speed'].values * units(df.units['speed']))



'''---------------------------------------------'''
'''
from datetime import datetime
from siphon.simplewebservice.igra2 import IGRAUpperAir

station = 'USM00072376'

print("Downloading data...")
df2, header = IGRAUpperAir.request_data(date, station)

print(df2.units)

df3 = df2.dropna(subset=['pressure'])


print(df3)
#print(df2.units)

u_wind_IGRA = df3['u_wind'].values * units(df2.units['u_wind'])
v_wind_IGRA = df3['v_wind'].values * units(df2.units['v_wind'])
height_IGRA = df3['height'].values * units(df2.units['height'])
'''


'''---------------------------------------------'''


def getWindPlotData():
#def getWindPlotData(hour_index,lat_i,lon_i):
    # Extract relevant u/v wind velocity, and altitude
    u = np.asarray(u_wind) #ugrdprs[hour_index,:,lat_i,lon_i]
    v = np.asarray(v_wind) #vgrdprs[hour_index,:,lat_i,lon_i]
    h = np.asarray(height) #hgtprs[hour_index,:,lat_i,lon_i]

    '''
    # Remove missing data
    u = np.ma.asarray(u)
    v = np.ma.asarray(v)
    h = np.ma.asarray(h)

    u = u.filled(np.nan)
    v = v.filled(np.nan)
    nans = ~np.isnan(u)
    u= u[nans]
    v= v[nans]
    h = h[nans]
    '''
    print(u)
    print(h)

    # Forecast data is sparse, so use a cubic spline to add more points
    cs_u = CubicSpline(h, u)
    cs_v = CubicSpline(h, v)
    #FIX THE RANGING ISSUES
    h_new = np.arange(h[0], h[-1], 1) # New altitude range #
    u = cs_u(h_new)
    v = cs_v(h_new)
    #h_new = h

    #u_interp = interp1d(h, u, assume_sorted=False, fill_value="extrapolate")
    #u = u_interp(h)

    #v_interp = interp1d(h, v, assume_sorted=False, fill_value="extrapolate")
    #v = v_interp(h)


    cs_dir = CubicSpline(h, df['direction'].values * units(df.units['direction']))
    cs_spd = CubicSpline(h, df['speed'].values * units(df.units['speed']))

    DIR = cs_dir(h_new)
    SPD = cs_spd(h_new)

    print("DIR", DIR)
    print("SPD", SPD)

    # Calculate altitude
    bearing = np.arctan2(v,u)
    bearing = np.unwrap(bearing)
    r = np.power((np.power(u,2)+np.power(v,2)),.5)

    # Set up Color Bar
    colors = h_new #h #h_new
    cmap=mpl.colors.ListedColormap(colors)

    return [bearing, r , colors, cmap, h, DIR, SPD]


#def plotWindVelocity(hour_index,lat,lon):
def plotWindVelocity():

    bearing0, r0 , colors0, cmap0, h, DIR, SPD = getWindPlotData() #getWindPlotData(hour_index,lat_i,lon_i)

    # Plot figure and legend
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(111, projection='polar')
    sc1 = ax1.scatter(bearing0[0:h.take(-1)], colors0[0:h.take(-1)], c=r0[0:h.take(-1)], cmap='autumn', alpha=0.75, s = 2)

    print(len(DIR), len(colors0), len(SPD))
    #sc1 = ax1.scatter(DIR, colors0, c=SPD, cmap='winter', alpha=0.75, s = 2)
    ax1.set_xticklabels(['E', '', 'N', '', 'W', '', 'S', ''])
    plt.colorbar(sc1, ax=ax1, label=" Wind Velocity (m/s)")
    ax1.title.set_text("3D Windrose for (" + "TEST" + ") on ")

    plt.figure()
    plt.plot(u_wind, height)
    plt.plot(v_wind, height)

    plt.plot(u2, height)
    plt.plot(v2, height)
    #plt.plot(v_wind, height)
    #plt.plot(u_wind_IGRA, height_IGRA)
    #plt.plot(v_wind_IGRA, height_IGRA)

    #print(u_wind_IGRA)
    #print(height_IGRA)
    plt.title("U-V Wind Plot")

plotWindVelocity()
plt.show()

'''
plotTempAlt(hour_index,LAT,LON)
file.close()
plt.show()
'''
