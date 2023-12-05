from datetime import datetime
from siphon.simplewebservice.wyoming import WyomingUpperAir
import pandas as pd
from termcolor import colored
import windrose
import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.cm as cm
from plot3DWindrose import polar_interpolated_scatter_plot
import config

'''
This file downloads an individual sounding from University of Wyoming and determines if the sounding meets the 4 criteria.  
Windroses can be plotted as well for visualization.

Fail :                            No Opposing Winds or Calm Winds
Calm Wind Station Keeping:        There is a region with winds under speed threshold (default is 1 m/s)
Opposing Wind Station Keeping:    There is a region with opposing winds
Full Winds Station Keeping:       There are 4 quadrants with opposing winds for full navigation


WMO is a 5 digit unique identifier and consistent between different data sources.
WBAN is a U.S. only 5 digit unique identifier. I'm not sure if it is consistent with WMO.  I suggest we use WMO because it includes U.S. + International. 
  
  
University of Wyoming uses FAA Airport Codes (no K) and WMO

RAOBS uses FAA Airport Codes (no K) and WMO

IGRA2 uses WMO

# Use WMO code or Airport Abbreviation/Nickname. Abbreviation is not consistent between UWY, IGRA2, and RAOBS but WMO code is.

# https://rucsoundings.noaa.gov/raob.short
# https://www1.ncdc.noaa.gov/pub/data/igra/igra2-station-list.txt
# https://artefacts.ceda.ac.uk/badc_datadocs/radiosglobe/stations_sorting_lists/stnlist-historical.html
# Also see emails from Larry Oolman and Joshua Webber

'''

def determine_calm_winds(df, speed_threshold = 1, alt_step = 500):
    '''
        Determine regions from the entire sounding dataset where there are calm winds (within speed_threshold; default is 0-1 m/s).

        Returns a list of altitudes rounded to nearest alt_step size where calm winds were recorderd

        :param speed_threshold: wind speed threshold for calm winds [Defaul is <=1 m/s)
        :type speed_threshold: float
        :param alt_step: altiude step size to round heights to.
        :type alt_step: int (typically 500 or 1000)
        :returns: array of altitudes (rounded to nears alt_step) where there are calm winds (winds < speed threshold)
        :rtype: [int]
    '''

    calm_winds = df[df['speed'] <= speed_threshold]
    calm_winds = calm_winds.copy() #this is needed to get rid of the SettingWithCopyWarning
    calm_winds['height'] = calm_winds.height.apply(lambda x: round(x/ alt_step) * alt_step) #Round calm wind regions to nearest alt_Step size (probably either 500 or 1000 m)

    return np.unique(calm_winds['height'].to_numpy()) #remove duplicate calm wind regions and return the array

def determine_opposing_winds(df , wind_bins, n_sectors, speed_threshold = 2):
    '''
        This function determines the opposing winds of an individual station for an indvidual sounding.

        The wind directions are broken into wind-bins and n_sectors

        n-sectors is the amount of

        :param wind_bins: How data is sperated for histogram,  should be either speed or altitude
        :type wind_bins: array[float]
        :param n_sectors: How data is sperated for histogram,  should be either speed or altitude
        :type n_sectors: int (even number)
        :returns:
        :rtype:
        '''


    #speed threshold
    df = df.drop(df[df['speed'] < speed_threshold].index)

    ws = np.asarray(df['speed'])
    wd = np.asarray(df['direction'])
    alt = np.asarray(df['height'])


    # Generate a histogram with wind directions and altitudes using windrose.historgram
    dir_edges, var_bins, table = windrose.windrose.histogram(wd, alt, bins = wind_bins, nsector=n_sectors)

    #Determine the sectors (directions) that contain non zero values (altitude levels that have wind)
    df = pd.DataFrame(table)
    altitude_lookup_idxs =  df.apply(np.flatnonzero, axis=0)

    opposing_wind_levels  = np.array([])
    opposing_wind_directions = np.array([])

    # Determine the sectors that have opposing winds by checking the current index and the complimentary index at n_sectors/2.
    # Also determine the altitudes contains in the opposing wind pairs for calculating probabilities later.
    for i in range (0,int(n_sectors/2)):
        if np.sum(table, axis=0)[i] != 0 and np.sum(table, axis=0)[i+int(n_sectors/2)] != 0:
            opposing_wind_directions = np.append(opposing_wind_directions, (i, i+8))
            #print(i, altitude_lookup_idxs[i], altitude_lookup_idxs[i+8])
            for idx in altitude_lookup_idxs[i]:
                #print(var_bins[idx])
                opposing_wind_levels = np.append(opposing_wind_levels, var_bins[idx])
            for idx in altitude_lookup_idxs[i+int(n_sectors/2)]:
                #print(var_bins[idx])
                opposing_wind_levels = np.append(opposing_wind_levels, var_bins[idx])

    # sort the opposing wind altitudes in ascending order and remove duplicates
    opposing_wind_levels = np.sort(np.unique(opposing_wind_levels))

    return opposing_wind_directions, opposing_wind_levels

def determine_full_winds(df , wind_bins, speed_threshold = 2):
    '''
        This function determines if an individual sounding has full navigation possibility in the filtered region.

        The winds are distributed into 8 sectors. Then we check if N-E-S-W or NE-SE-SW-NW have winds in every quadrant. If so, then "full navigation" is possible.


        :param wind_bins: How data is seperated for histogram,  should be either speed or altitude
        :type wind_bins: array[float]
        :param speed_threshold:  Minimum wind speed to consider for analysis
        :type speed_threshold: float
        :returns: full_winds
        :rtype: boolean
    '''

    df = df.drop(df[df['speed'] < speed_threshold].index)

    #ws = np.asarray(df['speed'])
    wd = np.asarray(df['direction'])
    alt = np.asarray(df['height'])

    dir_edges, var_bins, table = windrose.windrose.histogram(wd, alt, bins=wind_bins, nsector=8)

    full_winds = False

    #check if N-E-S-W or NE-SE-SW-NW have winds in every quadrant. If so, then "full navigation" is possible.
    if np.sum(table, axis=0)[0] != 0 and np.sum(table, axis=0)[2] != 0 and np.sum(table, axis=0)[4] != 0 and np.sum(table, axis=0)[6] != 0:
        full_winds = True
    if np.sum(table, axis=0)[1] != 0 and np.sum(table, axis=0)[3] != 0 and np.sum(table, axis=0)[5] != 0 and np.sum(table, axis=0)[7] != 0:
        full_winds = True

    return full_winds

#=======================================================================================
# MAIN
#=======================================================================================

if __name__=="__main__":
    # Download Individual Raidosnde Sounding from University of Wyoming.
    date = datetime(2023, 11, 27, 12)
    station = 'SBBV'

    #date = datetime(2018, 7, 27, 0)
    #station = 'AMA'

    print(station, date)

    # Make the request (a pandas dataframe is returned).
    # If 503 Error, server to busy -> run script again until it works. The script will only work if there's no data for that day, but that's a different error
    try:
        df = WyomingUpperAir.request_data(date, station)
    except ValueError as ve:
        print(colored(ve,"red"))
        sys.exit()

    #Default values for individual sounding data filtering
    min_alt = config.min_alt
    max_alt = config.max_alt
    alt_step = config.alt_step
    n_sectors = config.n_sectors
    speed_threshold = config.speed_threshold
    wind_bins = np.arange(min_alt, max_alt, alt_step)

    # Determine which way to visualize wind rose
    blowing = config.blowing  # 1 FOR (typical wind rose),  -1 for TO (where balloon will drift to)

    print()
    print()

    # PLOTTING BEFORE FILTERING)
    df.dropna(inplace=True)
    ws = np.asarray(df['speed'])
    wd = np.asarray(df['direction'])
    wd = wd * blowing % 360
    alt = np.asarray(df['height'])


    # '''
    # Wind Speed Windrose
    ax2 = windrose.WindroseAxes.from_ax()
    # ax.bar(wd, ws,  bins=np.arange(0, 50, 5), opening = 0.8, normed=False, edgecolor="white", nsector = 16)
    ax2.bar(wd, ws, opening=.8, bins = np.arange(0, 50, 5), nsector=n_sectors, cmap = cm.cool_r)
    ax2.set_legend()

    if blowing == -1:
        ax2.set_title("Speed Windrose (BLOWING TO) for Station " +str(station) + " on " + str(date))
    else:
        ax2.set_title("Speed Windrose (BLOWING FROM) for Station " + str(station) + " on " + str(date))
    #'''

    #Do some filtering of the dataframe
    # Only analyze between 10-25km  as well as speeds over 2m/s is the default for now
    #df.dropna(inplace=True)
    df = df.drop(df[df['height'] < min_alt].index)
    df = df.drop(df[df['height'] > max_alt].index)
    #df = df.drop(df[df['speed'] < 2].index) #we'll ad this in later on'

    #Determine Wind Statistics
    opposing_wind_directions, opposing_wind_levels = determine_opposing_winds(df, wind_bins = wind_bins, n_sectors = n_sectors, speed_threshold = speed_threshold)
    calm_winds = determine_calm_winds(df, alt_step = alt_step)
    full_winds = determine_full_winds(df , wind_bins = wind_bins, speed_threshold = speed_threshold)

    print()
    print("WIND STATISTICS")
    if not calm_winds.any() and not opposing_wind_directions.any():
        print(colored("Wind Diversity FAIL.", "red"))
    else:
        if not calm_winds.any():
            print(colored("No Calm Winds.", "yellow"))
        else:
            print(colored("Calm Winds.", "green"))

        if not opposing_wind_directions.any():
            print(colored("No Opposing Winds.", "yellow"))
        else:
            print(colored("Opposing Winds.", "green"))

        if not full_winds:
            print(colored("No Full Wind Diversity.", "yellow"))
        else:
            print(colored("Full Wind Diversity", "green"))

    '''
    if not calm_winds.any() and not opposing_wind_directions.any():
        print(colored("Station Keeping FAIL.","red"))
    if calm_winds.any() or opposing_wind_directions.any():
        print(colored("Station Keeping PASS.", "green"))
        if full_winds:
            print(colored("Full Navigation PASS.", "green"))
        else:
            print(colored("Full Navigation FAIL.", "yellow"))
    print("Calm Winds:", bool(calm_winds.any()))
    print("Opposing Winds", bool(opposing_wind_directions.any()))
    print("Full Winds:", full_winds)
    #print("opposing_wind_levels", bool(opposing_wind_levels.any()))
    '''

    print()
    print("Calm Winds Regions:", calm_winds)
    print("Opposing Wind Levels:", opposing_wind_levels)


    ### PLOTING AFTER FILTERING###

    print(df)
    #asfa
    ws = np.asarray(df['speed'])
    wd = np.asarray(df['direction'])
    wd = wd * blowing % 360
    alt = np.asarray(df['height'])

    #Altitude Windrose
    ax = windrose.WindroseAxes.from_ax()
    #ax.bar(wd, ws,  bins=np.arange(0, 50, 5), opening = 1, normed=False, edgecolor="white", nsector = 16) #opening = 0.8
    ax.bar(wd, alt, opening = 1, bins=wind_bins, nsector=n_sectors, cmap = cm.rainbow)
    ax.set_legend(loc = 'lower left')

    '''
    fig = plt.figure(figsize=(8, 8))
    table = ax._info["table"]
    direction = ax._info["dir"]
    wd_freq = np.sum(table, axis=0)

    plt.bar(np.arange(16), wd_freq, align="center")
    xlabels = (
        "N",
        "",
        "N-E",
        "",
        "E",
        "",
        "S-E",
        "",
        "S",
        "",
        "S-W",
        "",
        "W",
        "",
        "N-W",
        "",
    )
    xticks = np.arange(16)
    plt.gca().set_xticks(xticks)
    plt.gca().set_xticklabels(xlabels)
    '''





    if blowing == -1:
        ax.set_title("Altitude Windrose (BLOWING TO) for Station " +str(station) + " on " + str(date))
    else:
        ax.set_title("Altitude Windrose (BLOWING FROM) for Station " + str(station) + " on " + str(date))

    # Plot 3D Wind Rose
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')
    polar_interpolated_scatter_plot(df, fig, ax, num_interpolations=100, color='winter', blowing= config.blowing,
                                    station=station, date=date)

    # To plot the sounding datapoints on top of the interpolated plot:
    viridis = cm.get_cmap('Set1', 1)  # This is just to get red dots
    polar_interpolated_scatter_plot(df, fig, ax, num_interpolations=1, color=viridis, size=20,
                                    no_interpolation=True, blowing=-1, station=station, date=date)

    plt.show()