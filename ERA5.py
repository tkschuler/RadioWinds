
from geographiclib.geodesic import Geodesic
import numpy as np
import netCDF4
from termcolor import colored
import xarray as xr
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import opposing_wind_wyoming
import config
from plot3DWindrose import polar_interpolated_scatter_plot
import windrose

class ERA5:
    def __init__(self):
        self.geod = Geodesic.WGS84

    def import_forecast(self, filepath):
        self.file = netCDF4.Dataset(filepath)

        self.ds = xr.open_dataset(filepath, engine="netcdf4", decode_times=True)
        #print(self.ds)


    def get_statistics(self):
        #print(self.file)

        time_arr = self.file.variables['time']
        # Convert from epoch to human readable time. Different than GFS for now.
        self.time_convert = netCDF4.num2date(time_arr[:], time_arr.units, time_arr.calendar)

        netcdf_ranges = self.file.variables['u'][0, 0, :, :]
        self.determineRanges(netcdf_ranges)

        # smaller array of downloaded forecast subset
        self.lat = self.file.variables['latitude'][self.lat_max_idx:self.lat_min_idx]
        self.lon = self.file.variables['longitude'][self.lon_min_idx:self.lon_max_idx]

        # min/max lat/lon degree values from netcdf4 subset
        self.LAT_LOW = self.file.variables['latitude'][self.lat_min_idx - 1]
        self.LON_LOW = self.file.variables['longitude'][self.lon_min_idx]
        self.LAT_HIGH = self.file.variables['latitude'][self.lat_max_idx]
        self.LON_HIGH = self.file.variables['longitude'][self.lon_max_idx - 1]

        print()
        print("ERA5 Forecast Statistics")
        print("LAT RANGE: min: " + str(self.file.variables['latitude'][self.lat_min_idx - 1]),
              " max: " + str(self.file.variables['latitude'][self.lat_max_idx]) + " size: " + str(
                  self.lat_min_idx - self.lat_max_idx))
        print("LON RANGE: min: " + str(self.file.variables['longitude'][self.lon_min_idx]),
              " max: " + str(self.file.variables['longitude'][self.lon_max_idx - 1]) + " size: " + str(
                  self.lon_max_idx - self.lon_min_idx))

        print("TIME RANGE: start time: " + str(self.time_convert[self.start_time_idx]) +
                                " end time: " + str(self.time_convert[self.end_time_idx]))
        print()

    def determineRanges(self, netcdf_ranges):
        """
        Determine the dimensions of actual data. If you have columns or rows with missing data
        as indicated by NaN, then this function will return your actual shape size so you can
        resize your data being used.

        """

        results = np.all(~netcdf_ranges.mask)
        if results == False:
            timerange, latrange, lonrange = np.nonzero(~netcdf_ranges.mask)

            self.start_time_idx = timerange.min()
            self.end_time_idx = timerange.max()
            self.lat_min_idx = latrange.min()  # Min/Max are switched compared to with ERA5
            self.lat_max_idx = latrange.max()
            self.lon_min_idx = lonrange.min()
            self.lon_max_idx = lonrange.max()
        else:  # Typically this for ERA5 because they don't mask data, right?
            self.start_time_idx = 0
            self.end_time_idx = len(self.time_convert) - 1
            lati, loni = netcdf_ranges.shape
            # THese are backwards???
            self.lat_min_idx = lati
            self.lat_max_idx = 0
            self.lon_max_idx = loni
            self.lon_min_idx = 0

    def get_station(self, time, lat, lon):
        station_ds = self.ds.sel(time=time, longitude=lon, latitude=lat, method = "nearest")
        #print(alt)
        #print(alt.latitude, alt.longitude)

        station_df = self.get_dataframe(station_ds, time)

        #print(station_ds.z/config.g)

        #sdfs

        return station_df

    def get_dataframe(self, station, time):
        g = 9.80665  # gravitation constant used to convert geopotential height to height
        df = pd.DataFrame()

        df['height'] = station.z / g
        df['time'] = time
        df['u_wind'] = station.u * 1  # put in same as radiosonde format,  blowing from?
        df['v_wind'] = station.v * 1  # put in same as radiosonde format,  blowing from?
        df['speed'] = (df['u_wind'] ** 2 + df['v_wind'] ** 2) ** 0.5
        df['direction'] = (90 - np.rad2deg(np.arctan2(df['v_wind'], df['u_wind']))) % 360  # convert to meteolorological wind?
        df['pressure'] = station.level

        #Reverse order, so altitude is in ascending order
        df = df[::-1].reset_index(drop=True)
        return df

if __name__=="__main__":
    time = date = '2022-12-18  00:00:00'

    #PHTO
    lat = 19.72
    lon = -155

    #BUF
    #lat = 42.9
    #lon = -78.7

    #SLC
    #lat = 40.6
    #lon = -111.9

    #TBW
    #lat = 27.7
    #lon = -82.6

    #BIS
    #lat = 46.8
    #lon = -100.8

    #SBBV
    #lat = 2.8206
    #lon = -60.6738


    era5 = ERA5()
    #era5.import_forecast("forecasts/" + "western_hemisphere-08-08-23.nc")
    era5.import_forecast("forecasts/" + "western_hemisphere-2022.nc")
    era5.get_statistics()

    df = era5.get_station(time, lat, lon)

    print(df)
    #print("hey')")

    #sdfs

    #if config.altitude_type == "pres":
    #    df['height'] = df.index


    station = 'test'

    min_alt = config.min_alt
    max_alt = config.max_alt
    min_pressure = config.min_pressure
    max_pressure = config.max_pressure
    alt_step = config.alt_step
    n_sectors = config.n_sectors
    speed_threshold = config.speed_threshold


    print()
    print()

    # PLOTTING BEFORE FILTERING)
    if not config.blowing_to:
        df['direction'] = (df['direction'] - 180) % 360

        # ================= Filter Data Frame for Opposing WInd Analysis ===================

    if not config.by_pressure:
        df = df.drop(df[df['height'] < min_alt].index)
        df = df.drop(df[df['height'] > max_alt].index)
        # df = df.drop(df[df['speed'] < 2].index) #we'll ad this in later on'
    else:
        df = df.drop(df[df['pressure'] < min_pressure].index)
        df = df.drop(df[df['pressure'] > max_pressure].index)
        # df = df [::-1]

    if config.by_pressure:
        wind_bins = config.era5_pressure_levels[::-1]
        wind_bins = wind_bins[(wind_bins <= config.max_pressure)]
        wind_bins = wind_bins[(wind_bins >= config.min_pressure)]
        print(wind_bins)
    else:
        wind_bins = np.arange(min_alt, max_alt, alt_step)

    df.dropna(inplace=True)
    ws = np.asarray(df['speed'])
    wd = np.asarray(df['direction'])
    alt = np.asarray(df['height'])

    #print(df)

    # ============= Unfiltered Standard Windspeed Windrose ======================
    df.dropna(inplace=True)
    ws = np.asarray(df['speed'])
    wd = np.asarray(df['direction'])
    alt = np.asarray(df['height'])
    pressure = np.asarray(df['pressure'])

    ax2 = windrose.WindroseAxes.from_ax()
    ax2.bar(wd, ws, opening=.8, bins=np.arange(0, 50, 5), nsector=n_sectors, cmap=cm.cool_r)
    ax2.set_legend()

    if config.blowing_to: #ERA4 is flipped from Radiosonde
        ax2.set_title("Speed Windrose (BLOWING TO) for Station " + str(station) + " on " + str(date))
    else:
        ax2.set_title("Speed Windrose (BLOWING FROM) for Station " + str(station) + " on " + str(date))

    #Determine Wind Statistics
    opposing_wind_directions, opposing_wind_levels = opposing_wind_wyoming.determine_opposing_winds(df, wind_bins = wind_bins, n_sectors = n_sectors, speed_threshold = speed_threshold)
    calm_winds = opposing_wind_wyoming.determine_calm_winds(df, alt_step = alt_step)
    full_winds = opposing_wind_wyoming.determine_full_winds(df , wind_bins = wind_bins, speed_threshold = speed_threshold)

    print()
    print(df)

    print("WIND STATISTICS")
    if not calm_winds.any() and not opposing_wind_levels.any():
        print(colored("Wind Diversity FAIL.", "red"))
    else:
        if not calm_winds.any():
            print(colored("No Calm Winds.", "yellow"))
        else:
            print(colored("Calm Winds.", "green"))

        if not opposing_wind_levels.any():
            print(colored("No Opposing Winds.", "yellow"))
        else:
            print(colored("Opposing Winds.", "green"))

        if not full_winds:
            print(colored("No Full Wind Diversity.", "yellow"))
        else:
            print(colored("Full Wind Diversity", "green"))


    print()
    print("Calm Winds Regions:", calm_winds)
    print("Opposing Wind Levels:", opposing_wind_levels)


    ### PLOTING AFTER FILTERING###
    #asfa
    ws = np.asarray(df['speed'])
    wd = np.asarray(df['direction'])
    #wd = wd * blowing % 360
    alt = np.asarray(df['height'])

    #Altitude Windrose
    ax = windrose.WindroseAxes.from_ax()
    if not config.by_pressure:
        ax.bar(wd, alt, opening=1, bins=wind_bins, nsector=n_sectors, cmap=cm.rainbow)
    else:
        ax.bar(wd, pressure, opening=1, bins=wind_bins, nsector=n_sectors, cmap=cm.rainbow)
    ax.set_legend(loc = 'lower left')

    if config.blowing_to: #ERA5 is flipped from radiosonde (already default to blowing to)
        ax.set_title("Altitude Windrose (BLOWING TO) for Station " +str(station) + " on " + str(date))
    else:
        ax.set_title("Altitude Windrose (BLOWING FROM) for Station " + str(station) + " on " + str(date))

    # Plot 3D Wind Rose
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')
    polar_interpolated_scatter_plot(df, fig, ax, num_interpolations=100, color='winter', blowing_to = config.blowing_to,
                                    station=station, date=date)

    # To plot the sounding datapoints on top of the interpolated plot:
    viridis = cm.get_cmap('Set1', 1)  # This is just to get red dots
    polar_interpolated_scatter_plot(df, fig, ax, num_interpolations=1, color=viridis, size=20,
                                    no_interpolation=True, blowing_to = config.blowing_to, station=station, date=date)

    plt.figure()

    plt.plot(df['u_wind'],df['height'])
    plt.plot(df['v_wind'],df['height'])

    plt.show()
