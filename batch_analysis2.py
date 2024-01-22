from datetime import datetime
import pandas as pd
from termcolor import colored
import numpy as np
from pathlib import Path
import dataframe_image as dfi
import os
from os import listdir
from multiprocessing import Process, Manager

import opposing_wind_wyoming
import config
from ERA5 import ERA5

# maybe add this to utils.py?
def get_analysis_folder(FAA, WMO, year):
    return config.analysis_folder + str(FAA) + " - " + str(WMO) + "/" + str(year) + "_analysis/"

# maybe add this to utils.py?
def get_data_folder(FAA, WMO, year):
    return config.parent_folder + str(FAA) + " - " + str(WMO) + "/" + str(year) + "/"


def determine_wind_statistics(df, min_alt = 15000, max_alt = 24000, min_pressure = 20, max_pressure  = 125, alt_step = 500, n_sectors = 16, speed_threshold = 2):
    if config.type == "PRES":
        wind_bins = config.era5_pressure_levels[::-1]
        wind_bins = wind_bins[(wind_bins <= max_pressure)]
        wind_bins = wind_bins[(wind_bins >= min_pressure)]
        #print(wind_bins)
    if config.type == "ALT":
        wind_bins = np.arange(min_alt, max_alt, alt_step)

    # Do some filtering of the dataframe
    #df.dropna(inplace=True)

    if config.type == "ALT":
        df.dropna(subset=['height'], how='all', inplace=True)
        df = df.drop(df[df['height'] < min_alt].index)
        df = df.drop(df[df['height'] > max_alt].index)
        # df = df.drop(df[df['speed'] < 2].index) #we'll ad this in later on'
    if config.type == "PRES":
        df = df.drop(df[df['pressure'] < min_pressure].index)
        df = df.drop(df[df['pressure'] > max_pressure].index)

    # Determine Wind Statistics
    opposing_wind_directions, opposing_wind_levels = opposing_wind_wyoming.determine_opposing_winds(df,
                                                                                                    wind_bins=wind_bins,
                                                                                                    n_sectors=n_sectors,
                                                                                                    speed_threshold=speed_threshold)
    calm_winds = opposing_wind_wyoming.determine_calm_winds(df, alt_step=alt_step)
    full_winds = opposing_wind_wyoming.determine_full_winds(df, wind_bins=wind_bins, speed_threshold=speed_threshold)

    if config.logging:
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

    return wind_bins, opposing_wind_levels

def save_wind_probabilties(FAA, WMO, wind_probabilities, analysis_folder, date):
    '''

    Args:
        wind_probabilities: df of wind probabilities
        analysis_folder: path
        date: datetime

    Returns: Void, saves .csv and colored png table of the wind probabilities table for a particular station

    '''
    #print(wind_probabilities)
    wind_probabilities = pd.concat([wind_probabilities, wind_probabilities.apply(['average'])])

    print(colored(
        "Processing data for Station-" + str(FAA) + " - " + str(WMO) + "    Year-" + str(date.year) + "    Month-" + str(date.month),
        "cyan"))

    wind_probabilities = wind_probabilities.apply(pd.to_numeric)

    if config.type == "PRES":
        #reverse order of dataframes for pressure, since high pressure = low altitude
        wind_probabilities = wind_probabilities.iloc[:,::-1]
        #wind_probabilities = wind_probabilities[wind_probabilities.columns[::-1]]

    #print(wind_probabilities)
    wind_probabilities_styled = wind_probabilities.style.background_gradient(axis=None, vmin=0, vmax=1.0, cmap='RdYlGn')

    wind_probabilities_styled = wind_probabilities_styled.set_caption(
        'Opposing Wind Probabilities for Station ' + str(FAA) + " - " + str(WMO) + ' in Month ' + str(
            date.month) + ' 12Z ' + str(date.year)).set_table_styles([{
        'selector': 'caption',
        'props': [
            ('color', 'black'),
            ('font-weight', 'bold'),
            ('font-size', '20px')
        ]
    }])

    # Save table in csv form as well as a colored table png.
    wind_probabilities_styled = wind_probabilities_styled.format(precision=2)

    filepath_image = Path(
        analysis_folder + "images/" + str(FAA) + " - " + str(WMO) + "-" + str(date.year) + "-" + str(
            date.month) + '-wind_probabilities.png')
    filepath_image.parent.mkdir(parents=True, exist_ok=True)
    dfi.export(wind_probabilities_styled, filepath_image, max_rows=-1)

    filepath_dataframe = Path(
        analysis_folder + "dataframes/" + str(FAA) + " - " + str(WMO) + "-" + str(date.year) + "-" + str(
            date.month) + '-wind_probabilities.csv')
    filepath_dataframe.parent.mkdir(parents=True, exist_ok=True)
    wind_probabilities.to_csv(filepath_dataframe)


def anaylze_annual_data_era5(era5, lat, lon, FAA, WMO, year, min_alt = 15000, max_alt = 24000, min_pressure = 20, max_pressure  = 125, alt_step = 500, n_sectors = 16, speed_threshold = 2):


    current_month = 1

    analysis_folder = get_analysis_folder(FAA, WMO, year)

    # reinitialize dataframes
    if config.type == "PRES":
        wind_bins = config.era5_pressure_levels[::-1]
        wind_bins = wind_bins[(wind_bins <= max_pressure)]
        wind_bins = wind_bins[(wind_bins >= min_pressure)]
        #print(wind_bins)
    if config.type == "ALT":
        wind_bins = np.arange(min_alt, max_alt, alt_step)

    #print("wind bins", wind_bins)


    if config.type == "ALT":
        column_headers = np.char.mod("%.1f", wind_bins / 1000.)
    if config.type == "PRES":
        pressure_bins = wind_bins
        column_headers = np.char.mod("%.1f", pressure_bins)

    wind_probabilities = pd.DataFrame(columns=column_headers)

    #print(wind_probabilities)
    #print(wind_bins)

    #sdfs

    # Check if sounding data has already been analyzed.  If so, skip
    if check_monthly_analyzed(analysis_folder):
        return True

    for time in era5.time_convert:

        #print(time)

        station = era5.get_station(time, lat, lon)

        month = time.month
        day = time.day
        hour = time.hour

        station.dropna(subset=['direction', 'speed'], how='all', inplace=True)


        wind_bins, opposing_wind_levels = determine_wind_statistics(station, min_alt=min_alt, max_alt=max_alt, min_pressure= min_pressure, max_pressure = max_pressure,
                                                                    alt_step=alt_step, n_sectors=n_sectors,
                                                                    speed_threshold=speed_threshold)

        if config.logging:
            print("opposing_wind_levels", opposing_wind_levels)
        # Double check this when full forecast is downloaded
        #print(month, current_month)

        #print("new wind_bins", wind_bins)

        #Do I need to do this again?
        if month != current_month or (month == 12 and day== 31 and hour == 12):
            if config.logging:
                print(wind_probabilities)
            save_wind_probabilties(FAA, WMO, wind_probabilities, analysis_folder, date)
            current_month += 1

            # reinitialize dataframes
            if config.type == "PRES":
                wind_bins = config.era5_pressure_levels[::-1]
                wind_bins = wind_bins[(wind_bins <= max_pressure)]
                wind_bins = wind_bins[(wind_bins >= min_pressure)]
            if config.type == "ALT":
                wind_bins = np.arange(min_alt, max_alt, alt_step)

            if config.type == "ALT":
                column_headers = np.char.mod("%.1f", wind_bins / 1000.)
            if config.type == "PRES":
                #pressure_bins = [300, 250, 225, 200, 175, 150, 125, 100, 70, 50, 30, 20]  # LEAVE OFF 10 HPA
                column_headers = np.char.mod("%.1f", pressure_bins)
            wind_probabilities = pd.DataFrame(columns=column_headers)

        #print(wind_probabilities)
        #print(wind_bins)

        # there's probably a faster way to do this with numpy.  Or maybe I should change the output of opposing_wind_levels?
        mask = wind_bins
        for k in range(len(mask)):
            if wind_bins[k] in opposing_wind_levels:
                mask[k] = 1
            else:
                mask[k] = 0

        # Need to check if Dataframe is empty after dropping nan values was done on direction and speed

        #station.time = station['time']
        date = station.time.iat[0]
        if config.logging:
            print()
            print(date)
        wind_probabilities.loc[date, :] = mask

    return True


def anaylze_annual_data(FAA, WMO, year, min_alt = 15000, max_alt = 24000, min_pressure = 20, max_pressure  = 125, alt_step = 500, n_sectors = 16, speed_threshold = 2):
    '''
        Iterate through every sounding for a particular [station] in [year]

        Pseudocode for the algorithm:

        Iterate By Month for a [station] in a [year]:
            1. Create an empty wind_probabilties dataframe

            2. Iterate for each [day] within a [Month]:
                i.   **getsounding()**                  If the sounding isn't downloaded locally, download them from UofWy and continue the analysis.
                ii.  **determine_wind_statistics()**    Anaylze the opposing wind statistics for the sounding on this [day] and [station]
                iii.                                    Check if there are opposing winds at each [alt_step] and add the row [day] to the wind_probabilties dataframe

            3. **save_wind_probabilties()** Generate a consolidated dataframe of wind probabilties for each [day/rows]
            and [alt_step/columns] and output as a .csv and a colored dataframe image (png) for each month.
    '''

    data_folder = get_data_folder(FAA, WMO, year)
    analysis_folder = get_analysis_folder(FAA, WMO, year)

    #Check if sounding data has already been analyzed.  If so, skip
    if check_monthly_analyzed(analysis_folder):
        return True

    #Iterate by month and day for a particular year.
    for j in range (1,12+1):
        #reinitialize dataframes

        if config.type == "PRES":
            wind_bins = config.era5_pressure_levels[::-1]
            wind_bins = wind_bins[(wind_bins <= max_pressure)]
            wind_bins = wind_bins[(wind_bins >= min_pressure)]
            # print(wind_bins)
        if config.type == "ALT":
            wind_bins = np.arange(min_alt, max_alt, alt_step)

        # print("wind bins", wind_bins)

        if config.type == "ALT":
            column_headers = np.char.mod("%.1f", wind_bins / 1000.)
        if config.type == "PRES":
            pressure_bins = wind_bins
            column_headers = np.char.mod("%.1f", pressure_bins)

        wind_probabilities = pd.DataFrame(columns=column_headers)
        wind_probabilities_styled = pd.DataFrame()

        try:
            all_files = os.listdir(data_folder + str(j))
        except:
            print(colored(str(FAA) + " - " + str(WMO) + "/" + str(year) + " Not Downloaded.", "red"))
            return False
            #raise ValueError
        csv_files = list(filter(lambda f: f.endswith('.csv'), all_files))
        csv_files.sort() #sort the list of CSVs to have the table in the right order


        #Will need to check if there is missing data.
        if csv_files:
            for csv in csv_files:
                df = pd.read_csv(data_folder + str(j) + "/" + csv, index_col = 0)

                #df = df.dropna(axis='rows')  #change this to speed
                df.dropna(subset=['direction', 'speed'], how='all', inplace=True)

                wind_bins, opposing_wind_levels = determine_wind_statistics(df, min_alt=min_alt,
                                                                            max_alt=max_alt,
                                                                            min_pressure = min_pressure,
                                                                            max_pressure = max_pressure,
                                                                            alt_step=alt_step,
                                                                            n_sectors=n_sectors,
                                                                            speed_threshold=speed_threshold)

                # there's probably a faster way to do this with numpy.  Or maybe I should change the output of opposing_wind_levels?
                mask = wind_bins
                for k in range(len(mask)):
                    if wind_bins[k] in opposing_wind_levels:
                        mask[k] = 1
                    else:
                        mask[k] = 0

                #Need to check if Dataframe is empty after dropping nan values was done on direction and speed
                try:
                    df.time = pd.to_datetime(df['time'])
                    date = df.time.iat[0]
                    if config.logging:
                        print(date)
                    wind_probabilities.loc[date, :] = mask
                except:
                    print(colored("GOT AN EXCEPTION","yellow"))
                    pass

        else:
            #print(colored("MADE IT HERE", "magenta"))
            date = datetime(year, j, 1, 00)  #day and time shouldn't matter
            mask = np.empty(len(wind_bins))
            mask[:] = np.nan
            wind_probabilities.loc[date, :] = mask

            if config.logging:
                print(wind_probabilities)

        save_wind_probabilties(FAA, WMO, wind_probabilities, analysis_folder, date)

    return True

def cumulative_batch_analysis(FAA, WMO, year, min_alt = 15000, max_alt = 24000, min_pressure = 20, max_pressure  = 125, alt_step = 500):
    print("============CUMULATIVE WIND PROBABILITY ANALYSIS==============\n")

    analysis_folder = get_analysis_folder(FAA, WMO, year)

    files = [f for f in listdir(analysis_folder + "dataframes") if f.endswith(".csv")]

    if config.type == "PRES":
        wind_bins = config.era5_pressure_levels[::-1]
        wind_bins = wind_bins[(wind_bins <= max_pressure)]
        wind_bins = wind_bins[(wind_bins >= min_pressure)]
        # print(wind_bins)
    if config.type == "ALT":
        wind_bins = np.arange(min_alt, max_alt, alt_step)

    if config.type == "ALT":
        column_headers = np.char.mod("%.1f", wind_bins / 1000.)
    if config.type == "PRES":
        #reversed for pressure,  but already reversed in the original files.
        column_headers = np.char.mod("%.1f", wind_bins[::-1])

    cumulative = pd.DataFrame(columns=column_headers)


    for csv in files:
        # read csv for each month of individual station
        try:
            df = pd.read_csv(analysis_folder + "/dataframes/" + csv, index_col=0)

            str_date = df.iloc[0:1].index.values[0]
            date = datetime.strptime(str_date, '%Y-%m-%d %H:%M:%S')

            cumulative.loc[date.month, :] = df.iloc[-1:].values
        except:
            continue

    cumulative.sort_index(inplace=True, ascending=True)
    print(cumulative)

    cumulative = cumulative.apply(pd.to_numeric)

    cumulative_styled = cumulative.style.background_gradient(axis=None, vmin=0, vmax=1.0, cmap='RdYlGn')

    cumulative_styled = cumulative_styled.set_caption(
        'Opposing Wind Probabilities for Station ' + str(FAA) + " - " + str(WMO) + ' 12Z in ' + str(year)).set_table_styles([{
        'selector': 'caption',
        'props': [
            ('color', 'black'),
            ('font-weight', 'bold'),
            ('font-size', '20px')
        ]
    }])

    cumulative_styled = cumulative_styled.format(precision=2)

    #Put the cumulative probabilities in 2 spots for easy visual inspection and further decade-based cumulative analysis.
    filepath_image = Path(analysis_folder[:-14] + "analysis_" + str(year) + '-wind_probabilities-TOTAL.png')
    filepath_image.parent.mkdir(parents=True, exist_ok=True)
    dfi.export(cumulative_styled, filepath_image, max_rows=-1)

    filepath_image = Path(analysis_folder+ "analysis_" + str(year) + '-wind_probabilities-TOTAL.png')
    filepath_image.parent.mkdir(parents=True, exist_ok=True)
    dfi.export(cumulative_styled, filepath_image, max_rows=-1)

    filepath_dataframe = Path(analysis_folder[:-14]  + "analysis_" + str(year) + '-wind_probabilities-TOTAL.csv')
    filepath_dataframe.parent.mkdir(parents=True, exist_ok=True)
    cumulative.to_csv(filepath_dataframe)

    filepath_dataframe = Path(analysis_folder + "analysis_" + str(year) + '-wind_probabilities-TOTAL.csv')
    filepath_dataframe.parent.mkdir(parents=True, exist_ok=True)
    cumulative.to_csv(filepath_dataframe)

#Maybe add this to a utils.py function?
def check_monthly_analyzed(analysis_folder):
    '''

        Helper Function to see if annual sounding data has been analyzed yet

    '''

    isExist = os.path.exists(analysis_folder)
    if not isExist:
        print(colored(analysis_folder, "yellow"))
        return False

    print(colored(analysis_folder, "green"))
    return True

#Maybe add this to a utils.py function?
def check_annual_analyzed(FAA, WMO, year):
    '''

    Helper Function to see if annual sounding data has been analyzed yet

    '''

    analysis_folder = get_analysis_folder(FAA, WMO, year)
    file = analysis_folder[:-14] + "analysis_" + str(year) + '-wind_probabilities-TOTAL.csv'

    isExist = os.path.exists(file)
    if not isExist:
        print(colored(file, "yellow"))
        return False

    print(colored(file, "green"))
    return True



def analyze(era5,year,WMO,FAA,lat,lon,min_alt,max_alt,min_pressure,max_pressure,alt_step,n_sectors,speed_threshold):
    if config.mode == "era5":
        status = anaylze_annual_data_era5(era5, lat, lon, FAA, WMO, year, min_alt, max_alt, min_pressure, max_pressure,
                                          alt_step, n_sectors, speed_threshold)

    if config.mode == "radiosonde":
        status = anaylze_annual_data(FAA, WMO, year, min_alt=min_alt, max_alt=max_alt, min_pressure=min_pressure,
                                     max_pressure=max_pressure, alt_step=alt_step, n_sectors=n_sectors,
                                     speed_threshold=speed_threshold)
        # cumulative_batch_analysis(FAA, WMO, year, min_alt=min_alt, max_alt=max_alt, alt_step=alt_step, n_sectors=n_sectors,speed_threshold=speed_threshold)


    if not status:
        print(colored("Can't perform cumulative analysis because " + str(FAA) + "-" + str(WMO) + "/" + str(
            year) + " not downloaded.", "red"))

    if not check_annual_analyzed(FAA, WMO, year) and status:
        cumulative_batch_analysis(FAA, WMO, year, min_alt=min_alt, max_alt=max_alt, min_pressure=min_pressure,
                                  max_pressure=max_pressure, alt_step=alt_step)


def parallelize(era5, stations_df, year,min_alt,max_alt,min_pressure,max_pressure,alt_step,n_sectors,speed_threshold ):
    '''
    Parralel the process.  each station downlaod goes on it's own thread.  THis make the downloads much faster.

    Args:
        stations_df:
        year:

    Returns:

    '''

    queue = Manager().Queue()
    procs = [Process(target=analyze, args=(era5,year,  row.WMO, row.FAA, row.lat_era5, row.lon_era5,min_alt,max_alt,min_pressure,max_pressure,alt_step,n_sectors,speed_threshold )) for row in stations_df.itertuples(index=False)]
    for p in procs: p.start()
    for p in procs: p.join()

    results = []
    while not queue.empty():
        results.append(queue.get)

    return results


#mMain
if __name__=="__main__":

    continent = "North_America"
    stations_df = pd.read_csv('Radisonde_Stations_Info/CLEANED/' + continent + ".csv")
    #stations_df = stations_df.loc[stations_df["CO"] == "US"]  # Only do US Countries for now
    #stations_df = stations_df[-1:]

    era5 = ERA5()
    era5.import_forecast(config.era_file)
    era5.get_statistics()

    stations_df['lat_era5'] = stations_df.apply(lambda x: (-1 * x['  LAT'] if x['N'] == 'S' else 1 * x['  LAT']), axis=1)
    stations_df['lon_era5'] = stations_df.apply(lambda x: (-1* x[' LONG'] if x['E'] == 'W' else 1 * x[' LONG']), axis=1)

    print(stations_df)

    # Initialize Variables
    min_alt = config.min_alt
    max_alt = config.max_alt
    min_pressure = config.min_pressure
    max_pressure = config.max_pressure
    alt_step = config.alt_step
    n_sectors = config.n_sectors
    speed_threshold = config.speed_threshold

    '''
    for row in stations_df.itertuples(index=False):


        WMO = row.WMO
        FAA = row.FAA
        Name = row.Station_Name
        lat = row.lat_era5
        lon = row.lon_era5

        #Initialize Variables
        min_alt = config.min_alt
        max_alt = config.max_alt
        min_pressure = config.min_pressure
        max_pressure = config.max_pressure
        alt_step = config.alt_step
        n_sectors = config.n_sectors
        speed_threshold = config.speed_threshold

        #Station downloads
        #year = 2013
        #station = 'SKBO'
    '''

        # DO THE WIND PROBABILTIES ANALYSIS
    for year in range (config.start_year, config.end_year +1):

        if config.parallelize:
            print(colored("Downloading Radiosonde Datasets in Parallel [MultiThreading]", "cyan"))
            parallelize(era5, stations_df, year=year,
                        min_alt=config.min_alt,
                        max_alt=config.max_alt,
                        min_pressure=config.min_pressure,
                        max_pressure=config.max_pressure,
                        alt_step=config.alt_step,
                        n_sectors=config.n_sectors,
                        speed_threshold=config.speed_threshold)

        else:
            print(colored("Downloading Radiosonde Datasets in Sequence", "cyan"))
            for row in stations_df.itertuples(index=False):
                analyze(era5,year,
                            WMO = row.WMO,
                            FAA = row.FAA,
                            lat = row.lat_era5,
                            lon = row.lon_era5,
                            min_alt = config.min_alt,
                            max_alt = config.max_alt,
                            min_pressure = config.min_pressure,
                            max_pressure = config.max_pressure,
                            alt_step = config.alt_step,
                            n_sectors = config.n_sectors,
                            speed_threshold = config.speed_threshold)

    '''
        # ADD MONTHLY STATUS CHECK AND CUMULATIVE STATUS CHECK BACK IN!

        #status = anaylze_annual_data(FAA, WMO, year, min_alt = min_alt, max_alt = max_alt, alt_step = alt_step, n_sectors = n_sectors, speed_threshold = speed_threshold)

        if config.mode == "era5":
            status = anaylze_annual_data_era5(era5, lat, lon, FAA, WMO, year, min_alt, max_alt, min_pressure, max_pressure, alt_step, n_sectors, speed_threshold)
        if config.mode == "radiosonde":
            status = anaylze_annual_data(FAA, WMO, year, min_alt = min_alt, max_alt = max_alt, min_pressure = min_pressure, max_pressure = max_pressure, alt_step = alt_step, n_sectors = n_sectors, speed_threshold = speed_threshold)
            #cumulative_batch_analysis(FAA, WMO, year, min_alt=min_alt, max_alt=max_alt, alt_step=alt_step, n_sectors=n_sectors,speed_threshold=speed_threshold)
        if not status:
            print(colored("Can't perform cumulative analysis because + " + str(FAA) + "/" + str(WMO) + "/" + str(year) + " not downloaded.", "red"))

        if not check_annual_analyzed(FAA, WMO, year) and status:
            cumulative_batch_analysis(FAA, WMO, year, min_alt=min_alt, max_alt=max_alt, min_pressure = min_pressure, max_pressure = max_pressure, alt_step=alt_step)
    '''