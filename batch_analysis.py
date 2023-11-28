from datetime import datetime
from siphon.simplewebservice.wyoming import WyomingUpperAir
import pandas as pd
from termcolor import colored
import numpy as np
from pathlib import Path
import dataframe_image as dfi
import os
from os import listdir

import opposing_wind_wyoming

def get_sounding(station, date):
    filepath_sounding = Path(folder + str(station) + "-" + str(date.year) + "-" + str(
        date.month) + "-" + str(date.day) + "-" + str(date.hour) + '.csv')

    df = None
    missing_data = False

    #Check if data has been downloaded locally,  assume if the folder exists, all the data was properly downloaded for that particular station and year.
    if local_download:
        try:
            df = pd.read_csv(filepath_sounding)
        except:
            missing_data = True
            print(colored("Data doesn't exist for " + str(date), "red"))

    #If not make a request to University of Wyoming for each individual sounding
    else:
        # Make the request (a pandas dataframe is returned).
        # If 503 Error, server to busy -> run script again until it works. The script will only work if there's no data for that day, but that's a different error
        count = 0
        while df is None:
            try:
                df = WyomingUpperAir.request_data(date, station)
                # download Soundings locally if they don't exist yet.  That way analysis can be done faster next time.

                filepath_sounding = Path(folder + str(station) + "-" + str(date.year) + "-" + str(
                    date.month) + "-" + str(date.day) + "-" + str(date.hour) + '.csv')
                filepath_sounding.parent.mkdir(parents=True, exist_ok=True)

                df.to_csv(filepath_sounding)

            except ValueError as ve:
                print(colored(ve,"red"))
                missing_data = True
                df = pd.DataFrame()
                pass
            except:
                count += 1
                print(colored("server Error. Trying again" + str(count), "yellow"))
                df = None
                if count == 10:
                    print("SKIPPING")
                    missing_data = True
                    break



    return df, missing_data


def determing_wind_statistics(df , min_alt, max_alt, alt_step, n_sectors, speed_threshold):
    wind_bins = np.arange(min_alt, max_alt, alt_step)

    # Do some filtering of the dataframe
    # Only analyze between 10-25km  as well as speeds over 2m/s is the default for now
    df.dropna(inplace=True)
    df = df.drop(df[df['height'] < min_alt].index)
    df = df.drop(df[df['height'] > max_alt].index)

    # Determine Wind Statistics
    opposing_wind_directions, opposing_wind_levels = opposing_wind_wyoming.determine_opposing_winds(df,
                                                                                                    wind_bins=wind_bins,
                                                                                                    n_sectors=n_sectors,
                                                                                                    speed_threshold=speed_threshold)
    calm_winds = opposing_wind_wyoming.determine_calm_winds(df, alt_step=alt_step)
    full_winds = opposing_wind_wyoming.determine_full_winds(df, wind_bins=wind_bins, speed_threshold=speed_threshold)

    if not calm_winds.any() and not opposing_wind_directions.any():
        print(colored("Station Keeping FAIL.", "red"))
    if calm_winds.any() or opposing_wind_directions.any():
        print(colored("Station Keeping PASS.", "green"))
        if full_winds:
            print(colored("Full Navigation PASS.", "green"))
        else:
            print(colored("Full Navigation FAIL.", "yellow"))
    print("Calm Winds:", bool(calm_winds.any()))
    print("Opposing Winds", bool(opposing_wind_directions.any()))
    print("Full Winds:", full_winds)

    return wind_bins, opposing_wind_levels

def save_wind_probabilties(wind_probabilities, date):
    wind_probabilities = pd.concat([wind_probabilities, wind_probabilities.apply(['average'])])

    # print(wind_probabilities)
    print(colored(
        "Processing data for Station-" + str(FAA) + " - " + str(WMO) + "    Year-" + str(date.year) + "    Month-" + str(date.month),
        "cyan"))

    wind_probabilities = wind_probabilities.apply(pd.to_numeric)
    wind_probabilities_styled = wind_probabilities.style.background_gradient(axis=None, vmin=0, vmax=1.0, cmap='RdYlGn')

    wind_probabilities_styled = wind_probabilities_styled.set_caption(
        'Opposing Wind Probabilities for Station ' + str(FAA) + " - " + str(WMO) + ' in Month ' + str(
            date.month) + ' 12Z').set_table_styles([{
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
        folder[:-5] + "analysis_" + str(date.year) + "/images/" + str(FAA) + " - " + str(WMO) + "-" + str(date.year) + "-" + str(
            date.month) + '-wind_probabilities.png')
    filepath_image.parent.mkdir(parents=True, exist_ok=True)
    dfi.export(wind_probabilities_styled, filepath_image)

    filepath_dataframe = Path(
        folder[:-5] + "analysis_" + str(date.year) + "/dataframes/" + str(FAA) + " - " + str(WMO) + "-" + str(date.year) + "-" + str(
            date.month) + '-wind_probabilities.csv')
    filepath_dataframe.parent.mkdir(parents=True, exist_ok=True)
    wind_probabilities.to_csv(filepath_dataframe)

    #return wind_probabilities

#THIS IS THE MAIN FUNCTION THAT INCOPROATES ALL THE HELPER FUNCTIONS
def anaylze_annual_data(FAA, WMO, year, min_alt = 15000, max_alt = 24000, alt_step = 500, n_sectors = 16, speed_threshold = 2):
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

    #Iterate by month and day for a particular year.
    for j in range (1,12+1):
        #reinitialize dataframes
        wind_bins = np.arange(min_alt, max_alt, alt_step)
        column_headers = np.char.mod("%.1f", wind_bins / 1000.)
        wind_probabilities = pd.DataFrame(columns=column_headers)
        wind_probabilities_styled = pd.DataFrame()
        for i in range(1,31+1):

            #FIRST DO HOUR 0,  THEN DO HOUR 12.  ASSUME NO OTHER HOURS
            try:
                #places in florida have alterante hours.  I've seen 09Z, 15Z, and 18Z  but I don't want to have to check for these every time.
                date = datetime(year, j, i, 0)
            except:
                print(colored("Day + " + str(i) +  "doesn't exist in month " + str(j), "yellow"))
                continue

            print()
            print(FAA, " - ", WMO, " - ",  date)

            df, missing_data = get_sounding(WMO, date)

            if not missing_data:
                #print("missing data", i)
                #missing_date = False
                #continue

                wind_bins, opposing_wind_levels = determing_wind_statistics(df , min_alt = min_alt, max_alt = max_alt, alt_step = alt_step, n_sectors = n_sectors, speed_threshold = speed_threshold)

                #there's probably a faster way to do this with numpy.  Or maybe I should change the output of opposing_wind_levels?
                mask = wind_bins
                for k in range(len(mask)):
                    if wind_bins[k] in opposing_wind_levels:
                        mask[k] = 1
                    else:
                        mask[k] = 0

                wind_probabilities.loc[date, :] = mask

        #save_wind_probabilties(wind_probabilities, date)

            #DO IT AGAIN FOR HOUR 12!!!!!

            #FIRST DO HOUR 0,  THEN DO HOUR 12.  ASSUME NO OTHER HOURS
            #try:
            date = datetime(year, j, i, 12)
            #except:
            #    print(colored("Day + " + str(i) +  "doesn't exist in month " + str(j), "yellow"))
            #    continue

            print()
            print(FAA, " - ", WMO, " - ",  date)

            df, missing_data = get_sounding(WMO, date)

            if not missing_data:
                #print("missing data", i)
                #missing_date = False
                #continue

                wind_bins, opposing_wind_levels = determing_wind_statistics(df , min_alt = min_alt, max_alt = max_alt, alt_step = alt_step, n_sectors = n_sectors, speed_threshold = speed_threshold)

                #there's probably a faster way to do this with numpy.  Or maybe I should change the output of opposing_wind_levels?
                mask = wind_bins
                for k in range(len(mask)):
                    if wind_bins[k] in opposing_wind_levels:
                        mask[k] = 1
                    else:
                        mask[k] = 0

                wind_probabilities.loc[date, :] = mask
                #print(wind_probabilities)

        save_wind_probabilties(wind_probabilities, date)

def cumulative_batch_analysis(FAA, WMO, year, min_alt, max_alt, alt_step, n_sectors, speed_threshold):
    folder = 'SOUNDINGS_DATA/' + str(FAA) + " - " + str(WMO) + "/analysis_" + str(year) + "/dataframes/"

    files = [f for f in listdir(folder) if f.endswith(".csv")]
    #print(files)

    wind_bins = np.arange(min_alt, max_alt, alt_step)

    column_headers = np.char.mod("%.1f", wind_bins / 1000.)
    cumulative = pd.DataFrame(columns=column_headers)

    print("=========CUMULATIVE WIND PROBABILITY ANALYSIS==============\n")
    #print(cumulative)

    for csv in files:
        # read csv for each month of individual station
        try:
            df = pd.read_csv(folder + csv, index_col=0)

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
        'Opposing Wind Probabilities for Station ' + str(FAA) + " - " + str(WMO) + ' 12Z').set_table_styles([{
        'selector': 'caption',
        'props': [
            ('color', 'black'),
            ('font-weight', 'bold'),
            ('font-size', '20px')
        ]
    }])

    cumulative_styled = cumulative_styled.format(precision=2)

    filepath_image = Path(folder[:-5] + "analysis_" + str(year) + '-wind_probabilities-TOTAL.png')
    filepath_image.parent.mkdir(parents=True, exist_ok=True)
    dfi.export(cumulative_styled, filepath_image)

    filepath_dataframe = Path(folder[:-5] + "analysis_" + str(year) + '-wind_probabilities-TOTAL.csv')
    filepath_dataframe.parent.mkdir(parents=True, exist_ok=True)
    cumulative.to_csv(filepath_dataframe)


#main
if __name__=="__main__":

    continent = "South_America"
    stations_df = pd.read_csv('Radisonde_String_Parsing/CLEANED/' + continent + ".csv")
    #stations_df = stations_df.loc[stations_df["CO"] == "US"]  # Only do US Countries for now

    #stations_df = stations_df[10:]

    print(stations_df)

    for row in stations_df.itertuples(index=False):


        WMO = row.WMO
        FAA = row.FAA
        Name = row.Station_Name

        #Initialize Variables
        min_alt = 15000
        max_alt = 24000
        alt_step = 500
        n_sectors = 16
        speed_threshold = 2 #should we raise this,  it's in knots

        #Station downloads
        year = 2022
        #station = 'SKBO'
        local_download = False

        folder = 'SOUNDINGS_DATA/' + str(FAA) + " - " + str(WMO) + "/" + str(year) + "/"

        isExist = os.path.exists(folder)
        if isExist:
            print(colored("Soundings for " + str(Name) +  " - " + str(FAA) + " - " + str(WMO) + " in " + str(year) + " are downloaded locally", "green"))
            local_download = True
        else:
            print(colored("Soundings for " + str(Name) +  " - " + str(FAA) + " - " + str(WMO) + " in " + str(
                year) + " are not downloaded locally. \n Will continue to download for offline analysis", "yellow"))

        # DO THE WIND PROBABILTIES ANALYSIS
        anaylze_annual_data(FAA, WMO, year, min_alt = min_alt, max_alt = max_alt, alt_step = alt_step, n_sectors = n_sectors, speed_threshold = speed_threshold)
        cumulative_batch_analysis(FAA, WMO, year, min_alt=min_alt, max_alt=max_alt, alt_step=alt_step, n_sectors=n_sectors,speed_threshold=speed_threshold)