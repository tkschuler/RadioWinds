from datetime import datetime
import pandas as pd
from termcolor import colored
import numpy as np
from pathlib import Path
import dataframe_image as dfi
import os
from os import listdir
import opposing_wind_wyoming

'''
def get_soundings(FAA, WMO, year):
    data_folder = 'SOUNDINGS_DATA2/' + str(FAA) + " - " + str(WMO) + "/" + str(year) + "/"
    sub_folders = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
    print(sub_folders)
'''

def determing_wind_statistics(df , min_alt, max_alt, alt_step, n_sectors, speed_threshold):
    wind_bins = np.arange(min_alt, max_alt, alt_step)

    # Do some filtering of the dataframe
    # Only analyze between 10-25km  as well as speeds over 2m/s is the default for now
    #df.dropna(inplace=True)

    df.dropna(subset=['height'], how='all', inplace=True)
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

def save_wind_probabilties(wind_probabilities, analysis_folder, date):

    wind_probabilities = pd.concat([wind_probabilities, wind_probabilities.apply(['average'])])

    print(colored(
        "Processing data for Station-" + str(FAA) + " - " + str(WMO) + "    Year-" + str(date.year) + "    Month-" + str(date.month),
        "cyan"))

    wind_probabilities = wind_probabilities.apply(pd.to_numeric)
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

    data_folder = 'SOUNDINGS_DATA2/' + str(FAA) + " - " + str(WMO) + "/" + str(year) + "/"
    analysis_folder = 'SOUNDINGS_DATA2/' + str(FAA) + " - " + str(WMO) + "/" + str(year) + "_analysis/"

    #Check if sounding data has already been analyzed.  If so, skip
    if check_monthly_analyzed(analysis_folder):
        return True

    #Iterate by month and day for a particular year.
    for j in range (1,12+1):
        #reinitialize dataframes
        wind_bins = np.arange(min_alt, max_alt, alt_step)
        column_headers = np.char.mod("%.1f", wind_bins / 1000.)
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

                wind_bins, opposing_wind_levels = determing_wind_statistics(df, min_alt=min_alt, max_alt=max_alt,
                                                                            alt_step=alt_step, n_sectors=n_sectors,
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

                    #date = datetime(year, j, i, 12)
                    #date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
                    print()
                    print(date)
                    #asda

                    wind_probabilities.loc[date, :] = mask
                except:
                    print(colored("GOT AN EXCEPTION","yellow"))
                    pass

        else:
            print(colored("MADE IT HERE", "magenta"))
            date = datetime(year, j, 1, 00)  #day and time shouldn't matter
            mask = np.empty(len(wind_bins))
            mask[:] = np.nan
            wind_probabilities.loc[date, :] = mask

            print(wind_probabilities)

        save_wind_probabilties(wind_probabilities, analysis_folder, date)

        return True


def cumulative_batch_analysis(FAA, WMO, year, min_alt, max_alt, alt_step, n_sectors, speed_threshold):
    print("=========CUMULATIVE WIND PROBABILITY ANALYSIS==============\n")

    analysis_folder = 'SOUNDINGS_DATA2/' + str(FAA) + " - " + str(WMO) + "/" + str(year) + "_analysis/"

    files = [f for f in listdir(analysis_folder + "dataframes") if f.endswith(".csv")]

    wind_bins = np.arange(min_alt, max_alt, alt_step)

    column_headers = np.char.mod("%.1f", wind_bins / 1000.)
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

    #Put the cumulative stuff in 2 spots for easy visual inspection.
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

def check_annual_analyzed(FAA, WMO, year):
    '''

    Helper Function to see if annual sounding data has been analyzed yet

    '''

    analysis_folder = 'SOUNDINGS_DATA2/' + str(FAA) + " - " + str(WMO) + "/" + str(year) + "_analysis/"
    file = analysis_folder[:-14] + "analysis_" + str(year) + '-wind_probabilities-TOTAL.csv'

    isExist = os.path.exists(file)
    if not isExist:
        print(colored(file, "yellow"))
        return False

    print(colored(file, "green"))
    return True


#mMain
if __name__=="__main__":

    continent = "Antarctica"
    stations_df = pd.read_csv('Radisonde_Stations_Info/CLEANED/' + continent + ".csv")
    #stations_df = stations_df.loc[stations_df["CO"] == "US"]  # Only do US Countries for now

    #stations_df = stations_df.loc[stations_df["FAA"] == "PHLI"]

    #stations_df = stations_df[2:]

    print(stations_df)

    #sdfs

    for row in stations_df.itertuples(index=False):


        WMO = row.WMO
        FAA = row.FAA
        Name = row.Station_Name

        #WMO = 72206
        #FAA = 'JAX'
        #Name = "Somewhere in Alaska?"

        #Initialize Variables
        min_alt = 15000
        max_alt = 24000
        alt_step = 500
        n_sectors = 16
        speed_threshold = 2 #should we raise this,  it's in knots

        #Station downloads
        year = 2013
        #station = 'SKBO'
        local_download = False
        local_download = False

        folder = 'SOUNDINGS_DATA2/' + str(FAA) + " - " + str(WMO) + "/" + str(year) + "/"
        #print(folder)

        isExist = os.path.exists(folder)
        if isExist:
            print(colored("Soundings for " + str(Name) +  " - " + str(FAA) + " - " + str(WMO) + " in " + str(year) + " are downloaded locally", "green"))
            local_download = True
        else:
            print(colored("Soundings for " + str(Name) +  " - " + str(FAA) + " - " + str(WMO) + " in " + str(
                year) + " are not downloaded locally. \n Will continue to download for offline analysis", "yellow"))

        #get_soundings(FAA, WMO, year)

        #sdfs

        # DO THE WIND PROBABILTIES ANALYSIS

        for year in range (2012, 2023 +1):
            status = anaylze_annual_data(FAA, WMO, year, min_alt = min_alt, max_alt = max_alt, alt_step = alt_step, n_sectors = n_sectors, speed_threshold = speed_threshold)

            if not status:
                print(colored("Can't perform cumulative analysis because + " + str(FAA) + "/" + str(WMO) + "/" + str(year) + " not downloaded.", "red"))

            if not check_annual_analyzed(FAA, WMO, year) and status:
                cumulative_batch_analysis(FAA, WMO, year, min_alt=min_alt, max_alt=max_alt, alt_step=alt_step, n_sectors=n_sectors,speed_threshold=speed_threshold)
