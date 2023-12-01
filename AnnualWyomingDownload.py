from siphonMulti import SiphonMulti
import pandas as pd
import os
from termcolor import colored
from pathlib import Path

from multiprocessing import Process, Manager
import config

def save_monthly_soundings(df_monthly_list, FAA, WMO, year, month):

    folder = config.parent_folder + str(FAA) + " - " + str(WMO) + "/" + str(year) + "/" + str(month) + "/"

    # This takes care of empty months.
    isExist = os.path.exists(folder)
    if not isExist:
        os.makedirs(folder)

    if df_monthly_list is not None:
        for df in df_monthly_list:
            date = df.time[0]

            filepath_sounding = Path(folder + str(FAA) + "-" + str(date.year) + "-" + str(
                date.month) + "-" + str(date.day) + "-" + str(date.hour) + '.csv')
            filepath_sounding.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(filepath_sounding)


def get_yearly_soundings(FAA, WMO, year):
    #Check if Soundings Data Folder exists:

    folder = config.parent_folder + str(FAA) + " - " + str(WMO) + "/" + str(year) + "/"

    # Check if the soundings have already been downloaded.
    # not checking incomplete downloads yet.
    isExist = os.path.exists(folder)
    if isExist:
        print(colored("Soundings for " + str(FAA) + " - " + str(FAA) + " - " + str(WMO) + " in " + str(
            year) + " are downloaded locally", "green"))
        local_download = True
    else:
        print(colored("Soundings for " + str(FAA) + " - " + str(FAA) + " - " + str(WMO) + " in " + str(
            year) + " are not downloaded locally. \n Will continue to download for offline analysis", "yellow"))


        yearly_count = 0
        #Iterate by Month
        for i in range(1,12+1):

            df_monthly_list = None
            count = 0
            missing_data = False

            while df_monthly_list is None:
                try:
                    df_monthly_list = SiphonMulti.request_data(year=year, month=i, site_id=WMO)
                except ValueError as ve:
                    # I have never seen this error actually trigger with monthly downloads,  only individual sounging downloads in the standard siphon.
                    print(colored(ve, "red"))
                    missing_data = True
                    df = pd.DataFrame()
                    pass
                except:
                    #keep trying until the data is downloaded.  The highest server error I've seen is 500, but in the end everything shoudl download.
                    count += 1
                    df = None
                    if count %50 == 0:
                        print("SERVER ERRORS", count, "TRIES - ", FAA, WMO, "month", i)
                        missing_data = True

            save_monthly_soundings(df_monthly_list, FAA, WMO, year, month = i)

            if not missing_data:

                result = ('Number of Monthly Soundings for '
                        '{FAA}/{WMO} in {month}/{year} is {num_of_soundings}').format(FAA=FAA, WMO = WMO, year = year, month = i, num_of_soundings = len(df_monthly_list))

                #print(result)
                yearly_count += len(df_monthly_list)

        print("Total number of annual soundings download for", FAA, "-", WMO, "in", year, ":", yearly_count)


def parallel_something(stations_df, year):
    '''
    Parralel the process.  each station downlaod goes on it's own thread.  THis make the downloads much faster.

    Args:
        stations_df:
        year:

    Returns:

    '''

    queue = Manager().Queue()
    procs = [Process(target=get_yearly_soundings, args=(row.FAA, row.WMO, year)) for row in stations_df.itertuples(index=False)]
    for p in procs: p.start()
    for p in procs: p.join()

    results = []
    while not queue.empty():
        results.append(queue.get)

    return results

if __name__=="__main__":

    continent = "Antarctica"
    stations_df = pd.read_csv('Radisonde_Stations_Info/CLEANED/' + continent + ".csv")
    #stations_df = stations_df.loc[stations_df["CO"] == "US"]  # Only do US Countries for now
    #stations_df = stations_df.drop_duplicates(subset=['FAA'])


    print(stations_df)


    for i in range (2012, 2023+ 1):
        parallel_something(stations_df, year = i)
        #for row in stations_df.itertuples(index=False):
        #    get_yearly_soundings(row.FAA, row.WMO, year = i)
        #    #Do not move onto the next year until all the processes have finished.
        print(colored("MOVING ON TO YEAR" + str(i), "cyan"))
