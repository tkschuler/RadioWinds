import pandas as pd
import sys
import config
import os
from termcolor import colored
import numpy as np

'''NOTE: THIS PROGRAM ASSUMES ALL RADIOSONDES HAVE BEEN DOWNLOADED IN THE
 PROPER ORGANIZATION STRUCTURE USING ANNUALWYOMINGDOWNLOAD.PY.  IF MONTHS ARE MISSING FROM THE DOWNLOAD
 YOU MUST RE-DOWNLOAD FOR THAT STATION OR INCLUDE EMPTY FOLDERS. OTHERWISE THIS PROGRAM WILL NOT COMPLETE.
 
 Before running you can check if everything is organized correctly with checkRadiosondeDownloads.py'''



#x = date
#y = latitude
#z = zonal windrose

# Detemine zonal wind

def get_analysis_folder(FAA, WMO, year):
    return config.analysis_folder + str(FAA) + " - " + str(WMO) + "/" + str(year) + "_analysis_ZONAL/"

def get_data_folder(FAA, WMO, year):
    return config.parent_folder + str(FAA) + " - " + str(WMO) + "/" + str(year) + "/"



def filter_pres_bins(df):
    pres_bins = [100., 70., 50., 30., 20.]

    if df is not None:
        #Only keep relevant columns for data analysis.
        df = df[df.columns[df.columns.isin(['pressure', 'speed', 'u_wind', 'v_wind'])]]
        #Some radiosonde flights have duplicate mandatory pressure levels, which creates a weird bug with averaging
        df = df.drop_duplicates(subset=['pressure'])

        #print(df)

        #Create empty dataframe to merge filtered values with, in case any pressure values were not recorded by radiosonde
        df2 = pd.DataFrame(columns=['pressure', 'speed', 'u_wind', 'v_wind'])
        df2['pressure'] = pres_bins

        # filter rows of orginial radiosonde flight based on pressure bins
        mask = df['pressure'].isin(pres_bins)
        df3 = df[mask]

        #print(df3)
        #drop duplicate rows, only keep 1


        #print(df3)

        #Create new merged dataframe that has all pressure_bins,  and Nan's for rows not recorded.
        df4 = df2.merge(df3, on='pressure', how='outer', suffixes=('_y', ''))
        df4.drop(df4.filter(regex='_y$').columns, axis=1, inplace=True)
    else:
        df4 = pd.DataFrame(columns=['pressure', 'speed', 'u_wind', 'v_wind'])
        df4['pressure'] = pres_bins

    #print(df4)

    return df4



def getZonalWinds(year, FAA, WMO):
    data_folder = get_data_folder(FAA, WMO, year)
    analysis_folder = get_analysis_folder(FAA, WMO, year)

    avg_list = []

    for j in range(1, 12 + 1):
        try:
            all_files = os.listdir(data_folder + str(j))
        except:
            print(colored(str(FAA) + " - " + str(WMO) + "/" + str(year) + " Not Downloaded.", "red"))
            return False
            #raise ValueError
        csv_files = list(filter(lambda f: f.endswith('.csv'), all_files))
        csv_files.sort() #sort the list of CSVs to have the table in the right order


        df_list = []

        #Will need to check if there is missing data.
        if csv_files:
            for csv in csv_files:
                df = pd.read_csv(data_folder + str(j) + "/" + csv, index_col = 0)

                df = filter_pres_bins(df)
                df_list.append(df)

            # Average for the month
            #print("concat", pd.concat(df_list))
            avg = pd.concat(df_list).groupby(level=0).mean() #Double check this does what I think it does
            #print("average",avg)
            avg = avg.set_index('pressure', drop=True)
            avg = avg.add_suffix('_' + str(j) + '_' + str(year))
            avg_list.append(avg)
            #print(avg)

        else:
            avg = filter_pres_bins(None)
            avg = avg.set_index('pressure', drop=True)
            avg = avg.add_suffix('_' + str(j) + '_' + str(year))
            avg_list.append(avg)
            #print(avg)




    avg_annual = pd.concat(avg_list, axis=1)


    return avg_annual
                #100, 70, 50, 30, 20



#Main
if __name__=="__main__":
    #continent = config.continent
    #stations_df = pd.read_csv('Radisonde_Stations_Info/CLEANED/' + continent + ".csv")

    continent = "North_America"
    stations_df = pd.read_csv('Radisonde_Stations_Info/CLEANED/' + continent + ".csv")
    # stations_df = stations_df.loc[stations_df["CO"] == "US"]

    continent2 = "South_America"
    stations_df2 = pd.read_csv('Radisonde_Stations_Info/CLEANED/' + continent2 + ".csv")

    stations_df = pd.concat([stations_df, stations_df2])
    stations_df = stations_df.drop_duplicates(subset=['WMO'])
    stations_df = stations_df.reset_index()

    #print(stations_df)

    stations_df['lat_era5'] = stations_df.apply(lambda x: (-1 * x['  LAT'] if x['N'] == 'S' else 1 * x['  LAT']),
                                                axis=1)
    stations_df['lon_era5'] = stations_df.apply(lambda x: (-1 * x[' LONG'] if x['E'] == 'W' else 1 * x[' LONG']),
                                                axis=1)

    #pres_bins = [100., 70., 50., 30., 20.]
    #for pres in pres_bins:







    pres = 70
    print(stations_df)
    #print(stations_df)

    #stations_df= stations_df[:2]

    stations_df_pres = stations_df.copy()

    for year in range(config.start_year, config.end_year + 1):
        for i, row in enumerate(stations_df.itertuples(index=False)):
            print("Averaging", i, row.WMO, row. FAA, "-", year, "-", pres)
            avg_annual = getZonalWinds(year,
                            WMO = row.WMO,
                            FAA = row.FAA)
            #print(avg_annual)

            #only need to do this the first time a new year
            if i == 0:
                stations_df_pres[avg_annual.columns.tolist()] = len(avg_annual.columns.tolist()) * [np.nan]
                #print(stations_df_pres)

            #row_pres = stations_df.loc[i].append(avg_annual.loc[pres])
            row_pres = pd.concat([stations_df.loc[i],avg_annual.loc[pres]])
            stations_df_pres.loc[i] = row_pres
            #print(stations_df_pres)

        print(stations_df_pres)
        stations_df = stations_df_pres.copy()
        #asda

    stations_df_pres.to_csv('stations_df_' + str(pres) + '.csv', index=False)