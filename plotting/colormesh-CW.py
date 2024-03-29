"""
This creates a 2d colormesh by latitude and altidue for calm winds probabilities averaged monthly in 5 degree latitude intervals.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import config
import glob
import utils

font = {'size'   : 22}

plt.rc('font', **font)


overall_max = 0
overall_min = 1

continent = "North_America"
stations_df = pd.read_csv('Radiosonde_Stations_Info/CLEANED/' + continent + ".csv", index_col=1)

# Uncomment this if South America has been downloaded and Analyzed as well
#'''
continent2 = "South_America"
stations_df2 = pd.read_csv('Radiosonde_Stations_Info/CLEANED/' + continent2 + ".csv", index_col=1)

stations_df = pd.concat([stations_df, stations_df2])

stations_df = utils.convert_stations_coords(stations_df)

df_lat = stations_df.iloc[:0]

lats = {}

# lat way
for i in range (0,19+4):

    lat_range = stations_df.loc[stations_df['lat_era5'].between(i*5-55, i*5-55+5)] #.mean(numeric_only=True)
    print(lat_range)

    files = []

    for index, row in lat_range.iterrows():
        path = config.analysis_folder + str(row.FAA) + " - " + str(index)

        files += glob.glob(os.path.join(path, "*CALM.csv"))  # only include total probabilities maps

    dfs = [pd.read_csv(f, low_memory=False, index_col=0) for f in files]

    print(pd.concat(dfs).reset_index().groupby("index").mean())
    #sdfs

    # Calculate Decadal Mean for Each month in a stack of 10 annual dataframes
    decadal_mean = (
        # combine dataframes into a single dataframe
        pd.concat(dfs)
            # replace 0 values with nan to exclude them from mean calculation
            #.replace(0, np.nan)
            .reset_index()
            # group by the row within the original dataframe
            .groupby("index")
            # calculate the mean
            .mean()
    )

    #print(decadal_mean)

    # Calculate Decadal Standard Deviation for Each month in a stack of 10 annual dataframes
    decadal_var = (
        # combine dataframes into a single dataframe
        pd.concat(dfs)
            # replace 0 values with nan to exclude them from mean calculation
            .replace(0, np.nan)
            .reset_index()
            # group by the row within the original dataframe
            .groupby("index")
            # calculate the mean
            .std()
            #.rename("std")
    )

    #print(decadal_var)

    lats[i*5-55] = decadal_mean


month = lats[next(iter(lats))].iloc[:0]



#sdfs

for i in range(1,13):
    print(i)

    for key in lats:
        #print(lats[key].iloc[0])
        month.loc[key] = lats[key].loc[i]
        #print(month)
    print(month)

    calm_wind = month.to_numpy()

    levels = np.linspace(0.0, .3, 11)

    #Plotting
    #fig, ax = plt.subplots(1, 1 , figsize=(18,4))
    fig, ax = plt.subplots(1, 1, figsize=(12,10))

    im = ax.contourf(month.index, month.columns, calm_wind.T, levels=levels, cmap='rainbow', extend='max')
    fig.colorbar(im)

    Months = ['', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
              'November', 'December']
    plt.title(Months[i], fontsize=30)
    #plt.title("Decadal Calm Winds Probability Distribution - Month " + str(i)  + " [2012-2023]", fontsize=12)

    plt.ylabel('Altitude (km)', fontsize=20)
    plt.xlabel('Latitude', fontsize=20)
    plt.gca().set_ylim(bottom=20.) # wtf, why does 10=15 on the axis limits?  It's not index either?
    #plt.ylim((15,28))
    plt.tight_layout()
    plt.savefig(fname = "Pictures/Calm-Winds/Decadal-Calm-Winds-Contour-Month-" + str(i) + ".png", bbox_inches='tight')

    #plt.show()

#sdfs


#df_lat = df_lat.set_index('index')

print(df_lat)



# individual way

for dir in os.listdir(config.analysis_folder):
    #dir = "PABR - 70026"
    path = config.analysis_folder + dir
    files = glob.glob(os.path.join(path, "*CALM.csv"))  # only include total probabilities maps
    dfs = [pd.read_csv(f, low_memory=False, index_col=0) for f in files]
    print(dir)

    #Take the max of each month's opposing winds probability
    for df in dfs:
        df['max'] = df.max(axis = 1)


    #Calculate Decadal Mean for Each month in a stack of 10 annual dataframes
    decadal_mean = (
                    # combine dataframes into a single dataframe
                    pd.concat(dfs)
                    # replace 0 values with nan to exclude them from mean calculation
                    .replace(0, np.nan)
                    .reset_index()
                    # group by the row within the original dataframe
                    .groupby("index")
                    # calculate the mean
                    .mean()
                    )
    # Calculate Decadal Standard Deviation for Each month in a stack of 10 annual dataframes
    decadal_var = (
        # combine dataframes into a single dataframe
        pd.concat(dfs)
            # replace 0 values with nan to exclude them from mean calculation
            .replace(0, np.nan)
            .reset_index()
            # group by the row within the original dataframe
            .groupby("index")["max"]
            # calculate the mean
            .std()
            .rename("std")
         )

    print(decadal_mean)

    decadal_statistics = pd.concat([decadal_mean["max"], decadal_var], axis=1)
    decadal_statistics = decadal_statistics.rename(columns={"max": "mean"})

    #asdasd

    print(decadal_statistics)

    print("max:", decadal_var.to_numpy().max())
    print("min:", decadal_var.to_numpy().min())
    print()

    if decadal_var.to_numpy().max() > overall_max:
        overall_max = decadal_var.to_numpy().max()
    if decadal_var.to_numpy().min() < overall_min:
        overall_min = decadal_var.to_numpy().min()

    print("overall_max:", overall_max)
    print("overall_min:", overall_min)
    print()
    print()



    utils.export_colored_dataframes(decadal_mean,
                                    title = 'Calm Wind Probabilities MEANS for Station ' + dir + ' for 2012-2023',
                                    path = path,
                                    suffix = 'analysis-wind_probabilities-DECADAL-MEAN',
                                    export_color=False)


    utils.export_colored_dataframes(decadal_statistics,
                                    title='calm Wind Probabilities Decadal Statistics for Station ' + dir + ' for 2012-2023',
                                    path=path,
                                    suffix='analysis-wind_probabilities-DECADAL-STATISTICS',
                                    precision = 2,
                                    vmin = 0.0,
                                    vmax = 1,cmap = 'RdYlGn',
                                    export_color=False)