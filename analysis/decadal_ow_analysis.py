'''
This script analyzes the decadal mean and standard deviation for opposing wind probabilities.

Soundings data needs to already be downloaded (AnnualWyomingDownload.py) and analyzed (batchAnalysis.py).

'''

import pandas as pd
import numpy as np
import os
import sys
sys.path.append('../RadioWinds')
import config
import glob
import utils

overall_max = 0
overall_min = 1

for dir in os.listdir(config.analysis_folder):
    path = config.analysis_folder + dir # path should be where opposing winds analysis is stored
    print(path)
    files = glob.glob(os.path.join(path, "*TOTAL.csv"))  # only include total probabilities maps
    dfs = [pd.read_csv(f, low_memory=False, index_col=0) for f in files]
    print(dir)

    #Take the max of each month's opposing winds probability
    for df in dfs:
        df['max'] = df.max(axis = 1)

    print(dfs)

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

    decadal_statistics = pd.concat([decadal_mean["max"], decadal_var], axis=1)
    decadal_statistics = decadal_statistics.rename(columns={"max": "mean"})

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
                                    title = 'Opposing Wind Probabilities MEANS for Station ' + dir + ' for 2012-2023',
                                    path = path,
                                    suffix = 'analysis-wind_probabilities-DECADAL-MEAN',
                                    export_color=False)


    utils.export_colored_dataframes(decadal_statistics,
                                    title='Opposing Wind Probabilities Decadal Statistics for Station ' + dir + ' for 2012-2023',
                                    path=path,
                                    suffix='analysis-wind_probabilities-DECADAL-STATISTICS',
                                    precision = 2,
                                    vmin = 0.0,
                                    vmax = 1,cmap = 'RdYlGn',
                                    export_color=False)