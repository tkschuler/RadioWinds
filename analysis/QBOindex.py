import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('../RadioWinds')
import config
import glob

import os

############################################################
# Subplot 1
# First we plot the radiosonde QBO index, taken straight from stations_df_50-stripped.csv"
############################################################

#df = pd.DataFrame(data)
df = pd.read_csv('QBO-Decadal-Means/stations_df_50-stripped' + '.csv')

# Drop non-numeric columns
df = df.drop(['index', 'WMO', 'FAA', 'Station_Name', 'Continent', 'ICAO', 'RAOB STATION NAME', 'ST', 'CO', 'EL', 'lat_era5', 'lon_era5'], axis=1)

# Convert column headers to datetime format
df.columns = pd.to_datetime(df.columns, format='u_wind_%m_%Y')

# Concert knots to m-s
knots2ms = 0.514444

df = df.map(lambda x: x * knots2ms)

df.loc['mean'] = df.mean()


fig, axs = plt.subplots(4)

axs[0].plot(df.loc['mean'])
axs[0].set_xlabel('Date')
axs[0].set_ylabel('QBO Index from \n U Winds (m/s)')
axs[0].grid()

############################################################
# Subplot 2
# Next, convert the QBO index to absolute value and color code the sections by East and West
############################################################

#Determine when QBO is positive (West) or negative (East)
positive_means = df.loc['mean'][df.loc['mean'] >= 0]
negative_means = df.loc['mean'][df.loc['mean'] < 0]

# Identify the gaps where the date difference is more than one month
date_diff = positive_means.index.to_series().diff()
gaps = date_diff[date_diff > pd.Timedelta(days=31)].index

# Create a function to plot different sections (without connecting the lines between 2 non-continuous data ranges)
def seperate_colors(means, color = 'blue', label = "East"):
    positive_means = means

    # Identify the gaps where the date difference is more than one month
    date_diff = positive_means.index.to_series().diff()
    gaps = date_diff[date_diff > pd.Timedelta(days=31)].index

    positive_date_index = positive_means.index

    # Plot the data segments separately without lines between gaps
    start_idx = positive_means.index[0]
    for gap in gaps:
        start_pos = positive_means.index.get_loc(start_idx)
        end_pos = positive_means.index.get_loc(gap)
        axs[1].plot(positive_means.index[start_pos:end_pos], positive_means[start_pos:end_pos], color=color, marker='o', label = label)
        start_idx = gap
    start_pos = positive_means.index.get_loc(start_idx)
    axs[1].plot(positive_means.index[start_pos:], positive_means[start_pos:], color=color,
             marker='o', label = label)  # Plot the last segment

seperate_colors(positive_means, 'blue', "West")
seperate_colors(abs(negative_means), 'red', "East")

axs[1].set_xlabel('Date')
axs[1].set_ylabel('QBO Index from \n U Winds (m/s)')

############################################################
# Subplot 3
# Before this next step,make sure you have run analysis_ow_analysis.py with the following variables in config.py
#   type = "PRES"
#   mode = "radiosonde"

# Then, for each station, we concatentate the probabilities over the decadal range 2012-2023. And add that to an overall list of dataframes
# Then We take the mean of the 50.0 kpa column across that array of dataframes to get the mean opposing winds probability for 50 kpa between -5 and 5 degrees latitude
############################################################

# I've hardcoded the directories for the 9 Western Hemisphere stations we have between -5 and 5 degrees latitude.

dirs = [
'SKBO - 80222',
'SKCL - 80259',
'SBMQ - 82099',
'SBBV - 82022',
'nan - 82244',
'SBMN - 82332',
'nan - 82411',
'SBFN - 82400',
'SKLT - 80398']

dfs_total = []

# For Each station directory, concatenate the annual probabilities into a decadal probability dataframe,  Then append the dataframe to the list dfs_total
for dir in dirs:
    path = config.analysis_folder + dir # path should be where opposing winds analysis are stored
    files = glob.glob(os.path.join(path, "*TOTAL.csv"))

    dfs = [] #list of annulal probabilties

    # Iterate through each file
    for file_path in files:
        # Read the file into a DataFrame
        df = pd.read_csv(file_path)

        # Set the first column as the index
        df.set_index(df.columns[0], inplace=True)
        dfs.append(df)

    dfs = pd.concat(dfs, ignore_index=False) #convert to a decadal df for the station

    # Convert the index value to datetime
    starting_year = 2012
    date_index = pd.date_range(start=f'{starting_year}-01-01', periods=len(dfs), freq='MS')
    dfs.index = date_index

    # Added the formatted Decadal station df to a list of stations between -5 and 5 to take the mean of later.
    dfs_total.append(dfs)


# Initialize an empty list to store the selected columns from each DataFrame
column_values = []

# Iterate through each DataFrame in the list
for df in dfs_total:
    # Select the column from the current DataFrame and append it to the list
    column_values.append(df['50.0'])

# Concatenate all selected columns into a single DataFrame and then take the means
combined_df = pd.concat(column_values, axis=1)
mean_values = combined_df.mean(axis=1)

axs[2].plot(mean_values.index, mean_values, color='green')
axs[2].set_ylabel('Mean Opposing \n Winds Probability')

mean_values.to_csv('mean_values.csv')


############################################################
# Subplot 4
# Similar to subplot 2,  take the average probability for each East/West QBO section and plot the color-coded repeated mean.
############################################################

def seperate_colors_means(means, probabilities, color = 'blue', label = "East"):
    positive_means = means

    # Identify the gaps where the date difference is more than one month
    date_diff = positive_means.index.to_series().diff()
    gaps = date_diff[date_diff > pd.Timedelta(days=31)].index

    # Plot the data segments separately without lines between gaps
    start_idx = positive_means.index[0]
    for gap in gaps:
        start_pos = positive_means.index.get_loc(start_idx)
        end_pos = positive_means.index.get_loc(gap)

        mean_prob = probabilities[start_idx:gap].mean()

        axs[3].plot(positive_means.index[start_pos:end_pos],[mean_prob] * len(positive_means.index[start_pos:end_pos]), color=color, marker='o', label = label)
        start_idx = gap
    start_pos = positive_means.index.get_loc(start_idx)
    axs[3].plot(positive_means.index[start_pos:],[mean_prob] * len(positive_means.index[start_pos:]), color=color,
             marker='o', label = label)  # Plot the last segment


seperate_colors_means(positive_means, mean_values, 'blue', "West")
seperate_colors_means(negative_means, mean_values, 'red', "East")
axs[3].set_ylabel('Mean Opposing Winds \n Probability Seperated and \n Averaged by Direction')

plt.show()
