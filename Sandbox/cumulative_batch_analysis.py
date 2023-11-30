from datetime import datetime
from siphon.simplewebservice.wyoming import WyomingUpperAir
import pandas as pd
from termcolor import colored
import windrose
import numpy as np
from pathlib import Path
import dataframe_image as dfi

from os import listdir

station = 'IAD'
year = 2023

folder = 'SOUNDINGS_DATA/' + str(station) + "/analysis_" + str(year) + "/dataframes/"


#folder = str(station) +"/dataframes/"

files= [f for f in listdir(folder) if f.endswith(".csv")]

print(files)

min_alt = 15000
max_alt = 24000
alt_step = 500
n_sectors = 16
speed_threshold = 2
wind_bins = np.arange(min_alt, max_alt, alt_step)

column_headers = np.char.mod("%.1f", wind_bins/1000.)
cumulative = pd.DataFrame(columns=column_headers)

print(cumulative)

for csv in files:
    #read csv for each month of individual station
    try:
        df = pd.read_csv(folder + csv, index_col=0)

        str_date = df.iloc[0:1].index.values[0]
        date = datetime.strptime(str_date, '%Y-%m-%d %H:%M:%S')

        cumulative.loc[date.month, :] = df.iloc[-1:].values
    except:
        continue
cumulative.sort_index(inplace = True, ascending = True)
print(cumulative)

cumulative = cumulative.apply(pd.to_numeric)
cumulative_styled = cumulative.style.background_gradient(axis=None, vmin=0, vmax=1.0, cmap = 'RdYlGn')

cumulative_styled = cumulative_styled.set_caption('Opposing Wind Probabilities for Station ' + station + ' 12Z').set_table_styles([{
    'selector': 'caption',
    'props': [
        ('color', 'black'),
        ('font-weight', 'bold'),
        ('font-size', '20px')
    ]
}])

cumulative_styled = cumulative_styled.format(precision = 2)

filepath_image = Path(folder[:-5] + "analysis_" + str(year) + '-wind_probabilities-TOTAL.png')
filepath_image.parent.mkdir(parents=True, exist_ok=True)
dfi.export(cumulative_styled, filepath_image )

filepath_dataframe = Path(folder[:-5] + "analysis_" + str(year) + '-wind_probabilities-TOTAL.csv')
filepath_dataframe.parent.mkdir(parents=True, exist_ok=True)
cumulative.to_csv(filepath_dataframe)