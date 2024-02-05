import config
from os import listdir
import pandas as pd
import numpy as np
import datetime as dt
import utils

import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, YearLocator, WeekdayLocator, DateFormatter
import matplotlib.ticker as ticker

#FAA = 'SBBV'
#WMO = 82022

#FAA = 'ABQ'
#WMO = 72365

#FAA = 'ALB'
#WMO = 72518

FAA = 'PHTO'
WMO = 91285


#FAA = 'PAYA'
#WMO = 70361

#FAA = 'PABR'
#WMO = 70026


def getDecadalMonthlyMeans(FAA, WMO):
    decadal_df = None
    for year in range (config.start_year, config.end_year +1):
        analysis_folder = utils.get_analysis_folder(FAA, WMO, year)
        files = [f for f in listdir(analysis_folder) if f.endswith(".csv")]
        print(files)
        for file in files:
            df = pd.read_csv(analysis_folder+file, index_col=0)
            df.index = pd.to_datetime(dict(year=year, month=df.index, day=1))
            if decadal_df is None:
                print("here")
                decadal_df = df.copy()
                #print("asdasd")
            else:
                decadal_df = pd.concat([decadal_df, df])
            #print(df)

    return(decadal_df)


decadal_df = getDecadalMonthlyMeans(FAA, WMO)
print(decadal_df)


opposing_wind_probability = decadal_df.to_numpy()
#Create timestamps for plotting, that match dataset
#base = dt.datetime(2012, 1, 1)
dates = decadal_df.index #[base + dt.timedelta(x,'M') for x in range(0, 144)]
monthsx, altsx = np.meshgrid(dates,decadal_df.columns)
opposing_wind_probability = opposing_wind_probability.T

#Plotting
fig, ax = plt.subplots(1, 1 , figsize=(15,5))
#im = ax.pcolormesh(decadal_df.index, decadal_df.columns, opposing_wind_probability, cmap='RdYlGn', vmin=0, vmax=1)
im = ax.contourf(decadal_df.index, decadal_df.columns, opposing_wind_probability,levels = 10, cmap='RdYlGn', vmin=0, vmax=1)
fig.colorbar(im)

plt.title("Decadal Opposing Winds Probability for " + FAA + " [2012-2023]", fontsize=12)
plt.ylabel('Altitude')
plt.xlabel('Date')

'''
# make labels centered
ax.xaxis.set_major_locator(MonthLocator())
ax.xaxis.set_minor_locator(MonthLocator(bymonth=12))

ax.xaxis.set_major_formatter(ticker.NullFormatter())
ax.xaxis.set_minor_formatter(DateFormatter('%b'))

'''
ax.xaxis.set_minor_locator(YearLocator(1))
ax.xaxis.set_minor_formatter(DateFormatter('%Y'))
for tick in ax.xaxis.get_minor_ticks():
    tick.tick1line.set_markersize(0)
    tick.tick2line.set_markersize(0)
    tick.label1.set_horizontalalignment('center')

plt.show()