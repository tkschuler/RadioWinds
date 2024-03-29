import config
from os import listdir
import os
import pandas as pd
import numpy as np
import datetime as dt
import utils
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, YearLocator, WeekdayLocator, DateFormatter
import matplotlib.ticker as ticker

FAA = "SCCI"
WMO = utils.lookupWMO(FAA)
print(WMO)

def getDecadalMonthlyMeans(FAA, WMO):
    decadal_df = None
    #for year in range (config.start_year, config.end_year +1):
    analysis_folder = utils.get_analysis_folder(FAA, WMO, 2023)
    analysis_folder = os.path.dirname(os.path.dirname(analysis_folder)) + "/" #Go up one level to get annual
    #analysis_folder = analysis_folder[:-13]  #go up one directory
    files = [f for f in listdir(analysis_folder) if f.endswith(".csv")]
    print(analysis_folder)
    print(files)
    for file in files:
        year = int(file[9:-29])
        print(year)
        #sdfs
        df = pd.read_csv(analysis_folder+file, index_col=0)
        #df = df.drop(['average'])
        #print(df)
        #df.index = pd.to_datetime(dict(year=year, month=df.index, day=1))
        #df.index = pd.to_datetime(df.index)
        df.index = pd.to_datetime(dict(year=year, month=df.index, day=1))

        #print(df)

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
fig, ax = plt.subplots(1, 1 , figsize=(18,3))
#im = ax.pcolormesh(decadal_df.index, decadal_df.columns, opposing_wind_probability, cmap='RdYlGn', vmin=0, vmax=1)
im = ax.contourf(decadal_df.index, decadal_df.columns, opposing_wind_probability, levels=np.linspace(0, 1., 11), cmap='RdYlGn', vmin=0., vmax=1.)

#plt.title("Fairbanks, Alaska USA (65$^\circ$N)" +
#plt.title("Pittsburgh, Pennsylvania USA (40$^\circ$N)" +
#plt.title("Hilo, Hawaii USA (15$^\circ$N)" +
plt.title("Punta Arenas, Chile (53$^\circ$S)" +
          "\nDecadal Opposing Winds Probability for " + FAA +  " [2012-2023]", fontsize=12)
plt.ylabel('Altitude')
plt.xlabel('Date')
#fig.colorbar(im)


divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "1%", pad="3%")
#plt.colorbar(im, cax=cax)
#im.set_array(opposing_wind_probability)
im.set_clim(0.,1.)
fig.colorbar(im, cax=cax, boundaries=np.linspace(0, 1, 11))



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

for tick in ax.yaxis.get_major_ticks()[::2]:
    tick.set_visible(False)


fig.tight_layout()
plt.tight_layout()
plt.margins(0.1)
#plt.bbox_inches='tight'
plt.savefig("Pictures/Hovmoller/" +  str(FAA), bbox_inches='tight')
plt.show()