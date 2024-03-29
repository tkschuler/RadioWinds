import config
from os import listdir
import pandas as pd
import numpy as np
import datetime as dt

import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, YearLocator, WeekdayLocator, DateFormatter
from dateutil.relativedelta import relativedelta
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

#Plotting Variables:
pres = 10


df = pd.read_csv('stations_df_' + str(pres) + '.csv')
df_lat = df.iloc[:0]
print(df)

for i in range (0,19+4):

    row = df.loc[df['lat_era5'].between(i*5-55, i*5-55+5)].mean(numeric_only=True)
    #print(row)
    row['index'] = i*5-55

    #df_lat = df_lat.append(row, ignore_index = True) #deprecated
    df_lat = pd.concat([df_lat, pd.DataFrame([row])], ignore_index=True)

df_lat = df_lat.set_index('index')
df_lat = df_lat.filter(regex='u_wind')


u_wind = df_lat.to_numpy()
u_wind = u_wind

base = dt.datetime(2012, 1, 1)
dates = [base + relativedelta(months=x) for x in range(0, 144)]
df_lat.columns = dates

#Plotting
fig, ax = plt.subplots(1, 1 , figsize=(18,4))

if pres == 100 or pres == 70:
    levels = np.linspace(-25, 25, 11)
else:
    levels = np.linspace(-50.0, 50.0, 11)
print(levels)

im = ax.contourf(df_lat.columns, df_lat.index, u_wind,levels = levels, cmap='BrBG_r', extend='both')
#fig.colorbar(im)

plt.title(str(pres) + " mb", fontsize=12)
plt.ylabel('Latitude')

divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "1%", pad="3%")
#im.set_clim(0.,1.)
fig.colorbar(im, cax=cax)

#plt.title(str(pres) + " mb QBO from Monthly Averages of Radiosondes launched Over Land in the Western Hemisphere [2012-2023]", fontsize=12)
#plt.xlabel('Date')

ax.xaxis.set_major_locator(YearLocator(1))
ax.xaxis.set_major_formatter(DateFormatter('%Y'))

#ax.xaxis.set_minor_locator(MonthLocator(4))
ax.xaxis.set_minor_locator(MonthLocator(bymonth=[4,7,10]))
#ax.xaxis.set_minor_formatter(DateFormatter('%M'))

plt.tight_layout()
plt.savefig("Pictures/Hovmoller/QBO/" +  str(pres) +"mb", bbox_inches='tight')
plt.show()