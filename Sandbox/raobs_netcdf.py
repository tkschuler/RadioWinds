import xarray as xr
from datetime import datetime
import numpy as np


infile = "/home/schuler/Downloads/raob_soundings36764.cdf"
infile = "/home/schuler/Downloads/raob_soundings59511.cdf"
ds = xr.open_dataset(infile, engine="netcdf4", decode_times=False)




#print(len(ds['synTime'].values.astype(int)))
#print(ds.staName[5000])

#First convert the station name and time to coordiantes so we can do some searching of the dataset
ds = ds.set_coords(("staName", "synTime"))

print(ds)

print(ds.data_vars)

#ds2 = ds.sel(staElev=[5000])
ds['synTime'] = ds.synTime.astype('datetime64[s]')
#For organizing by month
#https://stackoverflow.com/questions/51976126/subset-an-xarray-dataset-or-data-array-in-the-time-dimension

date = datetime(2023, 11, 1, 00)

print(np.datetime64(date))

ds2 = ds.where((ds.staName == b'ITO   ') , drop = True)

#print(ds2)
ds2 = ds2.where(ds2.synTime > np.datetime64(date) , drop = True)

print(ds2)

print(ds2.synTime[dict(recNum=[0])])
print(ds2.htMan[dict(recNum=[0])])

asda
#print(ds2.wdSigT[dict(recNum=[0])])

#create a new array for individual analysis?

#station =

sdfs
#print(np.array(ds2.synTime, dtype='datetime64[s]'))

asda
small_ds = ds[["staName"]]
print(small_ds)
print()


#small_ds = small_ds.query(x=ds['staName'] == "ITO")
#print(small_ds)
print()
print();
print(small_ds.to_array())

#print(small_ds.isel(staName = [0]))

sfa

#print(ds.variables)
print()
print(ds.data_vars)
print()
print(ds.keys)
print()
print(ds['staName'])
#print(ds.loc[dict(staName="ITO")])

ds2 = ds_org = xr.tutorial.open_dataset("eraint_uvz")

print(ds2)
