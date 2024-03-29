'''Combine netcdf files, where the only difference is the variables downloaded. This won't work for different
coordinate systems.  It may work for different periods of time?

U-wind, V-wind, and Geopotential are all downloaded seperated for the region, pressure levels, and time period
of interest.

https://unseen-open.readthedocs.io/_/downloads/en/latest/pdf/

Recommended,  1 Hemisphere, 1 year, Pressure levels 300-10mb

Additional Info: https://docs.xarray.dev/en/stable/user-guide/dask.html
'''

import xarray
from dask.diagnostics import ProgressBar

#print(xarray.__version__)

#ds = xarray.open_mfdataset('forecasts/Western-Hemisphere-2023*.nc', combine='by_coords', chunks={"time": 10}, engine = "netcdf4")
#ds = xarray.open_mfdataset('Western-Hemisphere-2023*.nc', chunks={"time": 10})

x1 = xarray.open_dataset('forecasts/1.nc', chunks={"time": 10}, engine='netcdf4')
print(x1)

x2 = xarray.open_dataset('forecasts/2.nc', chunks={"time": 10}, engine='netcdf4')
print(x2)

x3 = xarray.open_dataset('forecasts/3.nc', chunks={"time": 10}, engine='netcdf4')
print(x3)

#asda

x4 = xarray.combine_by_coords([x1, x2, x3], combine_attrs='drop_conflicts')

print(x4)

#check if expver exists, for latest data:
x4 = x4.sel(expver=1).combine_first(x4.sel(expver=5))
print(x4)



#'''
# This takes about 30 minutes,  creating a 30 gig + file.
write_job = x4.to_netcdf("forecasts/COMBINED-Western-Hemisphere-2022.nc", compute=False, engine='netcdf4')
with ProgressBar():
    print(f"Writing to outfile")
    write_job.compute()
print("done")
#'''
