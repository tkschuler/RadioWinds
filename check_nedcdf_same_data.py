import xarray as xr
import numpy as np


filepath = "forecasts/" + "western_hemisphere-2022-North.nc"
ds1 = xr.open_dataset(filepath, engine="netcdf4", decode_times=True)
print(ds1)

filepath = "../../../../mnt/d/FORECASTS/" + "optimized_ERA5-2022-WH.nc"
ds2 = xr.open_dataset(filepath, engine="netcdf4", decode_times=True)

#ds2 = ds2.rename({'valid_time': 'time', 'pressure_level': 'level'})
#ds2 = ds2.reindex(level=ds2.level[::-1])
print(ds2)

#filepath = "../../../../mnt/d/cds_api/" + "2022-ERA5-Complete.nc"
#ds = xr.open_dataset(filepath, engine="netcdf4", decode_times=True)
#print(ds)

# Step 1: Find the overlapping coordinate ranges for longitude, latitude, level, time
overlap_lon = np.intersect1d(ds1.longitude.values, ds2.longitude.values)
overlap_lat = np.intersect1d(ds1.latitude.values, ds2.latitude.values)
overlap_level = np.intersect1d(ds1.level.values, ds2.level.values)
overlap_time = np.intersect1d(ds1.time.values, ds2.time.values)

print(overlap_lon)
print(overlap_lat)
print(overlap_time)
print(overlap_level)

# Step 2: Subset both datasets to the overlapping region
ds1_overlap = ds1.sel(longitude=overlap_lon, latitude=overlap_lat, level=overlap_level, time=overlap_time)
ds2_overlap = ds2.sel(longitude=overlap_lon, latitude=overlap_lat, level=overlap_level, time=overlap_time)

print(ds1_overlap)
print(ds2_overlap)

print(ds1_overlap.z.values[0,:,90,0])
print(ds2_overlap.z.values[0,:,90,0])
sdfsdf

# Step 3: Compare variables (e.g., "z", "u", "v", etc.)
# For example, comparing variable 'z' from both datasets
z_equal = np.allclose(ds1_overlap.z.values, ds2_overlap.z.values, atol=1e-5)

print(f"Are the 'z' variable values the same in the overlapping region? {z_equal}")