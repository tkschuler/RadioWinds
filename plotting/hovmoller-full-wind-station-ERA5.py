import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import config
from matplotlib.colors import LinearSegmentedColormap, rgb_to_hsv, hsv_to_rgb
import utils
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
from matplotlib.dates import YearLocator,  DateFormatter
import os
import pandas as pd

FAA = "SCCI"
WMO = utils.lookupWMO(FAA)
Station_Name = utils.lookupStationName(FAA)
CO = utils.lookupCountry(FAA)
lat,lon,el = utils.lookupCoordinate(FAA)
print(WMO, FAA, Station_Name, lat,lon,el)


def brighten_and_saturate_colormap(cmap, brightness_factor=1.5, saturation_factor=1.5):
    """
    Adjust both brightness and saturation of a colormap in HSV space.

    Parameters:
    - cmap: The original colormap to adjust.
    - brightness_factor: A multiplier for the brightness (default is 1.5).
    - saturation_factor: A multiplier for the saturation (default is 1.5).

    Returns:
    - A new colormap with enhanced brightness and saturation.
    """
    colors = cmap(np.linspace(0, 1, 256))  # Sample the colormap
    # Convert RGB to HSV
    hsv_colors = rgb_to_hsv(colors[:, :3])  # Ignore alpha channel
    # Scale brightness (Value) and saturation
    hsv_colors[:, 1] = np.clip(hsv_colors[:, 1] * saturation_factor, 0, 1)  # Saturation
    hsv_colors[:, 2] = np.clip(hsv_colors[:, 2] * brightness_factor, 0, 1)  # Brightness
    # Convert back to RGB
    adjusted_colors = hsv_to_rgb(hsv_colors)
    # Create a new colormap
    return LinearSegmentedColormap.from_list('bright_saturated_' + cmap.name, adjusted_colors)


def plot_hovmoller_wind_direction(nc_file, lat, lon):
    """
    Generates a Hovmöller plot of wind direction for a specific lat/lon coordinate over time.

    Args:
        nc_file (str): Path to the NetCDF file.
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.

    Returns:
        None (Displays the plot)
    """
    # Open the NetCDF file
    ds = xr.open_dataset(nc_file)

    if 'plev' in ds.coords:
        print("FIXING LON")
        ds = ds.assign_coords(lon=ds.lon - 360)
        print(ds)

    else:
        print("'plev' is not found in the dataset!")

    print(ds)

    # Select nearest grid point to the specified latitude and longitude
    ds_sel = ds.sel(lat=lat, lon=lon, method="nearest")

    # Extract variables
    time = ds_sel.time.values
    pressure_levels = ds_sel.plev.values  # Assuming 'plev' represents pressure levels
    z = ds_sel.z/9.81  # Geopotential height (assumed in meters)
    u = ds_sel.u  # U-wind component
    v = ds_sel.v  # V-wind component


    # Filter for altitudes between 15 km and 28 km
    altitude_mask = (z >= config.min_alt-10) & (z <= config.max_alt+10)
    z_filtered = z.where(altitude_mask, drop=True)
    u_filtered = u.where(altitude_mask, drop=True)
    v_filtered = v.where(altitude_mask, drop=True)

    # Compute wind direction in degrees (meteorological convention: add 180°)
    wind_direction = (180 / np.pi) * np.arctan2(u_filtered, v_filtered)  # Calculate wind direction
    wind_direction = (wind_direction + 180) % 360  # Convert to 0-360° and invert

    #altitude_bins = np.arange(15000, 28500, 500)
    #z_mean = altitude_bins
    # Use `z_filtered` as the y-axis (height in meters)
    z_mean = z_filtered.mean(dim="time").values  # Average geopotential height over time for plotting

    # Convert to 2D array for plotting (time x pressure level)
    wind_direction_2D = wind_direction.T  # Transpose so pressure is on the y-axis

    df = pd.DataFrame(wind_direction_2D.T.values, index=time, columns=z_mean)
    path = "Pictures/Data-Hovmoller/"
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
    df.to_csv(path + str(FAA) + "-" + str(config.start_year) + "-era5.csv")
    print(df)


    import colorcet as cc

    bs_csm = brighten_and_saturate_colormap(cc.cm.CET_C6s,
                                            brightness_factor=1.25,
                                            saturation_factor=1.25)

    # Create Hovmöller plot
    fig, ax = plt.subplots(1, 1 , figsize=(18,3))
    im = ax.contourf(time, z_mean, wind_direction_2D, levels=np.linspace(0, 360., 19),
                     cmap=bs_csm)

    if lat >= 0:
        plt.title(Station_Name + "- " + CO + " (Station #" + str(WMO).zfill(5) + ") - " + str(int(lat)) +"$^\circ$N", fontsize=12)
    else:
        plt.title(Station_Name + "- " + CO + " (Station #" + str(WMO).zfill(5) + ") - " + str(int(-1*lat)) +"$^\circ$N", fontsize=12)

            #"Salt Lake City, Utah USA (40$^\circ$N)" +
            #      "\nWind Directionality (ERA5) for Station #" + str(WMO).zfill(5) +  " in " + str(config.start_year), fontsize=13)

    plt.ylabel('Altitude (m)')
    plt.xlabel('Date')

    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "1%", pad="1%")
    im.set_clim(0., 360.)
    cbar = fig.colorbar(im, cax=cax, boundaries=np.linspace(0, 360, 13))

    cbar.set_label("Wind Direction (degrees)", labelpad=10)

    ax.xaxis.set_minor_locator(YearLocator(1))
    ax.xaxis.set_minor_formatter(DateFormatter('%Y'))
    for tick in ax.xaxis.get_minor_ticks():
        tick.tick1line.set_markersize(0)
        tick.tick2line.set_markersize(0)
        tick.label1.set_horizontalalignment('center')

    for tick in ax.yaxis.get_major_ticks()[::2]:
        tick.set_visible(False)

    #y_ticks = [16000, 20000, 24000, 28000]

    # Set the y-axis ticks on the subplot
    #ax.set_yticks(y_ticks)

    fig.tight_layout()
    plt.tight_layout()
    plt.margins(0.1)






# Example usage
nc_file = config.era_file # Replace with the actual NetCDF file path
#lat = 40.76  # Replace with desired latitude
#lon = -111.9  # Replace with desired longitude

plot_hovmoller_wind_direction(nc_file, lat, lon)

#fig.tight_layout()
path = "Pictures/Hovmoller-Full-Winds-ERA5/"
isExist = os.path.exists(path)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(path)
#plt.savefig("Pictures/Hovmoller/" +  str(FAA), bbox_inches='tight')
plt.savefig(path +  str(FAA) + "-" + str(config.start_year), bbox_inches='tight')
plt.show()
