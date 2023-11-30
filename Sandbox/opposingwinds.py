import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import matplotlib as mpl

def find_opposing_wind_ranges(df, threshold_speed, threshold_angle):
    """
    Find altitude ranges with opposing wind pairs at different altitudes.

    Args:
    - df: A DataFrame containing wind data with columns "Altitude", "Wind Speed", and "Wind Direction".
    - threshold_speed: Wind speed threshold for considering opposing winds.
    - threshold_angle: Maximum angle (in degrees) for two winds to be considered opposing.

    Returns:
    - A list of altitude ranges with opposing winds.
    Each range is represented as a tuple (start_altitude, end_altitude).
    """

    # Sort the DataFrame by altitude
    df = df.sort_values(by="Altitude")

    # Initialize a list to store altitude ranges with opposing winds
    opposing_wind_ranges = []

    # Get the altitude and wind data as arrays
    altitudes = df["Altitude"].values
    wind_speeds = df["Wind Speed"].values
    wind_directions = df["Wind Direction"].values

    # Iterate through altitude ranges
    num_altitudes = len(altitudes)
    for i in range(num_altitudes - 1):
        for j in range(i + 1, num_altitudes):
            # Initialize variables to track opposing wind pairs in the range
            opposing_pair_found = False
            start_altitude = altitudes[i]

            # Iterate through measurements within the altitude range
            #for m in range(len(wind_speeds)):
            speed_i = wind_speeds[i]
            direction_i = wind_directions[i]
            speed_j = wind_speeds[j]
            direction_j = wind_directions[j]

            # Calculate the absolute angular difference between the two directions
            angular_difference = abs(direction_i - direction_j)
            #print(start_altitude, altitudes[j], (direction_i, direction_j), angular_difference, 180 + threshold_angle)

            # Ensure that the angular difference is within the threshold
            if angular_difference <= 180 + threshold_angle and angular_difference >= 180 - threshold_angle:
                #print("made it here")
                #if not opposing_pair_found:
                    # Set the start altitude of the range
                    #start_altitude = altitudes[i]
                opposing_pair_found = True
                #print(opposing_pair_found)

            #print(opposing_pair_found)
            # If an opposing pair was found, add the range to the list
            if opposing_pair_found:
                #print("made it here 2")
                end_altitude = altitudes[j]
                opposing_wind_ranges.append((start_altitude, end_altitude))

    return opposing_wind_ranges

'''
# Example usage:
data = {
    "Altitude": [0, 100, 200, 300, 400],
    "Wind Speed": [10, 12, 8, 9, 14],
    "Wind Direction": [120, 320, 60, 150, 290]
}

df = pd.DataFrame(data)
'''


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing as p

def add_rotated_wind_components(df, rotation_angle=90):
    """
    Add x and y components of the wind to a DataFrame with altitude, wind direction, and wind speed.
    Optionally, rotate the wind components by a specified angle (default is 90 degrees).

    Args:
    - df: Input DataFrame with columns "Altitude", "Wind Direction" (in degrees), and "Wind Speed" (in m/s).
    - rotation_angle: Angle in degrees by which to rotate the wind components (default is 90 degrees).

    Returns:
    - DataFrame with added columns "Wind X Component" and "Wind Y Component."
    """
    # Convert rotation angle to radians
    rotation_rad = np.radians(rotation_angle)

    # Calculate x and y components of the wind
    wind_y_component = df["Wind Speed"] * np.cos(np.radians(df["Wind Direction"])) #np.sin(np.radians(df["Wind Direction"])) * np.cos(rotation_rad) + df["Wind Speed"] * np.cos(np.radians(df["Wind Direction"])) * np.sin(rotation_rad)
    wind_x_component = df["Wind Speed"] * np.sin(np.radians(df["Wind Direction"])) #np.sin(np.radians(df["Wind Direction"])) * np.sin(rotation_rad) - df["Wind Speed"] * np.cos(np.radians(df["Wind Direction"])) * np.cos(rotation_rad)

    df["Wind X Component"] = wind_x_component
    df["Wind Y Component"] = wind_y_component

    return df

def convert_angles(arr):
    """
    Convert angles in a NumPy array from [0, 360] to [-180, 180].

    Args:
    - arr: Input NumPy array containing angles in degrees.

    Returns:
    - NumPy array with angles converted to the range [-180, 180].
    """
    # Subtract 180 degrees to shift the range to [-180, 180]
    converted_arr = arr - 360

    # Wrap angles that fall outside the range [-180, 180]
    converted_arr = (converted_arr + 180) % 360 - 180

    return converted_arr

def polar_interpolated_scatter_plot(df, num_interpolations=500):
    """
    Create a polar scatter plot with altitude as the radius, wind direction in degrees, and a color bar for wind speed.

    Args:
    - df: A DataFrame containing wind data with columns "Altitude", "Wind Speed", and "Wind Direction" in degrees.
    - num_interpolations: Number of altitude interpolations between each altitude level (default is 20).
    """

    # Extract data from the DataFrame
    altitudes = df["Altitude"].values
    wind_speeds = df["Wind Speed"].values
    wind_directions_deg = df["Wind Direction"].values  # Wind direction in degrees

    # Create interpolated altitudes and corresponding wind data
    interpolated_altitudes = []
    interpolated_speeds = []
    interpolated_directions_deg = []


    for i in range(len(altitudes) - 1):

        #Do some angle wrapping checks
        interp_dir_deg = 0
        angle1 = wind_directions_deg[i] %360
        angle2 = wind_directions_deg[i + 1] %360
        angular_difference = abs(angle2-angle1)

        if angular_difference > 180:
            if (angle2 > angle1):
                angle1 += 360
            else:
                angle2 += 360


        for j in range(num_interpolations + 1):
            alpha = j / num_interpolations
            interp_alt = altitudes[i] + alpha * (altitudes[i + 1] - altitudes[i])
            interp_speed = np.interp(interp_alt, [altitudes[i], altitudes[i + 1]], [wind_speeds[i], wind_speeds[i + 1]])


            interp_dir_deg = np.interp(interp_alt, [altitudes[i], altitudes[i + 1]], [angle1, angle2]) % 360 #make sure in the range (0, 360)

            interpolated_altitudes.append(interp_alt)
            interpolated_speeds.append(interp_speed)
            interpolated_directions_deg.append(interp_dir_deg)

    print(len(interpolated_altitudes))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')

    # Create a scatter plot where radius is altitude, angle is wind direction (in radians), and color represents wind speed
    sc = ax.scatter(np.radians(interpolated_directions_deg), interpolated_altitudes, c=interpolated_speeds, cmap='magma', s=10)
    cbar = plt.colorbar(sc, label='Wind Speed (m/s)')

    # Set title
    plt.title('Windmap with Wind Angles Interpolated')

def windVectorToBearing(u, v, h):
    """ Converts U-V wind data at specific heights to angular and radial
    components for polar plotting.

    :param u: U-Vector Wind Component from Forecast
    :type u: float64 array
    :param v: V-Vector Wind Component from Forecast
    :type v: float64 array
    :param h: Corresponding Converted Altitudes (m) from Forecast
    :type h: float64 array
    :returns: Array of bearings, radius, colors, and color map for plotting
    :rtype: array

    """
    # Calculate altitude
    bearing = np.arctan2(v,u)
    bearing = np.unwrap(bearing)
    r = np.power((np.power(u,2)+np.power(v,2)),.5)

    # Set up Color Bar
    colors = h
    cmap=mpl.colors.ListedColormap(colors)

    return [bearing, r , colors, cmap]

def getWind(df):
    """ Calculates a wind vector estimate at a particular 3D coordinate and timestamp
    using a 2-step linear interpolation approach.

    Currently using scipy.interpolat.CubicSpline instead of np.interp like in GFS and ERA5.

    See also :meth:`GFS.GFS.wind_alt_Interpolate`

    :param hour_index: Time index from forecast file
    :type hour_index: int
    :param lat_i: Array index for corresponding netcdf lattitude array
    :type lat_i: int
    :param lon_i: Array index for corresponding netcdf laongitude array
    :type lon_i: int
    :returns: [U, V]
    :rtype: float64 2d array

    """


    v = df["Wind X Component"].values   #why are these switched
    u = df["Wind Y Component"].values
    h = df["Altitude"].values

    print(u)


    #Fix this interpolation method later, espcially for ERA5
    cs_u = CubicSpline(h, u)
    cs_v = CubicSpline(h, v)

    alts_new = np.arange(0, 7, .01) # New altitude range

    u = cs_u(alts_new)
    v = cs_v(alts_new)

    return windVectorToBearing(u, v, alts_new)

def plotWindVelocity(df):
    """ Plots a 3D Windrose for a particular coordinate and timestamp from a downloaded forecast.

    :param hour_index: Time index from forecast file
    :type hour_index: int
    :param lat: Latitude
    :type lat: float
    :param lon: Longitude
    :type lon: float
    :returns:

    """

    bearing1, r1 , colors1, cmap1 = getWind(df)

    # Plot figure and legend
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(111, projection='polar')
    sc2 = ax1.scatter(bearing1, colors1, c=r1, cmap='magma', alpha=0.75, s = 10)
    ax1.title.set_text("Windrose from windmap.py")

    #ax1.set_xticks([0,9,90,180, 180,270,270,360]) #Fixes the FixedLocator Warning for the line below
    ax1.set_xticks(ax1.get_xticks())
    ax1.set_xticklabels(['E', '', 'N', '', 'W', '', 'S', ''])

    plt.colorbar(sc2, ax=ax1, label=" Wind Velocity (m/s)")

    print("hey")


def polar_interpolated_scatter_plot_uv(df, num_interpolations=20):
    """
    Create a polar scatter plot with altitude as the radius, wind direction in radians, and a color bar for wind speed.

    Args:
    - df: A DataFrame containing wind data with columns "Altitude", "Wind X Component", and "Wind Y Component".
    - num_interpolations: Number of altitude interpolations between each altitude level (default is 20).
    """

    # Extract data from the DataFrame
    altitudes = df["Altitude"].values
    wind_x_components = df["Wind X Component"].values
    wind_y_components = df["Wind Y Component"].values

    interpolated_altitudes = []
    interpolated_x = []
    interpolated_y = []


    for i in range(len(altitudes) - 1):
        for j in range(num_interpolations + 1):
            alpha = j / num_interpolations
            interp_alt = altitudes[i] + alpha * (altitudes[i + 1] - altitudes[i])
            interp_x = np.interp(interp_alt, [altitudes[i], altitudes[i + 1]], [wind_x_components[i], wind_x_components[i + 1]])
            interp_y = np.interp(interp_alt, [altitudes[i], altitudes[i + 1]], [wind_y_components[i], wind_y_components[i + 1]])

            interpolated_altitudes.append(interp_alt)
            interpolated_x.append(interp_x)
            interpolated_y.append(interp_y)

    # Calculate interpolated directions
    print(len(interpolated_x), len(interpolated_y))
    interpolated_directions_rad = np.arctan2(interpolated_x, interpolated_y)
    interpolated_speeds = np.power((np.power(interpolated_x,2)+np.power(interpolated_y,2)),.5)

    # Create a polar scatter plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')

    sc = ax.scatter(interpolated_directions_rad, interpolated_altitudes, c=interpolated_speeds, cmap='magma', s=10)
    cbar = plt.colorbar(sc, label='Wind Speed (m/s)')

    # Set title
    plt.title('Windmap with Vector Components Interpolated')

    plt.figure()
    #plt.scatter(interpolated_altitudes, interpolated_y)
    plt.plot(interpolated_altitudes, interpolated_directions_rad)
    plt.plot(interpolated_altitudes, interpolated_speeds)
    #plt.plot(altitudes, wind_y_components, color = "black")
    #plt.scatter(interpolated_altitudes, interpolated_x)

    plt.figure()
    #plt.scatter(interpolated_altitudes, interpolated_y)
    plt.plot(interpolated_x, interpolated_altitudes)
    plt.plot(interpolated_y, interpolated_altitudes)



# Example usage:
altitudes = list(range(0, 10))  # Altitudes in 1 km increments from 0 to 30 km
wind_speeds = np.random.uniform(5, 20, 10)  # Random wind speeds between 5 and 20 m/s
wind_directions = np.random.uniform(0, 360, 10)  # Random wind directions between 0 and 360 degrees

data = {
    "Altitude": altitudes,
    "Wind Speed": wind_speeds,
    "Wind Direction": wind_directions
}


data = {
    "Altitude": [0, 1, 2, 3, 4, 5, 6, 7, 8], #, 6, 7, 8, 9, 10],
    "Wind Speed": [7, 12, 8, 9, 14, 11, 14, 9, 6], #, 15, 2, 4, 8, 10],
    "Wind Direction": [120, 90, 220, 150, 275, 350, 10, 282, 110] #, 5, 120, 330, 245, 25]  # Wind direction in degrees
}


df = pd.DataFrame(data)
df = add_rotated_wind_components(df, rotation_angle=90)

new_range = np.arange(0, 8.5, 0.25)
print(new_range)
df2 = pd.DataFrame(new_range, columns = ["Altitude"])


df['Wind Direction'] = np.rad2deg(np.unwrap(np.deg2rad(df['Wind Direction'])))
df3 = pd.merge(left=df2, right=df, on='Altitude', how='left').interpolate()
df3['Wind Direction'] %= 360
print(df3)

polar_interpolated_scatter_plot(df3, num_interpolations=1000)

#df = add_rotated_wind_components(df, rotation_angle=180)
#polar_interpolated_scatter_plot_uv(df, num_interpolations=200)

#plotWindVelocity(df)



threshold_speed = 0.1  # Threshold wind speed (m/s)
threshold_angle = 10  # Threshold angle (degrees)
opposing_ranges = find_opposing_wind_ranges(df3, threshold_speed, threshold_angle)

if len(opposing_ranges) > 0:
    print("Altitude ranges with opposing winds:")
    for range_pair in opposing_ranges:
        print(f"Start Altitude: {range_pair[0]} m, End Altitude: {range_pair[1]} m")
else:
    print("No altitude ranges with opposing winds found.")

plt.show()
