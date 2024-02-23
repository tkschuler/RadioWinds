import pandas as pd
import config
import dataframe_image as dfi
from pathlib import Path
import os
from termcolor import colored

"""
utils.py contains multiple helper and utility functions that are used across multiple other scripts in RadioWinds.
"""


def convert_stations_coords(stations_df):
    """
        Converts the station lat/long coordinates for mapping
    """

    stations_df['lat_era5'] = stations_df.apply(lambda x: (-1 * x['  LAT'] if x['N'] == 'S' else 1 * x['  LAT']),
                                                axis=1)
    # This works with 360-x or -1*x???
    stations_df['lon_era5'] = stations_df.apply(lambda x: (360 - x[' LONG'] if x['E'] == 'W' else 1 * x[' LONG']),
                                                axis=1)

    return stations_df


def get_analysis_folder(FAA, WMO, year):
    return config.analysis_folder + str(FAA) + " - " + str(WMO) + "/" + str(year) + "_analysis/"


def get_data_folder(FAA, WMO, year):
    return config.parent_folder + str(FAA) + " - " + str(WMO) + "/" + str(year) + "/"


def check_analyzed(FAA, WMO, year, path, category):
    """
        Check if the radiosonde data has been downloaded yet
    """
    isExist = os.path.exists(path)
    if not isExist:
        print(colored(str(FAA) + "-" + str(WMO) + "/" + str(
            year) + " " + category + " data not yet analyzed.", "yellow"))
        return False

    print(colored(str(FAA) + "-" + str(WMO) + "/" + str(
        year) + " " + category + " data already analyzed.", "green"))
    return True


def export_colored_dataframes(df, title, path, suffix, precision=2, export_color=True,
                              vmin=0.0, vmax=1.0, cmap='RdYlGn', mode=config.dfi_mode):

    """
    Exports up to 2 dataframes:
        * a .csv of the dataframe
        * if export_color is true, a colored .png of the table as well


    .. note::
        You may need to install the following if now already installed:
            * chrome driver
            * Selenium
        The default for dataframe_image is to use chrome. But this doesn't work well with WSL.
        Matplotlib is an option as well, but it doesn't allow for css customization, so no titles or captions
    """

    df.index.name = None

    df = df.apply(pd.to_numeric)

    df_styled = df.style.background_gradient(axis=None, vmin=vmin, vmax=vmax, cmap=cmap)

    if 'std' in df:
        cmaps = {'std': 'winter_r'}
        for col, cmap in cmaps.items():
            df_styled = df_styled.background_gradient(cmap, subset=col, vmin=0.0, vmax=.35)

    df_styled = df_styled.set_caption(
        title).set_table_styles([{
            'selector': 'caption',
            'props': [
                ('color', 'black'),
                ('font-weight', 'bold'),
                ('font-size', '20px')
            ]
        }])

    df_styled = df_styled.format(precision=precision)

    if export_color:
        filepath_image = Path(path + '/' + suffix + '.png')
        filepath_image.parent.mkdir(parents=True, exist_ok=True)
        dfi.export(df_styled, filepath_image, max_rows=-1, max_cols=-1, table_conversion=mode)

    filepath_dataframe = Path(path + '/' + suffix + '.csv')
    filepath_dataframe.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath_dataframe)
