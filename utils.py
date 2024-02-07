import pandas as pd
import config
import dataframe_image as dfi
from pathlib import Path

def convert_stations_coords(stations_df):
    '''Converts the station lat/long coordinates for mapping'''

    stations_df['lat_era5'] = stations_df.apply(lambda x: (-1 * x['  LAT'] if x['N'] == 'S' else 1 * x['  LAT']),
                                                axis=1)
    #This works with 360-x or -1*x???
    stations_df['lon_era5'] = stations_df.apply(lambda x: (360 - x[' LONG'] if x['E'] == 'W' else 1 * x[' LONG']),
                                                axis=1)

    return stations_df

def get_analysis_folder(FAA, WMO, year):
    return config.analysis_folder + str(FAA) + " - " + str(WMO) + "/" + str(year) + "_analysis/"

def get_data_folder(FAA, WMO, year):
    return config.parent_folder + str(FAA) + " - " + str(WMO) + "/" + str(year) + "/"



def export_colored_dataframes(df, title, path, suffix, precision = 2, export_color = True, vmin = 0.0, vmax = 1.0, cmap = 'RdYlGn', mode = "selenium"):

    '''
    Note you will need to install chrome driver
    #https://stackoverflow.com/questions/43397162/show-matplotlib-plots-and-other-gui-in-ubuntu-wsl1-wsl2


    :param df:
    :param title:
    :param path:
    :param suffix:
    :return:
    '''

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
        # Put the cumulative probabilities in 2 spots for easy visual inspection and further decade-based cumulative analysis.
        filepath_image = Path(path + '/' + suffix +'.png')
        filepath_image.parent.mkdir(parents=True, exist_ok=True)
        dfi.export(df_styled, filepath_image, max_rows=-1, table_conversion = mode) #this is slow on windows?

    filepath_dataframe = Path(path + '/' + suffix +'.csv')
    filepath_dataframe.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath_dataframe)