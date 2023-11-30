from siphon.simplewebservice.wyoming import WyomingUpperAir
from datetime import datetime
from io import StringIO
import warnings
import calendar
from termcolor import colored
from requests.exceptions import HTTPError
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

from siphon._tools import get_wind_components


class SiphonMulti(WyomingUpperAir):
    """Download and parse data from the University of Wyoming's upper air archive."""


    def __init__(self):
        """Set up endpoint."""
        super(WyomingUpperAir, self).__init__('http://weather.uwyo.edu/cgi-bin/sounding')


    class InvalidTimeParameter(Exception):
        "Server Error Invalid time parameter."
        pass


    @classmethod
    def request_data(cls, year, month, site_id, **kwargs):

        endpoint = cls()
        df = endpoint._get_monthly_data(year, month, site_id)  #override of original class
        #df = endpoint._get_data(start_time, end_time, site_id)
        return df

    def _get_monthly_data(self, year, month, site_id):
        '''
        This is a new class to download soundings by Month from UofWy.  It returns a list of all soundings for the months
        as pandas dataframes

        Args:
            year:
            month:
            site_id:

        Returns:

        '''

        raw_data = self._get_data_raw(year, month, site_id)
        soup = BeautifulSoup(raw_data, 'html.parser')
        sounding_titles = soup.find_all('h2')
        soundings = soup.find_all('pre')

        #print(soup)

        #sdfs


        monthly_soundings = []

        for i in range(0,len(soundings),2):
            df = self._get_data(i,soundings, sounding_titles, site_id)

            #Check if a df was returned
            if df is not None:
                monthly_soundings.append(df)

        return monthly_soundings

    def _get_data(self, i, soundings, sounding_titles,  site_id):
        r"""Download and parse upper air observations from an online archive.

        Parameters
        ----------
        time : datetime
            The date and time of the desired observation.

        site_id : str
            The three letter ICAO identifier of the station for which data should be
            downloaded.

        Returns
        -------
            :class:`pandas.DataFrame` containing the data

        """
        #raw_data = self._get_data_raw(start_time, end_time, site_id)
        #soup = BeautifulSoup(raw_data, 'html.parser')
        #tabular_data = StringIO(soup.find_all('pre')[0].contents[0])

        tabular_data = StringIO(soundings[i].contents[0])


        col_names = ['pressure', 'height', 'temperature', 'dewpoint', 'direction', 'speed']
        #print(tabular_data)

        #Check if there is incomplete data
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                df = pd.read_fwf(tabular_data,  widths=[7] * 8, skiprows=5, usecols=[0, 1, 2, 3, 6, 7], names=col_names)
            except :
                print(colored("Incomplete data for " + str(sounding_titles[i // 2]), "yellow"))
                #return None
                return None
                #raise HTTPError





        df['u_wind'], df['v_wind'] = get_wind_components(df['speed'],
                                                         np.deg2rad(df['direction']))

        # Drop any rows with all NaN values for T, Td, winds
        df = df.dropna(subset=('temperature', 'dewpoint', 'direction', 'speed',
                               'u_wind', 'v_wind'), how='all').reset_index(drop=True)



        # Parse metadata
        #meta_data = soup.find_all('pre')[1].contents[0]
        meta_data = soundings[i+1].contents[0]
        lines = meta_data.splitlines()

        #print(meta_data)



        # If the station doesn't have a name identified we need to insert a
        # record showing this for parsing to proceed.
        if 'Station number' in lines[1]:
            lines.insert(1, 'Station identifier: ')

        station = lines[1].split(':')[1].strip()
        station_number = int(lines[2].split(':')[1].strip())
        sounding_time = datetime.strptime(lines[3].split(':')[1].strip(), '%y%m%d/%H%M')
        #print(lines[3].split(':')[1])
        #print(sounding_time)

        #print(lines[6])

        #ANOTHER CHECK?

        #print(lines[4].split(':')[1].strip())


        #New Error for South America with some older data.  I don't think this affects batch analysis
        if (lines[4].split(':')[1].strip() == '******'):
            latitude = None
            longitude = None
            elevation = None
        else:
            latitude = float(lines[4].split(':')[1].strip())
            longitude = float(lines[5].split(':')[1].strip())
            elevation = float(lines[6].split(':')[1].strip())

        #print("here")

        df['station'] = station
        df['station_number'] = station_number
        df['time'] = sounding_time
        df['latitude'] = latitude
        df['longitude'] = longitude
        df['elevation'] = elevation

        # Add unit dictionary

        df.units = {'pressure': 'hPa',
                    'height': 'meter',
                    'temperature': 'degC',
                    'dewpoint': 'degC',
                    'direction': 'degrees',
                    'speed': 'knot',
                    'u_wind': 'knot',
                    'v_wind': 'knot',
                    'station': None,
                    'station_number': None,
                    'time': None,
                    'latitude': 'degrees',
                    'longitude': 'degrees',
                    'elevation': 'meter'}
        return df

    def _get_data_raw(self, year, month, site_id):
        """Download data from the University of Wyoming's upper air archive.

        Parameters
        ----------
        time : datetime
            Date and time for which data should be downloaded
        site_id : str
            Site id for which data should be downloaded

        Returns
        -------
        text of the server response

        """

        num_days = calendar.monthrange(year, month)[1]

        #print(num_days)

        #sds

        start_time = datetime(year, month, 1, 00)
        end_time = datetime(year, month, num_days, 23)

        #print(start_time)



        path = ('?region=naconf&TYPE=TEXT%3ALIST'
                '&YEAR={start_time:%Y}&MONTH={start_time:%m}&FROM={start_time:%d%H}&TO={end_time:%d%H}'
                '&STNM={stid}').format(start_time=start_time, end_time = end_time, stid=site_id)

        '''
        path =('?region=naconf&TYPE=TEXT%3ALIST'
               '&YEAR=2023'
               '&MONTH=10'
               '&FROM=1900'
               '&TO=3112'
               '&STNM=72402')
        '''

        #print(path)

        #path = ('?region=naconf&TYPE=TEXT%3ALIST&YEAR=2023&MONTH=04&FROM=0112&TO=0112&STNM=74794')

        #Do error handling to check if there is a 400 error (user error,  instead of server error). For UofWy server I've only found this invalid Time parameter error.
        server_400 = False
        try:
            resp = self.get_path(path)
        except HTTPError as http:
            if 'Server Error (400: Invalid TIME parameter.' in http.args[0]:
                server_400 = True
            else:
                raise(http)

        if server_400:
            raise self.InvalidTimeParameter

        #print(resp)

        # See if the return is valid, but has no data
        if resp.text.find('Can\'t') != -1:
            raise ValueError(
                'No data available for {end_time:%Y-%m-%d %HZ} '
                'for station {stid}.'.format(end_time = end_time, stid=site_id))


        if resp.text.find('Invalid') != -1:
            raise ValueError(
                'Invalid time range for {end_time:%Y-%m-%d %HZ} '
                'for station {stid}.'.format(end_time = end_time, stid=site_id))
        #dfdfdf

        #Do I need to add in a Forbidden error?
        if resp.text.find('Forbidden') != -1:
            print("FORBIDDEN, this is the error")
            #Do I raise an Http error?

        #if resp.text.find('******') != -1:
        #    print("WE FOUND A BUG!")


        #print(resp.text)

        return resp.text

#main
if __name__=="__main__":

    station = '74794'
    year = 2017
    month = 1
    df = SiphonMulti.request_data(year, month, station)

    print(df)