[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)

# RadioWinds

This software includes multiple files for analyzing historical winds from sounding data.  This includes batch downloading the soundings to a local machine for faster and repeatable analysis as well as the analysis itself
## Dependencies

RadioWinds relies on the following libraries:

```
netCDF4
numpy
pandas
termcolor
backports.datetime_fromisoformat
seaborn
scipy
xarray
cartopy
siphon
matplotlib
dataframe_image
```

**!IMORTANT: In Siphon, for downloading individual soundings, download the latest version of Wyoming.py from siphon Github,  newer than the 0.9 release.** This takes care of the occasional height folding issue. We take care of it in ``SiphonMulti.py``.


## Downloading Historical Sounding Data

There are 3 main sources for aaqquiring sounding data
* https://weather.uwyo.edu/upperair/sounding.html
* https://ruc.noaa.gov/raobs/
* https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C00975


We determined that RAOBs is missing wind data outside of the mandatory pressure levels. 

IGRA2 and UofWy have different wind readings for the same soundings.  


> "
    There are three types of reports at levels below 100 hPa and for levels above 100 hPa.  They are mandatory levels, significant temperature levels, and significant wind levels.  The mandatory levels report pressure, temperature, humidity, and winds at defined pressure levels (1000, 925, 850, 700, ... hPa).  Significant temperature levels report pressure, temperature, and humidity to fill in levels where an observation would be too different from an interpolation.  Significant wind levels report heights and winds.  These are merged into one report.  My software, GEMPAK, interpolates the heights and winds at significant temperature levels and interpolates the height and winds at significant wind levels.  IGRA reports these as missing.
    <br>
    <br>
    In the last few years, these is a new data format called BUFR.  It looks as if IGRA is using some of the data from this format.  I provide this data at:
    <br>
    <br>
    https://weather.uwyo.edu/upperair/bufrraob.shtml
    <br>
    <br>
    I believe the US issues two different types of BUFR reports.  IGRA and I may not be using the same reports.
    <br>
    <br>
    As far as accuracy, my feeling is the inaccuracy of the measurement and variation of the atmosphere are much greater than the method used to present the data.  The BUFR reports may use less smoothing that what is used to generate the old text style reports.
    "<br>
    <br>
    --Larry Oolman (Maintainer of Univerity of Wyoming Upper Air Sounding Dataset)



For this project we use UofWY because the dataset is easier to work with.  You can download an individual sounding or up to a month by modifying the URL,  whereas with IGRA2 you have to download a huge file of every sounding that station ever had and then parse through all the raw text to find the sounding you're looking for. 

## File Descriptions

``SiphonMulti.py`` This script extends the Siphon Library to be able to download a month of data from UofWy rather than only one sounding at a time.  This greatly speeds up the ability to bulk download soundings 

``AnnualWyomingDownload.py`` This script batch downloads soundings for an entire year locally and organizes the soundings by *year/month/sounding.csv*.  To automate this process we use *continent.csv* in *Radiosonde_String_Parsing/Cleaned/** for a lists of soundings to download from, and then do that for every year.  We use multiprocessing to asyncronously download [multiple stations by month] simultaneously rather than [one station one month] at a time.  

``opposing_wind_wyoming.py`` This script analyzes the wind diversity for an individual sounding.  It checks for 4 categories:  Fail, Calm Winds, Opposings Winds, and Full Wind Diversity. if running this file standalone, it all produces windrose plots of the sounding.

``batch_analysis2.py``  this script does opposing wind anaylsis for a year of soundings per station.  Currently, we assume that the soundings were downloaded properly and in full by ``AnnualWyomingDownload.py.``  (If the script finished running everything should be downloaded for the parameters). We should add some error handling for incomplete downloads. The script then generates binary opposing wind charts organized by date and altitude level (500m increments) for each month.  After all 12 months are analyzed and saved,  a final annual probability chart is saved by taking the max probability from each altitude level per month.

``Mapping/rainbnow.py`` This script creates colored mesh plots by interpolating the annual probability wind diversity charts. 

## Authors

* **Tristan Schuler** - *U.S. Naval Research Laboratory*
* **Craig Motell** - *NIWC Pacific*

## Acknowledgments

Hat Tip to [Raven Aerostar](https://www.dropbox.com/s/l5t9zw653nywuqh/Mike%20Smith%20-%20Mike_Smith_Presentation_2021.pdf?dl=0), who did similar analysis in the past and presented at the [2021 Scientific Ballooning Technologies Workshop](https://sites.google.com/umn.edu/2021-scientific-ballooning-tec)

