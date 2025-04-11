import os
from termcolor import colored

import sys
sys.path.append('../RadioWinds')
import config

'''
This checks if the Radiosondes Directory Downloads are organized correctly By Station, Year, and Months.  
If Months are missing the program will report red text of the missing months.  
You can try redownloading the months or adding empty months for missing data from the server.

If the radisonde downloads are organized correctly, this program does not print anything. 
'''

check_total_soundings = False

number_of_stations = 0
number_of_soundings = 0

for dir in os.listdir(config.parent_folder):
    number_of_stations +=1
    for year in os.listdir(config.parent_folder + dir):
        number_of_months = len(os.listdir(config.parent_folder + dir + "/" + str(year)))
        #print(year)
        if number_of_months != 12:
            print(colored((dir, year, number_of_months), "red"))
        if check_total_soundings:
            for month in range (1,12+1):
                number_of_soundings += len(os.listdir(config.parent_folder + dir + "/" + str(year) +'/' + str(month)))
            print("Total Number of Soundings Downloaded ", number_of_soundings)

print("Total Number of Stations Downloaded ", number_of_stations)