import os
from termcolor import colored
import config

'''
This checks if the Radiosondes Directory Downloads are organized correctly By Station, Year, and Months.  
If Months are missing the program will report red text of the missing months.  
You can try redownloading the months or adding empty months for missing data from the server.

If the radisonde downloads are organized correctly, this program does not print anything. 
'''

for dir in os.listdir(config.parent_folder):
    for year in os.listdir(config.parent_folder + dir):
        number_of_months = len(os.listdir(config.parent_folder + dir + "/" + str(year)))
        #print(year)
        if number_of_months != 12:
            print(colored((dir, year, number_of_months), "red"))
        #else:
        #    print(dir, year, len(os.listdir(config.parent_folder + dir + "/" + str(year))))
