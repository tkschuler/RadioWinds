import pandas as pd
import numpy as np

file1 = open('wyoming_html/' + 'north_america.txt', 'r')
continent = "North_America"
Lines = file1.readlines()

df = pd.DataFrame(columns=['WMO', 'FAA', 'Station_Name', 'Continent'])

count = 0

# Strips the newline character
for line in Lines:
    print(line)
    line  = line.strip()
    sub1 = line.split(":g('")
    sub2 = sub1[1].split("')\" ")
    WMO = sub2[0]
    sub3= sub2[1]
    name = sub3[7+6+1:-2]

    if name[-1] == ")":
        sub4 = name.rsplit("(" , 1)
        FAA = sub4[1][:-1]
        name = sub4[0]
    else:
        FAA = None

    print(WMO, " - ", FAA, " - ", name)

    df.loc[count, :] = [WMO , FAA, name, continent]
    count +=1

df['WMO'] = df['WMO'].astype(int)
#df = df.replace('None', np.NAN)
print(df)

df2 = pd.read_csv("raob_station_list_CLEANED2.csv")
df2 = df2.replace("----" , np.NAN)

print(df2)

continent_export = pd.merge(df, df2, left_on='WMO', right_on='WMO')
print(continent_export)

continent_export.to_csv("CLEANED/" + continent + ".csv")