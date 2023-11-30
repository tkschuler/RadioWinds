from datetime import datetime
from siphon.simplewebservice.igra2 import IGRAUpperAir
import pandas as pd

date = datetime(2023, 11, 1, 00)
station = 'USM00091285'

print("Downloading data...")
df, header = IGRAUpperAir.request_data(date, station)

df.dropna(axis='rows' , inplace = True)

pd.set_option("display.max_rows", None)

print(df.columns)

# Inspect metadata from the data headers
print(header.columns)

print(df)
