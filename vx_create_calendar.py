import sys
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np

#-----------------------------------------------------------------------------------------------------------------------

import f_folders as folder
from f_args import *
from f_date import *
from f_file import *
from f_dataframe import *
from f_rolldate import *

#-----------------------------------------------------------------------------------------------------------------------

# <module functions go here>

################################################################################

#print "Specify the Multichart export file (.txt) on the command line"
print "Creating prices data file for VX calendar (1-month)..."
print "Output file will be comma-delimeted pandas-ready dataframe (.csv)"
print
print "Rolling at first Wed on or before 30 days prior to 3rd Friday of month immediately following expiration month"
print


project_folder = join(data_folder, "vix_es")
symbol_root = '@VX'
#timeframe='1 Minute'
timeframe='Daily'

roll_date_filename = get_csv_filename(symbol_root + "_roll_dates", timeframe)
df = read_dataframe(join(project_folder, roll_date_filename), ['StartDate', 'EndDate'])

df_frames = []

for ix,row in df.iterrows():
    symbol1 = row.Symbol
    symbol2 = next_month_symbol(symbol1)
    calendar_symbol = symbol1 + "-" + symbol2
    print calendar_symbol

    pathname1 = get_df_pathname(symbol1, timeframe)
    pathname2 = get_df_pathname(symbol2, timeframe)

    if not (os.path.exists(pathname1) and os.path.exists(pathname2)):
        continue
    
    df1 = read_dataframe(join(project_folder, pathname1))
    df2 = read_dataframe(join(project_folder, pathname2))

    # merge the data for the front month and the back month of the calendar spread into a single dataframe
    df = pd.merge(df1, df2, on='DateTime')

    # only include the data within the date range specified in the roll_dates file
    dt1 = row.StartDate
    dt2 = row.EndDate
    dt2 += timedelta(days=1)
    df = df[df.DateTime >= dt1]
    df = df[df.DateTime < dt2]

    df['Open'] = df['Open_x'] - df['Open_y']
    df['Close'] = df['Close_x'] - df['Close_y']

    df.insert(0, 'Symbol', calendar_symbol)

    df.drop(['High_x', 'Low_x', 'High_y', 'Low_y'], 1)
    
    df_frames.append(df)
    
print

df = pd.concat(df_frames, ignore_index=True)

filename = get_csv_filename(symbol_root + "_calendar", timeframe)
df.to_csv(join(project_folder, filename), index=False)  #, cols=('A','B','sum'))

print
print "Prices output to file:", filename
print



