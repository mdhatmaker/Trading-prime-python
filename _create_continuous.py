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

########################################################################################################################

# FOR TESTING!!!!
if not is_arg('project'):
    set_args_from_file("create_continuous_es.args")
    project_folder = join(folder.data_folder, get_arg('project'))

symbol_root = get_arg('symbol')
timeframe = get_arg('timeframe')
start_year = int(get_arg('start_year'))
end_year = int(get_arg('end_year'))

(get_roll_dates, roll_description) = get_roll_function(symbol_root)


#roll_description = "Rolling at first Wed on or before 30 days prior to 3rd Friday of month immediately following expiration month"


#-----------------------------------------------------------------------------------------------------------------------

print "Creating prices data file for " + symbol_root + " continuous futures contract..."
print "Output files will be comma-delimeted pandas-ready dataframes (.csv)"
print
print roll_description
print

#csv_files = [ f for f in listdir(df_folder) if (isfile(join(df_folder, f)) and f.endswith('.csv')) ]
#files_for_symbol = [ f for f in csv_files if f.startswith(symbol_root) ]
#df = read_dataframe(csv_files[0])

roll_filename = get_csv_filename(symbol_root + "_roll_dates", timeframe)
f = open(join(project_folder, roll_filename), 'w')
f.write("Symbol,StartDate,EndDate,PriceAdjust\n")

########## CREATE CONTINUOUS PRICES DATA FILE ##########
print "-----" + symbol_root + " CONTINUOUS-----"
prev_close = None
df_frames = []
for year in range(start_year, end_year+1):
    for month in range(1, 12+1):
        #count = 0
        rows_list = []
        print "{0:2d} {1:4d}".format(month, year),
        symbol = get_symbol(symbol_root, month, year)
        #pathname = get_df_pathname(symbol)
        pathname = get_df_pathname(symbol, timeframe)
        print '   "' + get_split_filename(pathname) + '"     ',
        if isfile(pathname):
            df_all = read_dataframe(pathname)   #, [date_column])
            (dt0, dt1) = get_roll_dates(month, year)
            df = df_all[df_all.DateTime >= dt0]
            df = df[df.DateTime < dt1]
            if df.empty:
                print
                continue
            df.insert(0, 'Symbol', symbol)
            df_frames.append(df)

            df = df.sort_values('DateTime')
            #df.sort_values(['colA', 'colB'], ascending=[True, False])
            #mindate = df[date_column].min()
            #maxdate = df[date_column].max()
            if prev_close == None:
                prev_close = float(df.tail(1).Close)
                adjust = 0.0
            else:
                adjust = prev_close - float(df.head(1).Close)
                prev_close = float(df.tail(1).Close)
            #output = "{0},{1},{2},{3}".format(symbol, mindate.strftime("%Y-%m-%d"), maxdate.strftime("%Y-%m-%d"), adjust)
            output = "{0},{1},{2},{3}".format(symbol, dt0.strftime("%Y-%m-%d"), dt1.strftime("%Y-%m-%d"), adjust)
            f.write(output + '\n')
            print output
        else:
            print

f.close()

df = pd.concat(df_frames, ignore_index=True)

filename = get_csv_filename(symbol_root + "_continuous", timeframe)
df.to_csv(join(project_folder, filename), index=False)  #, cols=('A','B','sum'))

print
print "Prices output to file:", filename
print "Roll dates output to file:", roll_filename
print



