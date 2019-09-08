import json
from datetime import datetime
from datetime import timedelta
from os import listdir
from os.path import isfile, join, splitext
import sys

#-----------------------------------------------------------------------------------------------------------------------

import f_folders as folder
from f_args import *
from f_date import *
from f_file import *
from f_calc import *
from f_dataframe import *
from f_rolldate import *
from f_chart import *

#execfile("f_folders.py")
#execfile("f_args.py")
#execfile("f_date.py")
#execfile("f_file.py")
#execfile("f_dataframe.py")

#-----------------------------------------------------------------------------------------------------------------------

LME_str = "M.CU3=LX";

hist = {}

def print_ranges(value_counts):
    counts = {}
    for i in range(-4,5):
        if i in value_counts.keys():
            counts[i] = value_counts[i]
        else:
            counts[i] = 0
    print ">   3   :", counts[3]
    print " 2 to  3:", counts[2]
    print " 1 to  2:", counts[1]
    print " 0 to  1:", counts[0]
    print "-1 to  0:", counts[-1]
    print "-2 to -1:", counts[-2]
    print "-3 to -2:", counts[-3]
    print "<  -3   :", counts[-4]
    return

def get_range(x):
    if x < -3:
        return -4
    elif x < -2:
        return -3
    elif x < -1:
        return -2
    elif x < 0:
        return -1
    elif x < 1:
        return 0
    elif x < 2:
        return 1
    elif x < 3:
        return 2
    else:
        return 3

def read_symbol_data(folder):
    # Adjust times on LME data points by 6 hours (time is "exchange time")
    read_from_file(folder, LME_str, -6)
    # Adjust time on HG data points by 1 hour (time is "exchange time")
    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
    for name in onlyfiles:
        if name.startswith('QHG') and name.endswith('.txt'):
            symbol = name[0:-4]
            read_from_file(folder, symbol, -1)
    return

def read_from_file(folder, symbol, hour_delta):
    hist[symbol] = {}
    filename = join(folder, symbol + ".txt")
    print filename
    f = open(filename, 'r')
    line = f.readline()         # skip first line of file
    line = f.readline()         # start with second line of file
    count = 0
    while (line):
        v = line.split(',')
        dt = get_datetime_from_d_t(v[0], v[1])
        dt += timedelta(hours = hour_delta)
        o = float(v[2])
        h = float(v[3])
        l = float(v[4])
        c = float(v[5])
        volume = int(v[6])
        item = {'open': o, 'high': h, 'low': l, 'close': c, 'volume': volume}
        hist[symbol][dt] = item
        count += 1
        line = f.readline()
    f.close()
    return

def get_unique_dates(symbol):
    unique_dates = []
    for dt in hist[symbol].keys():
        dateonly = datetime(dt.year, dt.month, dt.day, 0, 0, 0)
        if not dateonly in unique_dates:
            unique_dates.append(dateonly)
    return unique_dates

def preprocess(folder):
    read_symbol_data(folder)

    copper_spread_file = open(folder + "copper_spread.csv", 'w')
    calendar_spread_file = open(folder + "calendar_spread.csv", 'w')
    premium_discount_file = open(folder + "premium_discount.csv", 'w')

    # Output column headers in the premium/discount file
    premium_discount_file.write("<Date>, <Time>, <HG>, <LME>, <Spread>, <Calendar>, <Discount/Premium>, <CalendarSymbol>")

    unique_dates = get_unique_dates(LME_str)
    unique_dates.sort()
    print len(unique_dates)

    copper_spread_file.close()
    calendar_spread_file.close()
    premium_discount_file.close()
    
    return


########################################################################################################################

"""
# Default to using entire date range
start_date = datetime(1900, 1, 1, 0, 0, 0)
end_date = datetime(2100, 1, 1, 0, 0, 0)

# The start_date and end_date should be modifiable from the command line
argv = sys.argv
argc = len(argv)

# Create dictionary of command-line arguments
if argc > 1:
    for xarg in argv:
        if xarg.startswith("-"): # and xarg.find("=") != -1:
            split = xarg.split('=')
            arg_id = split[0][1:]
            if len(split) < 2:
                arg_value = ""
            else:
                arg_value = split[1]
            args[arg_id] = arg_value

if 'd1' in args:
    start_date = get_date_from_yyyymmdd(args['d1'])
if 'd2' in args:
    end_date = get_date_from_yyyymmdd(args['d2'])
if not ('d1' in args or 'd2' in args):
    print "Use -d1 and -d2 command line args to specify date range in YYYYMMDD format"
    print "(ex: -d1=20170128 -d2=20171205)\n"
"""

#-----------------------------------------------------------------------------------------------------------------------


# Read processed data from JSON file
data = read_json(join(data_folder, 'copper', 'quandl_hg.json'))

dfx = {}
gfx = {}

dfx["Date"] = []
dfx["spread"] = []
dfx["discount"] = []
dfx["symbol"] = []

gfx["g_spread"] = []
gfx["g_discount"] = []

indexes = []
#index_names = ["date", "i"]
dfx_column_names_dfx = ["Date", "spread", "discount", "symbol"]
gfx_column_namez = ["g_spread", "g_discount"]

dates = {}
#mydata = {}

min_date = datetime(2100, 1, 1, 0, 0, 0)
max_date = datetime(1900, 1, 1, 0, 0, 0)

print "reading data",
count = 0
idx = 0
for dt_str in data.keys():
    if count % 1000 == 0: print ".",
    dt = get_datetime_from_dt(dt_str)
    if not (dt >= start_date and dt <= end_date):
        count += 1
        continue
    dates[dt] = count
    if dt > max_date: max_date = dt
    if dt < min_date: min_date = dt
    #indexes.append(idx)
    indexes.append(dt)
    d = data[dt_str]
    dfx["Date"].append(dt)
    dfx["spread"].append(d["spread"])
    dfx["discount"].append(d["discount"])
    dfx["symbol"].append(d["symbol"])
    gfx["g_spread"].append(get_range(d["spread"]))
    gfx["g_discount"].append(get_range(d["discount"]))
    count += 1
    idx += 1
print
print

#df = pd.DataFrame(dfx, index=pd.MultiIndex.from_tuples(indexes, index_names))
df_all = pd.DataFrame(dfx, index=indexes)
df_hist = pd.DataFrame(gfx, index=indexes)




