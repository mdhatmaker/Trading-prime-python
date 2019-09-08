import json
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
import pandas_datareader.data as web
from os import listdir
from os.path import isfile, join, splitext
import sys

#-----------------------------------------------------------------------------------------------------------------------

execfile("f_folders.py")
execfile("f_date.py")

#-----------------------------------------------------------------------------------------------------------------------

hist = {}

reds = ['rgb(164, 10, 19)', 'rgb(246, 15, 29)', 'rgb(255, 30, 58)']
blues = ['rgb(18, 77, 134', 'rgb(31, 161, 234)', 'rgb(47, 242, 255)']


def print_stats(col):
    fmt = "0.2f"
    print
    print col.upper()
    print "min:", format(df[col].min(), fmt)
    print "max:", format(df[col].max(), fmt)
    print "mean:", format(df[col].mean(), fmt)
    print "median:", format(df[col].median(), fmt)
    print "stddev:", format(df[col].std(), fmt)
    #print "mode:"
    #print df[col].mode()
    #print "quantile:"
    #print df[col].quantile([0.25, 0.75])
    return

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

"""
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
"""

def get_sorted_symbols(df):
    g = df.groupby('Symbol').groups
    keys = g.keys()
    sorted_symbols = sorted(keys, compare_calendar)
    return sorted_symbols


        
########################################################################################################################

# Default to using entire date range
start_date = datetime(1900, 1, 1, 0, 0, 0)
end_date = datetime(2100, 1, 1, 0, 0, 0)

# The start_date and end_date should be modifiable from the command line
argv = sys.argv
argc = len(argv)

# Create dictionary of command-line arguments
args = {}
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

#-----------------------------------------------------------------------------------------------------------------------


