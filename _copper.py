from __future__ import print_function
import urllib
import json
import requests
import hmac
import hashlib
from os.path import join
import sys

#-----------------------------------------------------------------------------------------------------------------------

from f_folders import *
from f_date import *
from f_plot import *
from f_stats import *
from f_dataframe import *
from f_file import *
"""
import f_quandl as q
from c_Trade import *
"""
import decimal

#-----------------------------------------------------------------------------------------------------------------------


# from (already created) copper data file ("copper_settle_discount.2.DF.csv")
# retrieve a dataframe with only the lines of data that match one of the given HG calendars
# where cal like 'NU'
def df_get_copper_data_for_calendar(df, cal):
    data_filename = "copper_settle_discount.2.DF.csv"       # copper data filename
    df = read_dataframe(join(df_folder, data_filename), display=False)
    symbol1 = 'QHG' + cal[0]
    symbol2 = 'QHG' + cal[1]
    _df = df[df['Symbol'].str.contains(symbol1) & df['Symbol'].str.contains(symbol2)]
    return _df

# from (already created) copper data file ("copper_settle_discount.2.DF.csv")
# retrieve a dataframe with only the lines of data that match one of the given HG calendars
# where cal1 like 'NQ' and cal2 like 'NU'
def df_get_copper_data_for_calendars(df, cal1, cal2):
    dfx = df_get_copper_data_for_calendar(df, cal1)
    dfy = df_get_copper_data_for_calendar(df, cal2)
    _df = dfx.append(dfy, ignore_index=True)
    _df = _df.sort_values(by=['DateTime'], ascending=[True])
    return _df

# Test the filtering of the existing copper data file to include only the specified calendar month (or months)
# where cal1 and cal2 like 'KM', 'KN', 'NQ', ...
def copper_filtercal(cal1, cal2 = None):
    cal1 = cal1.upper()
    if cal2 is None:
        # get copper data for one HG calendar
        dfx = df_get_copper_data_for_calendar(df, cal1)
        # print(dfx)
        write_dataframe(dfx, join(df_folder, "copper_calendar_{0}.DF.csv".format(cal1)))
    else:
        # get copper data for two HG calendars
        cal2 = cal2.upper()
        dfx = df_get_copper_data_for_calendars(df, cal1, cal2)
        # print(dfx)
        write_dataframe(dfx, join(df_folder, "copper_calendar_{0}_{1}.DF.csv".format(cal1, cal2)))


########################################################################################################################

# command-line arguments
args = sys.argv
nargs = len(sys.argv)
#print("{0} args: {1} {2}".format(nargs, args[0], args[1]))

update_historical = False

# Re-download the copper data to a local dataframe file:
#update_copper_dataframe()

data_filename = "copper_settle_discount.2.DF.csv"
#df = read_dataframe(join(df_folder, data_filename))
df = read_dataframe(join(df_folder, data_filename), display=False)

# --- TEST functions ---
#copper_filtercal('KM')
#copper_filtercal('KM', 'KN')

if nargs < 2:
    print('Try one of the following:\n')
    print('_copper.py filtercal')
    print('     Filter the copper data to generate a new data file containing only specified HG calendars')
    sys.exit()

#----------------------------------------------------------------------------------------------------------------------
# command-line: _copper filtercal NQ
#               _copper filtercal NQ NU
# Filtering of the existing copper data file to include only the specified calendar month (or months)
#----------------------------------------------------------------------------------------------------------------------
if nargs > 1 and args[1].lower() == "filtercal":
    # args[0] is the full path to the python script (ex: "C:/Users/Trader/Dropbox/dev/python/playground_copper.py")
    if nargs < 3:
        print('Filter for specified copper calendar data (i.e. "QHGK-QHGN") from "coppper_settle_discount.2.DF.csv"')
        print('Output filename like "copper_calendar_KN.DF.csv" or "copper_calendar_NQ_NU.DF.csv"\n')
        print('usage: _copper.py filtercal CC\n       _copper.py filtercal CC CC\n\nwhere CC is two monthcodes like "NQ" or "NU"')
        sys.exit()

    if nargs == 3:
        #print("filtercal: {0} {1}".format(args[1], args[2]))
        copper_filtercal(args[2])
    else:
        #print("filtercal: {0} {1} {2}".format(args[1], args[2], args[3]))
        copper_filtercal(args[2], args[3])

#----------------------------------------------------------------------------------------------------------------------

