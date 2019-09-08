from iqapi import IQHistoricData
from datetime import datetime, timedelta
from dateutil import relativedelta
import pandas as pd
from pandas.tseries.offsets import BDay
import numpy as np
import os.path
import math
import sys
"""
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import lstm, time  # helper libraries
"""

#-------------------------------------------------------------------------------
from f_folders import *
from f_date import *
from f_chart import *
from f_dataframe import *
from f_iqfeed import *

project_folder = join(data_folder, "vix_es")

#-------------------------------------------------------------------------------


# Given an interval representing daily, hour, minute, etc.
# Return a string representation of the interval that can be used in filenames, etc.
def str_interval(interval):
    if interval == INTERVAL_DAILY:
        return "DAILY"
    elif interval == INTERVAL_MINUTE:
        return "MINUTE"
    elif interval == INTERVAL1HOUR:
        return "HOUR"
    else:
        return str(interval)
    

# Given a symbol root (ex: "@VX") and a datetime object
# Return the mYY futures contract symbol (ex: "@VXM16") 
def mYY(symbol_root, dt):
    m = dt.month
    y = dt.year
    return symbol_root + monthcodes[m-1] + str(y)[-2:]



def show_contango_chart(df):
    # CREATE THE CHART
    lines = []
    lines.append(get_chart_line(df, 'contango1x3x2', line_color='rgb(255,0,0)')) #, line_dash='dot'))
    lines.append(get_chart_line(df, 'contango', line_color='rgb(0,0,255)')) #, line_dash='dot'))
    lines.append(get_chart_line(df, 'Close0', line_name='VX'))

    # Highlight the negative ranges on the chart
    #for dfx in rows:
    #    lines.append(get_chart_line(dfx, 'contango1x3x2', line_color='rgb(255,0,0)', line_width=3))

    show_chart(lines, 'VIX_contango')
    return

def get_es_quartiles(dt1=datetime(2003,1,1), dt2=datetime.now()):
    df_vix = get_historical_contract("VIX.XO", dt1, dt2)
    df_es = get_historical_contract("@ES#", dt1, dt2)                   # TODO: change this to use our continuous contract
    df_qes = get_quartiles_df(df_es, df_vix, suffixes=('_ES','_VIX'))
    return df_qes

def get_vx_quartiles(dt1=datetime(2003,1,1), dt2=datetime.now()):
    df_vvix = get_historical_contract("VVIX.XO", dt1, dt2)
    df_vx = get_historical_contract("@VX#", dt1, dt2)                   # TODO: change this to use our continuous contract
    df_qvx = get_quartiles_df(df_vx, df_vvix, suffixes=('_VX','_VVIX'))
    return df_qvx

def update_es_session_1min(dt1=datetime(2015,1,1), dt2=datetime.now()):
    pathname = join(project_folder, "@ES_session_1min.DF.csv")
    if exists(pathname):
        df = read_dataframe(pathname)
        last_dt = df.tail(1).squeeze()['DateTime']
        if dt2 > last_dt:
            df = get_historical_contract("@ES#", last_dt, dt2, INTERVAL_MINUTE, "083000", "150000")   # 8:30am to 3:00pm
            append_dataframe(df, pathname)
    else:
        df = get_historical_contract("@ES#", dt1, dt2, INTERVAL_MINUTE, "083000", "150000")   # 8:30am to 3:00pm
        write_dataframe(df, pathname)
    return df

def STOP(x=None):
    print("\n!!!STOP!!!\n")
    print(x)
    sys.exit()
    
    
################################################################################
################################################################################
################################################################################


# TASKS:
# 1. Risk indicator
# 2. Quartiles/Hits for ES/VIX/VVIX
# 3. Move from using the @ES# (and other) to our own continuous contracts
# 4. Contango 1x3x2
# 5. Calculate/Analyze Quartile ranges
# 6. Calculate/Analyize ES price movement by hour-of-day
# 7. Initial LSTM Neural Net results for ES quartiles
# 8. Test ES Neural Net adding other factors (contango, range, contango1x3x2, etc.)
# 9. Pat's spread analysis
# 10. Finalize clean up copper analysis for quick-and-easy addition of new price data
# 11. Live prices -- use IQFeed to request live prices and use these to perform some of our indicator calculations in real time
# 12. ARIMA
# 13. TTAPI -- get simple app working that gets us up and running with TTAPI automation
# 14. Buy GPU and build Linux box optimized for machine learning
# 15. Need to be polished and standardized: continuous, roll dates, trades (backtesting)


dt1 = datetime(2003, 1, 1)
dt2 = datetime.now()

#df = get_historical("QCL#", dt1, dt2)
#df = get_historical("@VX#", dt1, dt2)
#print df




# Download all historical ES futures and output to file '@ES.csv'
#df = create_historical_futures_df("@ES", 2003, 2017)
#df = create_historical_futures_df("@ES", 2016, 2017, INTERVAL_1MIN)

# Retrieve the 1-minute ES data and output to file '@ES_session_1min.DF.csv'
df = update_es_session_1min()



# Read the 1-minute ES data from file '@ES_session_1min.DF.csv' and output OHLC summary to file '@ES_session_ohlc.DF.csv'
df_1min = read_dataframe(join(project_folder, "@ES_session_1min.DF.csv"))
df_ohlc = get_ohlc_df(df_1min)
STOP(df_ohlc)
#filename = "@ES_session_ohlc.DF.csv"
#df_ohlc.to_csv(filename, index=False)
#print("Output OHLC summary data to '{0}'".format(filename))

# Request the daily VIX.XO and @ES data and output the Quartile values to '@ES_quartiles.DF.csv'
#dfq = get_es_quartiles()
#filename = "@ES_quartiles.DF.csv"
#dfq.to_csv(filename, index=False)
#print("Output quartiles to '{0}'".format(filename))
                 
dfq = pd.read_csv("@ES_quartiles.DF.csv", parse_dates=['DateTime'])
df_ohlc = pd.read_csv("@ES_session_ohlc.DF.csv", parse_dates=['DateTime'])

# Calculate the Quartile Hits from Quartiles and OHLC summary data and output to '@ES_quartile_hits.DF.csv'
#dfz = get_quartile_hits(dfq, df_ohlc)
#filename = "@ES_quartile_hits.DF.csv"
#dfz.to_csv(filename, index=False)
#print("Output quartile hits to '{0}'".format(filename))

df_hit = pd.read_csv("@ES_quartile_hits.DF.csv", parse_dates=['DateTime'])
df_vx = pd.read_csv("contango_@VX.DF.csv", parse_dates=['DateTime'])

# Calculate the Quartile Hit Ratios from Quartile Hits and VIX Contango data and output to '@ES_quartile_hit_ratios.DF.csv' 
#df_hr = get_hit_ratios(df_vx, df_hit, 5)
#filename = "@ES_quartile_hit_ratios.DF.csv"
#df_hr.to_csv(filename, index=False)
#print("Output quartile hit ratios to '{0}'".format(filename))

df_hr = pd.read_csv("@ES_quartile_hit_ratios.DF.csv", parse_dates=['DateTime'])


df = get_vix_quartiles()
i1 = df.columns.get_loc('Qd4')
i2 = df.columns.get_loc('Qu4')
cols = ['DateTime']
cols.extend(df.columns.values[i1:i2+1])
#df_vix = df.iloc[:,i1:i2]
df_vix = df.loc[:,cols]
#print(df_vix)
filename = "@VIX_quartiles.DF.csv"
df_vix.to_csv(filename, index=False)
print("Output quartiles to '{0}'".format(filename))



sys.exit()




symbol_root = '@VX'

# Create the '@VX.csv' file containing futures prices for the given year range
#create_historical_futures_df(symbol_root, 2003, 2017)

# Using the '@VX.csv' data file, create 'contango_@VX.raw.DF.csv' containing contango data
#dfz = create_contango_df(symbol_root)

# From the raw data, create the continuous contract contango file
input_filename = "contango_{0}.raw.DF.csv".format(symbol_root)
continuous_filename = "contango_{0}.DF.csv".format(symbol_root)
#dfz = create_continuous_df(input_filename, continuous_filename, get_roll_date_VX)




# Read in the 'contango_@VX.DF.csv' file
df = pd.read_csv(continuous_filename, parse_dates=['DateTime'])

#show_contango_chart(df)


"""
# Get datapoints where the 1x3x2 contango is negative
df_neg = df[df['contango1x3x2']<0]
rows = get_ranges_df(df, df_neg)            # get contiguous date ranges of the negative 1x3x2 values

# Find the slope leading into (before) a point where 1x3x2 is negative
# vs leading out (after)
li = []
for dfx in rows:
    dt1 = dfx.iloc[0].DateTime
    dt2 = dfx.iloc[dfx.shape[0]-1].DateTime
    #print dt1.strftime('%Y-%m-%d'), dt2.strftime('%Y-%m-%d'),
    ix = df[df.DateTime==dt1].index.values[0]
    length = 10
    ix1 = max(ix-length,0)
    ix2 = min(ix+length,df.shape[0]-1)
    dfy = df[ix1:ix]
    #print dfy.shape
    z = np.polyfit(dfy.index, dfy.Close0, 1)
    slope1 = round(z[0], 4)
    #print "{0:7.4f}".format(slope1),
    # f = np.poly1d(z)
    dfy = df[ix:ix2]
    if dfy.shape[0]==0:
        break
    z = np.polyfit(dfy.index, dfy.Close0, 1)
    slope2 = round(z[0], 4)
    #print "{0:7.4f}".format(slope2)
    li.append((slope1, slope2))
slopes = np.array(li)
s1 = abs(slopes[:,0])
s2 = abs(slopes[:,1])
print s1.mean()
print s2.mean()
"""














