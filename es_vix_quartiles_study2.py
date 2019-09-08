from __future__ import print_function
import sys
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import math
from datetime import time
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------------------------------------------------

from f_folders import *
from f_dataframe import *
from f_rolldate import *
from f_iqfeed import *

#-----------------------------------------------------------------------------------------------------------------------

# Create the dataframe files required for ES quartile data:
# "@ES_futures.daily.DF.csv"
# "@ES_continuous.daily.DF.csv"
# "@ES_futures.minute.DF.csv"
# "VIX.XO.daily.DF.csv"
def download_historical_data_for_es_quartiles():
    df_ = create_historical_futures_df("@ES")
    df_ = create_historical_futures_df("@ES", interval=INTERVAL_MINUTE, beginFilterTime='093000', endFilterTime='160000')
    df_ = create_continuous_df("@ES", get_roll_date_ES)
    df_ = create_historical_contract_df("VIX.XO")
    return

# determine the percentage of the time we hit the specified quartile column in the given dataframe
def pct_hit(df, qcolumn):
    #vu1 = df.loc[:,qcolumn].values
    vu1 = df[qcolumn].dropna().values
    total = len(vu1)
    count1 = sum(vu1)
    count0 = total-count1
    if total == 0:
        STOP("ZERO!!!")
        return 0.0
    else:
        return round(count1 / float(total) * 100.0, 2)

def get_hit_ratios(df, date_str):
    qhr = df[df['DateTime']==date_str].loc[:,'d4':'u4'].values[0]
    qhr_up = df[df['DateTime']==date_str].loc[:,'u1':'u4'].values[0]
    qhr_down = df[df['DateTime']==date_str].loc[:,'d4':'d1'].values[0]
    return (qhr, qhr_up, qhr_down)

def get_quartile_hit_percentages(df, date_str):
    #qhr = [0.2,0.2,0.2,0.4,0.4,0.4,0.4,0.2,0.2]
    qhr, qhr_up, qhr_down = get_hit_ratios(df, date_str)
    dfx_up = df[(df.u1==qhr_up[0]) & (df.u2==qhr_up[1]) & (df.u3==qhr_up[2]) & (df.u4==qhr_up[3])]
    dfx_down = df[(df.d4==qhr_down[0]) & (df.d3==qhr_down[1]) & (df.d2==qhr_down[2]) & (df.d1==qhr_down[3])]

    dfux = dfx_up.index
    dfdx = dfx_down.index

    # can't increment index beyond row count
    last_ix = df.shape[0]-1
    dfux.drop(last_ix, errors='ignore')
    dfdx.drop(last_ix, errors='ignore')

    # Next index is following day
    dfux = dfux.map(lambda x: x+1)
    dfdx = dfdx.map(lambda x: x+1)

    # Get these "following day" rows from original dataframe
    dfu = df.loc[dfux]
    dfd = df.loc[dfdx]

    df1 = dfu.loc[:,'Qd4':'Qu4']
    df2 = dfd.loc[:,'Qd4':'Qu4']

    up_pct = {}
    up_pct['Qu1'] = pct_hit(df1, 'Qu1')
    up_pct['Qu2'] = pct_hit(df1, 'Qu2')
    up_pct['Qu3'] = pct_hit(df1, 'Qu3')
    up_pct['Qu4'] = pct_hit(df1, 'Qu4')

    down_pct = {}
    down_pct['Qd1'] = pct_hit(df2, 'Qd1')
    down_pct['Qd2'] = pct_hit(df2, 'Qd2')
    down_pct['Qd3'] = pct_hit(df2, 'Qd3')
    down_pct['Qd4'] = pct_hit(df2, 'Qd4')

    up_count = dfu.shape[0]
    down_count = dfd.shape[0]
    return up_pct, down_pct, up_count, down_count

def analysis_for_dates(df, date_list):
    for dt in date_list:
        date_str = dt.strftime('%Y-%m-%d')
        qhr, qhr_up, qhr_down = get_hit_ratios(df, date_str)
        up_pct, down_pct, up_count, down_count = get_quartile_hit_percentages(df, date_str)
        print("{0} {1}  ({10:2}) d4:{2:5.1f}%  d3:{3:5.1f}%  d2:{4:5.1f}%  d1:{5:5.1f}% | u1:{6:5.1f}%  u2:{7:5.1f}%  u3:{8:5.1f}%  u4:{9:5.1f}% ({11:2})".format(date_str, qhr, down_pct['Qd4'], down_pct['Qd3'], down_pct['Qd2'], down_pct['Qd1'], up_pct['Qu1'], up_pct['Qu2'], up_pct['Qu3'], up_pct['Qu4'], up_count, down_count))
    return

################################################################################

print("Calculating quartile hits when hit ratios (upside and downside, separately) match a specified date")
print()

df = read_dataframe("ES_VIX_quartiles (5-day lookback).DF.csv")

dt1 = df['DateTime'].min().to_pydatetime().date()
dt2 = df['DateTime'].max().to_pydatetime().date()

dates = df['DateTime'].tail(20).map(pd.Timestamp.date).values
analysis_for_dates(df, dates)

# Research the HIT RATIOS for this date:

date_str = '2017-10-04'

up_pct, down_pct, up_count, down_count = get_quartile_hit_percentages(df, date_str)

qhr, qhr_up, qhr_down = get_hit_ratios(df, date_str)

print("\nExamining date range: {0} to {1}".format(dt1, dt2))
print("\nMatch quartile hit ratios for date: {0}  {1}".format(date_str, qhr))

# Calculate percentage hit when UPSIDE matches specified date hit ratios
print("\nUPSIDE matches hit ratios: {0}".format(qhr_up))
print("How likely are we to hit upper quartiles?")
print("u1: {0}%  u2: {1}%  u3: {2}%  u4: {3}%".format(up_pct['Qu1'], up_pct['Qu2'], up_pct['Qu3'], up_pct['Qu4']))

# Calculate percentage hit when DOWNSIDE matches specified date hit ratios
print("\nDOWNSIDE matches hit ratios: {0}".format(qhr_down))
print("How likely are we to hit lower quartiles?")
print("d1: {0}%  d2: {1}%  d3: {2}%  d4: {3}%".format(down_pct['Qd1'], down_pct['Qd2'], down_pct['Qd3'], down_pct['Qd4']))


STOP()




# Request the IQFeed data and create the dataframe files for the continuous ES and the VIX cash index (default is INTERVAL_DAILY)
download_historical_data_for_es_quartiles()
dfQ = create_quartile_df("@ES", "VIX.XO", "ES", "VIX", time(9,30), time(16,0), lookback_days=5, ticksize=0.25)
