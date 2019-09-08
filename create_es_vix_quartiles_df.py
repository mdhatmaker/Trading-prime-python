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
import scipy.stats as stats

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

################################################################################

update_historical = False

print("Calculating quartiles using volatility index to determine standard deviation")
print()

# 9-15 ,missed uq1
# 9-29 missed uq3

#set_default_dates_latest()

#df40 = read_dataframe("ES_VIX_quartiles (40-day lookback).DF.csv")
dfx = read_dataframe("ES_VIX_quartiles (5-day lookback).DF.csv")

#dfx = df40[['DateTime','d4','d3','d2','d1','unch','u1','u2','u3','u4']]
df5 = dfx[['DateTime','d4','d3','d2','d1','unch','u1','u2','u3','u4']]

lt_days = 20
for col in ['d4','d3','d2','d1','unch','u1','u2','u3','u4']:
    colavg = col + "avg"
    collt = col + "lt"
    colx = col + "x"
    coltr = col + "tr"
    df5[colavg] = df5[col].mean()
    df5[collt] = df5[col].rolling(lt_days).mean()
    #df5[colx] = (1 + df5[collt] - df5[col]) / 2.0
    #df5[colx] = (df5[collt] - df5[col]) / df5[col].std()
    #df5[colx] = (df5[collt] - df5[col]) / df5[col].rolling(lt_days).std()
    df5[colx] = df5[collt] - df5[col]
    df5[coltr] = df5[collt] - df5[colavg]
df5 = df5.round(4)

#df = df5[['DateTime','d4x','d3x','d2x','d1x','unchx','u1x','u2x','u3x','u4x','d4tr','d3tr','d2tr','d1tr','unchtr','u1tr','u2tr','u3tr','u4tr']].dropna()
#df = df5[['d4x','d3x','d2x','d1x','unchx','u1x','u2x','u3x','u4x','DateTime','d4tr','d3tr','d2tr','d1tr','unchtr','u1tr','u2tr','u3tr','u4tr']].dropna()
#df = df5[['DateTime','d4x','d3x','d2x','d1x','unchx','u1x','u2x','u3x','u4x', 'd4lt','d3lt','d2lt','d1lt','unchlt','u1lt','u2lt','u3lt','u4lt']].dropna()
df = df5[['DateTime','d4x','d3x','d2x','d1x','unchx','u1x','u2x','u3x','u4x']].dropna()

df_ = dfx[['DateTime','Qd4','Qd3','Qd2','Qd1','Qunch','Qu1','Qu2','Qu3','Qu4']]
df_.loc[:,'Qd4':'Qu4'] = df_.loc[:,'Qd4':'Qu4'].shift(-1)
df_ = df_.dropna()

df = df.merge(df_, on='DateTime')
df.loc[:,'Qd4':'Qu4'] = df.loc[:,'Qd4':'Qu4'].astype(int)
#df = df.round(2)

(corr, pvalue) = stats.pearsonr(df['d4x'], df['Qd4'])

r = 1
for col in ['d4','d3','d2','d1','unch','u1','u2','u3','u4']:
    #df['u4x'] = df['u4x']/(df5['u4'].mean()).round(r)
    colx = col + "x"
    #df[colx] = (df[colx]/(df5[col].mean())).round(r)
    #lbound = 0
    df[colx] = (df[colx] + (0 - df[colx].min())) / (df[colx].max() - df[colx].min())

df = df.round(2)

write_dataframe(df, 'ES_VIX_quartile_analysis.DF.csv')

STOP(df)


# Request the IQFeed data and create the dataframe files for the continuous ES and the VIX cash index (default is INTERVAL_DAILY)
if update_historical: download_historical_data_for_es_quartiles()
dfQ = create_quartile_df("@ES", "VIX.XO", "ES", "VIX", time(9,30), time(16,0), lookback_days=40, ticksize=0.25)
