import sys
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import math
from datetime import time

#-----------------------------------------------------------------------------------------------------------------------

import f_folders as folder
from f_args import *
from f_date import *
from f_file import *
from f_calc import *
from f_dataframe import *
from f_rolldate import *
from f_iqfeed import *
from f_chart import *
from f_spread import *

#-----------------------------------------------------------------------------------------------------------------------


########################################################################################################################

print "Playground for various HOGO trading ideas"
print

y1 = 2017
y2 = 2017

data_interval = INTERVAL_MINUTE
#data_interval = INTERVAL_HOUR
"""
df_ = create_historical_futures_df("QHO", y1, y2, interval=data_interval)
df_ = create_historical_futures_df("GAS", y1, y2, interval=data_interval)
# Use same roll date for both QHO and GAS, otherwise HOGO spread would be incorrect around roll
df_ = create_continuous_df("QHO", get_roll_date_QHO, interval=data_interval)
df_ = create_continuous_df("GAS", get_roll_date_QHO, interval=data_interval)
"""
df_ho = read_dataframe("QHO_continuous.{0}.DF.csv".format(str_interval(data_interval)))
df_go = read_dataframe("GAS_continuous.{0}.DF.csv".format(str_interval(data_interval)))

df = get_spread(df_ho, df_go, fn_price=spread_price_HOGO, suffixes=('_HO','_GO'))

# session times = 20:00 to 17:00 (first hour closes at 21:00, last hour closes at 17:00)
sessions = df_get_sessions(df, session_interval=timedelta(minutes=1))
#sessions = df_get_sessions(df, session_interval=timedelta(hours=1))

# make bars
print "Making bars ..."
bar_interval = timedelta(minutes=30)
unique_dates = df_get_unique_dates(df)
first_date = unique_dates[0]
last_date = unique_dates[len(unique_dates)-1]
last_dt = datetime.combine(last_date, time(0,0))
minutes = int(bar_interval.total_seconds() // 60)
#hours = bar_interval.total_seconds() // (60 * 60)
hours = int(minutes // 60)
minutes = minutes % 60
bars = {}
dt1 = datetime(first_date.year, first_date.month, first_date.day, hours, minutes, 0)
while dt1 < last_dt:
    dt2 = dt1 + bar_interval
    df_temp = df[(df['DateTime'] > dt1) & (df['DateTime'] <= dt2)].copy()
    bars[dt1] = df_temp
    dt1 += bar_interval
    if dt1.day == 1 and dt1.hour == 0 and dt1.minute == 0:
        print dt1, "    {0} bars".format(len(bars))
print "Done."

STOP(bars[0])

# calculate the spread OHLC for each daily session
df = calculate_spread_OHLC(df, sessions)

# create a dataframe that contains only the LAST row from each session
df_day = get_last_session_rows(df, sessions)

# calculate the ATRs (and chart)
df = calculate_spread_ATR(df, df_day, 10)
#quick_chart(df, ['Close_spread','spread_mean','atr_low','atr_high'], title='HOGO')

"""
df = get_spread(df_ho, df_go, fn_price=spread_price_HOGO, suffixes=('_HO','_GO'))

# modify the dataset to create a 'session_date'
session_start = time(20, 0)
session_end = time(17, 0)
# calculate the shift needed from 24-hour mark to session start time
shift_time = subtract_times(time(23,0), session_start)
shift_minutes = shift_time.hour * 60 + shift_time.minute + 60   # we add 60 because we subtract from 23 vs 24
#df['shift_dt'] = pd.to_datetime(df['DateTime'])
df['add_minutes'] = shift_minutes
df['session_date'] = (df['DateTime'] + df['add_minutes'].values.astype("timedelta64[m]")).dt.date
df['session_date'] = df['session_date'].astype('datetime64[ns]')
#df.drop(['add_minutes'], axis=1, inplace=True)

print("Making bars ...")
bars = Bars(df, bar_minutes=30)
#df = bars.df

# Let's filter our data to include only DAY SESSION of 8:30am-3:00pm (remove other bars)
# (bars 25-37 represent 8:30am-3:00pm)
bars.filter_bars(23, 34)
bars.filter_minimum_session_bars(10)
#bars.print_dates()

lookback = 40
bs = []
for i in range(100, 150):
    bs.append(bars.slice(i-lookback-1, i-1))
    #bslice.print()

pickle.dump(bs, open("temp.pkl", "wb"))

bs = pickle.load(open("temp.pkl", "rb" ))

b0 = bs[0]
print(b0.closes)
#print(b0.bar_list)

pred = predict_out_of_sample_ARMA(b0.closes)
"""