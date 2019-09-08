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

import f_folders as folder
from f_args import *
from f_date import *
from f_file import *
from f_calc import *
from f_dataframe import *
from f_rolldate import *
from f_iqfeed import *
from f_chart import *

project_folder = join(data_folder, "vix_es")

#-----------------------------------------------------------------------------------------------------------------------

q_columns = ['d4','d3','d2','d1','unch','u1','u2','u3','u4']

def calc_quartile_hit_ratios(df, lookback):
    for col in q_columns:
        Qcol = 'Q' + col
        df[col] = df[Qcol].shift(1).rolling(lookback).mean()
    return


def get_sessions(df, session_start=time(20,0), session_length=timedelta(hours=21), session_interval=timedelta(hours=1)):
    print "Splitting data into sessions ...",
    unique_dates = df_get_unique_dates(df)
    sessions = []
    for dt in unique_dates:
        dt1 = datetime(dt.year, dt.month, dt.day, session_start.hour, session_start.minute, 0)
        dt2 = dt1 + session_length - session_interval
        df_sess = df[(df['DateTime']>=dt1) & (df['DateTime']<=dt2)]     # all rows in session (between dt1 and dt2)
        if df_sess.shape[0] > 0:
            sessions.append(df_sess)
    print "Done."
    return sessions

def calculate_spread_OHLC(df, sessions):
    print "Calculating spread OHLC for each session ...",
    # Create 4 new columns in dataframe to hold spread open/high/low/close (for SESSION)
    df['spread_O'] = np.nan
    df['spread_H'] = np.nan
    df['spread_L'] = np.nan
    df['spread_C'] = np.nan
    for df_sess in sessions:
        r1, r2 = df_first_last(df_sess)
        df.loc[df_sess.index, 'spread_O'] = r1['Open_spread']
        df.loc[df_sess.index, 'spread_H'] = max(df_sess['Close_spread'].max(), r1['Open_spread'])
        df.loc[df_sess.index, 'spread_L'] = min(df_sess['Close_spread'].min(), r1['Open_spread'])
        df.loc[df_sess.index, 'spread_C'] = r2['Close_spread']
    print "Done."
    return df.dropna()

def get_last_session_rows(df, sessions):
    print "Getting last row from each session (effective close) ...",
    df_day = pd.DataFrame()
    for df_sess in sessions:
        df_temp = df.loc[df_sess.index, :]
        df_day = pd.concat([df_day, df_temp.tail(1)])
    print "Done."
    return df_day

def calculate_ATR(df, df_day, lookback=10):
    #lookback = 10
    df_day['spread_mean'] = df_day.spread_O.rolling(window=lookback).mean()
    df_day['range'] = df_day.spread_H - df_day.spread_L
    df_day['avg_range'] = df_day.range.rolling(window=lookback).mean()
    #df_day.dropna(inplace=True)
    df_day['atr_low'] = df_day.spread_O - df_day.avg_range
    df_day['atr_high'] = df_day.spread_O + df_day.avg_range
    #df_day['atr_low'] = df_day.spread_mean - df_day.avg_range
    #df_day['atr_high'] = df_day.spread_mean + df_day.avg_range

    df['spread_mean'] = np.nan
    df['atr_low'] = np.nan
    df['atr_high'] = np.nan

    print "Copying mean, ATR_low and ATR_high to each row ...",
    for df_sess in sessions:
        dt = df_sess.tail(1).squeeze()['DateTime']
        row = df_day[df_day['DateTime'] == dt]
        if row.shape[0] > 0:
            df.loc[df_sess.index, 'spread_mean'] = row.squeeze()['spread_mean']
            df.loc[df_sess.index, 'atr_low'] = row.squeeze()['atr_low']
            df.loc[df_sess.index, 'atr_high'] = row.squeeze()['atr_high']
    print "Done."
    return df

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

# Create the dataframe files required for VX quartile data:
# "@VX_futures.daily.DF.csv"
# "@VX_continuous.daily.DF.csv"
# "@VX_futures.minute.DF.csv"
# "VVIX.XO.daily.DF.csv"
def download_historical_data_for_vx_quartiles():
    df_ = create_historical_futures_df("@VX")
    df_ = create_historical_futures_df("@VX", interval=INTERVAL_MINUTE, beginFilterTime='093000', endFilterTime='160000')
    df_ = create_continuous_df("@VX", get_roll_date_VX)
    df_ = create_historical_contract_df("VVIX.XO")
    return

# Create the dataframe files required for VX quartile data:
# "@NQ_futures.daily.DF.csv"
# "@NQ_continuous.daily.DF.csv"
# "@NQ_futures.minute.DF.csv"
# "VXN.XO.daily.DF.csv"
def download_historical_data_for_nq_quartiles():
    df_ = create_historical_futures_df("@NQ")
    df_ = create_historical_futures_df("@NQ", interval=INTERVAL_MINUTE, beginFilterTime='093000', endFilterTime='160000')
    df_ = create_continuous_df("@NQ", get_roll_date_NQ)
    df_ = create_historical_contract_df("VXN.XO")
    return

# This function does the bulk of the work: It calculates the quartile hits/ratios and outputs them to a quartiles dataframe file.
# (output filename similar to "ES_VIX_quartiles (5-day lookback).DF.csv")
def create_quartile_df(symbol_root, vsymbol, sym, vsym, session_open_time, session_close_time, lookback_days=5, ticksize=0.0):
    filename1 = "{0}_continuous.daily.DF.csv".format(symbol_root)
    filename2 = "{0}_contract.daily.DF.csv".format(vsymbol)
    output_filename = "{0}_{1}_quartiles ({2}-day lookback).DF.csv".format(sym, vsym, lookback_days)

    print "project folder:", quoted(project_folder)
    print "input files:", quoted(filename1), quoted(filename2)
    print "lookback days:", lookback_days
    print

    df_es = read_dataframe(filename1)
    df_es = df_es.drop(['High', 'Low'], 1)
    df_es.rename(
        columns={'Symbol': 'Symbol_' + sym, 'Open': 'Open_' + sym, 'Close': 'Close_' + sym, 'Volume': 'Volume_' + sym,
                 'oi': 'oi_' + sym}, inplace=True)

    df_vix = read_dataframe(filename2)
    df_vix = df_vix.drop(['High', 'Low', 'Volume'], 1)
    df_vix.rename(
        columns={'Symbol': 'Symbol_' + vsym, 'Open': 'Open_' + vsym, 'Close': 'Close_' + vsym, 'oi': 'oi_' + vsym},
        inplace=True)

    df = pd.merge(df_es, df_vix, on=['DateTime'])

    # Output this quartile summary data (for potential analysis or debugging)
    summary_filename = "{0}_{1}_quartile_summary.DF.csv".format(sym, vsym)
    write_dataframe(df, join(project_folder, summary_filename))
    print
    print "{0}/{1} summary (daily data) output to file: {2}".format(vsym, sym, quoted(summary_filename))
    print

    symbol_column = 'Symbol_' + sym
    close_column = 'Close_' + sym
    vclose_column = 'Close_' + vsym

    symbols = df_get_sorted_symbols(df, symbol_column)  # get list of unique futures symbols in our data

    rows = []  # store our row tuples here (they will eventually be used to create a dataframe)

    df_1min = read_dataframe("{0}_futures.minute.DF.csv".format(symbol_root))  # read in the 1-minute data

    # For each specific future symbol, perform our quartile calculations
    for symbol in symbols:
        print "Processing future:", symbol
        df_es = df_1min[df_1min.Symbol == symbol]  # read_dataframe(get_df_pathname(es))
        dfx = df[df[symbol_column] == symbol]
        for (ix, row) in dfx.iterrows():
            if not (ix + 1) in dfx.index:  # if we are at the end of the dataframe rows (next row doesn't exist)
                continue

            # Use Close of ES and VIX
            es_close = row[close_column]  # ES close
            vix_close = row[vclose_column]  # VIX close
            std = round(Calc_Std(vix_close), 4)  # calculate standard deviation
            dt_prev = row['DateTime']  # date of ES/VIX close to use
            dt = dfx.loc[ix + 1].DateTime  # following date (date of actual quartile calculation)

            # Get the 1-minute bars for the day session (for date following the date of ES/VIX close)
            exchange_open = dt.replace(hour=session_open_time.hour, minute=session_open_time.minute)
            exchange_close = dt.replace(hour=session_close_time.hour, minute=session_close_time.minute)
            df_day = df_es[(df_es.DateTime > exchange_open) & (
            df_es.DateTime <= exchange_close)]  # ES 1-minute bars for day session

            # Get OHLC for the day session 1-minute data
            day_open = df_day.iloc[0]['Open']
            day_high = max(df_day.High.max(), day_open)  # check in case the open is higher
            day_low = min(df_day.Low.min(), day_open)  # check in case the open is lower
            row_count = df_day.shape[0]
            day_close = df_day.iloc[row_count - 1]['Close']

            # For each quartile, determine if it was hit during the day session
            hit_quartile = {}
            (q_list, q_dict) = Calc_Quartiles(es_close, std, ticksize=ticksize)
            for i in range(+4, -5, -1):
                quartile = q_dict[i]
                if day_low <= quartile and day_high >= quartile:
                    hit_quartile[i] = 1
                else:
                    hit_quartile[i] = 0
                    # print i, quartile, hit_quartile[i]

            rows.append((dt, symbol, es_close, vix_close, std, day_open, day_high, day_low, day_close, hit_quartile[-4],
                         hit_quartile[-3], hit_quartile[-2], hit_quartile[-1], hit_quartile[0], hit_quartile[1],
                         hit_quartile[2], hit_quartile[3], hit_quartile[4]))

    # Create new Quartile dataframe from the rows we have calculated
    dfQ = pd.DataFrame(rows, columns=['DateTime', 'Symbol', 'Prev_Close', 'Prev_VClose', 'Std', 'Open_Session',
                                      'High_Session', 'Low_Session', 'Close_Session', 'Qd4', 'Qd3', 'Qd2', 'Qd1',
                                      'Qunch', 'Qu1', 'Qu2', 'Qu3', 'Qu4'])
    calc_quartile_hit_ratios(dfQ, lookback_days)
    dfQ.dropna(inplace=True)
    write_dataframe(dfQ, join(project_folder, output_filename))
    print
    print "Quartile analysis output to file:", quoted(output_filename)
    print
    return dfQ

################################################################################

print "Calculating quartiles using volatility index to determine standard deviation"
print


#set_default_dates_latest()

# Request the IQFeed data and create the dataframe files for the continuous ES and the VIX cash index (default is INTERVAL_DAILY)
download_historical_data_for_es_quartiles()
dfQ = create_quartile_df("@ES", "VIX.XO", "ES", "VIX", time(9,30), time(16,0), lookback_days=5, ticksize=0.25)
STOP()


# Request the IQFeed data and create the dataframe files for the continuous NQ and the VXN cash index (default is INTERVAL_DAILY)
download_historical_data_for_nq_quartiles()
dfQ = create_quartile_df("@NQ", "VXN.XO", "NQ", "VXN", time(9,30), time(16,0), lookback_days=5)

# Request the IQFeed data and create the dataframe files for the continuous VX and the VVIX cash index (default is INTERVAL_DAILY)
download_historical_data_for_vx_quartiles()
dfQ = create_quartile_df("@VX", "VVIX.XO", "VX", "VVIX", time(9,30), time(16,0), lookback_days=5)




