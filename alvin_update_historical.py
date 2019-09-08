from __future__ import print_function
from iqapi import IQHistoricData
from datetime import datetime, time, timedelta
from dateutil import relativedelta
from pandas.tseries.offsets import BDay
from os.path import isfile, join
import numpy as np
import pandas as pd
import math
import sys

#-------------------------------------------------------------------------------
from f_folders import *
from f_chart import *
from f_dataframe import *
from f_rolldate import *
from f_date import *
from f_file import *
from f_calc import *
from f_iqfeed import *
from f_chart import *
from f_plot import *
#from f_args import *

#-------------------------------------------------------------------------------

# Create the dataframe files required for VX contango data:
# "@VX_futures.daily.DF.csv"
# "@VX_continuous.daily.DF.csv"
# "@VX_contango.daily.DF.csv"
def download_historical_data_for_vx_contango(interval=INTERVAL_DAILY, force_download=False):
    # To get contango data:
    # (1) retrieve latest futures data
    # (2) create continuous front-month from this futures data
    # (3) create contango from this continuous data (will infer one-month-out and two-months-out from front-month symbol)
    df_ = create_historical_futures_df("@VX", interval=interval, force_redownload=force_download)
    df_ = create_continuous_df("@VX", get_roll_date_VX, interval=interval)
    df_ = create_historical_calendar_futures_df("@VX", mx=0, my=1, interval=interval, force_redownload=force_download)
    df_ = create_continuous_calendar_ETS_df("@VX", 0, 1, interval=interval)
    df_ = create_historical_calendar_futures_df("@VX", mx=1, my=2, interval=interval, force_redownload=force_download)
    df_ = create_continuous_calendar_ETS_df("@VX", 1, 2, interval=interval)
    df_ = create_historical_contract_df("VIX.XO", force_redownload=force_download)
    df_ = create_contango_df("@VX", interval=interval)
    # Read the contango data from the newly-created .CSV dataframe file
    #df = read_dataframe("@VX_contango.{0}.DF.csv".format(str_interval(data_interval)))
    return

# Create the dataframe files required for ES quartile data:
# "@ES_futures.daily.DF.csv"
# "@ES_continuous.daily.DF.csv"
# "@ES_futures.minute.DF.csv"
# "VIX.XO.daily.DF.csv"
def download_historical_data_for_es_quartiles(force_download=False):
    df_ = create_historical_futures_df("@ES", force_redownload=force_download)
    df_ = create_historical_futures_df("@ES", interval=INTERVAL_MINUTE, beginFilterTime='093000', endFilterTime='160000', force_redownload=force_download)
    df_ = create_continuous_df("@ES", get_roll_date_ES)
    df_ = create_historical_contract_df("VIX.XO", force_redownload=force_download)
    return

########################################################################################################################

#set_default_dates_latest()

#---------- QUARTILES ----------

# Request the IQFeed data and create the dataframe files for ES/VIX quartiles ("ES_VIX_quartiles (5-day lookback).DF.csv")
download_historical_data_for_es_quartiles(force_download=True)
dfQ = create_quartile_df("@ES", "VIX.XO", "ES", "VIX", time(9,30), time(16,0), lookback_days=5, ticksize=0.25)
STOP(dfQ)


#---------- VX CONTANGO ----------

# Request the IQFeed data and create the dataframe file containing the VX contango data ("@VX_contango.daily.DF.csv")
download_historical_data_for_vx_contango(force_download=True)
contango_filename = "@VX_contango.daily.DF.csv"
df_vx = read_dataframe(contango_filename)
df_vx.rename(columns={'Close0':'Close_VX', 'Symbol':'Symbol_VX', 'Close1':'Close_VX1', 'Symbol1':'Symbol_VX1', 'Close2':'Close_VX2', 'Symbol2':'Symbol_VX2', 'm0_m1':'VX_m0_m1', 'm1_m2':'VX_m1_m2', '1x3x2':'VX_1x3x2'}, inplace=True)
df_vx = df_contango_ranges(df_vx)   # Add a 'contango_range' column to our dataframe
# Add a few contango-related columns to ES quartile data
#dfQ = pd.merge(dfQ, df_vx[['DateTime', 'contango', 'contango_1x3x2', 'contango_range']], on='DateTime')
df = read_dataframe(join(data_folder, "vix_es", "ES_VIX_quartiles (5-day lookback).DF.csv"))
df = pd.merge(df, df_vx, on='DateTime')
#df_vix = create_historical_contract_df("VIX.XO")
df_vix = read_dataframe("VIX.XO_contract.daily.DF.csv")
df = pd.merge(df, df_vix, on='DateTime')
#df_rename_columns(df, old_names_list=None, new_names_list=None)
df = df_rename_columns_dict(df, names_dict={'Symbol_x':'Symbol_ES', 'Symbol_y':'Symbol_VIX', 'Open':'Open_VIX', 'High':'High_VIX', 'Low':'Low_VIX', 'Close':'Close_VIX'})
df.drop(['Volume', 'oi'], axis=1, inplace=True)
df['VIX_discount'] = df['Close_VIX'] - df['Close_VX']

write_dataframe(df, "@VX_contango.EXPANDED.daily.DF.csv")

print("Done.")




"""
# This function does the bulk of the work: It calculates the quartile hits/ratios and outputs them to a quartiles dataframe file.
# (output filename similar to "ES_VIX_quartiles (5-day lookback).DF.csv")
def create_quartile_df(symbol_root, vsymbol, sym, vsym, session_open_time, session_close_time, lookback_days=5, ticksize=0.0):
    filename1 = "{0}_continuous.daily.DF.csv".format(symbol_root)
    filename2 = "{0}_contract.daily.DF.csv".format(vsymbol)
    output_filename = "{0}_{1}_quartiles ({2}-day lookback).DF.csv".format(sym, vsym, lookback_days)

    print("project folder: '{0}'".format(project_folder))
    print("input files: '{0}' '{1}'".format(filename1, filename2))
    print("lookback days: {0}".format(lookback_days))
    print()

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
    print()
    print("{0}/{1} summary (daily data) output to file: {2}".format(vsym, sym, quoted(summary_filename)))
    print()

    symbol_column = 'Symbol_' + sym
    close_column = 'Close_' + sym
    vclose_column = 'Close_' + vsym

    symbols = df_get_sorted_symbols(df, symbol_column)  # get list of unique futures symbols in our data

    rows = []  # store our row tuples here (they will eventually be used to create a dataframe)

    df_1min = read_dataframe("{0}_futures.minute.DF.csv".format(symbol_root))  # read in the 1-minute data

    # For each specific future symbol, perform our quartile calculations
    for symbol in symbols:
        print("Processing future: {0}".format(symbol))
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
    print()
    print("Quartile analysis output to file: '{0}'".format(output_filename))
    print()
    return dfQ
"""

