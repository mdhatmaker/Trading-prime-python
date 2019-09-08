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

#-----------------------------------------------------------------------------------------------------------------------


#-------------- DataFrameForm functions ----------------------------------------
def df_add_diff_column(filename, column_name):
    pathname = join(df_folder, filename)
    df = read_dataframe(pathname)
    col = 'diff_' + column_name
    df[col] = df[column_name].diff()
    write_dataframe(df, pathname)

def df_add_ema_column(filename, column_name, periods):
    pathname = join(df_folder, filename)
    df = read_dataframe(pathname)
    min = int(.80 * periods)                    # minimum datapoints required for calculation
    i = column_name.index('_')
    if i < 0:
        col = 'EMA_' + column_name              # prepend 'EMA_' for new column name
    else:
        col = 'EMA_' + column_name[i+1:]        # replace everything before '_' with 'EMA_'
    df[col] = df[column_name].ewm(span=periods, min_periods=min, adjust=False).mean()
    df[col] = df[col].round(decimals=6)         # round our result
    write_dataframe(df, pathname)               # save the updated dataframe

def df_add_sma_column(filename, column_name, periods):
    pathname = join(df_folder, filename)
    df = read_dataframe(pathname)
    min = int(.80 * periods)            # minimum datapoints required for calculation
    i = column_name.index('_')
    if i < 0:
        col = 'SMA_' + column_name      # prepend 'SMA_' for new column name
    else:
        col = 'SMA_' + column_name[i+1:]  # replace everything before '_' with 'SMA_'
    df[col] = df[column_name].rolling(window=periods, min_periods=min).mean()
    df[col].round(decimals=6)           # round our result
    write_dataframe(df, pathname)       # save the updated dataframe

#-------------------------------------------------------------------------------

def get_arg(i):
    args = sys.argv
    if (len(args) > i):
        return args[i]
    else:
        return None

########################################################################################################################

print("Utilizing PrimeTrader-specific functions")
print()

# COMMAND LINE ARGUMENTS (arg[0] is script name)
args = sys.argv
###args = ['KRAKEN_OHLC', 'BCHUSD', '60']      # FOR TESTING ONLY!!!
args = ['SMA', 'diff_VWAP', '5']      # FOR TESTING ONLY!!!
if (len(args) > 0):
    print(len(args))
    print(get_arg(1), get_arg(2), get_arg(3))

    # ARG: "DIFF" <filename> <column_name>
    # ADD DIFF COLUMN TO DATAFRAME (dataframe file should reside in DF_DATA folder)
    if get_arg(1) == 'DIFF' and get_arg(2) and get_arg(3):
        filename = get_arg(2)
        column_name = get_arg(3)
        print("'{0}' '{1}'".format(filename, column_name))
        df_add_diff_column(filename, column_name)
        STOP()

    # ARG: "EMA" <filename> <column_name> <periods>
    # ADD EXPONENTIAL MOVING AVERAGE TO DATAFRAME (dataframe file should reside in DF_DATA folder)
    if get_arg(1) == 'EMA' and get_arg(2) and get_arg(3) and get_arg(4):
        filename = get_arg(2)
        column_name = get_arg(3)
        periods = get_arg(4)
        print("'{0}' '{1}' [{2}]".format(filename, column_name, periods))
        df_add_ema_column(filename, column_name, int(periods))
        STOP()

    # ARG: "SMA" <filename> <column_name> <periods>
    # ADD MOVING AVERAGE TO DATAFRAME (dataframe file should reside in DF_DATA folder)
    if get_arg(1) == 'SMA' and get_arg(2) and get_arg(3) and get_arg(4):
        filename = get_arg(2)
        column_name = get_arg(3)
        periods = get_arg(4)
        print("'{0}' '{1}' [{2}]".format(filename, column_name, periods))
        df_add_sma_column(filename, column_name, periods)
        STOP()
# command line arguments

STOP()

# test: OHLC historical data
li = read_list(join(system_folder, "primary_symbols_kraken.txt"))
#for k in li:
#    print(k)
bar_minutes = 15    # (minutes) 1 (default), 5, 15, 30, 60, 240, 1440, 10080, 21600
ohlc = kraken_get_ohlc(li[:1], interval=bar_minutes, output_to_file=True)

STOP()


coinmarketcap_usd_url = "https://api.coinmarketcap.com/v1/ticker/?convert=USD"
coinmarketcap_usd_limit_url = "https://api.coinmarketcap.com/v1/ticker/?convert=USD&limit={0}"

js = "{'market_cap_usd': '71615060420.0', 'price_usd': '4313.54', 'last_updated': '1507072753', 'name': 'Bitcoin'}"
j = {'market_cap_usd': '71615060420.0', 'price_usd': '4313.54', 'last_updated': '1507072753', 'name': 'Bitcoin'}
js = json.dumps(j)
#js = '{"market_cap_usd": "71615060420.0", "price_usd": "4313.54", "last_updated": "1507072753", "name": "Bitcoin"}'
p = Payload(js)


json_text = get_request_text(coinmarketcap_usd_limit_url.format(20))
li = json.loads(json_text)

STOP()

d = {}
for x in li:
    print(x)
    xstr = str(x)
    p = Payload(xstr)
    #for k in x:
    #    print(k, x[k])
    #    d[k] = x[k]
    break


# dt, bids, asks = get_bitstamp_orderbook("btcusd")

# Only need to recreate the dataset files if/when a database has added/modified/removed a dataset
#q.create_dataset_codes_for_list(q.bitcoin_db_list)

# This will retrieve the data for ALL database names in the list provided
#q.create_database_data_for_list(q.bitcoin_db_list)

# Or you can retrieve data for individual databases
#q.create_database_data("GDAX")

STOP()


trd1 = buy('@ES', 1, 2024.25, datetime(2017, 9, 1, 9, 30), TRD_ENTRY)
trd2 = sell('@ES', 1, 2026.75, datetime(2017, 9, 3, 13, 45), TRD_EXIT, trd1.id)

print(trd1)
print(trd2)

rt = TradeRoundTrip(trd1, trd2)
print(rt.holding_period())
print(rt.days())
print(rt.profit())
df = rt.to_df()

write_trades_file([trd1, trd2])

STOP(df)


df = q.get_bitcoin_daily(days=365*1)
write_dataframe(df, "bitcoin.daily.DF.csv")
df_arima = alvin_arima(df)
df = df_arima.dropna()

plot_ARIMA(df, 'Bitcoin ARIMA')

