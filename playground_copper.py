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
    df = read_dataframe(join(df_folder, data_filename))
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

update_historical = True

print("Testing ideas for COPPER trade")
print()

# Re-download the copper data to a local dataframe file:
#update_copper_dataframe()

data_filename = "copper_settle_discount.2.DF.csv"
df = read_dataframe(join(df_folder, data_filename))

# --- TEST functions ---
#copper_filtercal('KM')
#copper_filtercal('KM', 'KN')

#print("{0} args: {1} {2}".format(nargs, args[0], args[1]))

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



STOP()


#x,y = kraken_get_server_time()
#print(x,y)
#di = kraken_get_asset_info()
#print_dict(di)
#pairs = kraken_get_tradable_asset_pairs()
#print_dict_keys(pairs)
#for k in pairs:
#    if 'USD' in k:
#        print(k)
#tickers = kraken_get_ticker(['XXBTZUSD'])
#print(tickers)

#-------------------------------------------------------------------------------
# COMMAND LINE ARGUMENTS (arg[0] is script name)
args = sys.argv
#args = ['KRAKEN_OHLC', 'BCHUSD', '60']      # FOR TESTING ONLY!!!
if (len(args) > 0):
    print(len(args))
    print(get_arg(1), get_arg(2), get_arg(3))
    #sys.exit()

    # ARG: "KRAKEN_SYMBOLS"
    # UPDATE KRAKEN SYMBOLS (output to 'kraken_symbols.txt' in SYSTEM data folder)
    if get_arg(1) == 'KRAKEN_SYMBOLS':
        symbols = kraken_get_symbols()
        STOP()

    # ARG: "KRAKEN_OHLC" <symbol> <bar_minutes>
    # GET OHLC KRAKEN HISTORICAL DATA (output to dataframe file in DF_DATA folder)
    if get_arg(1) == 'KRAKEN_OHLC' and get_arg(2) and get_arg(3):
        symbol = get_arg(2)
        bar_minutes = int(get_arg(3))   # 1, 5, 15, 30, 60, 240, 1440, 10080, 21600
        print("'{0}' [{1}]".format(symbol, bar_minutes));
        ohlc = kraken_get_ohlc([symbol], interval=bar_minutes, output_to_file=True)
        STOP()

# command line arguments -------------------------------------------------------

sys.exit()

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

