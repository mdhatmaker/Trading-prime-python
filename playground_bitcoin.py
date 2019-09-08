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

# USEFUL CRYPTO/JSON WEBSITES:
# https://stackoverflow.com/questions/12965203/how-to-get-json-from-webpage-into-python-script
# https://stackoverflow.com/questions/7706658/how-to-get-bitcoincharts-weighted-prices-via-json-and-jquery#7942091
# https://www.bitstamp.net/api/
# https://www.bitstamp.net/api/ticker/
# https://www.bitstamp.net/api/v2/ticker/{currency_pair}/
# https://www.bitstamp.net/api/v2/ticker_hour/{currency_pair}/
# http://api.bitcoincharts.com/v1/weighted_prices.json


def get_response(url):
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())
    print("get_response length:", len(data))
    return data

def get_request(url):
    r = requests.get(url)       #, auth=('user', 'pass'))
    print("status_code:", r.status_code)
    print("content-type:", r.headers['content-type'])
    print("encoding:", r.encoding)
    print("text length:", len(r.text))
    print("json length:", len(r.json()))
    return r.json()

def get_request_text(url):
    r = requests.get(url)       #, auth=('user', 'pass'))
    print("status_code:", r.status_code)
    print("content-type:", r.headers['content-type'])
    print("encoding:", r.encoding)
    print("text length:", len(r.text))
    print("json length:", len(r.json()))
    return r.text

def print_list(li):
    for item in li:
        print(item)
    return

def print_dict(di):
    for k in di:
        print(di[k])
    return

def print_dict_keys(di):
    for k in pairs:
        print(k)
    return

#-------------- BitcoinCharts --------------------------------------------------
def parse_weighted_prices(data):
    timestamp = None
    for k in data.keys():
        if k == 'timestamp':
            timestamp = data[k]
        else:
            weighted_7d = data[k]['7d'] if '7d' in data[k] else None
            weighted_30d = data[k]['30d'] if '30d' in data[k] else None
            weighted_24h = data[k]['24h'] if '24h' in data[k] else None
            print("{0}: 7d={1}  30d={2}  24h={3}".format(k, weighted_7d, weighted_30d, weighted_24h))
    print("timestamp: {0} ({1})".format(timestamp, ))
    return

def get_bcharts_weighted_prices():
    # Weighted Prices from BitcoinCharts
    data = get_response("http://api.bitcoincharts.com/v1/weighted_prices.json")
    parse_weighted_prices(data)
    return data

def get_bcharts_markets():
    # General Market Data from BitcoinCharts
    data = get_request("http://api.bitcoincharts.com/v1/markets.json")
    return data

def print_bcharts_symbols():
    data = get_bcharts_markets()
    # print symbols
    for d in data:
        print(d['symbol'])
    return data

#-------------------------------------------------------------------------------

#-------------- Bitstamp -------------------------------------------------------
bitstamp_ticker_hour_url = "https://www.bitstamp.net/api/v2/ticker_hour/{0}/"

def get_bitstamp(url, currency_pair):
    # Supported values for currency_pair:
    # btcusd, btceur, eurusd, xrpusd, xrpeur, xrpbtc,
    # ltcusd, ltceur, ltcbtc, ethusd, etheur, ethbtc
    request_url = url.format(currency_pair)
    #data = get_response(request_url)
    data = get_request(request_url)
    return data

def get_bitstamp_ticker(currency_pair):
    bitstamp_ticker_url = "https://www.bitstamp.net/api/v2/ticker/{0}/"
    data = get_bitstamp(bitstamp_ticker_url, currency_pair)
    return BitstampTicker(currency_pair, data)

def get_bitstamp_orderbook(currency_pair):
    bitstamp_order_book_url = "https://www.bitstamp.net/api/v2/order_book/{0}/"
    data = get_bitstamp(bitstamp_order_book_url, currency_pair)
    #dict_keys(['bids', 'asks', 'timestamp'])
    bids = data['bids']
    asks = data['asks']
    timestamp = int(data['timestamp'])
    dt = from_unixtime(timestamp)
    print("[{0}] bids_count:{1} asks_count:{2}".format(dt, len(bids), len(asks)))
    return dt, bids, asks

def bitstamp_generate_signature(nonce, customer_id, api_key):
    message = nonce + customer_id + api_key
    signature = hmac.new(
        API_SECRET,
        msg=message,
        digestmod=hashlib.sha256
    ).hexdigest().upper()
    return

class BitstampTicker:
    def __init__(self, symbol, data):
        self.data = data
        self.symbol = symbol
        self.timestamp = long(data['timestamp'])
        self.dt = from_unixtime(self.timestamp)
        self.open = float(data['open'])
        self.high = float(data['high'])
        self.low = float(data['low'])
        self.last = float(data['last'])
        self.bid = float(data['bid'])
        self.ask = float(data['ask'])
        self.volume = float(data['volume'])
        self.vwap = float(data['vwap'])

    def __repr__(self):
        return "[{0}] last={1}  b:{2} a:{3}  vwap:{4}  v:{5}  o:{6} h:{7} l:{8}".format(self.dt, self.last, self.bid, self.ask, self.vwap, self.volume, self.open, self.high, self.low)
        

def print_bitstamp_tickers():
    btc = get_bitstamp_ticker("btcusd")
    xrp = get_bitstamp_ticker("xrpusd")
    ltc = get_bitstamp_ticker("ltcusd")
    eth = get_bitstamp_ticker("ethusd")
    print()
    print('btc:', btc)
    print('xrp:', xrp)
    print('ltc:', ltc)
    print('etc:', eth)
    return
#-------------------------------------------------------------------------------

#-------------- CoinMarketCap --------------------------------------------------
def get_coinmarketcap_ticker():
    coinmarketcap_usd_url = "https://api.coinmarketcap.com/v1/ticker/?convert=USD"
    coinmarketcap_usd_limit_url = "https://api.coinmarketcap.com/v1/ticker/?convert=USD&limit={0}"
    data = get_request(coinmarketcap_usd_url)
    return data

# Retrieve all the info from coinmarketcap,com and output to a dataframe file
def create_coinmarketcap_df(pathname):
    data = get_coinmarketcap_ticker()
    columns = list(data[0].keys())
    # Use our own custom column order
    columns = ['symbol','name','id','rank','market_cap_usd','total_supply','available_supply','24h_volume_usd','percent_change_1h','percent_change_24h','percent_change_7d','price_usd','price_btc','last_updated']
    with open(pathname, 'wt') as f:
        f.write(','.join(columns) + '\n')
        for d in data:
            li = []
            for c in columns:
                if d[c] is None:
                    li.append('')
                elif c == 'last_updated':
                    li.append(str(from_unixtime(int(d[c]))))
                else:
                    li.append(d[c])
            f.write(','.join(li) + '\n')
    print("Output to dataframe file: '{0}'".format(pathname))
    return

# Re-download coinmarketcap.com data to dataframe "coinmarketcap.DF.csv"
def update_coinmarketcap_dataframe():
    filename = "coinmarketcap.DF.csv"
    create_coinmarketcap_df(join(df_folder, filename))
    return
#-------------------------------------------------------------------------------

#-------------- BlockChain.info ------------------------------------------------
def print_blockchain_info():
    data = get_request("https://blockchain.info/charts/market-price?format=json")
    print(data.keys())
    #dict_keys(['period', 'name', 'unit', 'status', 'values', 'description'])
    for d in data['values']:
        dt = from_unixtime(d['x'])
        print("x={0}  y={1}".format(strdate(dt), d['y']))
    return
#-------------------------------------------------------------------------------

#-------------- Kraken ---------------------------------------------------------
def kraken_get_server_time():
    data = get_request("https://api.kraken.com/0/public/Time")
    print(data.keys())
    result = data['result']
    error_list = data['error']
    if len(error_list) > 0: print_list(error_list)
    # {u'unixtime': 1509761955, u'rfc1123': u'Sat,  4 Nov 17 02:19:15 +0000'}
    return result['unixtime'], result['rfc1123']

def kraken_get_asset_info():
    data = get_request("https://api.kraken.com/0/public/Assets")
    print(data.keys())
    result = data['result']
    error_list = data['error']
    if len(error_list) > 0: print_list(error_list)
    # {u'aclass': u'currency', u'decimals': 10, u'display_decimals': 5, u'altname': u'BCH'}
    return result

def kraken_get_tradable_asset_pairs():
    data = get_request("https://api.kraken.com/0/public/AssetPairs")
    print(data.keys())
    result = data['result']
    error_list = data['error']
    if len(error_list) > 0: print_list(error_list)
    # {u'lot_multiplier': 1, u'fee_volume_currency': u'ZUSD', u'quote': u'ZUSD', u'aclass_base': u'currency', u'fees': [[0, 0.26], [50000, 0.24], [100000, 0.22], [250000, 0.2], [500000, 0.18], [1000000, 0.16], [2500000, 0.14], [5000000, 0.12], [10000000, 0.1]], u'margin_stop': 40, u'fees_maker': [[0, 0.16], [50000, 0.14], [100000, 0.12], [250000, 0.1], [500000, 0.08], [1000000, 0.06], [2500000, 0.04], [5000000, 0.02], [10000000, 0]], u'leverage_sell': [], u'base': u'BCH', u'leverage_buy': [], u'lot': u'unit', u'altname': u'BCHUSD', u'lot_decimals': 8, u'margin_call': 80, u'aclass_quote': u'currency', u'pair_decimals': 1}
    return result

# Retrieve list of Kraken symbols
# (output to 'kraken_symbols.txt' file in SYSTEM data folder if output_to_file is True)
def kraken_get_symbols(output_to_file=True):
    pairs = kraken_get_tradable_asset_pairs()
    li = pairs.keys()
    li.sort()
    if output_to_file:
        filename = "kraken_symbols.txt"
        pathname = join(system_folder, filename)
        with open(pathname, 'w') as f:
            for symbol in li:
                f.write("{0}\n".format(symbol))
        print("Data written to file: '{0}'".format(pathname))
        print('')
    return li

# 'XXBTZUSD'
def kraken_get_ticker(asset_pair_list = ['BCHUSD','DASHEUR']):
    url = "https://api.kraken.com/0/public/Ticker?pair={0}".format(','.join(asset_pair_list))
    print(url)
    data = get_request(url)
    print(data.keys())
    result = data['result']
    error_list = data['error']
    if len(error_list) > 0: print_list(error_list)
    # {u'XXBTZUSD': {u'a': [u'7150.60000', u'1', u'1.000'], u'c': [u'7060.00000', u'0.21257648'], u'b': [u'7058.30000', u'2', u'2.000'], u'h': [u'7182.50000', u'7449.00000'], u'l': [u'6934.10000', u'6934.10000'], u'o': u'7160.00000', u'p': [u'7064.50986', u'7243.32566'], u't': [3043, 22101], u'v': [u'674.46925821', u'5091.71427112']}}
    return result

# interval = time frame interval in minutes (optional): 1 (default), 5, 15, 30, 60, 240, 1440, 10080, 21600
# (output to 'kraken.SYMBOL.Xminute.DF.csv' file in DF_DATA data folder if output_to_file is True)
def kraken_get_ohlc(asset_pair_list=['XXBTZUSD'], interval=1, output_to_file=True):
    url = "https://api.kraken.com/0/public/OHLC?pair={0}&interval={1}".format(','.join(asset_pair_list), interval)
    print(url)
    data = get_request(url)
    print(data.keys())
    result = data['result']
    error_list = data['error']
    if len(error_list) > 0: print_list(error_list)
    # {u'XXBTZUSD': {u'a': [u'7150.60000', u'1', u'1.000'], u'c': [u'7060.00000', u'0.21257648'], u'b': [u'7058.30000', u'2', u'2.000'], u'h': [u'7182.50000', u'7449.00000'], u'l': [u'6934.10000', u'6934.10000'], u'o': u'7160.00000', u'p': [u'7064.50986', u'7243.32566'], u't': [3043, 22101], u'v': [u'674.46925821', u'5091.71427112']}}
    if (output_to_file): kraken_write_ohlc_df(result, interval)
    return result

# Write OHLC historical data to 'kraken.SYMBOL.Xminute.DF.csv' file in DF_DATA data folder
def kraken_write_ohlc_df(ohlc, interval=1):
    for symbol in ohlc.keys():
        if symbol == 'last':
            print('last = {0}'.format(from_unixtime(ohlc['last'])))
        else:
            filename = "kraken.{0}.{1}minute.DF.csv".format(symbol, interval)
            pathname = join(df_folder, filename)
            with open(pathname, 'w') as f:
                f.write("DateTime,Symbol,VWAP,Open,High,Low,Close,Volume,Count" + '\n')
                for i in range(len(ohlc[symbol])):
                    bar = KrakenOHLC(ohlc[symbol][i])
                    f.write(bar.to_csv(symbol) + '\n')
                    # print(bar)
            print("Data written to file: '{0}'".format(pathname))
        print('')


def cool_chart(dfx, title='', plot_filename='crypto_analysis'):
    df = dfx.set_index('DateTime')
    plt.figure()
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(14, 8))
    #fig.text(0.5, 0.04, 'DateTime', ha='center')
    #main_ax = ax[0]
    main_ax = ax
    """ts_ax = ax[1]
    roll_vx_ax = ax[2]
    roll_es_ax = ax[3]
    vx_ax = ax[4]
    es_ax = ax[5]"""

    plt.legend(loc='best')

    # main plot
    #df['Close'].plot(ax=main_ax, color='green');
    trace = go.Ohlc(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close)
    data = [trace]
    #trace.plot(ax=main_ax)
    plt.plot(data)

    #py.plot(data)   #, filename='simple_ohlc')
    #df['sell1'].plot(ax=main_ax, linestyle='dotted', color='red', alpha=0.5);
    #df['buy1'].plot(ax=main_ax, linestyle='dotted', color='blue', alpha=0.5);
    main_ax.set_title(title);
    # Draw some horizontal lines on the plot
    #prem_ax.axhline(y=3, xmin=0.0, xmax=1.0, linewidth=1, linestyle='dotted', color='black', alpha=0.5)
    #prem_ax.axhline(y=2, xmin=0.0, xmax=1.0, linewidth=1, linestyle='dotted', color='black', alpha=0.5)
    #prem_ax.axhline(y=1, xmin=0.0, xmax=1.0, linewidth=1, linestyle='dotted', color='black', alpha=0.5)
    main_ax.axhline(y=0, xmin=0.0, xmax=1.0, linewidth=1, linestyle='solid', color='black', alpha=0.5)
    #prem_ax.axhline(y=-1, xmin=0.0, xmax=1.0, linewidth=1, linestyle='dotted', color='black', alpha=0.5)
    #prem_ax.axhline(y=-2, xmin=0.0, xmax=1.0, linewidth=1, linestyle='dotted', color='black', alpha=0.5)
    #prem_ax.axhline(y=-3, xmin=0.0, xmax=1.0, linewidth=1, linestyle='dotted', color='black', alpha=0.5)

    """# ES_diff 5-day average
    #df['Close_VX'].plot(ax=ts_ax)
    rolling_mean = pd.rolling_mean(df['diff_ES'], window=5)     # changed this from window=12
    rolling_mean.plot(ax=ts_ax, color='darkslateblue');
    #plt.legend(loc='best')
    #ts_ax.set_title(title, fontsize=24);
    ts_ax.set_title("ES diff (5-day average)");
    ts_ax.axhline(y=5, xmin=0.0, xmax=1.0, linewidth=1, linestyle='dotted', color='black', alpha=0.5)
    ts_ax.axhline(y=-5, xmin=0.0, xmax=1.0, linewidth=1, linestyle='dotted', color='black', alpha=0.5)

    # VX and ES calculated from date of VX roll
    df['roll_price_VX'].plot(ax=roll_vx_ax, color='crimson')
    roll_vx_ax.set_title('VX Price Change from VX Roll')
    df['roll_price_ES'].plot(ax=roll_es_ax, color='darkslateblue')
    roll_es_ax.set_title('ES Price Change from VX Roll')

    # VX and ES prices
    df['Close_VX'].plot(ax=vx_ax, color='crimson')
    vx_ax.set_title('VX Close')
    df['Close_ES'].plot(ax=es_ax, color='darkslateblue')
    es_ax.set_title('ES Close')"""

    # save plot
    plt.tight_layout();
    pathname = join(misc_folder, '{}.png'.format(plot_filename))
    savefig(plt, pathname)
    #plt.show()
    plt.show(block=False)
    plt.clf()
    plt.cla()
    plt.close()
    #plt.gcf().clear()

#===============================================================================
class KrakenOHLC:
    def __init__(self, data):
        self.data = data
        self.timestamp = long(data[0])
        self.dt = from_unixtime(self.timestamp)
        self.open = float(data[1])
        self.high = float(data[2])
        self.low = float(data[3])
        self.close = float(data[4])
        self.vwap = float(data[5])
        self.volume = float(data[6])
        self.count = int(data[7])

    def __repr__(self):
        return "[{0}] vwap:{1}  v:{2}    O:{3} H:{4} L:{5} C:{6}    count:{7}".format(self.dt, self.vwap, self.volume, self.open, self.high, self.low, self.close, self.count)

    def columns(self):
        return "DateTime,Synbol,VWAP,Open,High,Low,Close,Volume,Count"

    def to_csv(self, symbol):
        return "{0},{1},{2},{3},{4},{5},{6},{7},{8}".format(self.dt, symbol, self.vwap, self.open, self.high, self.low, self.close, self.volume, self.count)

#===============================================================================
# p = Payload(jsonString)
class Payload(object):
    def __init__(self, j):
        #self.__dict__ = json.loads(j)
        self.__dict__ = json.loads(j)

########################################################################################################################

update_historical = True

print("Testing ideas for BITCOIN and other crypto currencies")
print()

# Re-download the coinmarketcap.com data to a local dataframe file:
#update_coinmarketcap_dataframe()

#symbol = "TRXETH"
#timeframe = "1h"
data_filename = "TRXETH.1h.DF.csv"
analysis_filename = "TRXETH_analysis.1h.DF.csv"

if update_historical:
    #df = read_dataframe()
    df = pd.read_csv(join(crypto_folder, data_filename), parse_dates=['DateTime'], dtype={'Open':np.str, 'High':np.str, 'Low':np.str, 'Close':np.str})
    #df['Open'] = df['Open'].astype(decimal.Decimal)
    #df['Close'] = df['Close'].astype(decimal.Decimal)
    dfz = df['Open'].astype(float)
    df['mean'] = df['Close'].rolling(7).mean()
    df['std'] = df['Close'].rolling(7).std()
    df['buy1'] = df['mean'] - df['std']
    df['buy2'] = df['mean'] - 2*df['std']
    df['buy3'] = df['mean'] - 3*df['std']
    df['sell1'] = df['mean'] + df['std']
    df['sell2'] = df['mean'] + 2*df['std']
    df['sell3'] = df['mean'] + 3*df['std']
    write_dataframe(df, analysis_filename)

df = read_dataframe(analysis_filename)
cool_chart(df, title="TFXETH (1h)")

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

