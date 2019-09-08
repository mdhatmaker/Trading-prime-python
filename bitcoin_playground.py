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
"""
import f_quandl as q
from c_Trade import *
"""

#-----------------------------------------------------------------------------------------------------------------------

# USEFUL CRYPTO/JSON WEBSITES:
# https://stackoverflow.com/questions/12965203/how-to-get-json-from-webpage-into-python-script
# https://stackoverflow.com/questions/7706658/how-to-get-bitcoincharts-weighted-prices-via-json-and-jquery#7942091
# https://www.bitstamp.net/api/
# https://www.bitstamp.net/api/ticker/
# https://www.bitstamp.net/api/v2/ticker/{currency_pair}/
# https://www.bitstamp.net/api/v2/ticker_hour/{currency_pair}/
# http://api.bitcoincharts.com/v1/weighted_prices.json


def STOP(x=None):
    print("****STOP****")
    if x is not None: print(x)
    sys.exit()


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


# p = Payload(jsonString)
class Payload(object):
    def __init__(self, j):
        #self.__dict__ = json.loads(j)
        self.__dict__ = json.loads(j)



########################################################################################################################

print("Testing ideas for BITCOIN and other crypto currencies")
print()

# Re-download the coinmarketcap.com data to a local dataframe file:
#update_coinmarketcap_dataframe()

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

STOP(df)




