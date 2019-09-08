from os.path import join, basename, splitext
import requests
import csv
import urlparse
import pandas as pd
import sys

#-----------------------------------------------------------------------------------------------------------------------
from f_folders import *
from f_date import *
from f_plot import *
from f_stats import *
from f_dataframe import *
#-----------------------------------------------------------------------------------------------------------------------



# get a JSON object from a specified URL
def get_json(url):
    # Both GET and POST can take a dictionary of parameters as an argument
    #userdata = {"firstname": "John", "lastname": "Doe", "password": "jdoe123"}
    #resp = requests.post('http://www.mywebsite.com/user', data=userdata)
    response = requests.get(url)
    return response.json()

# deal with Unicode strings
def enc(text):
     return text.encode('utf-8').strip()


# pass in currency symbol (like "CNY") to show BPI in a different currency
# default (None) returns USD, GBP, EUR
def get_bpi_current_price(to_currency=None):
    if to_currency is None:
        bpi = get_json("https://api.coindesk.com/v1/bpi/currentprice.json")
    else:
        bpi = get_json("https://api.coindesk.com/v1/bpi/currentprice/{0}.json".format(to_currency))
    time = bpi['time']
    disclaimer = bpi['disclaimer']
    chart_name = bpi['chartName']
    return bpi['bpi']

# index: 'USD', 'CNY'
# currency: any of the ISO 4217 currency codes
# start_date/end_date: YYYY-MM-DD format (like '2013-09-01')
def get_bpi_historical(index='USD', currency='USD', start_date=None, end_date=None, for_yesterday=False):
    if for_yesterday == True:
        hist = get_json("https://api.coindesk.com/v1/bpi/historical/close.json?index={0}&currency={1}&for=yesterday".format(index, currency))                        
    elif not (start_date is None or end_date is None):
        hist = get_json("https://api.coindesk.com/v1/bpi/historical/close.json?index={0}&currency={1}&start={2}&end={3}".format(index, currency, start_date, end_date))
    else:
        hist = get_json("https://api.coindesk.com/v1/bpi/historical/close.json?index={0}&currency={1}".format(index, currency))
    return hist['bpi']

def print_bpi(bpi):
    symbol = bpi['symbol']
    code = bpi['code']
    rate = bpi['rate']                      # rate formatted for display (commas, etc)
    rate_float = bpi['rate_float']
    description = bpi['description']
    print '{0} {1} {2} "{3}"'.format(symbol, code, rate_float, description)
    return

def print_bpi_hist(hist):
    for dt in hist:
        print dt, hist[dt]          # date (string), price (float)
    return

# Use the coindesk API to retrieve a list (actually a dictionary) of ISO 4217 currency codes
# (optional) output_pathname defaults to None which will NOT write to file; pass a pathname string to write currency data to file
# (optional) display defaults to True which will print the currency data; set to False for no printing
# Return dictionary<string currency,string country> (ex: {'AUD':'Australian Dollar', 'CNY':'Chinese Yuan'})
def get_currency_codes(output_pathname=None, display=True):
    currencies = get_json("https://api.coindesk.com/v1/bpi/supported-currencies.json")  # ISO 4217 currency codes
    tofile = False
    if output_pathname is not None:
        fout = open(output_pathname, 'w')
        tofile = True
    columns = "Currency,Country"
    if display: print columns
    if tofile: fout.write(columns + '\n')
    dict = {}
    for cur in currencies:
        currency = enc(cur["currency"])
        country = enc(cur["country"])
        dict[currency] = country
        currency_country = "{0},{1}".format(currency, country)
        if display: print currency_country
        if tofile: fout.write(currency_country + '\n')
    if display: print
    if tofile: fout.close()
    return dict


########################################################################################################################



#---------------------------------------------------------------------------------------------------
# https://www.coindesk.com/api/

# CoinDesk Currencies
#pathname = join(system_folder, "currency_codes_ISO_4217.DF.csv")
#d = get_currency_codes(output_pathname=pathname)

# CoinDesk BPI real-time data
bpi = get_bpi_current_price()
for k in bpi:
    print k,
    print_bpi(bpi[k])
print

# TODO: retrieve historical Coindesk data and store in DF_DATA dataframe files

# Coindesk BPI historical data
hist = get_bpi_historical()
print_bpi_hist(hist)

#---------------------------------------------------------------------------------------------------







