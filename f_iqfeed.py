from iqapi import *
import pandas as pd
import numpy as np
from os.path import join
from datetime import datetime, timedelta


#-------------------------------------------------------------------------------

from f_folders import *
from f_date import *
import f_dataframe

#-------------------------------------------------------------------------------

INTERVAL_DAILY = 'd' #0
INTERVAL_MINUTE = 's60' #60
INTERVAL_HOUR = 's3600' #3600

# default start/end years to use if none provided to function calls
default_y1 = 2010
default_y2 = datetime.now().year
if datetime.now().month >= 10:              # On or after Oct, use the following year as the ending year
    default_y2 = datetime.now().year + 1
else:                                       # otherwise, use the current year
    default_y2 = datetime.now().year

# default start/end dates to use if none provided to function calls
default_dt1 = datetime(default_y1,1,1)
default_dt2 = datetime.now()

# Set the default start/end years and start/end dates to be the "earliest" dates (start in 2002)
def set_default_dates_earliest():
    global default_y1, default_y2, default_dt1, default_dt2
    default_y1 = 2002
    default_y2 = datetime.now().year
    default_dt1 = datetime(default_y1,1,1)
    default_dt2 = datetime.now()
    print("Set EARLIEST default dates for f_iqfeed.py: y1={0} y2={1}  d1={2} d2={3}".format(default_y1, default_y2, default_dt1, default_dt2))
    return

# Set the default start/end years and start/end dates to be the "standard" dates (start in 2010)
def set_default_dates_standard():
    global default_y1, default_y2, default_dt1, default_dt2
    default_y1 = 2010
    default_y2 = datetime.now().year
    default_dt1 = datetime(default_y1,1,1)
    default_dt2 = datetime.now()
    print("Set STANDARD default dates for f_iqfeed.py: y1={0} y2={1}  d1={2} d2={3}".format(default_y1, default_y2, default_dt1, default_dt2))
    return

# Set the default start/end years and start/end dates to be the "latest" dates (start in 2017)
def set_default_dates_latest():
    global default_y1, default_y2, default_dt1, default_dt2
    default_y1 = 2017
    default_y2 = datetime.now().year
    default_dt1 = datetime(default_y1,1,1)
    default_dt2 = datetime.now()
    print("Set LATEST default dates for f_iqfeed.py: y1={0} y2={1}  d1={2} d2={3}".format(default_y1, default_y2, default_dt1, default_dt2))
    return

# Given an interval (in seconds)
# Return a string representation ("daily", "minute", "hour", etc.)--useful for creating filenames
def str_interval(interval):
    if interval == INTERVAL_DAILY:
        return "daily"
    elif interval == INTERVAL_MINUTE:
        return "minute"
    elif interval == INTERVAL_HOUR:
        return "hour"
    else:
        return str(interval)

# Given an IQFeed symbol and the start/end dates and interval in seconds (0=daily) and begin/end filter time ("HHmmSS")
# Return a dataframe containing the historical data for this symbol (using IQFeed)
def get_historical(symbol, dateStart, dateEnd, interval=INTERVAL_DAILY, beginFilterTime='', endFilterTime=''):
    iq = IQHistoricData(dateStart, dateEnd, interval, beginFilterTime, endFilterTime)
    df = iq.download_symbol(symbol)
    df['Volume'] = df['Volume'].astype(np.int)
    df['oi'] = df['oi'].astype(np.int)
    return df


# Given a contract symbol and date start/end and interval in seconds (0=daily) and begin/end filter time ("HHmmSS")
# Return a dataframe containing the historical data for this contract (using IQFeed)
def get_historical_contract(symbol, dateStart=default_dt1, dateEnd=default_dt2, interval=INTERVAL_DAILY, beginFilterTime='', endFilterTime=''):
    df = get_historical(symbol, dateStart, dateEnd, interval, beginFilterTime, endFilterTime)
    df = df.reset_index(drop=True)  # initially, the DateTime is the index
    df.sort_values(["DateTime"], ascending=True, inplace=True)
    return df


# Given month (1-12) and year (4-digit)
# Return a dataframe containing the historical data for this futures contract (using IQFeed)
def get_historical_future(symbol_root, m, y, interval=INTERVAL_DAILY, days_back=180, beginFilterTime='', endFilterTime=''):
    dateEnd = datetime(y, m, 28)  # use the 28th day of the future month (28th will be valid for every month)
    dateStart = dateEnd - timedelta(days=days_back)  # go back 365 days (by default)
    mYY = monthcodes[m - 1] + str(y)[-2:]
    symbol = symbol_root + mYY
    #print(symbol, dateStart, dateEnd)
    # iq = historicData(dateStart, dateEnd, INTERVAL_1HOUR)
    # symbolOneData = iq.download_symbol(symbolOne)
    symbolData = get_historical(symbol, dateStart, dateEnd, interval, beginFilterTime, endFilterTime)
    return symbolData


# Given two symbols "XXXmYY" and "ZZZmYY" (symbol0 and symbol1) which represent an ETS spread "XXXmYY-ZZZmYY"
# Return a dataframe containing the historical data for this calendar spread ETS futures contract (using IQFeed)
def get_historical_spread_future(symbol0, symbol1, interval=INTERVAL_DAILY, days_back=180, beginFilterTime='', endFilterTime=''):
    m0, y0 = get_month_year(symbol0[-3:])
    dateEnd = datetime(y0, m0, 28)  # use the 28th day of the future month (28th will be valid for every month)
    dateStart = dateEnd - timedelta(days=days_back)  # go back 365 days (by default)
    spread_symbol = symbol0 + '-' + symbol1
    #print(spread_symbol, dateStart, dateEnd)
    symbolData = get_historical(spread_symbol, dateStart, dateEnd, interval, beginFilterTime, endFilterTime)
    return symbolData


# Given a symbol for a contract and a start/end date (dateStart/dateEnd)
# create a dataframe file with the symbol as the name that contains price data for the requested date range
# (output file ex: "VIX.XO.daily.DF.csv"
def create_historical_contract_df(symbol, dateStart=default_dt1, dateEnd=default_dt2, interval=INTERVAL_DAILY, beginFilterTime='', endFilterTime='', force_redownload=True):
    #df = get_historical_contract(symbol, dateStart, dateEnd, interval, beginFilterTime, endFilterTime)
    if force_redownload == True:
        df = get_historical_contract(symbol, dateStart, dateEnd, interval, beginFilterTime, endFilterTime)
    else:
        df_exist = f_dataframe.read_dataframe("{0}_contract.{1}.DF.csv".format(symbol, str_interval(interval)))
        df_exist = df_exist.iloc[:-1,:]     # delete last row
        dt = max(dateStart, f_dataframe.df_last(df_exist)['DateTime'] - timedelta(days=7));
        df = get_historical_contract(symbol, dt, dateEnd, interval, beginFilterTime, endFilterTime)
        last_dt = f_dataframe.df_last(df_exist)['DateTime']
        df = df[df['DateTime'] > last_dt]
        df = pd.concat([df_exist, df], ignore_index=True)
    f_dataframe.write_dataframe(df, "{0}_contract.{1}.DF.csv".format(symbol, str_interval(interval)))
    return df


# Given a symbol root (ex: "@VX") and a starting/ending year (y1/y2)
# create a dataframe file with the symbol root as the name that contains price data for all futures in requested year range
# (output file ex: "@VX_futures.daily.DF.csv")
def create_historical_futures_df(symbol_root, y1=default_y1, y2=default_y2, interval=INTERVAL_DAILY, days_back=180, beginFilterTime='', endFilterTime='', force_redownload=True):
    today = datetime.now()
    df = pd.DataFrame()
    for year in range(y1, y2 + 1):
        for month in range(1, 12 + 1):
            if force_redownload == True or (year > today.year or (year==today.year and month >=today.month)):
                dfx = get_historical_future(symbol_root, month, year, interval, days_back, beginFilterTime, endFilterTime)
                if df.shape[0] == 0:
                    df = dfx
                else:
                    df = df.append(dfx)
    df = df.reset_index(drop=True)
    f_dataframe.df_sort_mYY_symbols_by_date(df)
    if force_redownload != True:
        df_exist = f_dataframe.read_dataframe("{0}_futures.{1}.DF.csv".format(symbol_root, str_interval(interval)))
        df_exist = f_dataframe.df_remove_duplicates(df_exist, df, ['DateTime','Symbol'])
        df = df[df_exist.columns]
        df = pd.concat([df_exist, df], ignore_index=True);
        df = df.reset_index(drop=True)
    f_dataframe.write_dataframe(df, "{0}_futures.{1}.DF.csv".format(symbol_root, str_interval(interval)))
    return df


# Given a symbol root (ex: "@VX") and front month number (mx) and back month number (my) and a starting/ending year (y1/y2)
# create a dataframe file that contains price data for all corresponding spread ETS futures in given year range
# (mx=0, my=1 is the default which represents front-month to next-month calendar spread)
# (output file ex: "@VX_calendar-m0-m1.daily.DF.csv")
def create_historical_calendar_futures_df(symbol_root, mx=0, my=1, y1=default_y1, y2=default_y2, interval=INTERVAL_DAILY, days_back=180, beginFilterTime='', endFilterTime='', force_redownload=True):
    today = datetime.now()
    df = pd.DataFrame()
    for year in range(y1, y2 + 1):
        for month in range(1, 12 + 1):
            if force_redownload == True or (year > today.year or (year==today.year and month >=today.month)):
                symbol0, symbol1, _ = get_calendar_symbols(symbol_root, month, year, mx, my)
                dfx = get_historical_spread_future(symbol0, symbol1, interval, days_back, beginFilterTime, endFilterTime)
                if df.shape[0] == 0:
                    df = dfx
                else:
                    df = df.append(dfx)
    #df = df.reset_index()
    f_dataframe.df_sort_mYY_symbols_by_date(df)
    if force_redownload != True:
        df_exist = f_dataframe.read_dataframe("{0}_calendar-m{1}-m{2}.{3}.DF.csv".format(symbol_root, mx, my, str_interval(interval)))
        df_exist = f_dataframe.df_remove_duplicates(df_exist, df, ['DateTime','Symbol'])
        df = df[df_exist.columns]
        df = pd.concat([df_exist, df], ignore_index=True);
        df = df.reset_index(drop=True)
    f_dataframe.write_dataframe(df, "{0}_calendar-m{1}-m{2}.{3}.DF.csv".format(symbol_root, mx, my, str_interval(interval)))
    return df


# Given search text
# Return list of symbols that match the given text
def search_symbols(text):
    iq = IQSymbolSearch()
    result = iq.symbol_search(text)
    #[','.join(s)]
    return result

# Given search text
# Return list of descriptions that match the given text
def search_descriptions(text):
    iq = IQSymbolSearch()
    result = iq.description_search(text)
    return result

# Given at least two digits in an existing SIC code
# Return list of symbols that match this SIC code
def search_SIC(digits):
    iq = IQSymbolSearch()
    result = iq.sic_code_search(digits)
    return result

# Given at least two digits in an existing NIAC code
# Return list of symbols that match this NIAC code
def search_NIAC(digits):
    iq = IQSymbolSearch()
    result = iq.niac_code_search(digits)
    return result

# Request various lists from IQFeed (Listed Markets, Security Types, Trade Conditions, SIC codes and NIAC codes)
# Return a dictionary with descriptive keys that describe each list
def get_iqfeed_lists():
    iq = IQSymbolSearch()
    results = iq.request_lists()
    return results

########################################################################################################################

# If we RUN this python script, run the following (good for testing)
if __name__ == "__main__":
    li = search_symbols("NVDA")
    li = search_descriptions("internet")
    li = search_SIC("34");
    li = search_NIAC("69");
    d = get_iqfeed_lists()
    print(d.keys)

