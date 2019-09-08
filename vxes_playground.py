from datetime import datetime, timedelta
from dateutil import relativedelta
import pandas as pd
from pandas.tseries.offsets import BDay
import numpy as np
import os.path
import math
import sys
"""
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import lstm, time  # helper libraries
"""

#-------------------------------------------------------------------------------
from f_folders import *
import f_date
from f_chart import *
from f_dataframe import *
from f_iqfeed import *

#-------------------------------------------------------------------------------

#mocodes = ['F','G','H','J','K','M','N','Q','U','V','X','Z']

interval_1min = 60
interval_1hour = 3600


# Given a symbol root (ex: "@VX") and a datetime object
# Return the mYY futures contract symbol (ex: "@VXM16") 
def mYY(symbol_root, dt):
    m = dt.month
    y = dt.year
    return symbol_root + monthcodes[m-1] + str(y)[-2:]


# Given a symbol in XXXmYY format (ex: "@VXM16")
# Return the datetime of the first day of that symbol month
# TODO: For now, we assume a XXXmYY (6-char) future symbol
# TODO: Also, we assume year is >= 2000
def get_symbol_date(symbol):
    m = monthcodes.index(symbol[3])+1
    y = 2000 + int(symbol[-2:])
    return datetime(y, m, 1)

    
# Given a symbol root (ex: "@VX") and a month (1-12) and a year (4-digit)
# Return the future symbols for the front month, one month out, and two months out (tuple of 3 values)
def get_symbols(symbol_root, m, y):
    dt0 = datetime(y, m, 1)
    dt1 = dt0 + relativedelta.relativedelta(months=1)
    dt2 = dt1 + relativedelta.relativedelta(months=1)
    return mYY(symbol_root, dt0), mYY(symbol_root, dt1), mYY(symbol_root, dt2)


# Given a datetime
# Return the first Thursday of that month
def first_thursday(dt):
    dtx = dt.replace(day=1)
    while dtx.weekday() != 3:               # weekday 3 is THURSDAY
        dtx += timedelta(days=1)
    return dtx

# Given a datetime
# Return the datetime of the third Friday of that month
def third_friday(dt):
    dtx = dt.replace(day=1)
    if dtx.weekday() == 4:                  # weekday 4 is FRIDAY
        count = 1
    else:
        count = 0
    while count < 3:
        dtx += timedelta(days=1)
        if dtx.weekday() == 4:
            count += 1
    return dtx


# The Final Settlement Date for a contract with the "VX" ticker symbol is on the Wednesday
# that is 30 days prior to the third Friday of the calendar month immediately following the
# month in which the contract expires.
# http://cfe.cboe.com/cfe-products/vx-cboe-volatility-index-vix-futures/contract-specifications
def get_final_settlement_date_VX(symbol):
    dt = get_symbol_date(symbol)
    dt_nextmonth = dt + relativedelta.relativedelta(months=1)
    dt_thirdfri = third_friday(dt_nextmonth)
    dt = dt_thirdfri - timedelta(days=30)
    while dt.weekday() != 2:                # weekday 2 is WEDNESDAY
        dt -= timedelta(days=1)
    return dt.replace(hour=8, minute=0, second=0)       # 8am? (TODO)

# Given a VX future symbol (ex: "@VXmYY")
# Return the roll date for this VX future
def get_roll_date_VX(symbol):
    return get_final_settlement_date_VX(symbol) - BDay(1)

# Given an ES future symbol (ex: "@ESmYY")
# Return the roll date for this ES future
def get_roll_date_ES(symbol):
    m, y = get_month_year(symbol[-3:])
    return first_thursday(datetime(y, m, 1))

# Given an input dataframe filename (that has a 'DateTime' and a 'Symbol' column) and a roll function
# create a dataframe with the given output filename with the correct continuous front-month contracts
def create_continuous_df(symbol_root, fn_roll_date, input_filename="", output_filename=""):
    # Read in the raw futures file (ex: '@VX.csv')
    input_filename = "{0}.csv".format(symbol_root)
    continuous_filename = "{0}_continuous.DF.csv".format(symbol_root)
    df = pd.read_csv(join(df_folder, input_filename), parse_dates=['DateTime'])

    # We will build a new DataFrame that has the rows for which the front-month contract is correct
    # (check roll dates)
    dfz = pd.DataFrame()

    # The dataframe should be sorted by Symbol-then-Datetime, so these unique Symbols should be sorted also
    unique = df.Symbol.unique()

    dt1 = df.DateTime.min()
    for symbol in unique:
        roll_date = fn_roll_date(symbol)
        dfx = df[(df.DateTime >= dt1) & (df.DateTime < roll_date) & (df.Symbol==symbol)]
        if dfz.shape[0] == 0:
            dfz = dfx.copy()
        else:
            dfz = dfz.append(dfx)
        #print symbol, dt1, roll_date
        dt1 = roll_date
    print()
    print("Rows in RAW: {0}    Rows in CONTINUOUS: {1}".format(df.shape[0], dfz.shape[0]))

    # No longer RAW...
    # We now have a file in which the dates have the correct front-month contract (ex: '@VX_continuous.DF.csv')
    dfz.to_csv(join(df_folder, continuous_filename), index=False)
    print("Output to '{0}'".format(continuous_filename))
    return dfz


        
def show_contango_chart(df):
    # CREATE THE CHART
    lines = []
    lines.append(get_chart_line(df, 'contango1x3x2', line_color='rgb(255,0,0)')) #, line_dash='dot'))
    lines.append(get_chart_line(df, 'contango', line_color='rgb(0,0,255)')) #, line_dash='dot'))
    lines.append(get_chart_line(df, 'Close0', line_name='VX'))

    # Highlight the negative ranges on the chart
    #for dfx in rows:
    #    lines.append(get_chart_line(dfx, 'contango1x3x2', line_color='rgb(255,0,0)', line_width=3))

    show_chart(lines, 'VIX_contango')
    return

# Given a symbol root (ex: "@VX")
# create a dataframe file containing the contango values (ex: "contango_@VX.csv")
# this file includes "front" contango (1m0 x 1m1), "next" contango (1m1 x 1m2), and the 1x3x2 contango
def create_contango_df(symbol_root):
    df = pd.read_csv(symbol_root + '.csv', parse_dates=True)
    unique = df.Symbol.unique()
    dfz = pd.DataFrame()
    y1 = 2003
    y2 = 2017
    for y in range(y1, y2+1):
        for m in range(1, 12+1):
            symbols = get_symbols(symbol_root, m, y)
            if symbols[0] in unique and symbols[1] in unique and symbols[2] in unique:
                df0 = df[df.Symbol == symbols[0]]
                df1 = df[df.Symbol == symbols[1]]
                df2 = df[df.Symbol == symbols[2]]
                dfx = pd.merge(df0, df1, on="DateTime")
                dfx = pd.merge(dfx, df2, on="DateTime")
                if dfx.shape[0] > 0:
                    print(dfx.head(1)['DateTime'].values[0], dfx.tail(1)['DateTime'].values[0], symbols[0], symbols[1], symbols[2], dfx.shape[0])
                    if dfz.shape[0] == 0:
                        dfz = dfx.copy()
                    else:
                        dfz = dfz.append(dfx)

    dfz.drop(['Open_x','High_x','Low_x','Volume_x','oi_x','Open_y','High_y','Low_y','Volume_y','oi_y','Open','High','Low','Volume','oi'], axis=1, inplace=True)
    dfz.rename(columns={'Close_x':'Close0', 'Symbol_x':'Symbol', 'Close_y':'Close1', 'Symbol_y':'Symbol1', 'Close':'Close2', 'Symbol':'Symbol2'}, inplace=True)
    dfz['contango'] = dfz.Close1 - dfz.Close0
    dfz['contango2'] = dfz.Close2 - dfz.Close1
    dfz['contango1x3x2'] = (2 * dfz.contango2) - dfz.contango
    dfz.contango = dfz.contango.round(2)
    dfz.contango2 = dfz.contango2.round(2)
    dfz.contango1x3x2 = dfz.contango1x3x2.round(2)
    filename = "contango_{0}.raw.DF.csv".format(symbol_root)
    dfz.to_csv(filename, index=False)
    return dfz



# Given a symbol root (ex: "@VX") and a starting/ending year (y1/y2)
# recreate the functionality of create_historical_futures_df, but ONLY for those symbols that
#  may have updated prices
def update_historical_futures_df(symbol_root, y1, y2):
    raise NotImplementedError
    return


# Given NQ price (float) and ES price (float)
# Return Pat's NQES spread value
def spread_price_NQES(nq, es):
    return (nq * 120 - es * 250) / 100.0

# Given a dataframe and column name to use in calculations (defaults to 'spread')
# Return dataframe with columns added for various SMA and EMA calculations
def get_spread_moving_averages(df, col='spread'):
    df['SMA5'] = df[col].rolling(window=5,center=False).mean()
    df['SMA10'] = df[col].rolling(window=10,center=False).mean()
    df['SMA15'] = df[col].rolling(window=15,center=False).mean()
    df['SMA20'] = df[col].rolling(window=20,center=False).mean()
    df['EMA5'] = df[col].ewm(span=5).mean()
    df['EMA10'] = df[col].ewm(span=10).mean()
    df['EMA15'] = df[col].ewm(span=15).mean()
    df['EMA20'] = df[col].ewm(span=20).mean()
    return df


# Given front-month VX price (float) and front-month ES price (float) and the price of each from 1-day prior to previous expiration
# Return the VXES spread value
# calculations are performed using the net change vs 1-day prior to previous expiry (vx_prev_expiry_close and es_prev_expiry_close)
def spread_price_VXES(vx, es):
    #return (vx * 1000 / 50) - (es / 100)
    return (vx * 10 / 50) - (es / 100)

# Given two dataframes representing CASH INDEX and FRONT-MONTH FUTURE
# Return a dataframe with column added for 'spot_discount' (spot - future)
def get_spot_discount(df_cash, df_front, suffixes=('_CASH','_FRONT')):
    df = pd.merge(df_cash, df_front, on="DateTime", suffixes=suffixes)
    col1 = 'Close' + suffixes[0]
    col2 = 'Close' + suffixes[1]
    df['spot_discount'] = df[col1] - df[col2]
    return df

# Given two dataframes and a pricing function
# Return a single dataframe that merges these two on DateTime and calculates a 'spread' column using the given pricing function
# (optional) suffixes for the dataframe columns default to ('_x','_y'), but you could pass, for instance, suffixes=('_NQ','_ES')
def get_spread(dfx, dfy, fn_price, suffixes=('_x','_y')):
    df = pd.merge(dfx, dfy, on="DateTime", suffixes=suffixes)
    col1 = 'Close' + suffixes[0]
    col2 = 'Close' + suffixes[1]
    df['spread'] = fn_price(df[col1], df[col2])
    return df

# Shortcut function to print a value and exit the script execution
def STOP(x=None):
    print x
    sys.exit()

    
################################################################################
################################################################################
################################################################################


dt1 = datetime(2003, 1, 1)
dt2 = datetime.now()



# Download raw historical VX futures data and output to file '@VX.csv'
dfvx = create_historical_futures_df("@VX", 2016, 2017)
# Using the appropriate roll date calculation, create a continuous contract file from the raw futures data
dfvx = create_continuous_df("@VX", get_roll_date_VX)


# TODO: For now, let's assume DAILY data (will have to modify when it's not daily)
dfvx = pd.read_csv(join(df_folder, "@VX_continuous.DF.csv"), parse_dates=['DateTime'])
unique_symbols = dfvx.Symbol.unique()
for i in range(1, len(unique_symbols)):       # skip the first symbol because we need a previous expiration from which to calculate net chg
    symbol = unique_symbols[i]
    prev_symbol = unique_symbols[i-1]
    df_prev = dfvx[dfvx.Symbol==prev_symbol]
    last_row = df_prev.iloc[df_prev.shape[0]-1]
    prev_expiry_close = last_row.squeeze()['Close']
    dfvx.loc[dfvx.Symbol==symbol,'diff'] = dfvx.Close - prev_expiry_close
dfvx.dropna(inplace=True)
dfvx.loc[:,'Close'] = dfvx.loc[:,'diff']
dfvx.drop(['diff'], axis=1, inplace=True)


# Download raw historical ES futures data and output to file '@ES.csv'
#dfes = create_historical_futures_df("@ES", 2016, 2017)
# Using the appropriate roll date calculation, create a continuous contract file from the raw futures data
#dfes = create_continuous_df("@ES", get_roll_date_ES)

# TODO: For now, let's assume DAILY data (will have to modify when it's not daily)
dfes = pd.read_csv(join(df_folder, "@ES_continuous.DF.csv"), parse_dates=['DateTime'])
unique_symbols = dfes.Symbol.unique()
for i in range(1, len(unique_symbols)):       # skip the first symbol because we need a previous expiration from which to calculate net chg
    symbol = unique_symbols[i]
    prev_symbol = unique_symbols[i-1]
    df_prev = dfes[dfes.Symbol==prev_symbol]
    last_row = df_prev.iloc[df_prev.shape[0]-1]
    prev_expiry_close = last_row.squeeze()['Close']
    dfes.loc[dfes.Symbol==symbol,'diff'] = prev_expiry_close - dfes.Close
dfes.dropna(inplace=True)
dfes.loc[:,'Close'] = dfes.loc[:,'diff']
dfes.drop(['diff'], axis=1, inplace=True)

# Get the VXES spread prices and drop unnecessary columns from the dataframe
df = get_spread(dfvx, dfes, spread_price_VXES, suffixes=('_VX','_ES'))
df.drop(['Open_VX','High_VX','Low_VX','Volume_VX','oi_VX','Open_ES','High_ES','Low_ES','Volume_ES','oi_ES'], axis=1, inplace=True)

df['sum'] = df.spread.cumsum(axis=0)
quick_chart(df, ['sum'], "VX-ES Spread")

STOP()


# Create DAILY VIX Spot Discount
dfvix = get_historical_contract("VIX.XO", dt1, dt2)
dfvx = get_historical_contract("@VX#", dt1, dt2)
df = get_spot_discount(dfvix, dfvx)
quick_chart(df, ['spot_discount'], "Spot Discount")


STOP(df)


# Create DAILY VXEQ Spread
dfvx = get_historical_contract("@VX#", dt1, dt2)
dfes = get_historical_contract("@ES#", dt1, dt2)
df = get_spread(dfvx, dfes, spread_price_VXES, suffixes=('_VX','_ES'))
quick_chart(df, ['spread'])

print df
sys.exit()

# Create 1-MINUTE VXEQ Spread
dt1a = datetime(2017, 5, 1)
dfvx = get_historical_contract("@VX#", dt1a, dt2, interval=interval_1min)
dfes = get_historical_contract("@ES#", dt1a, dt2, interval=interval_1min)
df = get_spread(dfvx, dfes, spread_price_VXES, suffixes=('_VX','_ES'))

df = calc_spread_moving_averages(df)
quick_chart(df, ['spread', 'EMA10'])
                     

                     
print df
sys.exit()


# Download all historical ES futures and output to file '@ES.csv'
#df = create_historical_futures_df("@ES", 2003, 2017)

# Retrieve the 1-minute ES data and output to file '@ES_session_1min.DF.csv'
#df = get_historical_contract("@ES#", dt1, dt2, interval_1min, "083000", "150000")   # 8:30am to 3:00pm
#filename = "@ES_session_1min.DF.csv"
#df.to_csv(filename, index=False)
#print("Output 1-minute data to '{0}'".format(filename))

# Read the 1-minute ES data from file '@ES_session_1min.DF.csv' and output OHLC summary to file '@ES_session_ohlc.DF.csv'
#df_1min = pd.read_csv("@ES_session_1min.DF.csv", parse_dates=['DateTime'])
#df_ohlc = get_ohlc(df_1min)
#filename = "@ES_session_ohlc.DF.csv"
#df_ohlc.to_csv(filename, index=False)
#print("Output OHLC summary data to '{0}'".format(filename))

# Request the daily VIX.XO and @ES data and output the Quartile values to '@ES_quartiles.DF.csv'
#dfq = get_es_quartiles()
#filename = "@ES_quartiles.DF.csv"
#dfq.to_csv(filename, index=False)
#print("Output quartiles to '{0}'".format(filename))
                 
dfq = pd.read_csv("@ES_quartiles.DF.csv", parse_dates=['DateTime'])
df_ohlc = pd.read_csv("@ES_session_ohlc.DF.csv", parse_dates=['DateTime'])

# Calculate the Quartile Hits from Quartiles and OHLC summary data and output to '@ES_quartile_hits.DF.csv'
#dfz = get_quartile_hits(dfq, df_ohlc)
#filename = "@ES_quartile_hits.DF.csv"
#dfz.to_csv(filename, index=False)
#print("Output quartile hits to '{0}'".format(filename))

df_hit = pd.read_csv("@ES_quartile_hits.DF.csv", parse_dates=['DateTime'])
df_vx = pd.read_csv("contango_@VX.DF.csv", parse_dates=['DateTime'])

# Calculate the Quartile Hit Ratios from Quartile Hits and VIX Contango data and output to '@ES_quartile_hit_ratios.DF.csv' 
#df_hr = get_hit_ratios(df_vx, df_hit, 5)
#filename = "@ES_quartile_hit_ratios.DF.csv"
#df_hr.to_csv(filename, index=False)
#print("Output quartile hit ratios to '{0}'".format(filename))

df_hr = pd.read_csv("@ES_quartile_hit_ratios.DF.csv", parse_dates=['DateTime'])


df = get_vix_quartiles()
i1 = df.columns.get_loc('Qd4')
i2 = df.columns.get_loc('Qu4')
cols = ['DateTime']
cols.extend(df.columns.values[i1:i2+1])
#df_vix = df.iloc[:,i1:i2]
df_vix = df.loc[:,cols]
#print(df_vix)
filename = "@VIX_quartiles.DF.csv"
df_vix.to_csv(filename, index=False)
print("Output quartiles to '{0}'".format(filename))



sys.exit()


"""
#seq_len = 50

print('> Loading data... ')

seq_len = 40
prediction_len = 20
filename = "@ES_quartile_hit_ratios.DF.csv"
#df = read_dataframe(join(data_folder, "vix_es", "es_vix_daily_summary.DF.csv"))
df = pd.read_csv(filename, parse_dates=['DateTime'])
input_data_filename = 'NN_INPUT.csv'
f = open(input_data_filename, 'w')
for ix,r in df.iterrows():
    # We need to adjust this so there are no ZEROS in the data
    f.write("{0},{1},{2},{3},{4},{5},{6}\n".format(r['DateTime'],r['contango'],r['d4'],r['d3'],r['d2'],r['d1'],r['unch']))
f.close()

sys.exit()
"""

global_start_time = time.time()
epochs  = 1

seq_len  = 50
prediction_len = 50
input_data_filename = 'sp500.csv'

X_train, y_train, X_test, y_test = lstm.load_data(input_data_filename, seq_len, True)

print("TRAINING ROWS: {0}     TEST ROWS: {1}".format(X_train.shape[0], X_test.shape[0]))

print('> Data Loaded. Compiling...')

#model = lstm.build_model([1, 50, 100, 1])
# Don't hardcode "50" but use seq_len instead because seq_len is the lookback length
# original model layers were [1, 50, 100, 1]
model = lstm.build_model([1, seq_len, seq_len*2, 1])

model.fit(
    X_train,
    y_train,
    batch_size=512,
    nb_epoch=epochs,
    validation_split=0.05)

# For now, set our prediction length to our (input) sequence length
#prediction_len = seq_len

predictions = lstm.predict_sequences_multiple(model, X_test, seq_len, prediction_len)
#predicted = lstm.predict_sequence_full(model, X_test, seq_len)
#predicted = lstm.predict_point_by_point(model, X_test)        

print('Training duration (s) : ', time.time() - global_start_time)

#lstm.plot_results_multiple(predictions, y_test, prediction_len)

predict = lstm.predict_point_by_point(model, X_test)
lstm.plot_results(predict, y_test)


sys.exit()





symbol_root = '@VX'

# Create the '@VX.csv' file containing futures prices for the given year range
#create_historical_futures_df(symbol_root, 2003, 2017)

# Using the '@VX.csv' data file, create 'contango_@VX.raw.DF.csv' containing contango data
#dfz = create_contango_df(symbol_root)

# From the raw data, create the continuous contract contango file
input_filename = "contango_{0}.raw.DF.csv".format(symbol_root)
continuous_filename = "contango_{0}.DF.csv".format(symbol_root)
#dfz = create_continuous_df(input_filename, continuous_filename, get_roll_date_VX)




# Read in the 'contango_@VX.DF.csv' file
df = pd.read_csv(continuous_filename, parse_dates=['DateTime'])

#show_contango_chart(df)


"""
# Get datapoints where the 1x3x2 contango is negative
df_neg = df[df['contango1x3x2']<0]
rows = get_ranges_df(df, df_neg)            # get contiguous date ranges of the negative 1x3x2 values

# Find the slope leading into (before) a point where 1x3x2 is negative
# vs leading out (after)
li = []
for dfx in rows:
    dt1 = dfx.iloc[0].DateTime
    dt2 = dfx.iloc[dfx.shape[0]-1].DateTime
    #print dt1.strftime('%Y-%m-%d'), dt2.strftime('%Y-%m-%d'),
    ix = df[df.DateTime==dt1].index.values[0]
    length = 10
    ix1 = max(ix-length,0)
    ix2 = min(ix+length,df.shape[0]-1)
    dfy = df[ix1:ix]
    #print dfy.shape
    z = np.polyfit(dfy.index, dfy.Close0, 1)
    slope1 = round(z[0], 4)
    #print "{0:7.4f}".format(slope1),
    # f = np.poly1d(z)
    dfy = df[ix:ix2]
    if dfy.shape[0]==0:
        break
    z = np.polyfit(dfy.index, dfy.Close0, 1)
    slope2 = round(z[0], 4)
    #print "{0:7.4f}".format(slope2)
    li.append((slope1, slope2))
slopes = np.array(li)
s1 = abs(slopes[:,0])
s2 = abs(slopes[:,1])
print s1.mean()
print s2.mean()
"""














