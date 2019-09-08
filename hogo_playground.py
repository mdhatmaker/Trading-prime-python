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

#-----------------------------------------------------------------------------------------------------------------------

def print_trades(trade_column, trade_rows, side, entry_price, exit_price, df_rolls):
    average = 0.0
    for (t_entry, t_exit, cal_adjust, discount_adjust) in trade_rows:
        day_count = (t_exit.Date - t_entry.Date).days
        roll_count = roll_exists_in_date_range(t_entry.Date, t_exit.Date, df_rolls)
        roll_count_indicator = '*' * roll_count
        output = get_output(t_entry, t_exit, cal_adjust, discount_adjust)
        output += roll_count_indicator
        print output
        average += day_count
    average /= len(trades)
    print
    print len(trades), "trades"
    print "average holding period (days): %.1f" % (average)
    return

def write_trades_file(filename, trade_column, trade_rows, side, entry_price, exit_price, df_rolls):
    f = open(folder + filename, 'w')
    f.write("Side,EntryDiscount,ExitDiscount,AdjustDiscount,EntrySpread,ExitSpread,AdjustSpread,EntryDate,ExitDate,HoldingDays,RollCount\n")
    for (t_entry, t_exit, cal_adjust, discount_adjust) in trade_rows:
        day_count = (t_exit.Date - t_entry.Date).days
        roll_count = roll_exists_in_date_range(t_entry.Date, t_exit.Date, df_rolls)
        output = get_output(t_entry, t_exit, cal_adjust, discount_adjust)
        output += str(roll_count)
        f.write(output + '\n')
    f.close()
    return

def calc_rolling_mean(df, lookback):
    df['d4'] = df['Qd4'].shift(1).rolling(lookback).mean()
    df['d3'] = df['Qd3'].shift(1).rolling(lookback).mean()
    df['d2'] = df['Qd2'].shift(1).rolling(lookback).mean()
    df['d1'] = df['Qd1'].shift(1).rolling(lookback).mean()
    df['unch'] = df['Qunch'].shift(1).rolling(lookback).mean()
    df['u1'] = df['Qu1'].shift(1).rolling(lookback).mean()
    df['u2'] = df['Qu2'].shift(1).rolling(lookback).mean()
    df['u3'] = df['Qu3'].shift(1).rolling(lookback).mean()
    df['u4'] = df['Qu4'].shift(1).rolling(lookback).mean()
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

################################################################################

print "Calculating quartiles for ES using VIX.XO to determine standard deviation"
print "Output file will be comma-delimeted pandas-ready dataframe (.csv)"
print

project_folder = join(data_folder, "vix_es")

y1 = 2017
y2 = 2017
#df_es = update_historical_futures_df("@ES")
#df_vx = update_historical_futures_df("@VX")

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
sessions = get_sessions(df, session_interval=timedelta(minutes=1))
#sessions = get_sessions(df, session_interval=timedelta(hours=1))

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
df = calculate_ATR(df, df_day, 10)
#quick_chart(df, ['Close_spread','spread_mean','atr_low','atr_high'], title='HOGO')



STOP(df)

filename1 = "@ES_continuous (Daily).csv"
filename2 = "VIX.XO (Daily).csv"
filename3 = "@VX_contango.csv"

lookback_days = 5
output_filename = "es_vix_quartiles ({0}-day lookback).DF.csv".format(lookback_days)

print "project folder:", quoted(project_folder)
print "input files:", quoted(filename1), quoted(filename2), quoted(filename3)
print "lookback days:", lookback_days
print

print "----------BEGIN CALCULATIONS----------"

df1 = read_dataframe(join(project_folder, filename1))
df2 = read_dataframe(join(project_folder, filename2))
df3 = read_dataframe(join(project_folder, filename3))

df1 = df1.drop(['High', 'Low'], 1)
df2 = df2.drop(['High', 'Low', 'Volume'], 1)
df3 = df3.drop(['Open_x', 'Close_x', 'Open_y', 'Close_y', 'Open', 'Close'], 1)

df = pd.merge(df1, df2, on=['DateTime'])
df = pd.merge(df, df3, on=['DateTime'])

df.rename(columns={'Symbol_x':'Symbol_ES', 'Open_x':'Open_ES', 'Close_x':'Close_ES', 'Volume':'Volume_ES', 'Open_y':'Open_VIX', 'Close_y':'Close_VIX', 'Symbol_y':'Symbol_VX', 'Volume_x':'Volume_VX1', 'Volume_y':'Volume_VX2', 'Contango_Open':'Open_Contango', 'Contango_Close':'Close_Contango'}, inplace=True)

# Output this daily VIX/ES summary data (for potential analysis or debugging)
summary_filename = "es_vix_daily_summary.DF.csv"
write_dataframe(df, join(project_folder, summary_filename))
print
print "VIX/ES summary (daily data) output to file:", quoted(summary_filename)
print

symbols = df_get_sorted_symbols(df, 'Symbol_ES')            # get list of unique ES symbols in our data

rows = []                                                   # store our row tuples here (they will eventually be used to create a dataframe)

# For each specific ES symbol, perform our quartile calculations
for es in symbols:  # symbols[-2:]: 
    print "Processing future:", es
    df_es = read_dataframe(get_df_pathname(es))
    dfx = df[df.Symbol_ES == es]
    for (ix,row) in dfx.iterrows():
        if not (ix+1) in dfx.index:                         # if we are at the end of the dataframe rows (next row doesn't exist)
            continue
        
        # Use Close of ES and VIX
        es_close = row['Close_ES']                          # ES close
        vix_close = row['Close_VIX']                        # VIX close
        #std = vix_close / math.sqrt(252)                    # calculate standard deviation
        std = round(Calc_Std(vix_close), 4)
        dt_prev = row['DateTime']                           # date of ES/VIX close to use
        dt = dfx.loc[ix+1].DateTime                         # following date (of actual quartile calculation)

        # Get the ES 1-minute bars for the day session (for date following date of ES/VIX close)
        exchange_open = dt.replace(hour=8, minute=30)
        exchange_close = dt.replace(hour=15, minute=0)
        df_day = df_es[(df_es.DateTime > exchange_open) & (df_es.DateTime <= exchange_close)]   # ES 1-minute bars for day session
        day_open = df_day.iloc[0]['Open']
        day_high = df_day.High.max()
        day_low = df_day.Low.min()
        row_count=df_day.shape[0]
        day_close = df_day.iloc[row_count-1]['Close']

        # For each quartile, determine if it was hit during the day session of ES
        hit_quartile = {}
        #print es_close, std
        (q_list, q_dict) = Calc_Quartiles(es_close, std)
        for i in range(+4, -5, -1):
            #quartile = Quartile(es_close, std, i)       # for the given close price and standard dev, calculate the quartiles
            #quartile = round(quartile, 4)
            quartile = q_dict[i]
            if day_low <= quartile and day_high >= quartile:
                hit_quartile[i] = 1
            else:
                hit_quartile[i] = 0
            #print i, quartile, hit_quartile[i]
            
        rows.append((dt, es, es_close, vix_close, std, day_open, day_high, day_low, day_close, hit_quartile[-4], hit_quartile[-3], hit_quartile[-2], hit_quartile[-1], hit_quartile[0], hit_quartile[1], hit_quartile[2], hit_quartile[3], hit_quartile[4]))

df_new = pd.DataFrame(rows, columns=['DateTime', 'Symbol_ES', 'Prev_Close_ES', 'Prev_Close_VIX', 'Std', 'SessionOpen_ES', 'SessionHigh_ES', 'SessionLow_ES', 'SessionClose_ES', 'Qd4', 'Qd3', 'Qd2', 'Qd1', 'Qunch', 'Qu1', 'Qu2', 'Qu3', 'Qu4'])        
#df_new = pd.DataFrame(rows, columns=['DateTime', 'Symbol_ES', 'Close_ES', 'Std', 'Qd4', 'Qd3', 'Qd2', 'Qd1', 'Qunch', 'Qu1', 'Qu2', 'Qu3', 'Qu4'])
# df_new = df_new.set_index(['some_col1', 'some_col2'])       # Possibly also this if these can always be the indexes

calc_rolling_mean(df_new, lookback_days)

calc_days_to_quartile_hit = False
if calc_days_to_quartile_hit:
    # Create columns that will hold the number of days until a quartile hit occurs
    undef_value = -1
    df_new['days_to_d4'] = undef_value
    df_new['days_to_d3'] = undef_value
    df_new['days_to_d2'] = undef_value
    df_new['days_to_d1'] = undef_value
    df_new['days_to_unch'] = undef_value
    df_new['days_to_u1'] = undef_value
    df_new['days_to_u2'] = undef_value
    df_new['days_to_u3'] = undef_value
    df_new['days_to_u4'] = undef_value

    print "\nIterating dataframe rows to determine actual days-to-hit numbers for each quartile . . .",

    # Now scan dataframe rows to determine how many days until each specific quartile is ACTUALLY hit
    for (ix,row) in df_new.iterrows():
        if ix % 100 == 0:
            print '.',
        if np.isnan(row['unch']):
            continue
        dt = row['DateTime']

        for col in ['d4', 'd3', 'd2', 'd1', 'unch', 'u1', 'u2', 'u3', 'u4']:
            qcol = "Q" + col
            dayscol = "days_to_" + col
            dfx = df_new[(df_new[qcol] == 1) & (df_new.DateTime >= dt)]
            # An empty dataframe means there were no rows where the specified quartile was hit that satisfied the date constraint
            if not dfx.empty:
                ix_hit = dfx.index[0]
                df_new.set_value(ix, dayscol, ix_hit-ix)
    print



write_dataframe(df_new, join(project_folder, output_filename))

print
print "Quartile analysis output to file:", quoted(output_filename)
print




"""
from __future__ import print_function
import sys
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import math
from datetime import time
import pickle
#from statsmodels.tsa.arima_model import _arma_predict_out_of_sample     # this is the nsteps ahead predictor function
import statsmodels.api as sm
#import pyflux as pf
#import matplotlib.pyplot as plt
#%matplotlib inline
from scipy.stats import linregress
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
import warnings
import urllib
import json
import hmac
import hashlib

q_columns = ['d4','d3','d2','d1','unch','u1','u2','u3','u4']

def calc_rolling_mean(df, lookback):
    for col in q_columns:
        Qcol = 'Q' + col
        df[col] = df[Qcol].shift(1).rolling(lookback).mean()
    return

def get_sessions(df, session_start=time(20,0), session_length=timedelta(hours=21), session_interval=timedelta(hours=1)):
    print("Splitting data into sessions ...", end='')
    unique_dates = df_get_unique_dates(df)
    sessions = []
    for dt in unique_dates:
        dt1 = datetime(dt.year, dt.month, dt.day, session_start.hour, session_start.minute, 0)
        dt2 = dt1 + session_length - session_interval
        df_sess = df[(df['DateTime']>=dt1) & (df['DateTime']<=dt2)]     # all rows in session (between dt1 and dt2)
        if df_sess.shape[0] > 0:
            sessions.append(df_sess)
    print("Done.")
    return sessions

def calculate_spread_OHLC(df, sessions):
    print("Calculating spread OHLC for each session ...", end='')
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
    print("Done.")
    return df.dropna()

def get_last_session_rows(df, sessions):
    print("Getting last row from each session (effective close) ...", end='')
    df_day = pd.DataFrame()
    for df_sess in sessions:
        df_temp = df.loc[df_sess.index, :]
        df_day = pd.concat([df_day, df_temp.tail(1)])
    print("Done.")
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

    print("Copying mean, ATR_low and ATR_high to each row ...", end='')
    for df_sess in sessions:
        dt = df_sess.tail(1).squeeze()['DateTime']
        row = df_day[df_day['DateTime'] == dt]
        if row.shape[0] > 0:
            df.loc[df_sess.index, 'spread_mean'] = row.squeeze()['spread_mean']
            df.loc[df_sess.index, 'atr_low'] = row.squeeze()['atr_low']
            df.loc[df_sess.index, 'atr_high'] = row.squeeze()['atr_high']
    print("Done.")
    return df

def fit_line1(x, y):
    """Return slope, intercept of best fit line."""
    # Remove entries where either x or y is NaN.
    clean_data = pd.concat([x, y], axis=1).dropna(axis=0) # row-wise
    (_, x), (_, y) = clean_data.iteritems()
    slope, intercept, r, p, stderr = linregress(x, y)
    return slope, intercept # could also return stderr

def fit_line2(x, y):
    """Return slope, intercept of best fit line."""
    X = sm.add_constant(x)
    model = sm.OLS(y, X, missing='drop') # ignores entires where x or y is NaN
    fit = model.fit()
    return fit.params[1], fit.params[0] # could also return stderr in each via fit.bse

def plot_fit_line(x, y):
    m, b = fit_line2(x, y)
    N = 100 # could be just 2 if you are only drawing a straight line...
    points = np.linspace(x.min(), x.max(), N)
    plt.plot(points, m*points + b)
    return

def linreg_scipy(x, y):
    return stats.linregress(x, y)

def linreg_np(x, y):
    return np.polynomial.polynomial.polyfit(x, y, 1)

def linreg_sm_ols(x, y):
    x = sm.add_constant(x)
    return sm.OLS(y, x)

def linreg(np_values):
    #print(np_values)
    x = range(len(np_values))
    lr1 = linreg_scipy(x, np_values)
    #lr2 = linreg_np(x, np_values)
    #lr3 = linreg_sm_ols(x, np_values)
    #print("linreg_scipy:", lr1)
    #print("linreg_np:", lr2)
    #print("linreg_sm_ols:", lr3)
    m = lr1.slope
    b = lr1.intercept
    #y_hat = m * -1 + b          # calculate forecast (y_hat)
    y_hat = m * (len(np_values)+1) + b          # calculate forecast (y_hat)
    #return y_hat, lr1.rvalue, lr1.pvalue, lr1.stderr
    return y_hat


def choose_optimal_lag(cors, correlation_sig):
    optimal_lag = -1
    for i in range(len(cors)):
        if abs(cors[i]) > correlation_sig:
            optimal_lag = i
            break
    #if optimal_lag == -1:
    #    raise ValueError("correlation_sig ({0}) not hit!".format(correlation_sig))
    return optimal_lag

def get_p_optimal(y, rmse_target=100.0):
    N = len(y)
    p_optimal = 0
    #for p in range(1,13+1):
    for p in range(1,4+1):
        model = sm.tsa.ARIMA(y, order=(p, 0, 0))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # Line that is not converging
            model_fit = model.fit(disp=False, transparams=False) #, trend='nc')
        yhat = model_fit.forecast()[0]
        #print(yhat)
        y_predicted = model_fit.predict(start=0, end=N-1) #, typ='levels')
        #print(y_predicted)
        rmse = math.sqrt(mean_squared_error(y, y_predicted))
        #print("RMSE for p={0}: {1:.6f}".format(p, rmse))
        if rmse < rmse_target:
            p_optimal = p
            break
        # plot forecasts against actual outcomes
        #plt.figure(p)
        #plt.plot(y)
        #plt.plot(y_predicted, color='red')
        #plt.title("p={0} (RMSE={1})".format(p, rmse))
    #plt.show()
    return p_optimal, yhat

def linregarray(price_value_array, tgtpos=0):
    size = len(price_value_array)
    var0 = 0
    var1 = 0
    var2 = 0
    var3 = 0
    var4 = 1.0 / 6.0
    var5 = 0

    if size <= 1:
        return None

    var2 = size * (size - 1 ) * .5
    var3 = size * (size - 1 ) * (2 * size - 1 ) * var4
    var5 = var2**2 - size * var3

    var0 = 0
    for i in range(size):
        var0 += i * price_value_array[i]

    for i in range(size):
        var1 += price_value_array[i]

    oLRSlope = ( size * var0 - var2 * var1) / var5
    oLRAngle = math.atan(oLRSlope)
    oLRIntercept = (var1 - oLRSlope * var2) / size
    oLRValueRaw = oLRIntercept + oLRSlope * (size - 1 - tgtpos)
    return oLRValueRaw

def alvin_coefficientX(independent_array, dependent_array):
    size = len(independent_array)
    sumX = 0
    sumY = 0

    if size <= 0:
        return 0

    # Get the sums of the array values
    for i in range(size):
        sumX += independent_array[i]
        sumY += dependent_array[i]

    # Get the averages of the array values
    avgX = sumX / size
    avgY = sumY / size

    prodxy = 0
    x2 = 0
    y2 = 0
    for i in range(size):
        diffX = independent_array[i] - avgX
        diffY = dependent_array[i] - avgY
        prodxy += (diffX * diffY)
        x2 += (diffX * diffX)
        y2 += (diffY * diffY)

    if prodxy != 0:
        return prodxy / math.sqrt(x2*y2)
    else:
        return 0    # TODO: Is this correct?!?

def alvin_coefficient(independent_array, dependent_array):
    size = len(independent_array)
    sumX = 0
    sumY = 0

    if size <= 0:
        return 0

    sumX = np.sum(independent_array)
    sumY = np.sum(dependent_array)

    # Get the averages of the array values
    avgX = sumX / size
    avgY = sumY / size

    diffX = np.subtract(independent_array, avgX)
    diffY = np.subtract(dependent_array, avgY)
    prodxy = np.multiply(diffX, diffY)
    diffX2 = np.multiply(diffX, diffX)
    diffY2 = np.multiply(diffY, diffY)
    prodxy = prodxy.sum()
    x2 = diffX2.sum()
    y2 = diffY2.sum()

    if prodxy != 0:
        return prodxy / math.sqrt(x2*y2)
    else:
        return 0    # TODO: Is this correct?!?


def optimal_coefficient(ac, correlation_sig = 0.33):
    for i in range(len(ac)):
        if ac[i] >= correlation_sig:
            return i
    return -1

def alvin_arima(df, lookback=40, correlation_sig = 0.33):
    # A three-point average is sometimes used to smooth the signal:
    #   avgclose = (close + close[1] + close[2]) / 3
    df['avg_close'] = df['Close']   # this could be set to average of last 3 close values (to smooth)
    df['diff'] = df['avg_close'] - df['avg_close'].shift(1)
    df['diff1'] = df['diff'] - df['diff'].shift(1)
    df['diff2'] = df['diff'] - df['diff'].shift(2)
    df['diff3'] = df['diff'] - df['diff'].shift(3)
    df['diff4'] = df['diff'] - df['diff'].shift(4)
    df['diff5'] = df['diff'] - df['diff'].shift(5)
    df['diff6'] = df['diff'] - df['diff'].shift(6)

    df.dropna(inplace=True)

    df['ARIMA'] = np.nan
    df.reset_index(drop=True, inplace=True)

    ix1 = 0
    ix2 = df.shape[0]-lookback+1
    print("Running regression to index {0}...".format(ix2))
    start_time = datetime.now()
    for ix in range(ix1, ix2):
        a1 = np.array(df.loc[ix:ix+lookback-1, 'diff1'])
        a2 = np.array(df.loc[ix:ix+lookback-1, 'diff2'])
        a3 = np.array(df.loc[ix:ix+lookback-1, 'diff3'])
        a4 = np.array(df.loc[ix:ix+lookback-1, 'diff4'])
        a5 = np.array(df.loc[ix:ix+lookback-1, 'diff5'])
        a6 = np.array(df.loc[ix:ix+lookback-1, 'diff6'])

        ac = []
        ac.append(alvin_coefficient(a1, a2))
        ac.append(alvin_coefficient(a1, a3))
        ac.append(alvin_coefficient(a1, a4))
        ac.append(alvin_coefficient(a1, a5))
        ac.append(alvin_coefficient(a1, a6))
        opti = optimal_coefficient(ac)
        if opti == 0:
            lra = linregarray(a1)
        elif opti == 1:
            lra = linregarray(a2)
        elif opti == 2:
            lra = linregarray(a3)
        elif opti == 3:
            lra = linregarray(a4)
        elif opti == 4:
            lra = linregarray(a5)
        elif opti == 5:
            lra = linregarray(a6)
        else:
            print("ERROR: No optimal lag found for ix={0}".format(ix))
            lra = linregarray(a1)

        if ix % 100 == 0:
            print("{0:2d}: {1} {2:.2f} {3:.2f} {4:.2f} {5:.2f} {6:.2f}   {7:.6f}".format(ix, opti, ac[0], ac[1], ac[2], ac[3], ac[4], lra))

        df.loc[ix+lookback,'ARIMA'] = lra

    elapsed = datetime.now() - start_time
    print("elapsed time: {0}".format(elapsed))

    # I *think* this is how to calculate the Exponential Moving Average
    df['EMA'] = df['diff'].rolling(window=lookback, win_type='bartlett').mean()

    return df

# Given a dataframe containing columns 'EMA' and 'ARIMA'
# scale the ARIMA values to match the scale of the EMA
def scale_arima_plot(df):
    abs_ema_max = max(abs(df.EMA.max()), abs(df.EMA.min()))
    abs_arima_max = max(abs(df.ARIMA.max()), abs(df.ARIMA.min()))
    arima_max = abs_ema_max * df.ARIMA.max() / abs_arima_max
    arima_min = abs_ema_max * df.ARIMA.min() / abs_arima_max
    df['ARIMA'] = abs_ema_max * df.ARIMA / abs_arima_max
    return df

# Given a dataframe containing columns 'EMA' and 'ARIMA' (and 'Close' for closing price)
# plot the EMA/ARIMA on the top subplot and underlying price on the bottom subplot
def plot_ARIMA(df, title="", scale_arima=False):
    if scale_arima == True:
        scale_arima_plot(df)
    plot_with_hline(df, subplot=211)
    ind = df.index
    plt.plot(ind, df['EMA'], color='yellow', linewidth=1)
    plt.bar(ind, df['ARIMA'], color='white')
    df['xtick'] = df['DateTime'].apply(lambda x: x.strftime('%m-%d') if (x.day % 2 == 0 and x.hour == 12 and x.minute == 0) else '')
    plt.xticks(ind, df['xtick'])
    plt.subplot(212)
    plt.title(title)
    plt.plot(ind, df['Close'], color='red', linewidth=2)
    plot_hline(df)
    plt.xticks(ind, df['xtick'])
    plt.show()
    return
"""