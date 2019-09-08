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
import matplotlib.pyplot as plt
#%matplotlib inline
from scipy.stats import linregress
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
import warnings


#-----------------------------------------------------------------------------------------------------------------------

import f_folders as folder
from f_args import *
from f_date import *
from f_file import *
from f_calc import *
from f_dataframe import *
from f_rolldate import *
from f_iqfeed import *
from f_plot import *
import f_quandl as q

from c_Trade import *

project_folder = join(data_folder, "vix_es")

#-----------------------------------------------------------------------------------------------------------------------

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




################################################################################

print("Testing ideas for BITCOIN and other crypto currencies")
print()


# Only need to recreate the dataset files if/when a database has added/modified/removed a dataset
#q.create_dataset_codes_for_list(q.bitcoin_db_list)

# This will retrieve the data for ALL database names in the list provided
#q.create_database_data_for_list(q.bitcoin_db_list)

# Or you can retrieve data for individual databases
#q.create_database_data("GDAX")

create_historical_futures_df("@HE")
create_continuous_df("@HE", fn_roll_date=get_roll_date_HE)
create_historical_futures_df("@LE")
create_continuous_df("@LE", fn_roll_date=get_roll_date_LE)

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



df = search_symbols("NVDA")
STOP(df)


d = get_iqfeed_lists()
print(d.keys)
STOP(d)



data_interval = "s1800"     # 30 minutes = 1800 seconds

# To get VX calendar ETS data:
# (1) retrieve latest VX futures data
# (2) create continuous front-month from this futures data
# (3) retrieve latest VX calendar ETS data (using VX continuous to infer one-month-out and two-months-out from front-month symbol)
# (4) create continuous calendar ETS
df_ = create_historical_futures_df("@VX", y1, y2, interval=data_interval, days_back=180, beginFilterTime='093000', endFilterTime='160000')
df_ = create_continuous_df("@VX", get_roll_date_VX, interval=data_interval)
df_ = create_historical_calendar_futures_df("@VX", 0, 1, y1, y2, interval=data_interval, days_back=180, beginFilterTime='093000', endFilterTime='160000')
df_ = create_continuous_calendar_ETS_df("@VX", 0, 1, interval=data_interval)


df = read_dataframe("@VX_continuous_calendar-m0-m1.{0}.DF.csv".format(str_interval(data_interval)))

df = alvin_arima(df).tail(300)
plot_ARIMA('VX Calendar ARIMA', scale_arima=True)



