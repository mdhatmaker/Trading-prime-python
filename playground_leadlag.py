import sys, os
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar import var_model
import statsmodels.tsa.stattools as sm_tools
import statsmodels.tsa.api as smt
import statsmodels
import datetime
import matplotlib.pyplot as plt
from numpy.fft import rfft, irfft
import random
from matlib import matzero, matInsertConstCol, matprint
from scipy import linalg
from scipy.stats import norm, mstats
import urllib2
from pandas.plotting import autocorrelation_plot

#-----------------------------------------------------------------------------------------------------------------------
from f_folders import *
from f_date import *
from f_plot import *
from f_stats import *
from f_dataframe import *

pd.set_option('display.width', 160)
fcout = None
#-----------------------------------------------------------------------------------------------------------------------

def multi_dim_granger(X_ts, Y_ts, maxlags=5, test='F-test'):
    ts = np.hstack((X, Y))
    VAR_model = var_model.VAR(ts)
    results = VAR_model.fit(ic='aic', maxlags=maxlags)
    #return var_results.coefs
    return results

def df_granger(df, maxlags=10, test='F-test'):
    ts = df.values
    VAR_model = var_model.VAR(ts)
    results = VAR_model.fit(ic='aic', maxlags=maxlags)
    return results

"""
def example_numpy_array_to_dataframe():
    dtype = [('Col1','int32'), ('Col2','float32'), ('Col3','float32')]
    values = np.zeros(20, dtype=dtype)
    index = ['Row'+str(i) for i in range(1, len(values)+1)]
    print values
    df = pd.DataFrame(values, index=index)
    return df

def example_numpy_array_to_dataframe_with_index():
    X_ts = np.random.randn(1000, 2)
    Y_ts = (np.arange(4000) * np.random.randn(4000)).reshape((1000, 4))
    ts = np.hstack((X_ts, Y_ts))
    df = pd.DataFrame({'x1':ts[:,0],'x2':ts[:,1],'y1':ts[:,2],'y2':ts[:,3],'y3':ts[:,4],'y4':ts[:,5]})
    date_range = previous_days_date_range(ts.shape[0])
    df1['DateTime'] = date_range
    df = df1.set_index('DateTime')
    return df

def example_excel_to_dataframe_lag():
    cwd = os.getcwd()
    old = os.path.join(cwd, 'dat','prod_lf_usuk_520.xlsx')
    new=old.replace('\\','/')    
    myList = ['UK','US']  
    for idx, region in enumerate(myList):
        # df is the normalized set of macroeconomic variables.
        # It is a 483x6 matrix.
        df = pd.read_excel(open(new,'rb'), sheetname = region)
        df = (df - np.mean(df, axis=0))/np.std(df, axis = 0)
        model = stats.VAR(df)
        model.select_order(15)     
        results = model.fit(maxlags=15, ic='aic')
        results.summary()
        lag_order = results.k_ar
        steps_to_forecast = 1
        results.forecast(df[-lag_order:], steps_to_forecast)
    return
"""

def crosscorr(datax, datay, lag=0):
    """ Lag-N cross correlation.
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    return datax.corr(datay.shift(lag))

def check_lags(dfx, col1, col2, half_count=9, display=False):
    if display: print "\nCross Correlation: {0} vs {1}      ({2} datapoints)".format(col1, col2, dfx.shape[0])
    corrs = {}
    for lag in range(-half_count, half_count+1):
        cc = crosscorr(dfx[col1], dfx[col2], lag)
        if display: print "lag={0:3}   crosscorr={1:7.2f}%".format(lag, cc*100)
        corrs[lag] = cc*100
    return corrs

# Run some basic lead/lag analysis on coinmetrics data
def coinmetrics_leadlag(df_dict, symbol1='btc', symbol2='eth'):
    sym1 = symbol1.upper()
    sym2 = symbol2.upper()
    dfx = pd.merge(df_dict[symbol1], df_dict[symbol2], on="DateTime", suffixes=('_'+sym1, '_'+sym2))
    col1 = 'PriceUSD_'+sym1
    col2 = 'PriceUSD_'+sym2
    check_lags(dfx, col1, col2, True)
    check_lags(dfx[-90:], col1, col2, display=True)
    check_lags(dfx[-60:], col1, col2, display=True)
    check_lags(dfx[-30:], col1, col2, display=True)

    min_indexes = []
    max_indexes = []

    count_for_avg = 10
    xx = 30
    for i in range(-720,-xx):
        corrs = check_lags(dfx[i:i+xx], col1, col2)
        e = corrs.keys()
        e.sort(cmp=lambda a,b: cmp(corrs[a],corrs[b]))
        dtstr = dfx.iloc[i].DateTime.strftime('%Y-%m-%d')
        zerolag = 0
        print "{0}   min: [{1:3}]={2:6.2f}    max: [{3:3}]={4:6.2f}      [{5}]={6:6.2f}".format(dtstr, e[0], corrs[e[0]], e[-1], corrs[e[-1]], zerolag, corrs[zerolag]),
        min_indexes.append(e[0])
        max_indexes.append(e[-1])
        li = min_indexes[-count_for_avg:-1]
        if len(li) > 0:
            min_avg = float(sum(li))/len(li)
            li = max_indexes[-count_for_avg:-1]
            max_avg = float(sum(li))/len(li)
            print "               min_avg: {0:5.1f}   max_avg: {1:5.1f}".format(min_avg, max_avg)
    print min_indexes
    print max_indexes
    return dfx

# For all coinmetrics symbols, perform lead/lag analysis
def analyze_coinmetrics_leadlag():
    coinmetrics_symbols = ['btc', 'bch', 'ltc', 'eth', 'xem', 'dcr', 'zec', 'dash', 'doge', 'etc', 'pivx', 'xmr']
    # Run some lead/lag analysis on the coinmetrics (daily) data
    df_cm = {}
    for symbol in coinmetrics_symbols:
        pathname = join(df_folder, "coinmetrics.{0}.daily.DF.csv".format(symbol))
        print "Reading dataframe '{0}' ...".format(pathname),
        df_cm[symbol] = pd.read_csv(pathname, parse_dates=['DateTime'])
        print "Done."
    dfx = coinmetrics_leadlag(df_cm, 'btc', 'eth')
    return

def normalize_df(dfx, display=False):
    df = (dfx - np.mean(dfx, axis=0))/np.std(dfx, axis = 0)
    means = np.mean(dfx, axis=0)
    stds = np.std(dfx, axis=0)
    if display:
        for col,mean,std in zip(dfx.columns, means, stds):
            print "{0} ==>  mean: {1:.2f}   std: {2:.2f}".format(col, mean, std)
        print "------------------------------------------------------"
    return df

def returns_df(dfx):
    x = dfx.iloc[:,0].values    # values (prices) of first time series
    y = dfx.iloc[:,1].values    # values (prices) of second time series
    rx = returns(x)
    ry = returns(y)
    df = dfx.copy()
    df.iloc[1:,0] = rx
    df.iloc[1:,1] = ry
    df = df.drop(df.index[[0]])
    return df

def prices_df(dfx):
    x = dfx.iloc[1:,0].values       # skip first value because it contains starting price
    y = dfx.iloc[1:,1].values       # all other values in time series are diff(log(price))
    px = prices(x, dfx.iloc[0,0])
    py = prices(y, dfx.iloc[0,1])
    df = dfx.copy()
    df.iloc[:,0] = px
    df.iloc[:,1] = py
    return df

def diff_df(dfx):
    x = dfx.iloc[:,0].values
    y = dfx.iloc[:,1].values
    dx = np.diff(x)
    dy = np.diff(y)
    df = dfx.copy()
    df.iloc[1:,0] = dx
    df.iloc[1:,1] = dy
    df = df.drop(df.index[[0]])
    return df
    
def lead_lag(dfx, maxlags=10, steps_to_forecast=1, preprocess=0, display=True):
    if preprocess == 0:
        df = dfx                                # preprocess:0 --> do nothing
    elif preprocess == 1:
        #df = returns_df(dfx)                    # preprocess:1 --> returns (diff of the log of the values)
        df = diff_df(dfx)
    elif preprocess == 2:
        df = normalize_df(dfx, display=display) # preprocess:2 --> normalize using mean and standard deviation
    model = smt.VAR(df)
    model.select_order(maxlags=maxlags, verbose=display)     
    results = model.fit(maxlags=maxlags, ic='aic')
    if display: print results.summary()
    lag_order = results.k_ar
    steps_to_forecast = 1
    #print "lag_order: {0}".format(lag_order)
    dfz = df[-lag_order:]
    forecast = results.forecast(dfz.values, steps_to_forecast)
    if display: print "forecast (lag_order={0}):\n{1}\n".format(lag_order, forecast)
    #df[['Forecast_btc','Forecast_eth']] = results.fittedvalues
    #df[['Forecast_X','Forecast_Y']] = res.fittedvalues
    size = res.fittedvalues.shape[0]
    df = df[-size:].copy()
    #df[['Forecast_x','Forecast_y']] = res.fittedvalues
    return results, forecast, df.dropna(), lag_order

def koyckZmatrix(Y, X, withconstcol=True):
    n = len(Y)
    if n != len(X):
        raise ValueError, "in koyckZmatrix():Y,X have different lengths!"
    Z = matzero(n-1, 2)
    for i in range(n-1):        # one less due to lag
        Z[i][0], Z[i][1] = X[i+1], Y[i]
    if withconstcol:
        matInsertConstCol(Z, 0, 1)
    return Y[1:], Z

def leviatan(Y, X):
    """
    Solves the Koyck distributed lag system by instrumental variable technique.
    Reference: pp.678-679, Gujarati, 4e, "Basic Econometrics"
    """
    # RHS vector.
    b = [0.0] * 3
    b[0] = sum(Y[1:])
    b[1] = sum([y * x for y, x in zip(Y[1:], X[1:])])
    b[2] = sum([y * x for y, x in zip(Y[1:], X[:-1])])

    # Least squares matrix.
    A = matzero(3, 3)
    A[0][0] = len(Y) - 1
    A[0][1] = A[1][0] = sum(X[:-1])
    A[0][2] = sum(Y[1:])

    A[1][0] = sum(X[:-1])
    A[1][1] = sum([x * x for x in X[1:]])
    A[1][2] = sum([y * x for y, x in zip(Y[:-1], X[1:])])

    A[2][0] = sum([x for x in X[:-1]])
    A[2][1] = sum([x * xtm1 for x, xtm1 in zip(X[1:], X[:-1])])
    A[2][2] = sum([ytm1 * xtm1 for ytm1, xtm1 in zip(Y[:-1], X[:-1])])

    # Call linear solver
    C = linalg.solve(A, b)
    return C

# Determine simple monotonic trend using Mann-Kendall test for trend.
# Given a vector of data (x)
# (optional) significance level (alpha) defaults to 0.05
# Return trend (increasing, decreasing, no trend), is_trend_present (True/False), p-value of significance test and normalized test statistics
def mk_test(x, alpha=0.05, display=True):
    """
    Input:
        x: a vector of data
        alpha: significance level (0.05 default)

    Output:
        trend: tells the trend (increasing, decreasing or no trend)
        h: True (if trend is present) or False (if trend is absent)
        p: p value of the significance test
        z: normalized test statistics

    Examples
    --------
      x = np.random.rand(100)
      trend,h,p,z = mk_test(x, 0.05)
    """
    n = len(x)

    # calculate S
    s = 0
    for k in range(n-1):
        for j in range(k+1, n):
            s += np.sign(x[j] - x[k])

    # calculate the unique data
    unique_x = np.unique(x)
    g = len(unique_x)

    # calculate the var(s)
    if n == g: # there is no tie
        var_s = (n * (n-1) * (2 * n + 5)) / 18
    else: # there are some ties in the data
        tp = np.zeros(unique_x.shape)
        for i in range(len(unique_x)):
            tp[i] = sum(unique_x[i] == x)
        var_s = (n * (n-1) * (2 * n + 5) + np.sum(tp * (tp-1) * (2 * tp + 5))) / 18

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    #elif s == 0:
    else:
        z = 0

    # calculate the p_value
    p = 2 * (1 - norm.cdf(abs(z))) # two tail test
    h = abs(z) > norm.ppf(1 - alpha/2)

    if (z < 0) and h:
        trend = 'decreasing'
    elif (z > 0) and h:
        trend = 'increasing'
    else:
        trend = 'no trend'

    if display: print "'{0}'  has_trend: {1}  p-value: {2}  normalized_stats: {3}".format(trend, h, p, z)
    return trend, h, p, z

def get_trend(x, alpha=0.05):
    trend, h, p, z = mk_test(x, alpha=alpha, display=False)
    if trend == 'decreasing':
        return -1
    elif trend == 'increasing':
        return 1
    else:
        return 0

# FOR TESTING: get sample data time series "streamflow"
def get_sample_series_streamflow():
    # Import the sample streamflow dataset
    data = urllib2.urlopen('https://raw.github.com/mps9506/Sample-Datasets/master/Streamflow/USGS-Monthly_Streamflow_Bend_OR.tsv')
    df = pd.read_csv(data, sep='\t')

    # The yyyy, mm and dd are in separate columns, so make this a single column
    df['dti'] = df[['year_nu','month_nu','dd_nu']].apply(lambda x: datetime(*x), axis=1)

    # Let's use this as our index since we are using pandas
    df.index = pd.DatetimeIndex(df['dti'])
    # Clean the dataframe a bit
    df = df.drop(['dd_nu','year_nu','month_nu','dti'], axis=1)
    df = df.resample('M', how='mean')
    print ">>> STREAMFLOW SAMPLE DATA:\n{0}\n".format(df.head())
    #fig,ax = plt.subplots(1, 1, figsize=(6,4))
    flow = df['mean_va']
    flow = flow['1949-01':]
    return flow

def get_series_kraken(symbol='BCHUSD'):
    filename = "kraken.{0}.5minute.DF.csv".format(symbol)
    pathname = join(df_folder, filename)
    df = read_dataframe(pathname)
    # Let's use this as our index since we are using pandas
    df.index = pd.DatetimeIndex(df['DateTime'])
    # Clean the dataframe a bit
    df = df.drop(['Symbol','DateTime','Open','High','Low','Close','Volume','Count'], axis=1)
    #df = df.resample('M', how='mean')
    #df = df.resample('5T', label='right', closed='right').mean()
    df[df.VWAP==0.0] = np.nan
    #df = df.resample('5Min', label='right', closed='right').mean()
    print ">>> DATAFRAME:\n{0}\n".format(df.head())
    return df['VWAP']

# Given a time series (x)
# (optional) filename of PNG saved in MISC data folder (plot_filename)
# Return (from decomposed time series): residual, seasonal, trend
def decompose_time_series(x, plot_filename=''):
    res = sm.tsa.seasonal_decompose(x)
    fig = res.plot()
    if plot_filename == '': plot_filename = 'decompose_time_series'
    filename = join(misc_folder, '{}.png'.format(plot_filename))
    fig.savefig(filename)
    fig.show()
    return res.resid, res.seasonal, res.trend

# Typical Pandas rolling SMA calculation
def sma1(s, span=40, min=None):
    if min is None: min = int(.80 * span)
    return s.rolling(window=span, min_periods=min).mean()

# Typical Pandas rolling EWMA calculation
def ema1(s, span=40, min=None):
    if min is None: min = int(.80 * span)
    return s.ewm(span=span, min_periods=min, adjust=False).mean()

# Slightly modified Pandas rolling EWMA calculation
def ema2(s, span=40, min=None):
    if min is None: min = int(.80 * span)
    sma = s.rolling(window=span, min_periods=min).mean()[:span]
    rest = s[span:]
    return pd.concat([sma, rest]).ewm(span=span, min_periods=min, adjust=False).mean()

# The Holt-Winters second order method attempts to incorporate the estimated trend into the
# smoothed data, using a b_t term that keeps track of the slope of the original signal.
# http://connor-johnson.com/2014/02/01/smoothing-with-exponentially-weighted-moving-averages/
def holt_winters_second_order_ewma(x, span, beta):
    N = x.size
    alpha = 2.0 / (1 + span)
    s = np.zeros((N,))
    b = np.zeros((N,))
    s[0] = x[0]
    for i in range(1, N):
        s[i] = alpha * x[i] + (1 - alpha) * (s[i - 1] + b[i - 1])
        b[i] = beta * (s[i] - s[i - 1]) + (1 - beta) * b[i - 1]
    return s

# The EWMA function is really only appropriate for stationary data, i.e. data
# without trends or seasonality. In particular, the EWMA function resists trends
# away from the current mean that it's already "seen". So, if you have a noisy
# hat function that goes from 0 to 1, and then back to 0, then the EWMA function
# will return low values on the up-hill side, and high values on the down-hill side.
# One way to circumvent this is to smooth the signal in both directions, marching
# forward, and then marching backward, and then average the two.
# TEST: example code to test "correcting" the EWMA function
def test_corrected_ewma():
    ewma = pd.stats.moments.ewma
    # make a hat function, and add noise
    x = np.linspace(0, 1, 100)
    x = np.hstack((x, x[::-1]))
    x += np.random.normal(loc=0, scale=0.1, size=200)
    plt.plot(x, alpha=0.4, label='Raw')
    # take EWMA in both directions with a smaller span term
    fwd = ewma(x, span=15)  # take EWMA in fwd direction
    bwd = ewma(x[::-1], span=15)  # take EWMA in bwd direction
    c = np.vstack((fwd, bwd[::-1]))  # lump fwd and bwd together
    c = np.mean(c, axis=0)  # average
    # regular EWMA, with bias against trend
    plt.plot(ewma(x, span=20), 'b', label='EWMA, span=20')
    # "corrected" (?) EWMA
    plt.plot(c, 'r', label='Reversed-Recombined')
    plt.legend(loc=8)
    plt.savefig(join(misc_folder, 'ewma_correction.png'), fmt='png', dpi=100)
    return

# TEST: example code to test the Holt-Winters function
def test_holt_winters():
    # make a hat function, and add noise
    x = np.linspace(0, 1, 100)
    x = np.hstack((x, x[::-1]))
    x += np.random.normal(loc=0, scale=0.1, size=200) + 3.0
    plt.plot(x, alpha=0.4, label='Raw')
    # holt winters second order ewma
    plt.plot(holt_winters_second_order_ewma(x, 10, 0.3), 'b', label='Holt-Winters')
    plt.title('Holt-Winters')
    plt.legend(loc=8)
    plt.savefig(join(misc_folder, 'holt_winters.png'), fmt='png', dpi=100)
    return



#---------------------------------------------------------------------------------------------------
def dpath(filename):
    #data_path = r"Z:\Dropbox\alvin\data\DF_DATA"
    data_path = df_folder
    return os.path.join(data_path, filename)

def cout(text):
    if not isinstance(text, str): text = str(text)
    print(text)
    if fcout is not None:
        fcout.write(text + '\n')     # also output to file
    return

#---------------------------------------------------------------------------------------------------
xcorr = lambda x,y : irfft(rfft(x) * rfft(y[::-1]))                         # cross-correlation of two numpy arrays
corr = lambda x,y : np.corrcoef(x, y)[0, 1]                                 # correlation for two numpy arrays
ccf = lambda x,y : sm_tools.ccf(np.array(x), np.array(y), unbiased=True)    # cross-correlation using statsmodels
returns = lambda x : np.diff(np.log(x))                                     # convert from prices to returns
prices_ = lambda x,price0 : np.exp(np.cumsum(x)) * price0                   # convert returns back to prices (price0 is initial price in original time series)
prices = lambda x,price0 : np.insert(prices_(x,price0), 0, price0, axis=0)  # same as above but includes initial price in array

########################################################################################################################

"""
# TEST: Decomposition of time series
x = get_sample_series_streamflow()
residual,seasonal,trend = decompose_time_series(x)
print trend['1950':'1951']

# TEST: Trend identification of time series
trend,h,p,z = mk_test(x['1969':'1972'], 0.05)
trend,h,p,z = mk_test(x['1976':'1981'], 0.05)
"""

s = get_series_kraken('BCHUSD')
s.name = 'Price'
s_diff = s.diff(periods=1)
s_diff.name = 'diff'
s_diff = s_diff.fillna(method='pad')
s_diff.dropna(inplace=True)

rsma = sma1(s_diff)
rema = ema1(s_diff)
rema2 = ema2(s_diff)
s_sma = pd.Series(rsma.values, index=s_diff.index, name='SMA')
s_ema = pd.Series(rema.values, index=s_diff.index, name='EMA')
s_ema2 = pd.Series(rema2.values, index=s_diff.index, name='EMA2')
df = pd.concat([s,s_diff,s_sma,s_ema,s_ema2], axis=1)

#s.rolling(40, min_periods=30)
df['trend'] = s.rolling(40, min_periods=30).apply(lambda x: get_trend(x))
#trend,h,p,z = mk_test(s, 0.05, display=False)

df['MaxMA'] = df['EMA'].rolling(400, min_periods=30).std()
df['MinMA'] = -df['EMA'].rolling(400, min_periods=30).std()

df.dropna(inplace=True)

#df['MinMA'] = df['EMA'].rolling(400, min_periods=1).min()
#df['MaxMA'] = df['EMA'].rolling(400, min_periods=1).max()


autocorrelation_plot(df['EMA'])

"""
fig = plt.figure(figsize=(8, 6))
# time series plot
df['MinMA'].plot(color='crimson', alpha=0.5)    # color='crimson');
df['MaxMA'].plot(color='crimson', alpha=0.5)    #color='darkslateblue');
df['EMA'].plot(color='darkslateblue')
pd.Series(0, index=s_diff.index).plot(color='black', alpha=0.5)
#plt.legend(loc='best')
plt.title("my chart", fontsize=24)
"""

plt.tight_layout();
plot_filename = "my_chart"
filename = join(misc_folder, '{}.png'.format(plot_filename))
plt.savefig(filename)
plt.show()  #block=False)



# fit model
model = ARIMA(s_diff, order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())


sys.exit()

Y = [random.random() for i in range(10)]
X = range(10)

print "The input Y, X vectors"
for i in range(len(X)):
    print i, Y[i], X[i]

Y, Z = koyckZmatrix(Y, X)
print "The input Koyck Y and Z matrix."
matprint(Z)


sys.exit()


"""
X = np.random.randn(1000, 2)
Y = (np.arange(4000) * np.random.randn(4000)).reshape((1000, 4))
results = multi_dim_granger(X, Y)
print results.coefs
results.test_causality('realgdp', ['realinv', 'realcons'], kind='f')
# H_0: ['realinv', 'realcons'] do not Granger-cause realgdp
# Conclusion: reject H_0 at 5.00% significance level
"""

#x = np.array([0,0,1,1,0,0,1,1])
#y = np.array([1,1,0,0,1,1,0,0])
#print xcorr(x, y)
#a = np.array([1.0, 2.0, 3.0, 2.0])
#b = np.array([789.0, 786.0, 788.0, 785.0])
#print np.corrcoef(a, b)
#print corr(a, b)
#print ccf(a, b)

#X = np.random.randn(1000, 2)
#Y = (np.arange(4000) * np.random.randn(4000)).reshape((1000, 4))
#ts = np.hstack((X, Y))
#results = multi_dim_granger(X, Y)
#print results.summary()


# BTC vs ETH (coinmetrics, daily)
df_btc = pd.read_csv(dpath("coinmetrics.btc.daily.DF.csv"), index_col='DateTime')
df_eth = pd.read_csv(dpath("coinmetrics.eth.daily.DF.csv"), index_col='DateTime')
dfx = pd.merge(df_btc, df_eth, left_index=True, right_index=True, suffixes=('_btc','_eth'))
dfx = dfx[['PriceUSD_btc','PriceUSD_eth']]

x = dfx.iloc[:,0].values    # values (prices) of first time series
y = dfx.iloc[:,1].values    # values (prices) of second time series

res = df_granger(dfx, maxlags=40)
print res.summary()

res,forecast,df,lag = lead_lag(dfx, maxlags=40, preprocess=1, display=True)
df[['Forecast_x','Forecast_y']] = res.fittedvalues
sys.exit()

res, forecast, df, lag = lead_lag(dfx, maxlags=10, preprocess=2, display=True)
#df[['Forecast_btc','Forecast_eth']] = res.fittedvalues  
#plt1 = df[['PriceUSD_btc', 'Forecast_btc']].plot(figsize=(16, 12))
#plt2 = df[['PriceUSD_eth', 'Forecast_eth']].plot(figsize=(16, 12))


print "Done."
#input("Press [enter] to continue.")
sys.exit()

model = statsmodels.tsa.arima_model.ARIMA(df['PriceUSD_btc'].iloc[1:], order=(1, 0, 0))  
results = model.fit(disp=-1)  
df['Forecast'] = results.fittedvalues  
df[['PriceUSD_btc', 'Forecast']].plot(figsize=(16, 12))



#c = np.array([88.23, 88.44, 88.55, 88.77, 88.99])
#print c
#a2 = returns(c)
#print a2
#a3 = prices(a2, c[0])
#print a3
#df_ret = returns_df(dfx)
#print df_ret
#df_prc = prices_df(df_ret)
#print df_prc
