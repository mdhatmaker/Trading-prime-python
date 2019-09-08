import pandas as pd
import numpy as np
from os.path import join
from pandas.tools.plotting import autocorrelation_plot
from matplotlib import pyplot
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
from plotly.graph_objs import Scatter, Layout
from statsmodels.tsa.arima_model import ARIMA, ARMA
from statsmodels.tsa.stattools import acf, pacf, adfuller
from sklearn.metrics import mean_squared_error
import datetime
import sys

# -----------------------------------------------------------------------------------------------------------------------
from f_folders import *
from f_date import *
from f_plot import *
from f_stats import *
from f_dataframe import *
from f_file import *

pd.set_option('display.width', 160)
# -----------------------------------------------------------------------------------------------------------------------

#path = os.path.dirname(os.path.realpath(__file__))
#path = "/Users/michael/Dropbox/ALVIN/data_sd_analysis"              # macbook
#path = "C:\\Users\\Michael\\Dropbox\\ALVIN\\data_sd_analysis"       # lenovo laptop
#path = "B:\\Users\\mhatmaker\\Dropbox\\ALVIN\\data_sd_analysis"     # apartment desktop
path = join(misc_folder, "sd_analysis")                             # path to data files


def get_datetime_from_d_t(d_str, t_str):
    vd = d_str.split('/')
    vt = t_str.split(':')
    return datetime.datetime(int(vd[2]), int(vd[0]), int(vd[1]), int(vt[0]), int(vt[1]), int(vt[2]))

def read_dataframe_csv(name):
    filename = name + ".csv"
    print "Reading CSV data file:", filename
    f = open(join(path, filename), 'r')
    line = f.readline()[:-1]
    col_names = line.split(',')
    f.close()
    col_names[0] = 'ID'
    df = pd.read_csv(join(path, filename), index_col='ID', names=col_names)
    return df

def read_dataframe_bars_csv(name, minutes):
    filename = name + " " + str(minutes) + " Minutes.csv"
    print "Reading CSV data file:", filename
    pathname = join(path, filename)
    df=pd.read_csv(pathname,
                   #header=None,
                   skiprows=1,
                   index_col="ID",
                   names=["ID", "DateTime","Open","High","Low","Close","Volume"],
                   dtype={"ID":"int", "DateTime":"str", "Open":"float", "High":"float", "Low":"float", "Close":"float", "Volume":"int"},
                   parse_dates = ['DateTime'],
                   sep=","
                )
    df.info()
    print
    return df

def read_dataframe_tick_csv(name):
    filename = name + " 1 Tick Bar.csv"
    print "Reading CSV data file:", filename, "(this may take a few minutes)"
    pathname = join(path, filename)
    df=pd.read_csv(pathname,
                   #header=None,
                   skiprows=1,
                   index_col="ID",
                   names=["ID", "DateTime","Price","Volume"],
                   dtype={"ID":"int", "DateTime":"str", "Price":"float", "Volume":"int"},
                   parse_dates = ['DateTime'],
                   sep=","
                )
    df.info()
    print
    return df

def get_min_date_str(df):
    x = str(df['DateTime'].min())
    return x[:10]

def get_max_date_str(df):
    x = str(df['DateTime'].max())
    return x[:10]

def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')

def get_plot_lines(df, color1):
    trace0 = go.Scatter(
        x=df.Month,
        y=df['Sales'],
        name='Sales',
        line = dict(color=(color1), width=1)
        )
#    trace1 = go.Scatter(
#        x=df.Month,
#        y=df['discount'],
#        name='discount',
#        line = dict(color=(blue), width=1)
#        )
    return [trace0]     #[trace0, trace1]

def show_plot(title):
    fig = pyplot.gcf()
    fig.canvas.set_window_title(title)
    pyplot.show()
    return

def plot_series(df):
    df.plot()
    show_plot('data series')
    return

def plot_autocorrelation(df):
    autocorrelation_plot(df)
    show_plot('autocorrelation')
    return

def plot_residual_errors(fit):
    residuals = pd.DataFrame(fit.resid)
    residuals.plot()
    show_plot('residual errors')
    residuals.plot(kind='kde')
    show_plot('error distribution')
    print
    print "residual errors description:"
    print(residuals.describe())
    return

# arima_params = (p,d,q)
def calc_arima_train_test(X, arima_params):
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_params)
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        #print('predicted=%f, expected=%f' % (yhat, obs))
    error = mean_squared_error(test, predictions)
    #print('Test MSE: %.3f' % error)
    return (test, predictions, error)

# arima_params = (p,d,q)
def calc_arma(values, arima_params):
    (p,d,q) = arima_params
    model = ARMA(values, order=(p,q));
    model_fit = model.fit(full_output=False, disp=0);
    yhat = model_fit.predict();
    print yhat
    #output = model_fit.forecast()
    #yhat = output[0]
    return yhat

# arima_params = (p,d,q)
def calc_arima(values, arima_params):
    model = ARIMA(values, order=arima_params)
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    #error = mean_squared_error(test, predictions)
    #print('Test MSE: %.3f' % error)
    return yhat

def test_stationarity(timeseries):
    # Acf/Pacf
    # Box-Ljung
    # Dickey-Fuller
    # KPSS
    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput
    return

# return optimal (p,d,q) arima parameters
def get_optimal_alvin(series, max_lag=5):
    significant_corr = 0.33
    correlations = acf(series)
    optimal_lag = None
    for lag in range(1, max_lag+1):
        if abs(correlations[lag]) > significant_corr:
            #print lag, correlations[lag]
            optimal_lag = lag
            break
    return (optimal_lag, 0, 0)

def plot_rolling_stats(timeseries, rolmean, rolstd):
    #Plot rolling statistics:
    timeseries.plot(color='blue', label='Original')
    rolmean.plot(color='black', label='Rolling Mean')
    rolstd.plot(color='red', label='Rolling Std')
    show_plot('rolling stats')
    #orig = pyplot.plot(timeseries, color='blue',label='Original')
    #mean = pyplot.plot(rolmean, color='red', label='Rolling Mean')
    #std = pyplot.plot(rolstd, color='black', label = 'Rolling Std')
    #pyplot.legend(loc='best')
    #pyplot.title('Rolling Mean & Standard Deviation')
    #pyplot.show(block=False)
    return

def get_dataframe(bar_minutes, symbol, lookback=40):
    #bar_minutes = 15            # period of each bar
    #symbol = "QCL#C"            # symbol (will be used to determine from which CSV file to read data)
    #lookback_count = 40         # number of values to use in our mean and SD calculations (lookback)

    # Read the dataframes from their respective CSV files
    df = read_dataframe_bars_csv(symbol, bar_minutes)

    # Add a column that contains the price change (close - open)
    #df['diff'] = df['Close'] - df['Open']
    df['diff'] = df.Close - df.Close.shift(1)

    timeseries = df['diff'].dropna(inplace=False)
       
    test_stationarity(timeseries)

    #Determing rolling statistics
    #rolmean = timeseries.rolling(window=lookback,center=False).mean()
    #rolstd = timeseries.rolling(window=lookback,center=False).std()
    #plot_rolling_stats(timeseries, rolmean, rolstd)
    
    # Add two more columns (for mean and std)
    df['mean'] = np.nan
    df['std'] = np.nan
    df['lag'] = np.nan
    df['forecast'] = np.nan
    df['error'] = np.nan

    dec_places = 2              # round calculated values to this decimal place
    
    corr_error_count = 0
    corr_errors = []

    # For each row (price bar) calculate the mean and std (standard deviation) of the previous [count] bars
    length = len(df.index)
    for i in range(lookback,length):
        series = df['diff'].ix[i-lookback:i-1]
        mean = round(series.mean(), dec_places)
        std = round(series.std(), dec_places)
        #df.set_value(i, 'mean', np.std(values))    # shows n vs (n-1) difference in numpy vs pandas std calculation
        df.set_value(i, 'mean', mean)
        df.set_value(i, 'std', std)

        (p, d, q) = get_optimal_alvin(series)
        
        if p != None:
            #print i, optimal_lag
            df.set_value(i, 'lag', p)
            #yhat = calc_arma(series.values, (p, d, q))
            yhat = calc_arima(series.values, (p, d, q))
            df.set_value(i, 'forecast', round(yhat, dec_places))
            df.set_value(i, 'error', round(yhat - df['diff'].ix[i], dec_places))
        else:
            #print "ERROR: [" + str(i) + "]  No optimal lag found in autocorrelations."
            corr_error_count += 1
            corr_errors.append(i)

    #print "Processed {0} lines ({1} autocorrelation errors)".format(length-lookback, corr_error_count)
    
    return df

########################################################################################################################


# TODO: create PROJECTS data folder and put "hogo", "copper", "vix_es" and any other project-related data folders here


#df = pd.read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
df = get_dataframe(30, "QCL#C")

print
print("A few data points...")
print(df.ix[40:61])
print

nan_rows = df[df.forecast.isnull()]

(nrows, ncols) = df.shape

print "total rows: {0} rows".format(nrows)
print "missing data: {0} rows".format(len(nan_rows))


"""
plot_series(df)
plot_autocorrelation(df)

# fit model
model = ARIMA(df, order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())

# plot residual errors
plot_residual_errors(model_fit)
"""


"""
# plot
pyplot.plot(actual)
pyplot.plot(predictions, color='red')
show_plot("lag {0}    (red=predicted  blue=actual)".format(lag))
"""


"""
data_acf_1 =  acf(series)[1:32]
data_acf_2 = [series.autocorr(i) for i in range(1,32)]
"""

"""
blue = 'rgb(0, 0, 255)'

chart_data = []
df = df.sort_values('Month')
chart_data.extend(get_plot_lines(df, blue))

layout = go.Layout(showlegend=False)
fig = go.Figure(data=chart_data, layout=layout)
plotly.offline.plot(fig, filename='shampoo.html')
"""
