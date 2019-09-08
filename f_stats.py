from pandas import read_csv
from pandas import datetime
from pandas import Series
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import _arma_predict_out_of_sample     # this is the nsteps ahead predictor function
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import scipy.stats as stats
import math

# IT is recommended that warnings be ignored for this code to avoid a lot of noise
# from running the procedure. This can be done as follows:
import warnings
warnings.filterwarnings("ignore")


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    start_time = datetime.now()
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    str_elapsed = "elapsed time: {0}".format(datetime.now() - start_time)
                    print("ARIMA{0} MSE={1:.3f}     {2}".format(order,mse,str_elapsed))
                except:
                    continue
    print("Best ARIMA%s MSE=%.3f\n" % (best_cfg, best_score))
    return

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error

def arima_grid_search(series):
    # evaluate parameters
    p_values = [0, 1, 2, 3, 4, 5, 6]   # suite of lag values (p)
    d_values = range(0, 0)              # difference iterations (d)
    q_values = range(0, 3)              # residual error lag values (q)
    warnings.filterwarnings("ignore")
    evaluate_models(series.values, p_values, d_values, q_values)
    return

#-------------------------- ARIMA GRID TEST SAMPLE CODE --------------------------------------------
# The timestamps in the time series do not contain an absolute year component. We can use a custom
# date-parsing function when loading the data and baseline the year from 1900, as follows:
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

def arima_grid_search_test1():
    # load dataset
    series = read_csv("./misc/sales-of-shampoo.csv", header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

    # evaluate parameters
    p_values = [0, 1, 2, 4, 6, 8, 10]   # suite of lag values (p)
    d_values = range(0, 3)              # difference iterations (d)
    q_values = range(0, 3)              # residual error lag values (q)
    warnings.filterwarnings("ignore")
    evaluate_models(series.values, p_values, d_values, q_values)
    return
# Output of grid_search_test1() should report the best parameters of ARIMA(4, 2, 1) at
# the end of the run with a mean squared error of 4,694.873:
"""
ARIMA(0, 0, 0) MSE=52425.268
ARIMA(0, 0, 1) MSE=38145.167
ARIMA(0, 0, 2) MSE=23989.567
ARIMA(0, 1, 0) MSE=18003.173
ARIMA(0, 1, 1) MSE=9558.410
ARIMA(0, 2, 0) MSE=67339.808
ARIMA(0, 2, 1) MSE=18323.163
ARIMA(1, 0, 0) MSE=23112.958
ARIMA(1, 1, 0) MSE=7121.373
ARIMA(1, 1, 1) MSE=7003.683
ARIMA(1, 2, 0) MSE=18607.980
ARIMA(2, 1, 0) MSE=5689.932
ARIMA(2, 1, 1) MSE=7759.707
ARIMA(2, 2, 0) MSE=9860.948
ARIMA(4, 1, 0) MSE=6649.594
ARIMA(4, 1, 1) MSE=6796.279
ARIMA(4, 2, 0) MSE=7596.332
ARIMA(4, 2, 1) MSE=4694.873
ARIMA(6, 1, 0) MSE=6810.080
ARIMA(6, 2, 0) MSE=6261.107
ARIMA(8, 0, 0) MSE=7256.028
ARIMA(8, 1, 0) MSE=6579.403
Best ARIMA(4, 2, 1) MSE=4694.873
"""


def arima_grid_search_test2():
    # load dataset
    series = Series.from_csv("./misc/daily-total-female-births.csv", header=0)

    # evaluate parameters
    p_values = [0, 1, 2, 4, 6, 8, 10]
    d_values = range(0, 3)
    q_values = range(0, 3)
    warnings.filterwarnings("ignore")
    evaluate_models(series.values, p_values, d_values, q_values)
    return
# For grid_search_test2(), the best mean parameters are reported as ARIMA(6, 1, 0)
# with a mean squared error of 53.187:
"""
ARIMA(0, 0, 0) MSE=67.063
ARIMA(0, 0, 1) MSE=62.165
ARIMA(0, 0, 2) MSE=60.386
ARIMA(0, 1, 0) MSE=84.038
ARIMA(0, 1, 1) MSE=56.653
ARIMA(0, 1, 2) MSE=55.272
ARIMA(0, 2, 0) MSE=246.414
ARIMA(0, 2, 1) MSE=84.659
ARIMA(1, 0, 0) MSE=60.876
ARIMA(1, 1, 0) MSE=65.928
ARIMA(1, 1, 1) MSE=55.129
ARIMA(1, 1, 2) MSE=55.197
ARIMA(1, 2, 0) MSE=143.755
ARIMA(2, 0, 0) MSE=59.251
ARIMA(2, 1, 0) MSE=59.487
ARIMA(2, 1, 1) MSE=55.013
ARIMA(2, 2, 0) MSE=107.600
ARIMA(4, 0, 0) MSE=59.189
ARIMA(4, 1, 0) MSE=57.428
ARIMA(4, 1, 1) MSE=55.862
ARIMA(4, 2, 0) MSE=80.207
ARIMA(6, 0, 0) MSE=58.773
ARIMA(6, 1, 0) MSE=53.187
ARIMA(6, 1, 1) MSE=57.055
ARIMA(6, 2, 0) MSE=69.753
ARIMA(8, 0, 0) MSE=56.984
ARIMA(8, 1, 0) MSE=57.290
ARIMA(8, 2, 0) MSE=66.034
ARIMA(8, 2, 1) MSE=57.884
ARIMA(10, 0, 0) MSE=57.470
ARIMA(10, 1, 0) MSE=57.359
ARIMA(10, 2, 0) MSE=65.503
ARIMA(10, 2, 1) MSE=57.878
ARIMA(10, 2, 2) MSE=58.309
Best ARIMA(6, 1, 0) MSE=53.187
"""

#-------------------------- MISC LINEAR REGRESSION CODE --------------------------------------------
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



#-------------------------- ALVIN ARIMA CODE -------------------------------------------------------
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

def alvin_arima(df0, lookback=40, avg_close_days=1, correlation_sig=0.33):
    df = df0.copy()
    # This could be set to average of last 3 close values to smooth (avg_close_days=3 -> avgclose=(close+close[1]+close[2])/3)
    if avg_close_days == 1:
        df['avg_close'] = df['Close']
    else:
        df['avg_close'] = (df['Close'] + sum([df['Close'].shift(x) for x in range(1,avg_close_days)])) / avg_close_days

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
        opti = optimal_coefficient(ac, correlation_sig)
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

    df.drop(['diff','diff1','diff2','diff3','diff4','diff5','diff6'], axis=1, inplace=True)
    forecast = df.iloc[-1]['ARIMA']
    df.dropna(inplace=True)

    return df, forecast

#-------------------------- OTHER ARIMA CODE -------------------------------------------------------
def predictARMA(data, AR=4, MA=4):
    arma = sm.tsa.ARMA(data, order=(AR,MA))
    results = arma.fit( full_output=False, disp=0)
    # Where data is a one-dimensional array. To get in-sample predictions:
    prediction = results.predict()
    return prediction

def predict_out_of_sample_ARMA(data, AR=3, MA=0):
    res = sm.tsa.ARMA(data, order=(AR,MA)).fit(trend="nc")
    # get what you need for predicting one-step ahead
    params = res.params
    residuals = res.resid
    p = res.k_ar
    q = res.k_ma
    k_exog = res.k_exog
    k_trend = res.k_trend
    steps = 1
    prediction = _arma_predict_out_of_sample(params, steps, residuals, p, q, k_trend, k_exog, endog=y, exog=None, start=len(data))
    return prediction



# These may have been some attempts to use other statistical methods to calculate the Alvin ARIMA formula
"""
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
"""


########################################################################################################################

# If we RUN this python script, run the following (good for testing)
if __name__ == "__main__":
    arima_grid_search_test1()
    arima_grid_search_test2()


