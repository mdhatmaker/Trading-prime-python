import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import acf, ccf
import statsmodels.tsa.arima_process as ap
from scipy.signal import lfilter
from os.path import join
import sys

tsa = sm.tsa    # as shorthand


# Compare results of the autocorrelation (acf) and correlation (ccf) functions
def compare_acf_ccf():
    #this is the data series that I want to analyze
    A = np.array([np.absolute(x) for x in np.arange(-1,1.1,0.1)])
    #This is the autocorrelation using statsmodels's autocorrelation function
    plt.plot(acf(A, fft=True), "r-")
    #This the autocorrelation using statsmodels's correlation function
    # MUST set unbiased=False to get same result as acf function
    plt.plot(ccf(A, A, unbiased=False), "go")
    plt.plot(ccf(A, A), "bx")
    plt.show()
    return

# Examples from statsmodels.org home page
def minimal_examples():
    # Use R-style formulas together with pandas dataframes to fit models.
    # Example using ordinary least squares:
    # Load data
    dat = sm.datasets.get_rdataset("Guerry", "HistData").data
    # Fit regression model 9using the natural log of the regressors)
    results = smf.ols('Lottery ~ Literacy + np.log(Pop1831)', data=dat).fit()
    # Inspect the results
    print "{0}\n".format(results.summary())

    # You can also use numpy arrays instead of formulas:
    # Generate artificial data (2 regressors + constant)
    nobs = 100
    X = np.random.random((nobs, 2))
    X = sm.add_constant(X)
    beta = [1, .1, .5]
    e = np.random.random(nobs)
    y = np.dot(X, beta) + e
    # Fit regression model
    results = sm.OLS(y, X).fit()
    # Inspect the results
    print results.summary()
    print
    # Have a look at dir(results) to see available results. Attributes are described
    # in results.__doc__ and results methods have their own docstrings.
    return

def hline(X, yvalue=0, alpha=0.75):
    xmin = X.min()
    xmax = X.max()
    plt.hlines(yvalue, xmin, xmax, colors='gray', linestyles='dashed', alpha=alpha, label='')
    return

# Plot original data along with this data processed through a filter
def plot_filter(X, y, y_c, y_t, color='b', title="", figure=1, alpha=0.5):
    plt.figure(figure)
    fig, ax1 = plt.subplots()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plot_title = title + " Filter"
    plt.title(plot_title.strip())
    #plt.plot(X, y, color + "-", alpha=alpha)
    ax1.plot(X, y, color + "-", alpha=alpha)
    ax1.set_xlabel('date')
    # Make the y-axis label, ticks and tick labels match the line color
    ax1.set_ylabel('', color='k')
    ax1.tick_params('y', colors='k')
    if y_t is not None:
        #plt.plot(X, y_t, "r:")
        ax1.plot(X, y_t, "r:")
    ax2 = ax1.twinx()
    #hline(X)  
    #plt.plot(X, y_c, "g:")
    ax2.plot(X, y_c, "g:")
    ax2.set_ylabel('', color='g')
    ax2.tick_params('y', colors='g')
    #plt.show()
    return

def plot_all_filters(X, y, hp, bk, cf, subtitle=""):
    plot_filter(X, y, hp[0], hp[1], title=subtitle+"Hodrick-Prescott", color="b", figure=1)
    plot_filter(X, y, bk[0], None, title=subtitle+"Baxter-King", color="b", figure=2)
    plot_filter(X, y, cf[0], cf[1], title=subtitle+"Christiano-Fitzgerald", color="b", figure=3)
    plt.show()
    return

# Return an array of length 'n' that contains np.nan values
def nans(n):
    return np.empty(n) * np.nan

# Prepend a value (or multiple values in an array) to an existing np.array
def np_prepend(array, x):
    return np.insert(array, 0, x, axis=0)

# Append a value (or multiple values in an array) to the end of an existing np.array
def np_append(array, x):
    return np.insert(array, 0, x, axis=0)

# Apply the bkfilter, then adjust results to match x-axis of source data
# (by prepending some np.nan values)
def bkfilter(y):
    bky = tsa.filters.bkfilter(y)
    half_diff = (len(y) - len(bky)) / 2
    bky = np_prepend(bky, nans(half_diff))
    bky = np_append(bky, nans(half_diff))
    return bky

# Run 3 different filters on the given values
# Return as a tuple ((hp_c,hp_t), (bk_c,None), (cf_c,cf_t))
def filter_values(y):
    hp_y_c, hp_y_t = tsa.filters.hpfilter(y)    # Hodrick-Prescott filter
    bk_y_c = bkfilter(y)                        # Baxter-King filter
    cf_y_c, cf_y_t = tsa.filters.cffilter(y)    # Christiano-Fitzgerald filter
    return ((hp_y_c, hp_y_t), (bk_y_c, None), (cf_y_c, cf_y_t))

# Given a dataframe of historical data and a 'column_name' of price data to filter
# Return the (X,y) of the selected historical data along with (hp, bd, cf) filtered data
# (optional) tail defaults to use the whole dataframe, but can be set to integer row count
def df_get_filter_values(df, column_name='Close', tail=None):
    if tail is not None: df = df.tail(tail)
    X = np.array(df['DateTime'])[1:]
    y = np.array(df[column_name])[1:]
    hp, bk, cf = filter_values(y)
    return X, y, hp, bk, cf

# Given an X and y np.array (npX and npy)
# Return the (X,y) of the selected historical data along with (hp, bd, cf) filtered data
# (optional) tail defaults to use the whole np.array, but it can be set to integer row count
def np_get_filter_values(npX, npy, tail=None):
    if tail is not None:
        npX = npX.tail(tail)
        npy = npy.tail(tail)
    hp, bk, cf = filter_values(npy)
    return npX, npy, hp, bk, cf

# Print formtted Dickey-Fuller test output
def print_adf(adf):
    print "Dickey-Fuller: {0}".format(adf)
    print " adf-statistic={0:.4f}  p-value={1:.4f}\n".format(adf[:2][0], adf[:2][1])
    return

#################################################################################

# Comparison of acf and ccf functions (and why default results are different)
#compare_acf_ccf()

# Examples from the Statsmodels home page
minimal_examples()

#--------------------------------------------------------------------------------
mdata = sm.datasets.macrodata.load().data
endog = np.log(mdata['m1'])
exog = np.column_stack([np.log(mdata['realgdp']), np.log(mdata['cpi'])])
exog = sm.add_constant(exog, prepend=True)
res1 = sm.OLS(endog, exog).fit()

print "{0}\n".format(res1.summary())
# Very low Durbin-Watson statistic indicates there is a strong autocorrelation in the
# residuals. Plotting the residuals would show a similar strong autocorrelation.
# As a more formal test we can calculate the autocorrelation, the Ljung-Box Q-statistic
# for the test of zero autocorrelation and the associated p-values:

#acf, ci, Q, pvalue = tsa.acf(res1.resid, nlags=4, qstat=True, unbiased=True)
acf, ci, pvalue = tsa.acf(res1.resid, nlags=4, qstat=True, unbiased=True)
print "acf: {0}".format(acf)
print "pvalue: {0}".format(pvalue)

# To see how many autoregressive coefficients might be relevant, we can also look
# at the partial autocorrelation coefficients:
print "pacf: {0}".format(tsa.pacf(res1.resid, nlags=4))

# Similar regression diagnostics, for example for heteroscedas-ticity, are available
# in statsmodels.stats.diagnostic.

# The strong autocorrelation indicates that either our model is misspecified or there is
# strong autocorrelation in the errors. If we assume that the second is correct, then we
# can estimate the model with GLSAR.
# As an example, let us assume we consider four lags in the autoregressive error.
mod2 = sm.GLSAR(endog, exog, rho=4)
res2 = mod2.iterative_fit()
# iterativve_fit alternates between estimating the autoregressive process of the error
# term using tsayule_walker and feasible sm.GLS.
print "OLS params: {0}".format(res1.params)
print "GLSAR params: {0}".format(res2.params)
print "GLSAR rho: {0}\n".format(mod2.rho)
# Looking at the results shows two things:
# 1) the parameter estimates are very different between OLS and GLS
# 2) the autocorrelation in the residual is close to a random walk
# This indicates that the short run and long run dynamics might be very different and
# that we should consider a richer dynamic model, and that the variables might now be
# stationary and that there might be unit roots.
#--------------------------------------------------------------------------------
adf = tsa.adfuller(endog, regression="ct")
print_adf(adf)
# Testing the log of the stock of money with a null hypothesis of unit roots against
# an alternative of stationarity around a linear trend shows an adf-statistic of -1.5
# and a p-value of 0.8, so we are far away from rejecting the unit root hypothesis.

adf = tsa.adfuller(np.diff(endog), regression="c")
print_adf(adf)
# If we test the differenced series, that is the growth rate of moneystock, with a
# Null hypothesis of Random Walk with drift, then we can strongly reject the hypothesis
# that the growth rate has a unit root (p-value 0.0002)
#--------------------------------------------------------------------------------
# To choose the number of lagged terms, p and q, for ARIMA(p,d,q) processes, use the
# Box-Jenkins methodology to look at the pattern in the autocorrelation (acf) and
# partial autocorrelation (pacf) functions.
# scikits.statsmodels.tsa.arima_process contains a class that provides several properties
# of ARMA processes and a random process generator. This allows easy comparison of the
# theoretical properties of an ARMA process with their empirical counterparts.
# For exmaple, define the lag coefficients for an ARMA(2,2) process, generate a random
# process and compare the observed and theoretical pacf:
ar = np.r_[1., -0.5, -0.2]; ma = np.r_[1., 0.2, -0.2]
np.random.seed(123)
x = ap.arma_generate_sample(ar, ma, 20000, burnin=1000)
print "observed pacf: {0}".format(sm.tsa.pacf(x, 5))

theo_ap = ap.ArmaProcess(ar, ma)
print "    theo pacf: {0}\n".format(theo_ap.pacf(6))
# We can see that observed and theoretical pacf are very close in a large generated
# sample like this.
#--------------------------------------------------------------------------------
# We can use Statsmodels Autoregressive Moving Average (ARMA) time-series models to
# simulate a series:
ar_coef = [1, .75, -.25]
ma_coef = [1, -.5]
nobs = 100
y = ap.arma_generate_sample(ar_coef, ma_coef, nobs)
y += 4  # add in a constant
# Estimate an ARMA model of the series
mod = tsa.ARMA(y, (2,1))
res = mod.fit(order=(2,1), trend='c', method='css-mle', disp=-1)
print "Estimated ARMA model params: {0}\n".format(res.params)
# The estimation method 'css-mle' indicates the starting parameters from the optimization
# are to be obtained from the conditional sum of squares estimator and then the exact
# likelihood is optimized. The exact likelihood is implemented using the Kalman Filter.
#--------------------------------------------------------------------------------

"""
folder = r"C:\Users\Michael\Dropbox\ALVIN\data\DF_DATA"

filepath = join(folder, "@VX_contango.EXPANDED.daily.DF.csv")
df = pd.read_csv(filepath, parse_dates=['DateTime'])
X, y, hp, bk, cf = df_get_filter_values(df, column_name='Close_VX', tail=600)
plot_all_filters(X, y, hp, bk, cf, subtitle="VX: ")

filepath = join(folder, "@ES_continuous.daily.DF.csv")
df = pd.read_csv(filepath, parse_dates=['DateTime'])
X, y, hp, bk, cf = df_get_filter_values(df, tail=600)
plot_all_filters(X, y, hp, bk, cf, subtitle="ES: ")
"""

#--------------------------------------------------------------------------------
# Demonstrate the API and resultant filtered series for each method.
# Use series for unemployment and inflation to demonstrate
data = sm.datasets.macrodata.load()
infl = mdata['infl'][1:]
# Get 4-quarter moving average
infl = lfilter(np.ones(4)/4, 1, infl)[4:]
unemp = mdata['unemp'][1:]

"""
X, y, hp, bk, cf = np_get_filter_values(np.arange(len(infl)), infl)
plot_all_filters(X, y, hp, bk, cf, subtitle="Inflation: ")
X, y, hp, bk, cf = np_get_filter_values(np.arange(len(unemp)), unemp)
plot_all_filters(X, y, hp, bk, cf, subtitle="Unemployment: ")

sys.exit()
"""
hp_infl, bk_infl, cf_infl = filter_values(infl)
hp_unemp, bk_unemp, cf_unemp = filter_values(unemp)

plot_all_filters(np.arange(len(infl)), infl, hp_infl, bk_infl, cf_infl, subtitle="Inflation: ")
plot_all_filters(np.arange(len(unemp)), unemp, hp_unemp, bk_unemp, cf_unemp, subtitle="Unemployment: ")

sys.exit()

plot_filter(infl, hp_infl[0], hp_infl[1], title="Hodrick-Prescott", color="b", figure=1)
plot_filter(infl, bk_infl[0], None, title="Baxter-King", color="b", figure=2)
plot_filter(infl, cf_infl[0], cf_infl[1], title="Christiano-Fitzgerald", color="b", figure=3)
plt.show()

plot_filter(unemp, hp_unemp[0], hp_unemp[1], title="Hodrick-Prescott", color="c", figure=1)
plot_filter(unemp, bk_unemp[0], None, title="Baxter-King", color="c", figure=2)
plot_filter(unemp, cf_unemp[0], cf_unemp[1], title="Christiano-Fitzgerald", color="c", figure=3)
plt.show()
#--------------------------------------------------------------------------------





