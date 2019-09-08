from math import *
from scipy.stats import norm

def BlackScholes(CallPutFlag, S, K, T, r, d, v):
    d1 = (log(float(S)/K)+((r-d)+v*v/2.0)*T)/(v*sqrt(T))
    d2 = d1-v*sqrt(T)
    if CallPutFlag.lower() == 'c':
        return S*exp(-d*T)*norm.cdf(d1)-K*exp(-r*T)*norm.cdf(d2)
    else:
        return K*exp(-r*T)*norm.cdf(-d2)-S*exp(-d*T)*norm.cdf(-d1)
    
# Cumulative distribution function for the standard normal distribution (phi)
def normsdist(x):   
    return (1.0 + erf(x / sqrt(2.0))) / 2.0

def prob_density(d1):
    return (1.0 / sqrt(2.0 * pi)) * exp(-d1**2/2.0)

# Calculate a standard deviation given a price, time-to-expiry and implied vol
# S = spot price
# T = time to expiry (years)
# v = implied volatility
def standard_dev(S, T, v):
    return S * v * sqrt(T)

# Create options T (time to expiry in years) from days to expiry
def fT(daysToExpiry):
    return daysToExpiry/365.0

# S = spot price
# T = time to expiry (years)
# C = call price
def implied_vol_approx(S, T, C):
    return sqrt(2 * pi / T) * (C / S)
    
# S = spot price
# X = strike price
# T = time to expiry (years)
# r = risk-free rate
# d = dividend yield
# v = implied volatility
def black_scholes_call(S, X, T, r, d, v):
    d1 = (log(S/X) + (r - d + v**2 / 2.0) * T) / v / sqrt(T)
    d2 = d1 - v * sqrt(T)
    ert = exp(-r * T)
    edt = exp(-d * T)  # or eqt
    return edt * S * normsdist(d1) - X * ert * normsdist(d2)

def black_scholes_put(S, X, T, r, d, v):
    d1 = (log(S/X) + (r - d + v**2 / 2.0) * T) / v / sqrt(T)
    d2 = d1 - v * sqrt(T)
    ert = exp(-r * T)
    edt = exp(-d * T)  # or eqt
    return ert * X * normsdist(-d2) - S * edt * normsdist(-d1)

def black_scholes_delta(S, X, T, r, d, v):
    d1 = (log(S/X) + (r - d + v**2 / 2) * T) / v / sqrt(T)
    ert = exp(-r * T)
    edt = exp(-d * T)  # or eqt
    call_delta = edt * normsdist(d1)
    put_delta = edt * (normsdist(d1)-1)
    return (call_delta, put_delta)

def black_scholes_gamma(S, X, T, r, d, v):
    d1 = (log(S/X) + (r - d + v**2 / 2) * T) / v / sqrt(T)
    edt = exp(-d * T)  # or eqt
    gamma = edt / (S * v * sqrt(T)) * prob_density(d1)
    return gamma

def black_scholes_vega(S, X, T, r, d, v):
    d1 = (log(S/X) + (r - d + v**2 / 2) * T) / v / sqrt(T)
    edt = exp(-d * T)
    vega = 1 / 100.0 * S * edt * sqrt(T) * prob_density(d1)
    return vega

def black_scholes_theta(S, X, T, r, d, v):
    d1 = (log(S/X) + (r - d + v**2 / 2) * T) / v / sqrt(T)
    d2 = d1 - v * sqrt(T)
    ert = exp(-r * T)
    edt = exp(-d * T)
    call_theta = (1.0/T) * (-(S * v * edt / (2.0 * sqrt(T)) * prob_density(d1)) - r * X * ert * normsdist(d2) + d * S * edt * normsdist(d1))
    put_theta = (1.0/T) * (-(S * v * edt / (2.0 * sqrt(T)) * prob_density(d1)) + r * X * ert * normsdist(-d2) - d * S * edt * normsdist(-d1))
    return (call_theta, put_theta)

def black_scholes_rho(S, X, T, r, d, v):
    d1 = (log(S/X) + (r - d + v**2 / 2) * T) / v / sqrt(T)
    d2 = d1 - v * sqrt(T)
    ert = exp(-r * T)
    call_rho = 1/100.0 * X * T * ert * normsdist(d2)
    put_rho = -1/100.0 * X * T * ert * normsdist(-d2)
    return (call_rho, put_rho)

"""
# The Black Scholes Formula
# CallPutFlag - This is set to 'c' for all option, anything else for put
# S - Underlying price
# K - Strike price (X in some models)
# T - Time to maturity
# r - Riskfree interest rate
# d - Dividend yield
# v - Volatility
"""

def black_scholes(S, X, T, r, d, v, print_it=False):
    call = black_scholes_call(S, X, T, r, d, v)
    put = black_scholes_put(S, X, T, r, d, v)
    (call_delta, put_delta) = black_scholes_delta(S, X, T, r, d, v)
    gamma = black_scholes_gamma(S, X, T, r, d, v)
    (call_theta, put_theta) = black_scholes_theta(S, X, T, r, d, v)
    vega = black_scholes_vega(S, X, T, r, d, v)
    (call_rho, put_rho) = black_scholes_rho(S, X, T, r, d, v)
    if print_it:
        print_all(call, put, call_delta, put_delta, gamma, call_theta, put_theta, vega, call_rho, put_rho)
    return (call, put, call_delta, put_delta, gamma, call_theta, put_theta, vega, call_rho, put_rho)

def print_all(call, put, call_delta, put_delta, gamma, call_theta, put_theta, vega, call_rho, put_rho):
    print "call: {0:.2f}    put: {1:.2f}".format(call, put)
    print "delta_c: {0:.2f}    delta_p: {1:.2f}".format(call_delta, put_delta)
    print "gamma: {0:.2f}".format(gamma)
    print "theta_c: {0:.2f}    theta_p: {1:.2f}".format(call_theta, put_theta)
    print "vega: {0:.2f}".format(vega)
    print "rho_c: {0:.2f}    rho_p: {1:.2f}".format(call_rho, put_rho)
    print
    return

# X = S + (-C+P)
# P = X - S + C
#def black_scholes_put(S, X, T, r, d, v):
#    return X - S + black_scholes_call(S, X, T, r, d, v)

print "C: {0}".format(black_scholes_call(490, 470, 0.08, 0.033, 0.0, 0.2))
print "P: {0}".format(black_scholes_put(490, 470, 0.08, 0.033, 0.0, 0.2))
print
print implied_vol_approx(946.98, fT(8), 8.10)
print
print standard_dev(946.98, fT(8), .14482)
print standard_dev(323.62, fT(22), .316)
print
print prob_density(1.333333)

print
(c, p, cdlt, pdlt, gamma, ctht, ptht, vega, crho, prho) = black_scholes(490, 470, .08, .033, 0.0, .32094, True)
(c, p, cdlt, pdlt, gamma, ctht, ptht, vega, crho, prho) = black_scholes(490, 470, .08, .033, 0.0, .2, True)

(c, p, cdlt, pdlt, gamma, ctht, ptht, vega, crho, prho) = black_scholes(46.30, 50.00, (135/365.0), .0475, 0.0634, .348, True)


print "C: {0:.2f}".format(BlackScholes('c', 490, 470, 0.08, 0.033, 0.0, 0.32094))
print "P: {0:.2f}".format(BlackScholes('p', 490, 470, 0.08, 0.033, 0.0, 0.32094))

