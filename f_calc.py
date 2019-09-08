import math
import pandas as pd

# Return the calculated quartile value
# Given the mean of the values
# Given the standard deviation of the values
# Given a quartile identifier i (-4, -3, -2, -1, 0, +1, +2, +3, +4) and mean and standard deviation
# Given the decimal places to round (default=4)
def Calc_Quartile(mean, std, i, round_places=4, ticksize=0.0):
    qvalue = (std / 100.0 * mean) / 4.0
    quartile_price = mean + i * qvalue
    if ticksize > 0.0:
        if i < 0:
            quartile_price = round_to_tick(quartile_price, ticksize, True)
        else:
            quartile_price = round_to_tick(quartile_price, ticksize)
    return round(quartile_price, round_places)

# Return ALL the quartiles (including 'unchanged' = 9 total) in both DICTIONARY and LIST form
# Given the mean of the value
# Given the standard deviation of the value
# Given the decimal places to round (default=4)
def Calc_Quartiles(prev_underlying_close, std, round_places=4, ticksize=0.0):
    Quartiles = ['d4','d3','d2','d1','unch','u1','u2','u3','u4']
    dict_Q = {}
    list_Q = []
    for xi in range(-4, 5):
        quartile = Calc_Quartile(prev_underlying_close, std, xi, round_places, ticksize)
        list_Q.append(quartile)
        dict_Q[xi] = quartile
    return (list_Q, dict_Q)

# Return the Standard Deviation (std)
# Given the previous implied volatility close (i.e. VIX for ES)
def Calc_Std(prev_implied_vol_close):
    std = float(prev_implied_vol_close) / math.sqrt(252)                    # calculate standard deviation
    return std

# Given a decimal price and a ticksize
# Return an adjusted price that is evenly divisible by ticksize
# (optional) round_up defaults to False (which means we will round DOWN to the lower tick by default)
def round_to_tick(price, ticksize, round_up=False):
    xf = price / float(ticksize)
    xi = int(price / float(ticksize))
    if xi == xf:
        return price
    elif round_up == True:
        return (xi + 1) * ticksize
    else:
        return (xi - 1) * ticksize


#-------------------------------------------------------------------------------
