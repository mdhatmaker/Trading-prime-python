from __future__ import print_function
from os.path import join, basename, splitext
import pandas as pd
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


# <functions go here>




########################################################################################################################


# --------------------------------------------------------------------------------------------------
# https://<informational links here>

# <main code goes here>

"""
futures:
@ENY#       (mini)
@NIY#
@JE#        (mini)    
@JY#

SEY#    EUROYEN TIBOR (continuous)
SEYZ17  EUROYEN TIBOR (December 2017)

options:
NNJ18C20000 (example of J18 Call with strike price of 20000)
NNJ18P20000 (example of J18 Put with strike price of 20000)
"""

dt1 = datetime(2017, 11, 1)
dt2 = datetime.now()

#symbol = "NNJ18C20000"
callput = 'P'
expiry = 'Z17'
li = []
for i in range(20):
    K = 22000 + i * 125
    symbol = "NN{0}{1}{2}".format(expiry, callput, K)
    df = get_historical_contract(symbol, dateStart=dt1, dateEnd=dt2)
    if df_last(df) is not None:
        li.append([symbol, df_last(df)['Close']])

for o in li:
    print("{0}\t{1}".format(o[0], o[1]))

STOP()
