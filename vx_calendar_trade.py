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
from f_chart import *
from f_plot import *

project_folder = join(data_folder, "vix_es")

#-----------------------------------------------------------------------------------------------------------------------




########################################################################################################################

print("Testing ideas for VX calendar trade (front-month to next-month)")
print()

y1 = 2017
y2 = 2017

data_interval = "s1800"

# To get VX calendar ETS data:
# (1) retrieve latest VX futures data
# (2) create continuous front-month from this futures data
# (3) retrieve latest VX calendar ETS data (using VX continuous to infer one-month-out and two-months-out from front-month symbol)
# (4) create continuous calendar ETS
df_ = create_historical_futures_df("@VX", y1, y2, interval=data_interval, days_back=180, beginFilterTime='093000', endFilterTime='160000')
df_ = create_continuous_df("@VX", get_roll_date_VX, interval=data_interval)
df_ = create_historical_calendar_futures_df("@VX", 0, 1, y1, y2, interval=data_interval, days_back=180, beginFilterTime='093000', endFilterTime='160000')
df_ = create_continuous_calendar_ETS_df("@VX", 0, 1, interval=data_interval)
#df_ = create_contango_df("@VX", interval=data_interval)

#df = read_dataframe("@VX_contango.{0}.DF.csv".format(str_interval(data_interval)))
df = read_dataframe("@VX_continuous_calendar-m0-m1.{0}.DF.csv".format(str_interval(data_interval)))

df = alvin_arima(df)
