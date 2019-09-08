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




########################################################################################################################

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
