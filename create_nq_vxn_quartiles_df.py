from __future__ import print_function
import sys
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import math
from datetime import time
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------------------------------------------------

from f_folders import *
from f_dataframe import *
from f_rolldate import *
from f_iqfeed import *

#-----------------------------------------------------------------------------------------------------------------------

# Create the dataframe files required for VX quartile data:
# "@NQ_futures.daily.DF.csv"
# "@NQ_continuous.daily.DF.csv"
# "@NQ_futures.minute.DF.csv"
# "VXN.XO.daily.DF.csv"
def download_historical_data_for_nq_quartiles():
    df_ = create_historical_futures_df("@NQ")
    df_ = create_historical_futures_df("@NQ", interval=INTERVAL_MINUTE, beginFilterTime='093000', endFilterTime='160000')
    df_ = create_continuous_df("@NQ", get_roll_date_NQ)
    df_ = create_historical_contract_df("VXN.XO")
    return

################################################################################

print("Calculating quartiles using volatility index to determine standard deviation")
print()

#set_default_dates_latest()

# Request the IQFeed data and create the dataframe files for the continuous NQ and the VXN cash index (default is INTERVAL_DAILY)
download_historical_data_for_nq_quartiles()
dfQ = create_quartile_df("@NQ", "VXN.XO", "NQ", "VXN", time(9,30), time(16,0), lookback_days=5)
