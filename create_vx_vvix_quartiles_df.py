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
# "@VX_futures.daily.DF.csv"
# "@VX_continuous.daily.DF.csv"
# "@VX_futures.minute.DF.csv"
# "VVIX.XO.daily.DF.csv"
def download_historical_data_for_vx_quartiles():
    df_ = create_historical_futures_df("@VX")
    df_ = create_historical_futures_df("@VX", interval=INTERVAL_MINUTE, beginFilterTime='093000', endFilterTime='160000')
    df_ = create_continuous_df("@VX", get_roll_date_VX)
    df_ = create_historical_contract_df("VVIX.XO")
    return

################################################################################

print("Calculating quartiles using volatility index to determine standard deviation")
print()

#set_default_dates_latest()

# Request the IQFeed data and create the dataframe files for the continuous VX and the VVIX cash index (default is INTERVAL_DAILY)
download_historical_data_for_vx_quartiles()
dfQ = create_quartile_df("@VX", "VVIX.XO", "VX", "VVIX", time(9,30), time(16,0), lookback_days=5)
