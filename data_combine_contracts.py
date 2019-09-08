import sys
import os
from os import listdir
from os.path import isfile, join
import glob, os
import pandas as pd
import numpy as np
import math
import re

#-----------------------------------------------------------------------------------------------------------------------

import f_folders as folder
from f_args import *
from f_date import *
from f_file import *
from f_calc import *
from f_dataframe import *
from f_rolldate import *
from f_chart import *
from f_iqfeed import *

#-----------------------------------------------------------------------------------------------------------------------


    

############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################

print "Looking at files in the folder: '{0}'".format(df_folder)
print
print "Combine files that end with a date range (*.YYYYmmdd-YYYYmmdd.csv):"
print

# GET THE FILES IN THE DATA FOLDER
filenames = []
match_groups = []
os.chdir(df_folder)
for f in glob.glob("*.csv"):
    #print(f)
    filenames.append(f)
    # Find filenames that end with date range (ex: "XXXmYY (Daily).20170101-20170824.csv")
    matchObj = re.match( r'(.*)\.([0-9]+)\-([0-9]+)\.csv$', f, re.M|re.I)
    if matchObj:
        match_groups.append(matchObj.groups())
print
print "Out of {0} files, {1} end with date range (*.YYYYmmdd-YYYYmmdd.csv).".format(len(filenames), len(match_groups))
print

# Out of the filename prefixes, identify the unique ones
unique = set()
for g in match_groups:
    unique.add(g[0])

os.chdir(df_folder)
for pre in unique:
    df = pd.DataFrame()
    for f in glob.glob(pre + "*.csv"):
        print f
        dfx = read_dataframe(join(df_folder, f))
        if df.shape[0] == 0:
            df = dfx
        else:
            df = df.append(dfx)
    df.sort_values("DateTime", axis=0, ascending=True, inplace=True)
    filename = pre + ".csv"
    print "OUTPUT: '{0}'".format(filename)
    write_dataframe(df, join(df_folder, filename))
print

        
