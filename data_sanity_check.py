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

#-----------------------------------------------------------------------------------------------------------------------


# Given the pathname to a dataframe containing OHLC columns
# Return a tuple (passed, df_errant) where passed is True/False and df_errant is the dataframe of rows that fail sanity check
def pass_sanity_check_ohlc(pathname):
    df_all = pd.read_csv(pathname)
    dfx = df_all[(df_all.High<df_all.Open) | (df_all.High<df_all.Low) | (df_all.High<df_all.Close) | (df_all.Low>df_all.Open) | (df_all.Low>df_all.High) | (df_all.Low>df_all.Close)]
    if dfx.shape[0] > 0:
        return False, dfx
    else:
        return True, dfx

def check_ohlc(folder, pattern):
    print "The following files FAIL sanity check (comparing O,H,L,C prices):"
    print
    # GET THE FILES IN THE DATA FOLDER
    filenames = []
    fail_count = 0
    os.chdir(folder)
    for f in glob.glob(pattern):
        #print(f)
        filenames.append(f)
        passed, dfx = pass_sanity_check_ohlc(join(folder, f))
        if not passed:
            print "'{0}'".format(f)
            print dfx.head(25)
            fail_count += 1
    print
    print "Out of {0} files, {1} FAILED sanity check.".format(len(filenames), fail_count)
    print
    return


############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################


print "Looking at files in the folder: '{0}'".format(df_folder)
print



################# CHECK THE OPEN/HIGH/LOW/CLOSE TO ENSURE THEY SEEM VALID #########
check_ohlc(df_folder, "*.csv")
sys.exit()



################# COMPARE THE 1-MINUTE TIMES BETWEEN HG AND LME ###################
print "The following files FAIL sanity check in comparing last trade time within each 1 Hour bar:"
print

df = read_dataframe(join(df_folder, "M.CU3=LX (1 Minute).csv"))
                    
os.chdir(df_folder)
for f in glob.glob("QHG*.csv"):
    match = re.match(r"(.*)\((.*)\).*\.csv", f, re.M|re.I)
    if match:
        prefix = match.group(1).strip()
        timeframe = match.group(2).strip()
        filename_1min = "{0} ({1}).csv".format(prefix, "1 Minute")
        if timeframe == "1 Hour" and exists(join(df_folder, filename_1min)):
            print f,
            dfh = read_dataframe(join(df_folder, f))                # 1 Hour
            dfm = read_dataframe(join(df_folder, filename_1min))    # 1 Minute
            drop_indexes = []
            for ix,r in dfh.iterrows():
                dt2 = r.DateTime
                dt1 = r.DateTime - timedelta(hours=1)
                dfx = dfm[(dfm.DateTime >= dt1) & (dfm.DateTime < dt2)]
                hg_last = dfx.tail(1).squeeze()['DateTime']
                dfy = df[(df.DateTime >= dt1) & (df.DateTime < dt2)]
                lme_last = dfy.tail(1).squeeze()['DateTime']
                if dfx.shape[0] == 0 and dfy.shape[0] == 0:
                    # No 1-minute data for either HG or LME
                    pass
                elif dfx.shape[0] == 0 or dfy.shape[0] == 0:
                    # One or the other (HG or LME) has no 1-minute data
                    drop_indexes.append(ix)                    
                #elif (dt2 - dt_last) >= timedelta(minutes=15):
                elif abs(lme_last - hg_last) >= timedelta(minutes=15):
                    #print f, dt2    #, (dt2 - dt_last)
                    drop_indexes.append(ix)
                    
            if len(drop_indexes) == 0:
                print
            else:
                prev_row_count = dfh.shape[0]
                dfh.drop(drop_indexes, inplace=True)
                dfh.reset_index()
                row_count = dfh.shape[0]                
                print "{0:7} {1:7}     {2:.0f}%".format(prev_row_count, row_count, (row_count-prev_row_count)/float(prev_row_count)*100)
                write_dataframe(dfh, join(df_folder, f))
    
print
#print "Out of {0} files, {1} FAILED sanity check.".format(len(filenames), fail_count)








