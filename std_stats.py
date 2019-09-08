import pandas as pd
import numpy as np
from os.path import join
import datetime

#import os
#path = os.path.dirname(os.path.realpath(__file__))
#path = "/Users/michael/CLionProjects/ALVIN/data_sd_analysis"    # macbook
#path = "C:\\Users\\Michael\\Dropbox\\ALVIN\\data_sd_analysis"   # lenovo laptop
path = "B:\Users\mhatmaker\Dropbox\ALVIN\data_sd_analysis"      # apartment desktop

filename = "SD40 QCL#C 15 2017-03-03 2017-06-02.csv"
pathname = join(path, filename)

df = pd.read_csv(pathname)

####################################################################################################

def calc_autocorrelations(column_name, count, significant):
    print 'Running "' + column_name + '" autocorrelations with lookback ' + str(count),
    for (index, row) in df.iterrows():
        if index % 100 == 0:
            print ".",
        if index < count:
            continue
        s = df[column_name][index-count:index-1]
        df.set_value(index, 'autocorr_1', round(s.autocorr(lag=7),2))
        df.set_value(index, 'autocorr_2', round(s.autocorr(lag=8),2))
        df.set_value(index, 'autocorr_3', round(s.autocorr(lag=9),2))
        df.set_value(index, 'autocorr_4', round(s.autocorr(lag=10),2))
        df.set_value(index, 'autocorr_5', round(s.autocorr(lag=11),2))
        
        #df['autocorr_1'][index] = s.autocorr(lag=1)
        #df['autocorr_2'][index] = s.autocorr(lag=2)
        #df['autocorr_3'][index] = s.autocorr(lag=3)
        #df['autocorr_4'][index] = s.autocorr(lag=4)
        #df['autocorr_5'][index] = s.autocorr(lag=5)
        #print s.autocorr(lag=1)
        #print index
    print

    df1 = df[df['autocorr_1'] >= significant]
    df2 = df[df['autocorr_2'] >= significant]
    df3 = df[df['autocorr_3'] >= significant]
    df4 = df[df['autocorr_4'] >= significant]
    df5 = df[df['autocorr_5'] >= significant]

    print "lag 1: count =", len(df1)
    print "lag 2: count =", len(df2)
    print "lag 3: count =", len(df3)
    print "lag 4: count =", len(df4)
    print "lag 5: count =", len(df5)

    print "Results in Dataframes df1, df2, df3, df4, df5"
    print
    return (df1, df2, df3, df4, df5)

def look_for_runs(column_name, runlen):
    run_zeros = 0
    run_zeros_success = 0
    run_ones = 0
    run_ones_success = 0
    for (index, row) in df.iterrows():
        if index < count + runlen:
            continue
        found_run = True
        for i in range(1, runlen):
            if not (df[column_name][index-i] == 0):
                found_run = False
        if found_run == True:
            run_zeros += 1
            if df[column_name][index] == 0:
                run_zeros_success += 1
        found_run = True
        for i in range(1, runlen):
            if not (df[column_name][index-i] == 1):
                found_run = False
        if found_run == True:
            run_ones += 1
            if df[column_name][index] == 1:
                run_ones_success += 1

    print 'For "' + column_name + '":'
    zeros_pct = float(run_zeros_success)/run_zeros
    ones_pct = float(run_ones_success)/run_ones
    print str(runlen) + " consecutive zeros: %d %d %.2f" % (run_zeros, run_zeros_success, zeros_pct) 
    print str(runlen) + " consecutive ones:  %d %d %.2f" % (run_ones, run_ones_success, ones_pct)
    print
    
    return

####################################################################################################

# Modifiy the hitcount columns so they contain only 1 or 0 (based on original hitcount values)
#df.hitcount_pos.loc[df.hitcount_pos > 0] = 1
#df.hitcount_neg.loc[df.hitcount_neg < 0] = 1

# Create 5 columns that will hold autocorrelations from lag 1 to lag 5
df['autocorr_1'] = np.nan
df['autocorr_2'] = np.nan
df['autocorr_3'] = np.nan
df['autocorr_4'] = np.nan
df['autocorr_5'] = np.nan

# use Google to find significance of lookback periods other than 10:
# correlation statistical significance calculator
# want p-value to be .05 or less

lookback = 15
significant = .63


#(df1, df2, df3, df4, df5) = calc_autocorrelations('hitcount_neg', count, significant)
#(df1, df2, df3, df4, df5) = calc_autocorrelations('hitcount_pos', count, significant)
#look_for_runs('hitcount_neg', 2)
#look_for_runs('hitcount_pos', 3)


(df1, df2, df3, df4, df5) = calc_autocorrelations('diff', lookback, significant)





    
