from __future__ import print_function
from iqapi import IQHistoricData
from datetime import datetime, timedelta
from dateutil import relativedelta
import pandas as pd
from pandas.tseries.offsets import BDay
import numpy as np
import os.path
import math
import sys
#import matplotlib.pyplot as plt

"""
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import lstm, time  # helper libraries
"""

#-------------------------------------------------------------------------------
from f_folders import *
from f_chart import *
from f_dataframe import *
from f_rolldate import *
from f_plot import *

project_folder = join(data_folder, "vix_es")

#-------------------------------------------------------------------------------


# range 0: contango < 0
# range 1:  0   <= contango <  2.5
# range 2:  2.5 <= contango <  5
# range 3:  5   <= contango <  7.5
# range 4:  7.5 <= contango < 10
# range 5: 10   <= contango < 12.5
# range 6: 12.5 <= contango < 15
# range 7: contango >= 15
contango_ranges = [(None,0.0), (0.0,2.5), (2.5,5.0), (5.0, 7.5), (7.5,10.0), (10.0, 12.5), (12.5, 15.0), (15.0,None)]


# Given a dataframe, a column name in that dataframe, and a range of values (tuple)
# Return a dataframe containing only those rows where the specified column value falls in the given range (range[0] <= x <= range[1])
# (optional) padnan defaults to True which will include ALL rows from the original dataframe but set their values to np.nan (useful for charting)
# if one of the tuple values is None, it will function as (essentially) -infinity/+infinity (range=(None,15) returns all values >=15)
def get_range(df, column='contango', range=(None, None), padnan=True):
    range0 = -1000000
    range1 = 1000000
    if range[0] is not None: range0 = range[0]
    if range[1] is not None: range1 = range[1]
    dfx = df[(df[column] >= range0) & (df[column] <= range1)]
    if padnan == True:
        dfy = df[~df['DateTime'].isin(dfx['DateTime'])]
        dfy.loc[:,'Close_VX':] = np.nan
        dfx = pd.concat([dfx, dfy]).sort_values(['DateTime'])
    return dfx



def chart_contango(df):
    # CREATE THE CHART
    lines = []
    lines.append(get_chart_line(df, 'contango1x3x2', line_color='rgb(255,0,0)')) #, line_dash='dot'))
    lines.append(get_chart_line(df, 'contango', line_color='rgb(0,0,255)')) #, line_dash='dot'))
    lines.append(get_chart_line(df, 'Close0', line_name='VX'))

    # Highlight the negative ranges on the chart
    #for dfx in rows:
    #    lines.append(get_chart_line(dfx, 'contango1x3x2', line_color='rgb(255,0,0)', line_width=3))

    show_chart(lines, 'VIX_contango')
    return


# Plot 'contango' and 'contango_1x3x2'
def plot_contango(df):
    plot_with_hline(df)
    plt.plot(df['DateTime'], df['contango'], "gx")
    plt.plot(df['DateTime'], df['contango_1x3x2'], "ro")
    #plt.plot(df['DateTime'], df['contango_1x3x2'], color='red', linewidth=1, linestyle='dotted')
    #plt.plot(df['DateTime'], df['contango'], color='green', linewidth=1, linestyle='dotted')
    #plt.plot(df['DateTime'], df['VX_1x3x2'], color='orange', linewidth=1)
    plot_roll_dates(df, 'Symbol_VX', ymin=-10, ymax=+30)
    red_patch = mpatches.Patch(color='red', label='VX contango 1x3x2')
    #orange_patch = mpatches.Patch(color='orange', label='VX 1x3x2')
    green_patch = mpatches.Patch(color='green', label='VX contango')
    #plt.legend(handles=[red_patch, orange_patch, green_patch])
    plt.legend(handles=[red_patch, green_patch])
    plt.show()
    return

"""
# Not the most interesting chart, but plots blue dots representing how we exit the range 10-12.5 when we enter it from below
def plot_contango_range_exits(df):
    df['temp'] = df['contango_range'].shift(-1) # contango range index of following trading day
    dfx = df[(df['contango_range'] == 5) & (df['temp'] != 5)]  # end of range
    dfy = df[(df['contango_range'] < 5) & (df['temp'] == 5)]  # coming into range from below
    for ix, ry in dfy.iterrows():
        dt = ry['DateTime']
        rx = dfx[dfx['DateTime'] > dt].iloc[0]
        dfy.loc[ix, 'temp'] = rx['temp']
    dfy = dfy[['DateTime', 'temp']]
    # Plot the blue dots representing whether we exited the range above or below
    plot_with_hline(dfy, 5)
    plt.plot(dfy['DateTime'], dfy['temp'], "bo")
    plt.show()
    df.drop(['temp'], axis=1, inplace=True)     # delete the 'temp' column
    return
"""

# Plot 'contango' and display different colored markers for ranges 10-12.5, 12.5-15, and above 15
def plot_contango_ranges(df, N=None):
    if N is None: N = df.shape[0]
    df5 = get_range(df, 'contango', contango_ranges[5])  # range 5: 10 <= x < 12.5
    df6 = get_range(df, 'contango', contango_ranges[6])  # range 6: 12.5 <= x < 15
    df7 = get_range(df, 'contango', contango_ranges[7])  # range 7: x >= 15
    plt.plot(df['DateTime'], df['contango'], color='gray', linewidth=1)
    dfx = df.tail(N)
    plt.plot(dfx['DateTime'], dfx['contango'], color='gray', linewidth=3)
    plt.plot(df5['DateTime'], df5['contango'], "gx") #color='red', linewidth=1)
    plt.plot(df6['DateTime'], df6['contango'], "yo") #color='yellow', linewidth=1)
    plt.plot(df7['DateTime'], df7['contango'], "r+") #color='orange', linewidth=1)
    legend_list = []
    legend_list.append(mpatches.Patch(color='green', label='VX contango 10-12.5'))
    legend_list.append(mpatches.Patch(color='yellow', label='VX contango 12.5-15'))
    legend_list.append(mpatches.Patch(color='red', label='VX contango >15'))
    plt.legend(handles=legend_list)
    plt.show()
    return

def query_two_in_a_row_dont_hit(df, contango_target, symbol_column='Symbol'):
    print("Searching for two consecutive contango that do not hit {0}:".format(contango_target))
    symbols = df[symbol_column].unique()
    pair_count = len(symbols)-1
    hit_count = 0
    for i in range(pair_count):
        #print(symbols[i], symbols[i+1])
        df1 = df[df[symbol_column]==symbols[i]]
        df2 = df[df[symbol_column]==symbols[i+1]]
        max1 = round(df1['contango'].max(), 2)
        max2 = round(df2['contango'].max(), 2)
        if max1 < contango_target and max2 < contango_target:
            hit_count += 1
            print("{0} max={1}   {2} max={3}".format(symbols[i], max1, symbols[i+1], max2))
    print("Missed target [{0}]  {1}/{2} times  ({3:.2f}%)".format(contango_target, hit_count, pair_count, float(hit_count)/pair_count*100.0))
    return

def query_132_performance_when_contango_under(df, contango_value):
    dfx = df[(df['contango']<contango_value) & (df['contango']>=0)]
    dfx = dfx[['DateTime','Symbol_VX','Symbol_VX1','Symbol_VX2','contango','VX_1x3x2']]
    plot_with_hline(dfx)
    #plt.axhline(y=0, xmin=dt1, xmax=dt2, c="blue", linewidth=0.5, zorder=0)
    plt.plot(dfx['DateTime'], dfx['contango'], 'gx')
    plt.plot(dfx['DateTime'], dfx['VX_1x3x2'], 'ro')
    red_patch = mpatches.Patch(color='red', label='VX 1x3x2')
    green_patch = mpatches.Patch(color='green', label='VX contango')
    plt.legend(handles=[red_patch, green_patch])
    #plt.plot(df5['DateTime'], df5['contango'], "gx") #color='red', linewidth=1)
    plt.show()
    return dfx

# RUN CUSTOM ONE-OFF QUERIES HERE
def run_query(x):
    if x == 1:
        df = read_dataframe("@VX_contango.EXPANDED.daily.DF.csv")
        dfx = query_132_performance_when_contango_under(df, 5.0)
        write_dataframe(dfx, "query_132_performance_contango_0_to_5.DF.csv")
    elif x == 2:
        df = read_dataframe("@VX_contango.EXPANDED.daily.DF.csv")
        query_two_in_a_row_dont_hit(df, 10.0, symbol_column='Symbol_VX')
        #write_dataframe(dfx, "query_two_in_a_row_dont_hit.DF.csv")
    return

def add_days_to_roll_column(df, symbol_column='Symbol_VX'):
    roll_dates = df_get_roll_dates(df, symbol_column)
    df['days_to_roll'] = 0  # add column to contain 'days_to_roll'
    for symbol,first_date,last_date in roll_dates:
        # print(symbol, rdate)
        dfx = df[df[symbol_column] == symbol]
        count = dfx.shape[0]
        ix1 = dfx.index[count - 1] + 1
        # print(ix1)
        dfx['days_to_roll'] = ix1 - dfx.index
        df.loc[dfx.index, 'days_to_roll'] = dfx['days_to_roll']
        # STOP(df_vx)
    # Do a separate calculation for the roll date of the last continous symbol
    dfx = df[df.days_to_roll == 0]
    if not dfx.empty:
        last_symbol = dfx.iloc[0][symbol_column]
        #first_zero_index = dfx.index[0]
        #last_zero_index = dfx.index[dfx.shape[0] - 1]
        last_roll_date = get_roll_date_VX(last_symbol)
        last_roll_date = last_roll_date.replace(hour=0, minute=0, second=0)
        dfx['days_to_roll'] = dfx['DateTime'].apply(lambda x: diff_business_days(x, last_roll_date))
        df.loc[dfx.index, 'days_to_roll'] = dfx['days_to_roll']
    # If the ES symbol column is still mis-named as 'Symbol_x', fix it by changing it to 'Symbol_ES'
    df = df_rename_columns_dict(df, names_dict={'Symbol_x': 'Symbol_ES'})
    return df

########################################################################################################################
########################################################################################################################
########################################################################################################################

df_vx = read_dataframe("@VX_contango.EXPANDED.daily.DF.csv")

# RUN RANDOM QUERIES HERE:
#download_historical_data_for_vx_contango(interval=INTERVAL_HOUR)
#df = read_dataframe("@VX_contango.hour.DF.csv")
#run_query(1)
#run_query(2)
#STOP()

#plot_contango_chart(df.tail(600))
#STOP()


df = add_days_to_roll_column(df_vx, 'Symbol_VX')
write_dataframe(df, "@VX_contango.EXPANDED.daily.DF.csv")
STOP(df)


# Some VX Contango Plots:
#plot_contango_chart(df.tail(600))
#plot_with_hline(df_vx, subplot=211)
plot_with_hline(df_vx)
plot_contango_ranges(df_vx, N=100)
STOP(df_vx)




dfQ = read_dataframe(join(data_folder, "vix_es", "ES_VIX_quartiles (5-day lookback).DF.csv"))

df = dfQ.tail(N).copy()

addit = 0.01
df['d4'] = df.d4 + addit
df['d3'] = df.d3 + addit
df['d2'] = df.d2 + addit
df['d1'] = df.d1 + addit
df['unch'] = df.unch + addit
df['u1'] = df.u1 + addit
df['u2'] = df.u2 + addit
df['u3'] = df.u3 + addit
df['u4'] = df.u4 + addit
df['spacer'] = 2.4 - (df.d4 + df.d3 + df.d2 + df.d1) - (df.unch/2)


N = df.shape[0]
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

#plot_without_hline()

plt.subplot(212)
p0 = plt.bar(ind, df.spacer, width, color='black')
p1 = plt.bar(ind, df.d4, width, color='#d62728', bottom=df.spacer) #, yerr=menStd)
p2 = plt.bar(ind, df.d3, width, color='green', bottom=df.spacer+df.d4) #, yerr=womenStd)
p3 = plt.bar(ind, df.d2, width, color='blue', bottom=df.spacer+df.d3+df.d4) #, yerr=womenStd)
p4 = plt.bar(ind, df.d1, width, color='purple', bottom=df.spacer+df.d2+df.d3+df.d4)
p5 = plt.bar(ind, df.unch, width, color='yellow', bottom=df.spacer+df.d1+df.d2+df.d3+df.d4)
p6 = plt.bar(ind, df.u1, width, color='purple', bottom=df.spacer+df.unch+df.d1+df.d2+df.d3+df.d4)
p7 = plt.bar(ind, df.u2, width, color='blue', bottom=df.spacer+df.u1+df.unch+df.d1+df.d2+df.d3+df.d4)
p8 = plt.bar(ind, df.u3, width, color='green', bottom=df.spacer+df.u2+df.u1+df.unch+df.d1+df.d2+df.d3+df.d4)
p9 = plt.bar(ind, df.u4, width, color='#d62728', bottom=df.spacer+df.u3+df.u2+df.u1+df.unch+df.d1+df.d2+df.d3+df.d4)
plt.title('Quartile Hit Ratios')
df['xtick'] = df['DateTime'].apply(lambda x: x.strftime('%m-%d') if x.day == 1 else '')
plt.xticks(ind, df['xtick'])


plt.show()


STOP()






"""
#seq_len = 50

print('> Loading data... ')

seq_len = 40
prediction_len = 20
filename = "@ES_quartile_hit_ratios.DF.csv"
#df = fdf.read_dataframe(join(data_folder, "vix_es", "es_vix_daily_summary.DF.csv"))
df = pd.read_csv(filename, parse_dates=['DateTime'])
input_data_filename = 'NN_INPUT.csv'
f = open(input_data_filename, 'w')
for ix,r in df.iterrows():
    # We need to adjust this so there are no ZEROS in the data
    f.write("{0},{1},{2},{3},{4},{5},{6}\n".format(r['DateTime'],r['contango'],r['d4'],r['d3'],r['d2'],r['d1'],r['unch']))
f.close()

sys.exit()
"""

global_start_time = time.time()
epochs  = 1

seq_len  = 50
prediction_len = 50
input_data_filename = 'sp500.csv'

X_train, y_train, X_test, y_test = lstm.load_data(input_data_filename, seq_len, True)

print("TRAINING ROWS: {0}     TEST ROWS: {1}".format(X_train.shape[0], X_test.shape[0]))

print('> Data Loaded. Compiling...')

#model = lstm.build_model([1, 50, 100, 1])
# Don't hardcode "50" but use seq_len instead because seq_len is the lookback length
# original model layers were [1, 50, 100, 1]
model = lstm.build_model([1, seq_len, seq_len*2, 1])

model.fit(
    X_train,
    y_train,
    batch_size=512,
    nb_epoch=epochs,
    validation_split=0.05)

# For now, set our prediction length to our (input) sequence length
#prediction_len = seq_len

predictions = lstm.predict_sequences_multiple(model, X_test, seq_len, prediction_len)
#predicted = lstm.predict_sequence_full(model, X_test, seq_len)
#predicted = lstm.predict_point_by_point(model, X_test)        

print('Training duration (s) : ', time.time() - global_start_time)

#lstm.plot_results_multiple(predictions, y_test, prediction_len)

predict = lstm.predict_point_by_point(model, X_test)
lstm.plot_results(predict, y_test)









"""
# Get datapoints where the 1x3x2 contango is negative
df_neg = df[df['contango1x3x2']<0]
rows = get_ranges_df(df, df_neg)            # get contiguous date ranges of the negative 1x3x2 values

# Find the slope leading into (before) a point where 1x3x2 is negative
# vs leading out (after)
li = []
for dfx in rows:
    dt1 = dfx.iloc[0].DateTime
    dt2 = dfx.iloc[dfx.shape[0]-1].DateTime
    #print dt1.strftime('%Y-%m-%d'), dt2.strftime('%Y-%m-%d'),
    ix = df[df.DateTime==dt1].index.values[0]
    length = 10
    ix1 = max(ix-length,0)
    ix2 = min(ix+length,df.shape[0]-1)
    dfy = df[ix1:ix]
    #print dfy.shape
    z = np.polyfit(dfy.index, dfy.Close0, 1)
    slope1 = round(z[0], 4)
    #print "{0:7.4f}".format(slope1),
    # f = np.poly1d(z)
    dfy = df[ix:ix2]
    if dfy.shape[0]==0:
        break
    z = np.polyfit(dfy.index, dfy.Close0, 1)
    slope2 = round(z[0], 4)
    #print "{0:7.4f}".format(slope2)
    li.append((slope1, slope2))
slopes = np.array(li)
s1 = abs(slopes[:,0])
s2 = abs(slopes[:,1])
print s1.mean()
print s2.mean()
"""














