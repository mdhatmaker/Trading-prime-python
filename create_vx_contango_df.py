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

mocodes = ['F','G','H','J','K','M','N','Q','U','V','X','Z']

interval_1min = 60
interval_1hour = 3600

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

# Given a dataframe with a 'contango' column
# Return the dataframe with an added 'range' column that contains an integer index representing the contango range in contango_ranges tuple list
def df_contango_ranges(df):
    for i in range(len(contango_ranges)):
        dfx = get_range(df, 'contango', contango_ranges[i], padnan=False)
        df.loc[dfx.index, 'contango_range'] = int(i)
    df['contango_range'] = df['contango_range'].astype('int')
    return df


# Create the dataframe files required for VX contango data:
# "@VX_futures.daily.DF.csv"
# "@VX_continuous.daily.DF.csv"
# "@VX_contango.daily.DF.csv"
def download_historical_data_for_vx_contango(interval=INTERVAL_DAILY):
    # To get contango data:
    # (1) retrieve latest futures data
    # (2) create continuous front-month from this futures data
    # (3) create contango from this continuous data (will infer one-month-out and two-months-out from front-month symbol)
    df_ = create_historical_futures_df("@VX", interval=interval)
    df_ = create_continuous_df("@VX", get_roll_date_VX, roll_day_adjust=-4, interval=interval)
    df_ = create_historical_calendar_futures_df("@VX", mx=0, my=1, interval=interval)
    df_ = create_continuous_calendar_ETS_df("@VX", 0, 1, interval=interval)
    df_ = create_historical_calendar_futures_df("@VX", mx=1, my=2, interval=interval)
    df_ = create_continuous_calendar_ETS_df("@VX", 1, 2, interval=interval)
    df_ = create_historical_contract_df("VIX.XO")
    df_ = create_contango_df("@VX", interval=interval)
    # Read the contango data from the newly-created .CSV dataframe file
    #df = read_dataframe("@VX_contango.{0}.DF.csv".format(str_interval(data_interval)))
    return

def show_contango_chart(df):
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
def plot_contango_chart(df):
    plot_with_hline(df)
    plt.plot(df['DateTime'], df['contango_1x3x2'], "ro")
    plt.plot(df['DateTime'], df['contango'], "gx")
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
    #plt.show()
    return

def backtest_vx_contango(df, open_buy=None, open_sell=None, close_buy=None, close_sell=None, chart=True):
    roll_dates = [x[1] for x in df_get_roll_dates(df, 'Symbol')]    # df_get_roll_dates => (x[0]:symbol,x[1]:first_date,x[2]:last_date)
    adj = 0.00
    prev_vx = None
    vx1 = np.nan
    li = []
    pos = 0    # start with position flat (pos:+1 for long, pos:-1 for short)
    for ix, row in df.iterrows():
        if pos != 0 and row['DateTime'] in roll_dates:
            dt2 = row['DateTime']
            vx2 = prev_vx  # + adj
            days = (dt2 - dt1).days
            if pos == 1:
                profit = (vx2 - vx1) * 1000 + (adj * 1000)
                adj -= row['m0_m1'] - prev_vx
            elif pos == -1:
                profit = (vx1 - vx2) * 1000 + (adj * 1000)
                adj += row['m0_m1'] - prev_vx
            print("VX contango roll: [{0}] vx_contango:{1:5.2f}    days:{2:3}  profit:${3:8.2f}  profit/day:${4:6.2f}    adj:{5:5.2f}  vx_front_month:{6}=>{7}".format(row['DateTime'].strftime("%Y-%m-%d"), row['contango'], days, profit, profit / days, adj, prev_vx, row['m0_m1']))
            #print("vx contango roll: [{0}]  adj:{1}".format(row['DateTime'], adj))
        prev_vx = row['m0_m1']
        if pos == 0:
            #if open_buy is not None and (row['contango']+adj) <= open_buy:      # to buy: short VX, short ES
            if open_buy is not None and row['contango'] <= open_buy:  # to buy: short VX, short ES
                dt1 = row['DateTime']
                vx1 = row['m0_m1']
                adj = 0.00
                print("\nopen  BUY: [{0}] vx_contango:{1:.2f} vx_front_month:{2}".format(row['DateTime'].strftime("%Y-%m-%d"), row['contango'], vx1))
                pos = 1
            elif open_sell is not None and row['contango'] >= open_sell:   # to sell: long VX, long ES
                dt1 = row['DateTime']
                vx1 = row['m0_m1']
                adj = 0.00
                print("\nopen  SELL: [{0}] vx_contango:{1:.2f} vx_front_month:{2}".format(row['DateTime'].strftime("%Y-%m-%d"), row['contango'], vx1))
                pos = -1
        elif pos == 1:
            if close_buy is not None and row['contango'] >= close_buy:
                dt2 = row['DateTime']
                vx2 = row['m0_m1']   # + adj
                profit = (vx2 - vx1) * 1000 + (adj * 1000)
                days = (dt2 - dt1).days
                print("close BUY: [{0}] vx_contango:{1:.2f} vx_front_month:{2}  days:{3}  profit:${4:.2f}  profit/day:${5:.2f}".format(row['DateTime'].strftime("%Y-%m-%d"), row['contango'], vx2, days, profit, profit/days))
                li.append([row['DateTime'], profit, days])
                pos = 0
                adj = 0.00
        elif pos == -1:
            if close_sell is not None and row['contango'] <= close_sell:
                dt2 = row['DateTime']
                vx2 = row['m0_m1']   # + adj
                profit = (vx1 - vx2) * 1000 + (adj * 1000)
                days = (dt2 - dt1).days
                print("close SELL: [{0}] vx_contango:{1:.2f} vx_front_month:{2}  days:{3}  profit:${4:.2f}  profit/day:${5:.2f}".format(row['DateTime'].strftime("%Y-%m-%d"), row['contango'], vx2, days, profit, profit/days))
                li.append([row['DateTime'], profit, days])
                pos = 0
                adj = 0.00
    print('\n')
    df_profit = pd.DataFrame(data=li, index=None, columns=['DateTime','Profit','Days'])
    df_profit = df_profit.set_index('DateTime')
    df_profit['TotalProfit'] = df_profit['Profit'].cumsum()
    if chart: chart_vx_contango_profit(df_profit, open_buy=open_buy, open_sell=open_sell, close_buy=close_buy, close_sell=close_sell)
    return df_profit

def backtest_vx_contango_by_month(df, month, chart=True):
    roll_first_dates = [x[1] for x in df_get_roll_dates(df, 'Symbol')]    # df_get_roll_dates => (x[0]:symbol,x[1]:first_date,x[2]:last_date)
    roll_last_dates = [x[2] for x in df_get_roll_dates(df, 'Symbol')]    # df_get_roll_dates => (x[0]:symbol,x[1]:first_date,x[2]:last_date)
    adj = 0.00
    #prev_vx = None
    vx1 = np.nan
    li = []
    pos = 0    # start with position flat (pos:+1 for long, pos:-1 for short)
    for ix, row in df.iterrows():
        if pos != 0 and row['DateTime'] in roll_last_dates:
            if pos == 1:
                dt2 = row['DateTime']
                vx2 = row['m0_m1']   # + adj
                profit = (vx2 - vx1) * 1000 + (adj * 1000)
                days = (dt2 - dt1).days
                print("(roll) close BUY {0}: [{1}] vx_contango:{2:.2f} vx_front_month:{3}  days:{4}  profit:${5:.2f}  profit/day:${6:.2f}".format(row['Symbol'], row['DateTime'].strftime("%Y-%m-%d"), row['contango'], vx2, days, profit, profit/days))
                li.append([row['DateTime'], profit, days])
                pos = 0
                adj = 0.00
            elif pos == -1:
                dt2 = row['DateTime']
                vx2 = row['m0_m1']   # + adj
                profit = (vx1 - vx2) * 1000 + (adj * 1000)
                days = (dt2 - dt1).days
                print("(roll) close SELL {0}: [{1}] vx_contango:{2:.2f} vx_front_month:{3}  days:{4}  profit:${5:.2f}  profit/day:${6:.2f}".format(row['Symbol'], row['DateTime'].strftime("%Y-%m-%d"), row['contango'], vx2, days, profit, profit/days))
                li.append([row['DateTime'], profit, days])
                pos = 0
                adj = 0.00
            #print("VX contango roll: [{0}] vx_contango:{1:5.2f}    days:{2:3}  profit:${3:8.2f}  profit/day:${4:6.2f}    adj:{5:5.2f}  vx_front_month:{6}=>{7}".format(row['DateTime'].strftime("%Y-%m-%d"), row['contango'], days, profit, profit / days, adj, prev_vx, row['m0_m1']))
            #print("vx contango roll: [{0}]  adj:{1}".format(row['DateTime'], adj))
        #prev_vx = row['m0_m1']
        else:
            if pos == 0:
                symbol = row['Symbol']
                m = get_month_number(symbol[3])
                if row['DateTime'] in roll_first_dates and m == month:   # to sell: long VX, long ES
                    dt1 = row['DateTime']
                    vx1 = row['m0_m1']
                    adj = 0.00
                    print("\nopen  SELL {0}: [{1}] vx_contango:{2:.2f} vx_front_month:{3}".format(row['Symbol'], row['DateTime'].strftime("%Y-%m-%d"), row['contango'], vx1))
                    pos = -1
    print('\n')
    df_profit = pd.DataFrame(data=li, index=None, columns=['DateTime','Profit','Days'])
    df_profit = df_profit.set_index('DateTime')
    df_profit['TotalProfit'] = df_profit['Profit'].cumsum()
    #if chart: chart_vx_contango_profit(df_profit, open_buy=open_buy, open_sell=open_sell, close_buy=close_buy, close_sell=close_sell)
    return df_profit

def backtest_vx_contango_no_roll(df, open_buy=None, open_sell=None, close_buy=None, close_sell=None, chart=True):
    roll_dates = [x[2] for x in df_get_roll_dates(df, 'Symbol')]    # df_get_roll_dates => (x[0]:symbol,x[1]:first_date,x[2]:last_date)
    adj = 0.00
    #prev_vx = None
    vx1 = np.nan
    li = []
    pos = 0    # start with position flat (pos:+1 for long, pos:-1 for short)
    for ix, row in df.iterrows():
        if pos != 0 and row['DateTime'] in roll_dates:
            if pos == 1:
                dt2 = row['DateTime']
                vx2 = row['m0_m1']   # + adj
                profit = (vx2 - vx1) * 1000 + (adj * 1000)
                days = (dt2 - dt1).days
                print("(roll) close BUY {0}: [{1}] vx_contango:{2:.2f} vx_front_month:{3}  days:{4}  profit:${5:.2f}  profit/day:${6:.2f}".format(row['Symbol'], row['DateTime'].strftime("%Y-%m-%d"), row['contango'], vx2, days, profit, profit/days))
                li.append([row['DateTime'], profit, days])
                pos = 0
                adj = 0.00
            elif pos == -1:
                dt2 = row['DateTime']
                vx2 = row['m0_m1']   # + adj
                profit = (vx1 - vx2) * 1000 + (adj * 1000)
                days = (dt2 - dt1).days
                print("(roll) close SELL {0}: [{1}] vx_contango:{2:.2f} vx_front_month:{3}  days:{4}  profit:${5:.2f}  profit/day:${6:.2f}".format(row['Symbol'], row['DateTime'].strftime("%Y-%m-%d"), row['contango'], vx2, days, profit, profit/days))
                li.append([row['DateTime'], profit, days])
                pos = 0
                adj = 0.00
            #print("VX contango roll: [{0}] vx_contango:{1:5.2f}    days:{2:3}  profit:${3:8.2f}  profit/day:${4:6.2f}    adj:{5:5.2f}  vx_front_month:{6}=>{7}".format(row['DateTime'].strftime("%Y-%m-%d"), row['contango'], days, profit, profit / days, adj, prev_vx, row['m0_m1']))
            #print("vx contango roll: [{0}]  adj:{1}".format(row['DateTime'], adj))
        #prev_vx = row['m0_m1']
        else:
            if pos == 0:
                #if open_buy is not None and (row['contango']+adj) <= open_buy:      # to buy: short VX, short ES
                if open_buy is not None and row['contango'] <= open_buy:  # to buy: short VX, short ES
                    dt1 = row['DateTime']
                    vx1 = row['m0_m1']
                    adj = 0.00
                    print("\nopen  BUY {0}: [{1}] vx_contango:{2:.2f} vx_front_month:{3}".format(row['Symbol'], row['DateTime'].strftime("%Y-%m-%d"), row['contango'], vx1))
                    pos = 1
                elif open_sell is not None and row['contango'] >= open_sell:   # to sell: long VX, long ES
                    dt1 = row['DateTime']
                    vx1 = row['m0_m1']
                    adj = 0.00
                    print("\nopen  SELL {0}: [{1}] vx_contango:{2:.2f} vx_front_month:{3}".format(row['Symbol'], row['DateTime'].strftime("%Y-%m-%d"), row['contango'], vx1))
                    pos = -1
            elif pos == 1:
                if close_buy is not None and row['contango'] >= close_buy:
                    dt2 = row['DateTime']
                    vx2 = row['m0_m1']   # + adj
                    profit = (vx2 - vx1) * 1000 + (adj * 1000)
                    days = (dt2 - dt1).days
                    print("close BUY {0}: [{1}] vx_contango:{2:.2f} vx_front_month:{3}  days:{4}  profit:${5:.2f}  profit/day:${6:.2f}".format(row['Symbol'], row['DateTime'].strftime("%Y-%m-%d"), row['contango'], vx2, days, profit, profit/days))
                    li.append([row['DateTime'], profit, days])
                    pos = 0
                    adj = 0.00
            elif pos == -1:
                if close_sell is not None and row['contango'] <= close_sell:
                    dt2 = row['DateTime']
                    vx2 = row['m0_m1']   # + adj
                    profit = (vx1 - vx2) * 1000 + (adj * 1000)
                    days = (dt2 - dt1).days
                    print("close SELL {0}: [{1}] vx_contango:{2:.2f} vx_front_month:{3}  days:{4}  profit:${5:.2f}  profit/day:${6:.2f}".format(row['Symbol'], row['DateTime'].strftime("%Y-%m-%d"), row['contango'], vx2, days, profit, profit/days))
                    li.append([row['DateTime'], profit, days])
                    pos = 0
                    adj = 0.00
    print('\n')
    df_profit = pd.DataFrame(data=li, index=None, columns=['DateTime','Profit','Days'])
    df_profit = df_profit.set_index('DateTime')
    df_profit['TotalProfit'] = df_profit['Profit'].cumsum()
    if chart: chart_vx_contango_profit(df_profit, open_buy=open_buy, open_sell=open_sell, close_buy=close_buy, close_sell=close_sell)
    return df_profit

def check_vx_contango_level(df, above=None, below=None, chart=True):
    roll_dates = [x[1] for x in df_get_roll_dates(df, 'Symbol')]
    adj = 0.00
    prev_vx = None
    vx1 = np.nan
    li = []
    pos = 0    # start with position flat (pos:+1 for long, pos:-1 for short)
    print(" , Date, Contango, 1x3x2")
    for ix, row in df.iterrows():
        if pos == 0:
            if row['contango'] >= above:
                dt1 = row['DateTime']
                vx1 = row['m0_m1']
                print("\nopen ABOVE {0}:, [{1}], {2:.2f}, {3:.2f}".format(above, dt1, row['contango'], row['1x3x2']))
                adj = 0.00
                pos = 1
        elif pos == 1:
            if row['contango'] <= above:
                dt2 = row['DateTime']
                vx2 = row['m0_m1']
                print("close ABOVE {0}:, [{1}], {2:.2f}, {3:.2f}".format(above, dt2, row['contango'], row['1x3x2']))
                pos = 0
            else:
                print(" , [{0}], {1}, {2}".format(row['DateTime'], row['contango'], row['1x3x2']))

        """
        if pos != 0 and row['DateTime'] in roll_dates:
            dt2 = row['DateTime']
            vx2 = prev_vx  # + adj
            days = (dt2 - dt1).days
            if pos == 1:
                profit = (vx2 - vx1) * 1000 + (adj * 1000)
                adj -= row['m0_m1'] - prev_vx
            elif pos == -1:
                profit = (vx1 - vx2) * 1000 + (adj * 1000)
                adj += row['m0_m1'] - prev_vx
            print("VX contango roll: [{0}] vx_contango:{1:5.2f}    days:{2:3}  profit:${3:8.2f}  profit/day:${4:6.2f}    adj:{5:5.2f}  vx_front_month:{6}=>{7}".format(row['DateTime'].strftime("%Y-%m-%d"), row['contango'], days, profit, profit / days, adj, prev_vx, row['m0_m1']))
            #print("vx contango roll: [{0}]  adj:{1}".format(row['DateTime'], adj))
        prev_vx = row['m0_m1']
        if pos == 0:
            #if open_buy is not None and (row['contango']+adj) <= open_buy:      # to buy: short VX, short ES
            if open_buy is not None and row['contango'] <= open_buy:  # to buy: short VX, short ES
                dt1 = row['DateTime']
                vx1 = row['m0_m1']
                adj = 0.00
                print("\nopen  BUY: [{0}] vx_contango:{1:.2f} vx_front_month:{2}".format(row['DateTime'].strftime("%Y-%m-%d"), row['contango'], vx1))
                pos = 1
            elif open_sell is not None and row['contango'] >= open_sell:   # to sell: long VX, long ES
                dt1 = row['DateTime']
                vx1 = row['m0_m1']
                adj = 0.00
                print("\nopen  SELL: [{0}] vx_contango:{1:.2f} vx_front_month:{2}".format(row['DateTime'].strftime("%Y-%m-%d"), row['contango'], vx1))
                pos = -1
        elif pos == 1:
            if close_buy is not None and row['contango'] >= close_buy:
                dt2 = row['DateTime']
                vx2 = row['m0_m1']   # + adj
                profit = (vx2 - vx1) * 1000 + (adj * 1000)
                days = (dt2 - dt1).days
                print("close BUY: [{0}] vx_contango:{1:.2f} vx_front_month:{2}  days:{3}  profit:${4:.2f}  profit/day:${5:.2f}".format(row['DateTime'].strftime("%Y-%m-%d"), row['contango'], vx2, days, profit, profit/days))
                li.append([row['DateTime'], profit, days])
                pos = 0
                adj = 0.00
        elif pos == -1:
            if close_sell is not None and row['contango'] <= close_sell:
                dt2 = row['DateTime']
                vx2 = row['m0_m1']   # + adj
                profit = (vx1 - vx2) * 1000 + (adj * 1000)
                days = (dt2 - dt1).days
                print("close SELL: [{0}] vx_contango:{1:.2f} vx_front_month:{2}  days:{3}  profit:${4:.2f}  profit/day:${5:.2f}".format(row['DateTime'].strftime("%Y-%m-%d"), row['contango'], vx2, days, profit, profit/days))
                li.append([row['DateTime'], profit, days])
                pos = 0
                adj = 0.00
    """
    print('\n')
    return
    df_profit = pd.DataFrame(data=li, index=None, columns=['DateTime','Profit','Days'])
    df_profit = df_profit.set_index('DateTime')
    df_profit['TotalProfit'] = df_profit['Profit'].cumsum()
    if chart: chart_vx_contango_profit(df_profit, open_buy=open_buy, open_sell=open_sell, close_buy=close_buy, close_sell=close_sell)
    return df_profit

def chart_vx_contango_profit(df, open_buy='', open_sell='', close_buy='', close_sell='', plot_filename='vx_contango_profit'):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 12))
    #fig = plt.figure(figsize=(14, 6))
    ax[0].set_title("VX Contango: [Buy {0}, Sell {1}] [Sell {2}, Buy {3}]".format(open_buy, close_buy, open_sell, close_sell));
    df['TotalProfit'].plot(ax=ax[0], color='green')
    ax[1].set_title('Trade Holding Days');
    df['Days'].plot(ax=ax[1], kind='hist', bins=10);        # histogram plot
    plt.tight_layout();
    filename = '{0}_{1}_{2}_{3}_{4}'.format(plot_filename, open_buy, close_buy, open_sell, close_sell)
    filename = join(misc_folder, '{}.png'.format(filename))
    savefig(plt, filename)
    plt.show(block=False)

def chart_vx_contango_histogram(df, plot_filename='contango_histogram'):
    # VX contango histogram
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 12))
    ax.set_title("VX Contango");
    df['contango'].plot(ax=ax, kind='hist', bins=25);        # histogram plot
    plt.tight_layout();
    filename = join(misc_folder, '{}.png'.format(plot_filename))
    savefig(plt, filename)
    plt.show(block=False)

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

########################################################################################################################

update_historical = False
if update_historical: download_historical_data_for_vx_contango()   # update "@VX_contango.daily.DF.csv"

df = read_dataframe("@VX_contango.daily.DF.csv")

#df_profit_sell = backtest_vx_contango_no_roll(df, open_sell=12.5, close_sell=0)
df_profit_sell = backtest_vx_contango_by_month(df, 1)

STOP()

df_profit_sell = backtest_vx_contango(df, open_sell=12.5, close_sell=0)
df_profit_buy = backtest_vx_contango(df, open_buy=0, close_buy=10)

df_new = pd.concat([df_profit_buy, df_profit_sell])
df_new.sort_index(inplace=True)
df_new['TotalProfit'] = df_new['Profit'].cumsum()
chart_vx_contango_profit(df_new, open_buy=0, open_sell=12.5, close_buy=10, close_sell=0)

STOP()

#check_vx_contango_level(df, above=15)


### Print VX Contango last day before roll date
print('VX Contango Last Day Before Roll:')
li = df_get_roll_dates(df, 'Symbol')
ix_list = []
ix_dict = {}
for symbol,first_date,last_date in li:
    row = df[df['DateTime'] == last_date]
    #contango = row['contango'].values[0]
    ix_dict[last_date] = row.index
    ix_list.append(row.index)
count = 0
for symbol, last_date in li:
    row = df[df['DateTime'] == last_date]
    contango = row['contango'].values[0]
    print(symbol, strdate(last_date), contango)
    if contango >= 15: count += 1
print()
print('{0} >= contango 15 on day before roll (out of {1})   {2:.0f}%'.format(count, len(li), float(count)/len(li)*100.0))

### Print contango when above 15% with 5 days (or less) to roll date
print('Symbol,DateTime,contango,DaysToRoll')
for ix,row in df.iterrows():
    contango = row['contango']
    if contango >= 15:
        for i in ix_list:
            if i > ix:
                days_to_roll = ix - i
                days_to_roll = days_to_roll.values[0]
                break
        if days_to_roll >= -5:
            print("{0},{1},{2},{3}".format(row['Symbol'], strdate(row['DateTime']), row['contango'], days_to_roll))
    #print(ix)


STOP()

df_profit = backtest_vx_contango(df, open_sell=12.5, close_sell=0)
df_profit = backtest_vx_contango(df, open_buy=0, close_buy=10)

chart_vx_contango_histogram(df)

STOP()


# RUN RANDOM QUERIES HERE:
#download_historical_data_for_vx_contango(interval=INTERVAL_HOUR)
#df = read_dataframe("@VX_contango.hour.DF.csv")
#run_query(1)
#run_query(2)
#STOP()

#plot_contango_chart(df.tail(600))
#STOP()


#set_default_dates_latest()


# Request the IQFeed data and create the dataframe file containing the VX contango data ("@VX_contango.daily.DF.csv")
download_historical_data_for_vx_contango()
contango_filename = "@VX_contango.daily.DF.csv"
df_vx = read_dataframe(contango_filename)
df_vx.rename(columns={'Close0':'Close_VX', 'Symbol':'Symbol_VX', 'Close1':'Close_VX1', 'Symbol1':'Symbol_VX1', 'Close2':'Close_VX2', 'Symbol2':'Symbol_VX2', 'm0_m1':'VX_m0_m1', 'm1_m2':'VX_m1_m2', '1x3x2':'VX_1x3x2'}, inplace=True)
df_vx = df_contango_ranges(df_vx)   # Add a 'contango_range' column to our dataframe
# Add a few contango-related columns to ES quartile data
#dfQ = pd.merge(dfQ, df_vx[['DateTime', 'contango', 'contango_1x3x2', 'contango_range']], on='DateTime')
#df = read_dataframe(join(data_folder, "vix_es", "ES_VIX_quartiles (5-day lookback).DF.csv"))
df = read_dataframe("ES_VIX_quartiles (5-day lookback).DF.csv")
df = pd.merge(df, df_vx, on='DateTime')
df_vix = create_historical_contract_df("VIX.XO")
df_vix['DateTime'] = df_vix['DateTime'].astype('datetime64[ns]')
dfz = pd.merge(df, df_vix, on='DateTime')
#df_rename_columns(df, old_names_list=None, new_names_list=None)
dfz = df_rename_columns_dict(dfz, names_dict={'Symbol_y':'Symbol_VIX', 'Open':'Open_VIX', 'High':'High_VIX', 'Low':'Low_VIX', 'Close':'Close_VIX'})
dfz = df_rename_columns_dict(dfz, names_dict={'Symbol_x':'Symbol'})
dfz.drop(['Volume', 'oi'], axis=1, inplace=True)
dfz['Close_VIX'] = dfz['Close_VIX'].astype('float64')
dfz['Close_VX'] = dfz['Close_VX'].astype('float64')
dfz['VIX_discount'] = dfz['Close_VIX'] - dfz['Close_VX']

write_dataframe(dfz, "@VX_contango.EXPANDED.daily.DF.csv", date_only=True)

STOP(df)


N = 100

# Some VX Contango Plots:
#plot_contango_chart(df.tail(600))
plot_with_hline(df_vx, subplot=211)
plot_contango_ranges(df_vx, N)


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



dt1 = datetime(2003, 1, 1)
dt2 = datetime.now()

#df = get_historical("QCL#", dt1, dt2)
#df = get_historical("@VX#", dt1, dt2)
#print df


# Download all historical ES futures and output to file '@ES.csv'
#df = create_historical_futures_df("@ES", 2003, 2017)

# Retrieve the 1-minute ES data and output to file '@ES_session_1min.DF.csv'
#df = get_historical_contract("@ES#", dt1, dt2, interval_1min, "083000", "150000")   # 8:30am to 3:00pm
#filename = "@ES_session_1min.DF.csv"
#df.to_csv(filename, index=False)
#print("Output 1-minute data to '{0}'".format(filename))

# Read the 1-minute ES data from file '@ES_session_1min.DF.csv' and output OHLC summary to file '@ES_session_ohlc.DF.csv'
#df_1min = pd.read_csv("@ES_session_1min.DF.csv", parse_dates=['DateTime'])
#df_ohlc = get_ohlc(df_1min)
#filename = "@ES_session_ohlc.DF.csv"
#df_ohlc.to_csv(filename, index=False)
#print("Output OHLC summary data to '{0}'".format(filename))

# Request the daily VIX.XO and @ES data and output the Quartile values to '@ES_quartiles.DF.csv'
#dfq = get_es_quartiles()
#filename = "@ES_quartiles.DF.csv"
#dfq.to_csv(filename, index=False)
#print("Output quartiles to '{0}'".format(filename))
                 
dfq = pd.read_csv("@ES_quartiles.DF.csv", parse_dates=['DateTime'])
df_ohlc = pd.read_csv("@ES_session_ohlc.DF.csv", parse_dates=['DateTime'])

# Calculate the Quartile Hits from Quartiles and OHLC summary data and output to '@ES_quartile_hits.DF.csv'
#dfz = get_quartile_hits(dfq, df_ohlc)
#filename = "@ES_quartile_hits.DF.csv"
#dfz.to_csv(filename, index=False)
#print("Output quartile hits to '{0}'".format(filename))

df_hit = pd.read_csv("@ES_quartile_hits.DF.csv", parse_dates=['DateTime'])
df_vx = pd.read_csv("contango_@VX.DF.csv", parse_dates=['DateTime'])

# Calculate the Quartile Hit Ratios from Quartile Hits and VIX Contango data and output to '@ES_quartile_hit_ratios.DF.csv' 
#df_hr = get_hit_ratios(df_vx, df_hit, 5)
#filename = "@ES_quartile_hit_ratios.DF.csv"
#df_hr.to_csv(filename, index=False)
#print("Output quartile hit ratios to '{0}'".format(filename))

df_hr = pd.read_csv("@ES_quartile_hit_ratios.DF.csv", parse_dates=['DateTime'])


df = get_vix_quartiles()
i1 = df.columns.get_loc('Qd4')
i2 = df.columns.get_loc('Qu4')
cols = ['DateTime']
cols.extend(df.columns.values[i1:i2+1])
#df_vix = df.iloc[:,i1:i2]
df_vix = df.loc[:,cols]
#print(df_vix)
filename = "@VIX_quartiles.DF.csv"
df_vix.to_csv(filename, index=False)
print("Output quartiles to '{0}'".format(filename))



sys.exit()


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


sys.exit()





symbol_root = '@VX'

# Create the '@VX.csv' file containing futures prices for the given year range
#create_historical_futures_df(symbol_root, 2003, 2017)

# Using the '@VX.csv' data file, create 'contango_@VX.raw.DF.csv' containing contango data
#dfz = create_contango_df(symbol_root)

# From the raw data, create the continuous contract contango file
input_filename = "contango_{0}.raw.DF.csv".format(symbol_root)
continuous_filename = "contango_{0}.DF.csv".format(symbol_root)
#dfz = create_continuous_df(input_filename, continuous_filename, get_roll_date_VX)




# Read in the 'contango_@VX.DF.csv' file
df = pd.read_csv(continuous_filename, parse_dates=['DateTime'])

#show_contango_chart(df)


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














