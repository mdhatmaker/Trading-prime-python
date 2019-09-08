from __future__ import print_function
from os.path import join, basename, splitext
import pandas as pd
import sys
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------------------------------------------------
from f_folders import *
from f_date import *
from f_plot import *
from f_stats import *
from f_dataframe import *
from f_file import *
from f_rolldate import *

pd.set_option('display.width', 160)
# -----------------------------------------------------------------------------------------------------------------------


# Create the dataframe files required for VX/ES premium data:
# "@VX_futures.daily.DF.csv"
# "@VX_continuous.daily.DF.csv"
# "@ES_futures.daily.DF.csv"
# "@ES_continuous.daily.DF.csv"
def download_historical_for_vx_es_premium(interval=INTERVAL_DAILY):
    # To get contango data:
    # (1) retrieve latest futures data (both VX and ES)
    # (2) create continuous front-month from this futures data (both VX and ES)
    # (3) VX/ES premium from this data
    df_ = create_historical_futures_df("@VX", interval=interval)
    df_ = create_continuous_df("@VX", get_roll_date_VX, interval=interval)
    df_ = create_historical_futures_df("@ES", interval=interval)
    df_ = create_continuous_df("@ES", get_roll_date_ES, interval=interval)

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

def add_roll_price_column(df):
    roll_dates = df_get_roll_dates(df, 'Symbol_VX')
    df['roll_price_VX'] = np.nan  # add column to contain price diff of VX vs VX at time of roll
    df['roll_price_ES'] = np.nan  # add column to contain price diff of ES vs ES at time of VX roll
    df['insurance_premium'] = np.nan    # add column to contain insurance premium calculation (
    for i in range(1, len(roll_dates)):
        prev_symbol, prev_dt1, prev_dt2 = roll_dates[i-1]
        symbol, dt1, dt2 = roll_dates[i]
        dfx = df[df['Symbol_VX'] == symbol]         # TODO: change this to use Close price of previous day
        roll_price_VX = df_first(dfx)['Open_VX']
        roll_price_ES = df_first(dfx)['Open_ES']
        #df.loc[dfx.index, ['roll_price_VX', 'roll_price_ES']] = roll_price_VX, roll_price_ES
        #df.loc[dfx.index, ['roll_price_VX', 'roll_price_ES']] = dfx['Close_VX']-roll_price_VX, dfx['Close_ES']-roll_price_ES
        df.loc[dfx.index, 'roll_price_VX'] = (dfx['Close_VX'] - roll_price_VX)  # * 1000   # VX is $1,000 per point
        df.loc[dfx.index, 'roll_price_ES'] = (dfx['Close_ES'] - roll_price_ES)  # * 50     # ES is $1,000 per 20 points
    df['insurance_premium'] = ((df['roll_price_VX'] * 1000/50 + df['roll_price_ES'])/df['Close_ES'] * 100.0).round(2)
    #df['insurance_premium'].round(2)
    return df.dropna()

def chart_vix_insurance_premium(dfx, title='VX/ES Premium', plot_filename='vix_insurance_premium_analysis'):
    df = dfx.set_index('DateTime')
    #fig = plt.figure(figsize=(14, 12))
    layout = (6, 2)
    #fig = plt.figure(nrows=5, ncols=2, sharex=True, figsize=(14, 12))
    plt.figure()
    fig, ax = plt.subplots(nrows=6, ncols=1, sharex=True, figsize=(14, 14))
    fig.text(0.5, 0.04, 'DateTime', ha='center')
    #ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    #roll_vx_ax = plt.subplot2grid(layout, (1, 0), colspan=2)
    #roll_es_ax = plt.subplot2grid(layout, (2, 0), colspan=2)
    #vx_ax = plt.subplot2grid(layout, (3, 0), colspan=2)
    #es_ax = plt.subplot2grid(layout, (4, 0), colspan=2)
    prem_ax = ax[0]
    ts_ax = ax[1]
    roll_vx_ax = ax[2]
    roll_es_ax = ax[3]
    vx_ax = ax[4]
    es_ax = ax[5]

    plt.legend(loc='best')

    # insurance premium (%)
    df['insurance_premium'].plot(ax=prem_ax, color='green');
    prem_ax.set_title("VIX Insurance Premium (%)");
    prem_ax.axhline(y=3, xmin=0.0, xmax=1.0, linewidth=1, linestyle='dotted', color='black', alpha=0.5)
    prem_ax.axhline(y=2, xmin=0.0, xmax=1.0, linewidth=1, linestyle='dotted', color='black', alpha=0.5)
    prem_ax.axhline(y=1, xmin=0.0, xmax=1.0, linewidth=1, linestyle='dotted', color='black', alpha=0.5)
    prem_ax.axhline(y=0, xmin=0.0, xmax=1.0, linewidth=1, linestyle='solid', color='black', alpha=0.5)
    prem_ax.axhline(y=-1, xmin=0.0, xmax=1.0, linewidth=1, linestyle='dotted', color='black', alpha=0.5)
    prem_ax.axhline(y=-2, xmin=0.0, xmax=1.0, linewidth=1, linestyle='dotted', color='black', alpha=0.5)
    prem_ax.axhline(y=-3, xmin=0.0, xmax=1.0, linewidth=1, linestyle='dotted', color='black', alpha=0.5)

    # ES_diff 5-day average
    #df['Close_VX'].plot(ax=ts_ax)
    rolling_mean = pd.rolling_mean(df['diff_ES'], window=5)     # changed this from window=12
    rolling_mean.plot(ax=ts_ax, color='darkslateblue');
    #plt.legend(loc='best')
    #ts_ax.set_title(title, fontsize=24);
    ts_ax.set_title("ES diff (5-day average)");
    ts_ax.axhline(y=5, xmin=0.0, xmax=1.0, linewidth=1, linestyle='dotted', color='black', alpha=0.5)
    ts_ax.axhline(y=-5, xmin=0.0, xmax=1.0, linewidth=1, linestyle='dotted', color='black', alpha=0.5)

    # VX and ES calculated from date of VX roll
    df['roll_price_VX'].plot(ax=roll_vx_ax, color='crimson')
    roll_vx_ax.set_title('VX Price Change from VX Roll')
    df['roll_price_ES'].plot(ax=roll_es_ax, color='darkslateblue')
    roll_es_ax.set_title('ES Price Change from VX Roll')

    # VX and ES prices
    df['Close_VX'].plot(ax=vx_ax, color='crimson')
    vx_ax.set_title('VX Close')
    df['Close_ES'].plot(ax=es_ax, color='darkslateblue')
    es_ax.set_title('ES Close')

    # save plot
    plt.tight_layout();
    pathname = join(misc_folder, '{}.png'.format(plot_filename))
    savefig(plt, pathname)
    #plt.show()
    plt.show(block=False)
    plt.clf()
    plt.cla()
    plt.close()
    #plt.gcf().clear()

def backtest_es_diff(df, open_buy=-5, open_sell=+5, close_buy=0, close_sell=0, plot=True, plot_filename='vx_diff_es_profit'):
    #df = dfx.set_index('DateTime')
    li = []
    pos = 0    # start with position flat (pos:+1 for long, pos:-1 for short)
    for ix, row in df.iterrows():
        if pos == 0:
            if row['diff_ES'] <= open_buy:      # to buy: short VX, short ES
                dt1 = row['DateTime']
                vx1 = row['roll_price_VX']
                es1 = row['roll_price_ES']
                print("open  BUY: [{0}] vx:{1} es:{2}".format(row['DateTime'], vx1, es1))
                pos = 1
            elif row['diff_ES'] >= open_sell:   # to sell: long VX, long ES
                dt1 = row['DateTime']
                vx1 = row['roll_price_VX']
                es1 = row['roll_price_ES']
                print("open SELL: [{0}] vx:{1} es:{2}".format(row['DateTime'], vx1, es1))
                pos = -1
        elif pos == 1:
            if row['diff_ES'] >= close_buy:
                dt2 = row['DateTime']
                vx2 = row['roll_price_VX']
                es2 = row['roll_price_ES']
                profit = (vx1 - vx2)*1000 + (es1 - es2)*20
                days = (dt2 - dt1).days
                print("close  BUY: [{0}] vx:{1} es:{2}    days:{3}  profit:${4:.2f}".format(row['DateTime'], vx2, es2, days, profit))
                li.append([row['DateTime'], profit, days])
                pos = 0
        elif pos == -1:
            if row['diff_ES'] <= close_sell:
                dt2 = row['DateTime']
                vx2 = row['roll_price_VX']
                es2 = row['roll_price_ES']
                profit = (vx2 - vx1)*1000 + (es2 - es1)*20
                days = (dt2 - dt1).days
                print("close SELL: [{0}] vx:{1} es:{2}    days:{3}  profit:${4:.2f}".format(row['DateTime'], vx2, es2, days, profit))
                li.append([row['DateTime'], profit, days])
                pos = 0
    df_profit = pd.DataFrame(data=li, index=None, columns=['DateTime','Profit','Days'])
    df_profit = df_profit.set_index('DateTime')
    df_profit['TotalProfit'] = df_profit['Profit'].cumsum()
    if plot: chart_vix_insurance_premium_profit(df_profit, open_buy=open_buy, open_sell=open_sell, close_buy=close_buy, close_sell=close_sell, plot_filename=plot_filename)
    return df_profit

def backtest_vix_insurance_premium(df, open_buy=-2, open_sell=+2, close_buy=-1, close_sell=1, close_at_roll_date=True, plot=True, plot_filename='vx_insurance_premium_profit'):
    roll_dates = df_get_roll_dates(df, 'Symbol_VX')     # roll_dates[0]='<symbol>', roll_dates[1]=first_date_using_symbol, roll_dates[2]=last_date_using_symbol
    last_dates = [rd[2] for rd in roll_dates]
    #df = dfx.set_index('DateTime')
    li = []
    pos = 0    # start with position flat (pos:+1 for long, pos:-1 for short)
    for ix, row in df.iterrows():
        if pos == 0:
            if open_buy is not None and row['insurance_premium'] <= open_buy:      # to buy: short VX, short ES
                dt1 = row['DateTime']
                vx1 = row['roll_price_VX']
                es1 = row['roll_price_ES']
                #print("open  BUY: [{0}] vx:{1} es:{2}".format(row['DateTime'], vx1, es1))
                print("{0},VX_INSURANCE_PREMIUM,{1},OPEN_BUY,{2},{3},{4},{5}".format(row['DateTime'], row['insurance_premium'], row['Symbol_VX'], vx1, row['Symbol_ES'], es1))
                pos = 1
            elif open_sell is not None and row['insurance_premium'] >= open_sell:   # to sell: long VX, long ES
                dt1 = row['DateTime']
                vx1 = row['roll_price_VX']
                es1 = row['roll_price_ES']
                #print("open SELL: [{0}] vx:{1} es:{2}".format(row['DateTime'], vx1, es1))
                print("{0},VX_INSURANCE_PREMIUM,{1},OPEN_SELL,{2},{3},{4},{5}".format(row['DateTime'], row['insurance_premium'], row['Symbol_VX'], vx1, row['Symbol_ES'], es1))
                pos = -1
        elif pos == 1:
            if (close_buy is not None and row['insurance_premium'] >= close_buy) or (close_at_roll_date == True and row['DateTime'] in last_dates):
                dt2 = row['DateTime']
                vx2 = row['roll_price_VX']
                es2 = row['roll_price_ES']
                profit = (vx1 - vx2)*1000 + (es1 - es2)*20
                days = (dt2 - dt1).days
                #print("close  BUY: [{0}] vx:{1} es:{2}    days:{3}  profit:${4:.2f}".format(row['DateTime'], vx2, es2, days, profit))
                print("{0},VX_INSURANCE_PREMIUM,{1},CLOSE_BUY,{2},{3},{4},{5}".format(row['DateTime'], row['insurance_premium'], row['Symbol_VX'], vx2, row['Symbol_ES'], es2))
                li.append([row['DateTime'], profit, days])
                pos = 0
        elif pos == -1:
            if (close_sell is not None and row['insurance_premium'] <= close_sell)  or (close_at_roll_date == True and row['DateTime'] in last_dates):
                dt2 = row['DateTime']
                vx2 = row['roll_price_VX']
                es2 = row['roll_price_ES']
                profit = (vx2 - vx1)*1000 + (es2 - es1)*20
                days = (dt2 - dt1).days
                #print("close SELL: [{0}] vx:{1} es:{2}    days:{3}  profit:${4:.2f}".format(row['DateTime'], vx2, es2, days, profit))
                print("{0},VX_INSURANCE_PREMIUM,{1},CLOSE_SELL,{2},{3},{4},{5}".format(row['DateTime'], row['insurance_premium'], row['Symbol_VX'], vx2, row['Symbol_ES'], es2))
                li.append([row['DateTime'], profit, days])
                pos = 0
    df_profit = pd.DataFrame(data=li, index=None, columns=['DateTime','Profit','Days'])
    df_profit = df_profit.set_index('DateTime')
    df_profit['TotalProfit'] = df_profit['Profit'].cumsum()
    if plot: chart_vix_insurance_premium_profit(df_profit, open_buy=open_buy, open_sell=open_sell, close_buy=close_buy, close_sell=close_sell, plot_filename=plot_filename)
    return df_profit

def backtest_vix(df, plot=True, plot_filename='vx_monthly_profit'):
    roll_dates = df_get_roll_dates(df, 'Symbol')     # roll_dates[0]='<symbol>', roll_dates[1]=first_date_using_symbol, roll_dates[2]=last_date_using_symbol
    first_dates = [rd[1] for rd in roll_dates]
    last_dates = [rd[2] for rd in roll_dates]
    #df = dfx.set_index('DateTime')
    li = []
    pos = 0    # start with position flat (pos:+1 for long, pos:-1 for short)
    for ix, row in df.iterrows():
        if pos == 0:
            """if row['DateTime'] in first_dates:      # to buy: short VX, short ES
                dt1 = row['DateTime']
                vx1 = row['Close0']
                #print("open  BUY: [{0}] vx:{1} es:{2}".format(row['DateTime'], vx1, es1))
                print("{0},VX_MONTHLY,{1},OPEN_BUY,{2},{3},{4},{5}".format(row['DateTime'], row['contango'], row['Symbol'], vx1, row['Symbol_ES'], es1))
                pos = 1"""
            if row['DateTime'] in first_dates:   # to sell: long VX, long ES
                dt1 = row['DateTime']
                vx1 = row['m0_m1']
                #print("open SELL: [{0}] vx:{1} es:{2}".format(row['DateTime'], vx1, es1))
                print("{0},VX_MONTHLY,{1},OPEN_SELL,{2},{3},{4}".format(row['DateTime'], row['contango'], row['Symbol'], row['Symbol1'], vx1))
                pos = -1
        elif pos == 1:
            if row['DateTime'] in last_dates:
                dt2 = row['DateTime']
                vx2 = row['m0_m1']
                profit = (vx1 - vx2)*1000
                days = (dt2 - dt1).days
                #print("close  BUY: [{0}] vx:{1} es:{2}    days:{3}  profit:${4:.2f}".format(row['DateTime'], vx2, es2, days, profit))
                print("{0},VX_MONTHLY,{1},CLOSE_BUY,{2},{3},{4}".format(row['DateTime'], row['contango'], row['Symbol'], row['Symbol1'], vx2))
                li.append([row['DateTime'], profit, days])
                pos = 0
        elif pos == -1:
            if row['DateTime'] in last_dates:
                dt2 = row['DateTime']
                vx2 = row['m0_m1']
                profit = (vx2 - vx1)*1000
                days = (dt2 - dt1).days
                #print("close SELL: [{0}] vx:{1} es:{2}    days:{3}  profit:${4:.2f}".format(row['DateTime'], vx2, es2, days, profit))
                print("{0},VX_MONTHLY,{1},CLOSE_SELL,{2},{3},{4}".format(row['DateTime'], row['contango'], row['Symbol'], row['Symbol1'], vx2))
                li.append([row['DateTime'], profit, days])
                pos = 0
    df_profit = pd.DataFrame(data=li, index=None, columns=['DateTime','Profit','Days'])
    df_profit = df_profit.set_index('DateTime')
    df_profit['TotalProfit'] = df_profit['Profit'].cumsum()
    if plot: chart_vix_insurance_premium_profit(df_profit, open_buy=None, open_sell=None, close_buy=None, close_sell=None, plot_filename=plot_filename)
    return df_profit

def chart_vix_insurance_premium_profit(df, open_buy=-5, open_sell=+5, close_buy=0, close_sell=0, plot_filename='vix_insurance_premium_test'):
    #plt.figure()
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 12))
    #fig = plt.figure(figsize=(14, 6))
    ax[0].set_title("VX/ES Premium: [Buy {0}, Sell {1}] [Sell {2}, Buy {3}]".format(open_buy, close_buy, open_sell, close_sell));
    df['TotalProfit'].plot(ax=ax[0], color='green')
    df['Days'].plot(ax=ax[1], kind='hist', bins=10);        # histogram plot
    ax[1].set_title('Trade Holding Days');
    plt.tight_layout();
    filename = '{0}_{1}_{2}_{3}_{4}'.format(plot_filename, open_buy, close_buy, open_sell, close_sell)
    pathname = join(misc_folder, '{}.png'.format(filename))
    savefig(plt, pathname)
    plt.show(block=False)


########################################################################################################################

update_historical = False

# --------------------------------------------------------------------------------------------------
# https://<informational links here>

if update_historical: download_historical_for_vx_es_premium()
df_vx = read_dataframe("@VX_continuous.daily.DF.csv")
df_es = read_dataframe("@ES_continuous.daily.DF.csv")

vx_rolls = df_get_roll_dates(df_vx)     # symbol,first_date,last_date
es_rolls = df_get_roll_dates(df_es)     # symbol,first_date,last_date

df = pd.merge(df_vx, df_es, on='DateTime', suffixes=('_VX','_ES'))

df = add_roll_price_column(df)
df['diff_ES'] = df['Close_ES'].diff(1)
df.dropna(inplace=True)

dfx = df[['DateTime','Symbol_VX','Close_VX','roll_price_VX','Symbol_ES','Close_ES','roll_price_ES','diff_ES','insurance_premium']]
write_dataframe(dfx, 'vix_insurance_premium.daily.DF.csv')

#chart_vix_insurance_premium(dfx)
#chart_vix_insurance_premium(dfx[dfx['DateTime']>='2017-01-01'])
chart_vix_insurance_premium(dfx[dfx['DateTime']>='2015-01-01'])

dfz = read_dataframe("@VX_contango.daily.DF.csv")
df_profit = backtest_vix(dfz)

STOP()

df_profit = backtest_vix_insurance_premium(dfx, open_buy=-2, open_sell=None, close_buy=-1, close_sell=None)
df_profit = backtest_vix_insurance_premium(dfx, open_buy=-3, open_sell=None, close_buy=-1, close_sell=None)
df_profit = backtest_vix_insurance_premium(dfx, open_buy=None, open_sell=2, close_buy=None, close_sell=1)
df_profit = backtest_vix_insurance_premium(dfx, open_buy=None, open_sell=3, close_buy=None, close_sell=1)

STOP()

df_profit = backtest_es_diff(dfx, open_buy=-5, open_sell=+5, close_buy=0, close_sell=0)
df_profit = backtest_es_diff(dfx, open_buy=-5, open_sell=+5, close_buy=+5, close_sell=-5)


STOP()

