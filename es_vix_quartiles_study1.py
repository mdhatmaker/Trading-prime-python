import sys
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import math

#-----------------------------------------------------------------------------------------------------------------------

import f_folders as folder
from f_args import *
from f_date import *
from f_file import *
from f_calc import *
from f_dataframe import *
from f_rolldate import *
from f_chart import *

project_folder = join(data_folder, "vix_es")

#-----------------------------------------------------------------------------------------------------------------------

# Description of the argument list for this Python script
ARGS_LIST = \
"""
REQUIRED COMMAND-LINE ARGUMENTS
start_year : int
end_year : int
quartile : ['d4'|'d3'|'d2'|'d1'|'unch'|'u1'|'u2'|'u3'|'u4']
[lte | gte] : float         (initiate when quartile hit_ratio <= lte OR hit_ratio >= gte)
lookback: int               (specify X-day lookback input file)
chart_netchg : <nothing>    (create the NetChg chart)
chart_return : <nothing>    (create the Return chart)
chart_quartile : <nothing>  (create the Quartile bar chart)
contango: <nothing>         (perform contango analysis rather than backtest for quartile trades)
"""

# Using the arguments passed to this Python script, initialize some values for our analysis
# for 'iq': 0=d4  1=d3  2=d2  3=d1  4=unch  5=u1  6=u2  7=u3  8=u4
#iq = 3                  # which quartile (from down4 to up4)
#lte = 0.2               # initiate analysis when hit_ratio is <= this 'lte' value
#lte = None
#gte = 0.8               # initiate analysis when hit_ratio is >= this 'gte' value
#gte = None
#lookback_days = 5       # determines which file we use for analysis (hit_ratio in files are created based on lookback)
def get_args():
    y1 = int(get_arg('start_year'))
    y2 = int(get_arg('end_year'))
    if 'lte' in args:
        lte = float(get_arg('lte'))
        gte = None
    else:
        gte = float(get_arg('gte'))
        lte = None
    iq = quartiles.index(get_arg('quartile'))
    lookback_days = int(get_arg('lookback'))
    return (y1, y2, lte, gte, iq, lookback_days)

def print_trades(trade_column, trade_rows, side, entry_price, exit_price, df_rolls):
    average = 0.0
    for (t_entry, t_exit, cal_adjust, discount_adjust) in trade_rows:
        day_count = (t_exit.Date - t_entry.Date).days
        roll_count = roll_exists_in_date_range(t_entry.Date, t_exit.Date, df_rolls)
        roll_count_indicator = '*' * roll_count
        output = get_output(t_entry, t_exit, cal_adjust, discount_adjust)
        output += roll_count_indicator
        print(output)
        average += day_count
    average /= len(trades)
    print()
    print(len(trades), "trades")
    print("average holding period (days): %.1f" % (average))
    return

def write_trades_file(filename, trade_column, trade_rows, side, entry_price, exit_price, df_rolls):
    f = open(folder + filename, 'w')
    f.write("Side,EntryDiscount,ExitDiscount,AdjustDiscount,EntrySpread,ExitSpread,AdjustSpread,EntryDate,ExitDate,HoldingDays,RollCount\n")
    for (t_entry, t_exit, cal_adjust, discount_adjust) in trade_rows:
        day_count = (t_exit.Date - t_entry.Date).days
        roll_count = roll_exists_in_date_range(t_entry.Date, t_exit.Date, df_rolls)
        output = get_output(t_entry, t_exit, cal_adjust, discount_adjust)
        output += str(roll_count)
        f.write(output + '\n')
    f.close()
    return

# Given a quartile identifier (-4, -3, -2, -1, 0, +1, +2, +3, +4) and mean and standard deviation
# Return the value for the requested quartile
#def Q(i, mean, std):
#    qvalue = (std / 100.0 * mean) / 4.0
#    return mean + i * qvalue

def calc_rolling_mean(df, lookback):
    df['d4'] = df['Qd4'].shift(1).rolling(lookback).mean()
    df['d3'] = df['Qd3'].shift(1).rolling(lookback).mean()
    df['d2'] = df['Qd2'].shift(1).rolling(lookback).mean()
    df['d1'] = df['Qd1'].shift(1).rolling(lookback).mean()
    df['unch'] = df['Qunch'].shift(1).rolling(lookback).mean()
    df['u1'] = df['Qu1'].shift(1).rolling(lookback).mean()
    df['u2'] = df['Qu2'].shift(1).rolling(lookback).mean()
    df['u3'] = df['Qu3'].shift(1).rolling(lookback).mean()
    df['u4'] = df['Qu4'].shift(1).rolling(lookback).mean()
    return

"""
# Display chart for times when specified Quartile is LESS-THAN-OR-EQUAL to 'lte' value until specified Quartile is GREATER-THAN-OR-EQUAL to mean
# Filter chart for a single year
def show_chart_lte(dfx, columnA, columnB, year, iq, lte):
    print "Creating {0} chart for {1}:".format(columnA, year)
    dt1 = datetime(year, 1, 1)
    dt2 = datetime(year, 12, 31)
    df = dfx[(dfx.DateTime >= dt1) & (dfx.DateTime <= dt2)]

    # Store hit_ratio mean values in a list
    df_hitratio = df_all.iloc[:,18:27]
    hitratio_means = df_hitratio.mean().values

    traces = []
    labels = []
    
    wait_for_mean = None

    for (ix,row) in df.iterrows():
        # skip initial rows which will not contain valid data
        if row.isnull().any():
            continue
        if wait_for_mean != None:
            if row[Q] >= wait_for_mean:
                ix2 = ix
                print row[Q], row['DateTime'].strftime(" %b %d %Y"), "    ", int(ix2-ix1)
                wait_for_mean = None
                dfx = df.loc[ix1:ix2]
                #dfx['Label'] = dfx.apply(lambda s: s['DateTime'].strftime("%b %d"), axis=1)     # axis=1 means apply by columns (axis=0 for rows)
                dfx['Label'] = dfx.apply(lambda s: s['DateTime'].strftime("%b %d") + "  ctgo:{0}={1}".format(s[columnB], round(s[columnA],2)), axis=1)     # axis=1 means apply by columns (axis=0 for rows)
                i = len(traces) % len(colors_vga)
                line_color = colors_vga[i]
                #traces.append(get_trace(dfx, 'test', 'Close_ES', 'Contango_Close', line_color))
                traces.append(get_trace(dfx, 'test', columnA, columnB, line_color, marker_labels='Label'))
                #print dfx
        else:
            if row[Q] <= lte:
                ix1 = ix
                print row['DateTime'].strftime("%b %d %Y "), row[Q], "   ",
                wait_for_mean = hitratio_means[iq]
    print
    chart_filename = join(html_folder, columnA + "_vs_contango_" + str(year) + ".html")
    #trace1 = get_trace(df, 'test', 'Close_ES', mode='markers')
    display_chart(traces, chart_filename, "{0} vs VIX Contango {1} ({2}<={3})".format(columnA, year, Q, lte))
    return
"""

# Display chart for times when specified Quartile is GREATER-THAN-OR-EQUAL to 'gte' value until specified Quartile is LESS-THAN-OR-EQUAL to mean
# Filter chart for a single year
def show_chart(dfx, columnA, columnB, year, iq, f1, f2, x):
    print("Creating {0} chart for {1}:".format(columnA, year))
    dt1 = datetime(year, 1, 1)
    dt2 = datetime(year, 12, 31)
    df = dfx[(dfx.DateTime >= dt1) & (dfx.DateTime <= dt2)]

    # Store hit_ratio mean values in a list
    df_hitratio = df_all.iloc[:,18:27]
    hitratio_means = df_hitratio.mean().values

    traces = []
    labels = []

    wait_for_mean = None

    for (ix,row) in df.iterrows():
        # skip initial rows which will not contain valid data
        if row.isnull().any():
            continue
        if wait_for_mean != None:
            #if row[Q] <= wait_for_mean:
            if f2(row[Q_colname], wait_for_mean):
                ix2 = ix
                print(row[Q_colname], row['DateTime'].strftime(" %b %d %Y"), "    ", int(ix2 - ix1))
                wait_for_mean = None
                dfx = df.loc[ix1+1:ix2].copy()
                #dfx['Label'] = dfx.apply(lambda s: s['DateTime'].strftime("%b %d"), axis=1)     # axis=1 means apply by columns (axis=0 for rows)
                #dfx['Label'] = dfx.apply(lambda s: s['DateTime'].strftime("%b %d") + "  ctgo:{0}={1}".format(s[columnB], round(s[columnA],2)), axis=1)     # axis=1 means apply by columns (axis=0 for rows)
                dfx['Label'] = dfx.apply(lambda s: s['DateTime'].strftime("%b %d") + "  ctgo:{0}={1}".format(s['Contango_Close'], round(s['Return_ES'],2)), axis=1)     # axis=1 means apply by columns (axis=0 for rows)
                i = len(traces) % len(colors_vga)
                line_color = colors_vga[i]
                #traces.append(get_trace(dfx, 'test', 'Close_ES', 'Contango_Close', line_color))
                traces.append(get_trace(dfx, 'test', columnA, columnB, line_color, marker_labels='Label'))
                #print dfx
        else:
            #if row[Q] >= gte:
            if f1(row[Q_colname], x):
                ix1 = ix
                print(row['DateTime'].strftime("%b %d %Y "), row[Q_colname], "   ",)
                wait_for_mean = hitratio_means[iq]
    print()
    chart_filename = join(html_folder, columnA + "_vs_contango_" + str(year) + ".html")
    #trace1 = get_trace(df, 'test', 'Close_ES', mode='markers')
    if f1 == fn_lte:
        compare_symbol = "<="
    else:
        compare_symbol = ">="
    display_chart(traces, chart_filename, "{0} vs VIX Contango {1} ({2}{3}{4})".format(columnA, year, Q_colname, compare_symbol, x))
    return

def get_high(row):
    col_unch = 13
    for i in range(col_unch+4, col_unch-5, -1):
        #print "high:", i-col_unch, row[i]
        if row[i] == 1:
            return i-col_unch

def get_low(row):
    col_unch = 13
    for i in range(col_unch-4, col_unch+5):
        #print "low:", i-col_unch, row[i]
        if row[i] == 1:
            return i-col_unch

def get_open(row):
    if row['SessionOpen_ES'] < row['SessionClose_ES']:
        return row['Low']
    else:
        return row['High']

def get_close(row):
    if row['SessionOpen_ES'] < row['SessionClose_ES']:
        return row['High']
    else:
        return row['Low']

def show_chart_ohlc(dfx, year, iq, f1, f2, x):
    print("Creating OHLC chart for {0}:".format(year))
    dt1 = datetime(year, 1, 1)
    dt2 = datetime(year, 12, 31)
    df = dfx[(dfx.DateTime >= dt1) & (dfx.DateTime <= dt2)]

    # Store hit_ratio mean values in a list
    df_hitratio = df_all.iloc[:,18:27]
    hitratio_means = df_hitratio.mean().values

    traces = []
    labels = []

    wait_for_mean = None

    for (ix,row) in df.iterrows():
        # skip initial rows which will not contain valid data
        if row.isnull().any():
            continue
        if wait_for_mean != None:
            if f2(row[Q_colname], wait_for_mean):
                ix2 = ix
                print(row[Q_colname], row['DateTime'].strftime(" %b %d %Y"), "    ", int(ix2 - ix1))
                wait_for_mean = None
                dfx = df.loc[ix1+1:ix2].copy()
                #dfx['Label'] = dfx.apply(lambda s: s['DateTime'].strftime("%b %d"), axis=1)     # axis=1 means apply by columns (axis=0 for rows)
                #dfx['Label'] = dfx.apply(lambda s: s['DateTime'].strftime("%b %d") + "  ctgo:{0}={1}".format(s[columnB], round(s[columnA],2)), axis=1)     # axis=1 means apply by columns (axis=0 for rows)
                dfx['Label'] = dfx.apply(lambda s: s['DateTime'].strftime("%b %d") + "  ctgo:{0}={1}".format(s['Contango_Close'], round(s['Return_ES'],2)), axis=1)     # axis=1 means apply by columns (axis=0 for rows)
                #dfx['Open'] = dfx['Open_ES']
                #dfx['Close'] = dfx['Close_ES']
                dfx['High'] = dfx.apply(lambda s: get_high(s), axis=1)
                dfx['Low'] = dfx.apply(lambda s: get_low(s), axis=1)
                dfx['Open'] = dfx.apply(lambda s: get_open(s), axis=1)
                dfx['Close'] = dfx.apply(lambda s: get_close(s), axis=1)
                #i = len(traces) % len(colors_vga)
                #line_color = colors_vga[i]
                #traces.append(get_trace(dfx, 'test', 'Close_ES', 'Contango_Close', line_color))
                traces.append(get_trace_ohlc(dfx))
                #print dfx
        else:
            if f1(row[Q_colname], x):
                ix1 = ix
                print(row['DateTime'].strftime("%b %d %Y "), row[Q_colname], "   ",)
                wait_for_mean = hitratio_means[iq]
    print()
    chart_filename = join(html_folder, "Quartile_Bars_ES_" + str(year) + ".html")
    display_chart(traces, chart_filename, "Quartile Bars ({0}>={1})".format(Q_colname, gte))
    return

# LOOK AT QUARTILE HIT_RATIOS (ex: hit_ratio for lower-quartile 1 is .8 until this hit_ratio returns to its mean)
# ix1 is the first index where our condition is met
# so (ix1+1) is the first day where we would check the trade
# and ix2 is the last day we would check the trade (hit_ratio has reverted to mean)
def get_quartile_indexes(dfx, year, iq, f1, f2, x):
    dt1 = datetime(year, 1, 1)
    dt2 = datetime(year, 12, 31)
    df = dfx[(dfx.DateTime >= dt1) & (dfx.DateTime <= dt2)]

    # Get hit_ratio values from global
    global df_hitratio
    hitratio_means = df_hitratio.mean().values

    date_ranges = []

    wait_for_mean = None

    for (ix,row) in df.iterrows():
        # skip initial rows which will not contain valid data
        if row.isnull().any():
            continue
        if wait_for_mean != None:
            if f2(row[Q_colname], wait_for_mean):
                ix2 = ix
                #dt1 = df.loc[ix1+1].DateTime
                #dt2 = df.loc[ix2].DateTime
                #tup = (dt1, dt2, int(ix2-ix1))
                tup = (ix1, ix2)
                date_ranges.append(tup)
                wait_for_mean = None
        else:
            if f1(row[Q_colname], x):
                ix1 = ix
                wait_for_mean = hitratio_means[iq]
    return date_ranges


"""
# LOOK AT QUARTILE HIT_RATIOS (ex: hit_ratio for lower-quartile 1 is .8 until this hit_ratio returns to its mean)
def fn_row1(row):
    # for 'iq': 0=d4  1=d3  2=d2  3=d1  4=unch  5=u1  6=u2  7=u3  8=u4
    return row['d1'] >= .8 and row['u1'] == .2

def fn_row2(row):
    global hitratio_means   # use hit_ratio values from global
    # for 'iq': 0=d4  1=d3  2=d2  3=d1  4=unch  5=u1  6=u2  7=u3  8=u4
    return row['d1'] <= hitratio_means['d1']
"""

# Given a dataframe, an integer year, and two functions (f1=start_range, f2=end_range)
# Return a list of tuples (ix1, ix2) which represent start/end indexes in the given dataframe
# ix1 is the first index where our condition is met
# so (ix1+1) is the first day where we would check the trade
# and ix2 is the last day we would check the trade (exit condition has been met)
def get_quartile_indexes2(dfx, dt1, dt2, f1, f2):
    df = dfx[(dfx.DateTime >= dt1) & (dfx.DateTime <= dt2)]

    date_ranges = []
    is_range_started = False

    for (ix,row) in df.iterrows():
        # skip initial rows which will not contain valid data
        if row.isnull().any():
            continue
        if is_range_started:
            if f2(row):
                ix2 = ix
                tup = (ix1, ix2)
                date_ranges.append(tup)
                is_range_started = False
        else:
            if f1(row):
                ix1 = ix
                is_range_started = True
    return date_ranges

# LOOK AT CONTANGO (ex: crossing below zero then back)
# ix1 is the first index where our condition is met
# so (ix1+1) is the first day where we would check the trade
# and ix2 is the last day we would check the trade (hit_ratio has reverted to mean)
def get_contango_indexes(dfx, year, column_name, f1, x1, f2, x2):
    dt1 = datetime(year, 1, 1)
    dt2 = datetime(year, 12, 31)
    df = dfx[(dfx.DateTime >= dt1) & (dfx.DateTime <= dt2)]

    # Get hit_ratio values from global
    global df_hitratio
    hitratio_means = df_hitratio.mean().values

    date_ranges = []

    wait_for_condition = None

    for (ix,row) in df.iterrows():
        # skip initial rows which will not contain valid data
        if row.isnull().any():
            continue
        if wait_for_condition != None:
            if f2(row[column_name], x2):
                ix2 = ix
                tup = (ix1, ix2)
                date_ranges.append(tup)
                wait_for_condition = None
        else:
            if f1(row[column_name], x1):
                ix1 = ix
                wait_for_condition = True
    return date_ranges

# Default to ES session (8:30am-3:00pm)
def trades_for_date(dfx, symbol, dt, enter_quartile, exit_quartile):
    #rows = []                                                   # store our row tuples here (they will eventually be used to create a dataframe)
    if enter_quartile < exit_quartile:
        buy = True
    else:
        buy = False     # sell

    r = dfx[dfx.DateTime == dt]
    std = Calc_Std(r.Prev_Close_VIX)
    print(r, std)
    q = Calc_Quartiles(r.Prev_Close_ES, std)
    # Convert the enter/exit quartiles (-4,-3,-2,...,2,3,4) into underlying price values
    enter_level = q[enter_quartile]
    exit_level = q[exit_quartile]
    print("E X:", enter_quartile, exit_quartile)

    print("Processing future:", symbol, dt.strftime("%b %d %Y"))
    df_day = get_session_1min_bars(symbol, dt)
    high = df_day.High.max()
    low = df_day.Low.min()

    dfx = dfx[dfx.DateTime==dt]
    print(dfx)

    print(df_day.shape)

    # df_day contains the 1-minute bars for the DAY SESSION
    in_trade = None
    for (ix,row) in df_day.iterrows():
        print(row)
        last_price = row.Last
        print("last price:", last_price)
        if buy:
            if in_trade != None:
                if row.max() >= exit_level:
                    print("TRADE!!!", enter_level, exit_level)
                    in_trade = None
            elif row.min() <= enter_level:
                in_trade = row.min()
        else:
            pass
            if row.max() >= enter_level:
                in_trade = True

        print(row.DateTime)

        #if not (ix+1) in dfx.index:                         # if we are at the end of the dataframe rows (next row doesn't exist)
        #    continue

        # Use Close of ES and VIX
        #es_close = row['Close_ES']                          # ES close
        #vix_close = row['Close_VIX']                        # VIX close
        #std = vix_close / math.sqrt(252)                    # calculate standard deviation
        #dt_prev = row['DateTime']                           # date of ES/VIX close to use
        #dt = dfx.loc[ix+1].DateTime                         # following date (of actual quartile calculation)

        # For each quartile, determine if it was hit during the day session of ES
        #hit_quartile = {}
        #print es_close, std
        #for i in range(+4, -5, -1):
            #quartile = round(Quartile(i, prev_close, std), 2)
            #if low <= quartile and high >= quartile:
            #    hit_quartile[i] = 1
            #else:
            #    hit_quartile[i] = 0

        #rows.append((dt, es, es_close, vix_close, std, hit_quartile[-4], hit_quartile[-3], hit_quartile[-2], hit_quartile[-1], hit_quartile[0], hit_quartile[1], hit_quartile[2], hit_quartile[3], hit_quartile[4]))

    #df_new = pd.DataFrame(rows, columns=['DateTime', 'Symbol_ES', 'Prev_Close_ES', 'Prev_Close_VIX', 'Std', 'Qd4', 'Qd3', 'Qd2', 'Qd1', 'Qunch', 'Qu1', 'Qu2', 'Qu3', 'Qu4'])
    return df_day

# Return the composite dataframe (df_all) that will be used for the analysis
# Given the quartile filename
# Given the continous symbol
# Given the contango symbol
def create_dataframe(quartile_filename="es_vix_quartiles (5-day lookback).csv", continuous_symbol='@ES', contango_symbol='@VX'):
    global project_folder
    #filename1 = "es_vix_quartiles ({0}-day lookback).csv".format(lookback_days)
    filename2 = continuous_symbol + "_continuous (Daily).csv"
    filename3 = contango_symbol + "_contango.csv"

    print("INPUT FILES")
    print("'{0}'".format(quartile_filename))
    print("'{0}'".format(filename2))
    print("'{0}'".format(filename3))

    # Read files into dataframe(s)
    df = read_dataframe(join(project_folder, quartile_filename))
    df2 = read_dataframe(join(project_folder, filename2))
    df3 = read_dataframe(join(project_folder, filename3))

    df2 = df2[['DateTime', 'Close']]
    df2.rename(columns={'Close':'Close_ES'}, inplace=True)
    df3 = df3[['DateTime', 'Contango_Close']]

    df = pd.merge(df, df2, on=['DateTime'])
    df = pd.merge(df, df3, on=['DateTime'])

    df['NetChg_ES'] = df['Close_ES'] - df['Prev_Close_ES']
    df['Return_ES'] = df['NetChg_ES'] / df['Prev_Close_ES'] * 100.0
    df['Ratio_Contango_ES'] = df['Contango_Close'] / df['Return_ES']
    return df

# Return the rows in a given dataframe that fall within the given start index (ix1) and end index (ix2)
# Return a tuple containing the dates associated with the start/end rows (d1/d2) and the rows themselves (as a list)
def get_rows(df, ix1, ix2):
    rows = []
    d1 = df_all.loc[ix1].DateTime
    d2 = df_all.loc[ix2].DateTime
    for ix in range(ix1, ix2 + 1):
        row = df_all.loc[ix]
        #df2.loc[ix] = row
        rows.append(row)
    return (d1, d2, rows)

# ranges_or_step can either be a list of range cutoffs [-5, 0, 5, 10] or an integer representing the step to use when calculating ranges
def get_distribution(df, column_name, ranges_or_step=1):
    dfc = df.loc[:, column_name:column_name]
    if dfc.empty:
        print("EMPTY!", dfc.empty, dfc)
    minimum = dfc.min().values[0]
    maximum = dfc.max().values[0]
    xmin = int(math.floor(minimum))
    xmax = int(math.ceil(maximum))

    if type(ranges_or_step) is int:
        ranges = []
        #ranges.append(xmin)
        #step_count = math.ceil((xmax - xmin) / ranges_or_step)
        for x in range(xmin, xmax, ranges_or_step):
            #x1 = x
            #x2 = x + ranges_or_step
            ranges.append(x)
        ranges.append(xmax)
    else:
        ranges = ranges_or_step
        ranges.insert(0, xmin)
        ranges.append(xmax)

    counts = []
    for i in range(0, len(ranges)-1):
        x1 = ranges[i]
        x2 = ranges[i+1]
        tup = ((x1, x2), dfc[(dfc.Contango_Close > x1) & (dfc.Contango_Close <= x2)].count().values[0])
        counts.append(tup)

    return (minimum, maximum, counts)

# Return the index of the range into which x falls
def get_range_index(counts, x):
    i = 0
    for ((x1,x2),count) in counts:
        if x > x1 and x <= x2:
            return i
        i += 1
    return -1

# The description of the analysis (study) we're running differs depending on the value of the "lte" arg
def get_study_criteria(lte):
    if lte != None:
        comparison_text = 'lte'
        comparison_symbol = '<='
        comparison_value = lte
    else:
        comparison_text = 'gte'
        comparison_symbol = '>='
        comparison_value = gte

    study_criteria = "When hit_ratio of quartile '{0}' {1} {2}, until '{0}' returns to mean ({3:.2f})".format(Q_colname, comparison_symbol, comparison_value, hitratio_mean)

    print()
    print("When hit_ratio of quartile '{0}' {1} {2}, show scatterplots:".format(Q_colname, comparison_symbol, comparison_value))
    print("(1) ES vs VIX contango")
    print("(2) ES vs hit_ratio")
    print("...until '{0}' hit_ratio returns to mean ('{0}' hit_ratio mean = {1:.2f})".format(Q_colname, hitratio_mean))
    print()
    return study_criteria

# Given the results of previously called fucntion get_distribution, print a representation of the distribution to the screen
def print_distribution(counts, show_numeric=False):
    maxlen = 0
    maxcount = 0
    for ((x1,x2),count) in counts:
        str1 = "{0}".format(x1)
        str2 = "{0}".format(x2)
        maxlen = int(max(maxlen, len(str1), len(str2)))
        maxcount = max(maxcount, count)
        max_cols_to_display = 60
        if maxcount > max_cols_to_display:
            count_divisor = maxcount / max_cols_to_display
        else:
            count_divisor = 1.0
    for ((x1,x2),count) in counts:
        if show_numeric == True:
            print("{0:4} |".format(count),)
        print("{x1:{width}} >{x2:{width}}  | {dots}".format(x1=x1, x2=x2, width=maxlen+1, dots='.'*int(count/count_divisor)))
    return

# Given a symbol and date, retrieve the 1-minute historical data for the day session trading hours
def get_session_1min_bars(symbol, dt):
    # TODO: these are ES session times (8:30am-3:00pm); change the hardcoded session times to read them from a function based on the SYMBOL
    session_open_hour=8; session_open_minute=30; session_close_hour=15; session_close_minute=0
    df_1min = read_dataframe(get_df_pathname(symbol))
    # Get the 1-minute bars for the day session (for specified date)
    session_open = dt.replace(hour=session_open_hour, minute=session_open_minute)
    session_close = dt.replace(hour=session_close_hour, minute=session_close_minute)
    df_day = df_1min[(df_1min.DateTime > session_open) & (df_1min.DateTime <= session_close)]   # 1-minute bars for day session
    return df_day

# Given a symbol and date, retrieve the 1-minute historical data for period prev-day-close to this-day-close
def get_close_to_close_1min_bars(symbol, dt):
    # TODO: these are ES session times (8:30am-3:00pm); change the hardcoded session times to read them from a function based on the SYMBOL
    session_open_hour=8; session_open_minute=30; session_close_hour=15; session_close_minute=0
    df_1min = read_dataframe(get_df_pathname(symbol))
    # Get the 1-minute bars from the previous day close until this day close
    session_close = dt.replace(hour=session_close_hour, minute=session_close_minute)

    # We need to do a little work to calculate a datetime for the close of the previous day
    df_1min['date_only'] = df_1min['DateTime'].map(lambda x: x.strftime('%Y-%m-%d'))
    li_dates = df_1min.date_only.unique()
    find = np.where(li_dates == dt.strftime('%Y-%m-%d'))   # result is a tuple with first all the row indices, then all the column indices
    irows = find[0]
    index = irows[0]  # first occurrence
    previous_date = li_dates[index - 1]
    dt_prev = datetime.strptime(previous_date, '%Y-%m-%d')
    prev_session_close = dt_prev.replace(hour=session_close_hour, minute=session_close_minute)

    df_day = df_1min[(df_1min.DateTime > prev_session_close) & (df_1min.DateTime <= session_close)]   # 1-minute bars for close-to-close
    return df_day

def tick_price(price, ticksize=.25, mult=100, round_up=False):
    p = price * mult
    tick = ticksize * mult
    mod = p % tick
    if round_up:
        p += (tick-mod)
    else:
        p -= mod
    return p / mult

def run_backtest(dfx, enter_quartile, exit_quartile, print_output=False):
    # TODO: these are ES session times (8:30am-3:00pm); change the hardcoded session times to read them from a function based on the SYMBOL
    session_open_hour=8; session_open_minute=30; session_close_hour=15; session_close_minute=0

    initial_contango = dfx.iloc[0].Contango_Close
    initial_ES = dfx.iloc[0].Close_ES                   # we're buying ES on the close of the first day in the dataframe
    dt = dfx.iloc[0].DateTime
    dt_enter_trade = dt.replace(hour=session_open_hour, minute=session_open_minute)
    #print dt_enter_trade, initial_contango, initial_ES

    if enter_quartile < exit_quartile:
        buy = True
    else:
        buy = False     # sell

    trades = []

    # trade_dict = { "enter": (symbol1, datetime1, 'B', '2452.75'), "exit": (symbol2, datetime2, 'S', '2455.80'), "result": 'close', "max_drawdown": 1.50 }
    # "result" can be 'close', 'stop', 'end_trade' (if we come to the end of our trade without stop or exit)
    trade_dict = {}

    in_trade = None
    max_drawdown = 0.0

    #for (ix,r) in dfx.loc[2:,:].iterrows():             # start with the second day in the provided dataframe
    for i in range(1, dfx.shape[0]):
        r = dfx.iloc[i]
        symbol = r.Symbol_ES
        dt = r.DateTime

        std = Calc_Std(r.Prev_Close_VIX)
        (qli,qdict) = Calc_Quartiles(r.Prev_Close_ES, std)
        # Convert the enter/exit quartiles (-4,-3,-2,...,2,3,4) into underlying price values
        enter_level = tick_price(qdict[enter_quartile])
        exit_level = tick_price(qdict[exit_quartile])
        #print "{0}  std={1:.2f}    E X: {2} {3}".format(strdate(dt,'-'), std, enter_level, exit_level)

        # Now we need to load the 1-min data for both this day AND the previous day
        # We will process the 1-min data from session close of previous day until session close of this day
        if print_output == True: print("Processing future: {0} {1}   (contango={2:.1f})".format(symbol, dt.strftime("%b-%d-%Y"), r.Contango_Close))
        df_day = get_close_to_close_1min_bars(symbol, dt)
        high = df_day.High.max()
        low = df_day.Low.min()

        # df_day contains the 1-minute bars for the CLOSE-TO-CLOSE
        for (ix2, r_1min) in df_day.iterrows():
            #print r_1min.DateTime
            if buy:
                if in_trade == None:                                                   # NOT in a BUY trade
                    if r_1min.Low <= enter_level:                                       # check if we hit our BUY trade ENTRY
                        in_trade = enter_level
                        trade_dict["enter"] = (symbol, r_1min.DateTime, 'B', in_trade)
                        trade_dict["contango"] = r.Contango_Close
                else:                                                                   # in a BUY trade
                    max_drawdown = max(max_drawdown, in_trade - r_1min.Low)
                    if r_1min.High >= exit_level:                                       # check if we hit our BUY trade EXIT
                        #print "TRADE!!!", enter_level, exit_level
                        in_trade = None
                        trade_dict["exit"] = (symbol, r_1min.DateTime, 'S', exit_level)
                        trade_dict["max_drawdown"] = -max_drawdown
                        trade_dict["result"] = 'close'
                        break
            else:
                if in_trade == None:                                                   # NOT in a SELL trade
                    if r_1min.High >= enter_level:                                      # check if we hit our SELL trade ENTRY
                        in_trade = enter_level
                        trade_dict["enter"] = (symbol, r_1min.DateTime, 'S', in_trade)
                        trade_dict["contango"] = r.Contango_Close
                else:                                                                   # in a SELL trade
                    max_drawdown = max(max_drawdown, r_1min.High - in_trade)
                    if r_1min.Low <= exit_level:                                        # check if we hit our SELL trade EXIT
                        #print "TRADE!!!", enter_level, exit_level
                        in_trade = None
                        trade_dict["exit"] = (symbol, r_1min.DateTime, 'B', exit_level)
                        trade_dict["max_drawdown"] = -max_drawdown
                        trade_dict["result"] = 'close'
                        break

        if 'result' in trade_dict:
            trades.append(trade_dict)
            # TODO: For now, only perform ONE trade within this range (can be updated later for multiple trades)
            break

    if 'enter' in trade_dict and not('exit' in trade_dict):
        r_1min = df_day.iloc[-1]
        if buy:
            exit_side = 'S'
            max_drawdown = max(max_drawdown, in_trade - r_1min.Low)
        else:
            exit_side = 'B'
            max_drawdown = max(max_drawdown, r_1min.High - in_trade)
        trade_dict["exit"] = (symbol, r_1min.DateTime, exit_side, r_1min.Close)
        trade_dict["max_drawdown"] = -max_drawdown
        trade_dict["result"] = 'end_trade'
        trades.append(trade_dict)

    return trades

# Print the summary of beginning our calculations
def begin_calculations():
    global df_hitratio
    print("----------BEGIN CALCULATIONS----------")
    print("hit_ratio of each quartile:")
    print("MEAN:")
    print(df_hitratio.mean())
    #print "\nMIN:"
    #print df_hitratio.min()
    #print "\nMAX:"
    #print df_hitratio.max()
    print()
    return

# Print the summary of beginning our chart creation (if any charts have been requested in the args)
def begin_charts(df):
    print("----------BEGIN CHARTS----------")
    if is_arg('chart_netchg'):
        for y in range(y1, y2+1):
            if lte != None:
                show_chart(df, 'NetChg_ES', 'Contango_Close', y, iq, fn_lte, fn_gte, lte)
            else:
                show_chart(df, 'NetChg_ES', 'Contango_Close', y, iq, fn_gte, fn_lte, gte)
    if is_arg('chart_return'):
        for y in range(y1, y2+1):
            if lte != None:
                show_chart(df, 'Return_ES', 'Contango_Close', y, iq, fn_lte, fn_gte, lte)
            else:
                show_chart(df, 'Return_ES', 'Contango_Close', y, iq, fn_gte, fn_lte, gte)
    if is_arg('chart_quartile'):
        for y in range(y1, y2+1):
            if lte != None:
                show_chart_ohlc(df, y, iq, fn_lte, fn_gte, lte)
            else:
                show_chart_ohlc(df, y, iq, fn_gte, fn_lte, gte)
    print()
    return

def begin_get_indexes(lte, gte):
    print("----------GETTING INDEXES---------")
    if lte != None:
        params = (fn_lte, fn_gte, lte)
    else:
        params = (fn_gte, fn_lte, gte)
    return params

# Print the trades from the backtest on the rows in the given dataframe
def print_backtest_trades(li_trades):
    for trd in li_trades:
        (symbol1,dt1,side1,price1) = trd['enter']
        (symbol2,dt2,side2,price2) = trd['exit']
        str_dt1 = dt1.strftime('%Y-%m-%d %H:%M')
        str_dt2 = dt2.strftime('%Y-%m-%d %H:%M')
        if side1 == 'B':
            profit = price2-price1
        else:
            profit = price1-price2
        print("TRADE: [{0}  {1}  {2}  {3:.2f}] [{4}  {5}  {6}  {7:.2f}]  {8:10}  P/L={9:7.2f}    max_drawdown={10:7.2f}".format(symbol1, str_dt1, side1, price1, symbol2, str_dt2, side2, price2, trd['result'], profit, trd['max_drawdown']))
    #print
    return

# Output (to file) the trades from the backtest on the rows in the given dataframe
def file_output_backtest_trades(description, li_trades):
    for trd in li_trades:
        file_output("{0},{1},{2}".format(description.strip(), trd["contango"], str_trade(trd)))
    return

# Given a trade_dict dictionary, return a string representing a summary of the trade
# trade_dict = { "enter": (datetime1, 'B', '2452.75'), "exit": (datetime2, 'S', '2455.80'), "result": 'close', "max_drawdown": 1.50 }
# "result" can be 'close', 'stop', 'end_trade' (if we come to the end of our trade without stop or exit)
def str_trade(trade):
    (symbol1,dt1,side1,price1) = trade['enter']
    (symbol2,dt2,side2,price2) = trade['exit']
    str_dt1 = dt1.strftime('%Y-%m-%d %H:%M:%S')
    str_dt2 = dt2.strftime('%Y-%m-%d %H:%M:%S')
    if side1 == 'B':
        profit = price2-price1
    else:
        profit = price1-price2
    return "[{0}  {1}  {2}  {3:.2f}] [{4}  {5}  {6}  {7:.2f}] [{8:10}] [P/L={9:7.2f}] [max_drawdown={10:7.2f}]".format(symbol1, str_dt1, side1, price1, symbol2, str_dt2, side2, price2, trade['result'], profit, trade['max_drawdown'])

# When calculating the periods where contango makes specific moves (ex: drops below zero, moves above zero)...
def print_contango_rows(df_rows):
    print(" " * 10, "d4   d3   d2   d1 unch   u1   u2   u3   u4   contango     ES")
    initial_ES = None
    for (ix,row) in df_rows.iterrows():
        file_output(get_contango_output(row, initial_ES))
        if initial_ES == None:
            initial_ES = row['Close_ES']
        print(row['DateTime'].strftime('%b-%d   '),)
        for x in row['d4':'u4'].values:
            print("{0:.1f} ".format(x),)
        print("{0:7.2f}".format(row['Contango_Close']),)
        print("    {0:7.2f}".format(row['Close_ES']),)
        print()
    print("_" * 90)
    return

# Create a comma-delimeted output string from a dataframe row
def get_contango_output(row, init_ES):
    # output("date,d4,d3,d2,d1,unch,u1,u2,u3,u4,contango,ES")
    text = row['DateTime'].strftime('%Y-%m-%d') + ','
    for x in row['d4':'u4'].values:
        text += "{0:.1f}".format(x) + ','
    text += "{0:.2f}".format(row['Contango_Close']) + ','
    text += "{0:.2f}".format(row['Close_ES']) + ','
    if init_ES != None:
        pct = 100 * (row['Close_ES'] - init_ES) / init_ES
        text += "{0:.2f}".format(pct)
    return text

# Output to file
def file_output(text):
    global fcout
    fout.write(text + '\n')
    return

# Print a (horizontal) distribution of the Contango values for the given dataframe
# Segregate the distribution based on the given dist_ranges list
# For example, dist_ranges=[0, 5, 10, 12.5, 15] represents (x < 0), (0 <= x < 5), (5 <= x < 10), (10 <= x < 12.5), (12.5 <= x < 15), (x >= 15)
def display_contango_distribution(df, dist_ranges, description):
    df.dropna(inplace=True)     # remove the N/A values from the dataframe
    (minimum, maximum, xcounts) = get_distribution(df, 'Contango_Close', dist_ranges)
    print("{0}    min: {1}  max: {2}".format(description, minimum, maximum))
    print_distribution(xcounts, True)
    print("#" * 90)
    print()
    return

# Compare function that performs < (less-than)
def fn_lt(x, y):
    return x < y

# Compare function that performs > (greater-than)
def fn_gt(x, y):
    return x > y

# Compare function that performs <= (less-than-or-equal-to)
def fn_lte(x, y):
    return x <= y

# Compare function that performs >= (greater-than-or-equal-to)
def fn_gte(x, y):
    return x >= y

def dash(n):
    return '-' * n
    
def quartile_backtest(df_all, dt1, dt2, f1, f2, description=" "):
    date_ranges = get_quartile_indexes2(df_all, dt1, dt2, f1, f2)
    match_count = len(date_ranges)
    if match_count > 0:
        print("{0} QUARTILE BACKTEST{1}{2}  {3} to {4}  {5}".format(dash(11), description, dash(4), strdate(dt1), strdate(dt2), dash(4)),)
        print("{0:3d} matches {1}".format(match_count, dash(53-len(description))))

    # RUN BACKTEST ON EACH DATE RANGE THAT MATCHED OUR CRITERIA
    #print
    df2 = pd.DataFrame(columns=df_all.columns, index=df_all.index)  # create df2 as an empty copy of df_all (structure only)
    for (ix1, ix2) in date_ranges:
        (d1, d2, rows) = get_rows(df_all, ix1, ix2)
        # Create a dataframe from the rows produced by the get_rows function (and concat it to df2)
        df_rows = pd.DataFrame(data=rows, columns=df_all.columns)  # , index=df_all.index)
        df2 = pd.concat([df2, df_rows])

        print("{0} to {1}  ({2} days)".format(strdate(d1, '-'), strdate(d2, '-'), ix2 - ix1 + 1))

        li_trades = run_backtest(df_rows, -1, 2)
        print_backtest_trades(li_trades)
        file_output_backtest_trades(description, li_trades)
    return df2

############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################

quartiles = ['d4', 'd3', 'd2', 'd1', 'unch', 'u1', 'u2', 'u3', 'u4']

#-------------------------------------------------------------------------------
#                               begin args
#-------------------------------------------------------------------------------
# Comment out next line when NOT testing:
args['test'] = 1

# If args['test'] exists, then initialize args with some test values:
if 'test' in args:
    #args.update(dict(start_year=2010, end_year=2017, quartile='d1', lte=0.2, lookback=5))  # lower-quartile 1 (d1) is <= 0.2
    args.update(dict(start_year=2010, end_year=2017, quartile='d1', gte=0.8, lookback=5))   # lower-quartile 1 (d1) is >= 0.8
    #args.update(dict(start_year=2010, end_year=2017, quartile='d1', gte=0.8, lookback=10)) # lookback of 10 days instead of 5 days
    #args.update(dict(chart_netchg=True, chart_return=True, chart_quartile=True))   # this will print the charts
    #args.update(dict(contango=True))    # perform contango analysis (or perform standard quartile analysis)
    print("\nTEST ARGS")
    print(args)
elif argc <= 1:
    print(ARGS_LIST)

print("date range: [{0}] to [{1}]".format(strdate(get_arg('start_date')), strdate(get_arg('end_date'))))
print("year range: {0} to {1}".format(get_arg('start_year'), get_arg('end_year')))
print()

(y1, y2, lte, gte, iq, lookback_days) = get_args()
#-------------------------------------------------------------------------------
#                                end args
#-------------------------------------------------------------------------------

# for 'iq': 0=d4  1=d3  2=d2  3=d1  4=unch  5=u1  6=u2  7=u3  8=u4

Q_colname = quartiles[iq]       # text of selected quartile (for use as column_name in dataframe)

# Create the combined dataframe used for this analysis given the quartile file, continous symbol, and contango symbol
quartile_filename = "es_vix_quartiles ({0}-day lookback).csv".format(lookback_days)
continuous_symbol = "@ES"
contango_symbol = "@VX"
df_all = create_dataframe(quartile_filename, continuous_symbol, contango_symbol)

df_hitratio = df_all.loc[:, 'd4':'u4']          # get the data (all the columns) containing hit ratios
hitratio_means = df_hitratio.mean()
hitratio_mean = hitratio_means[Q_colname]           # get the hit_ratio mean for the given quartile

# Create a 'study_criteria' string that shows an abbreviated form of the inputs used for this analysis
study_criteria = get_study_criteria(lte)
#print study_criteria

begin_calculations()
begin_charts(df_all)
params = begin_get_indexes(lte, gte)

dt1 = datetime(y1, 1, 1)
dt2 = datetime(y2, 12, 31)

# FOR CONTANGO ANALYSIS, CREATE AN OUTPUT FILE WITH A SUMMARY OF THE RESULTS
trades_filename = "setup+trades.vix_es.{0}.csv".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
output_filename = join(project_folder, trades_filename)
#df = pd.read_csv(join(project_folder, "setup+trades.vix_es.2017-08-18-13-35-16.csv"))
#sys.exit()
fcout = open(output_filename, 'w')
file_output("d4,d3,d2,d1,unch,u1,u2,u3,u4,Contango,Trade")

print()
hitratio_range = range(0, 10+1, 2)
for x4 in hitratio_range:
    x_u4 = (x4 / 10.0)
    for x3 in hitratio_range:
        x_u3 = (x3 / 10.0)
        for x2 in hitratio_range:
            x_u2 = (x2 / 10.0)
            for x1 in hitratio_range:
                x_u1 = (x1 / 10.0)
                initial_hitratio = 0.8
                fn_1 = lambda row: row['d1']>=initial_hitratio and row['u1']==x_u1 and row['u2']==x_u2 and row['u3']==x_u3 and row['u4']==x_u4
                fn_2 = lambda row: row['d1']<=hitratio_means['d1']
                #description = " d4={0},d3={1},d2={2},d3={3},unch={4},u1={5},u2={6},u3={7},u4={8} ".format('?','?','?',initial_hitratio, '?', x_u1, x_u2, x_u3, x_u4)
                description = " {0},{1},{2},{3},{4},{5},{6},{7},{8} ".format('NaN','NaN','NaN',initial_hitratio, 'NaN', x_u1, x_u2, x_u3, x_u4)
                quartile_backtest(df_all, dt1, dt2, fn_1, fn_2, description)
                #fout.close()
                #sys.exit()

fcout.close()
print("Trades output to file: '{0}'".format(trades_filename))
print()

sys.exit()
# FOR CONTANGO ANALYSIS, CREATE AN OUTPUT FILE WITH A SUMMARY OF THE RESULTS
output_filename = join(project_folder, "contango_analysis.csv")
fcout = open(output_filename, 'w')
file_output("date,d4,d3,d2,d1,unch,u1,u2,u3,u4,contango,ES,%ES")

df2 = pd.DataFrame(columns=df_all.columns, index=df_all.index)      # create df2 as an empty copy of df_all (structure only)

# DEAL WITH THE DATA ONE YEAR AT A TIME. WHY? BECAUSE I SAID SO.
total_match_count = 0
for year in range(y1, y2+1):
    df_year = pd.DataFrame(columns=df_all.columns, index=df_all.index)
    print("Getting date ranges for {0}:".format(year),)
    if is_arg('contango'):
        date_ranges = get_contango_indexes(df_all, year, 'Contango_Close', fn_lt, 0, fn_gt, 0)
        print("*** CONTANGO ANALYSIS ***")
    else:
        dt1 = datetime(year, 1, 1)
        dt2 = datetime(year, 12, 31)
        #date_ranges = get_quartile_indexes(df_all, year, iq, params[0], params[1], params[2])
        #date_ranges = get_quartile_indexes2(df_all, dt1, dt2, fn_row1, fn_row2)
        date_ranges = get_quartile_indexes2(df_all, dt1, dt2, fn_1, fn_2)
        print("*** QUARTILE BACKTEST ***")

    match_count = len(date_ranges)
    total_match_count += match_count
    print("{0:3d} matches".format(match_count))

    # PROCESS EACH DATE RANGE FOR THIS YEAR
    print()
    for (ix1,ix2) in date_ranges:
        (d1, d2, rows) = get_rows(df_all, ix1, ix2)
        # Create a dataframe from the rows produced by the get_rows function (and concat it to df2)
        df_rows = pd.DataFrame(data=rows, columns=df_all.columns)   #, index=df_all.index)
        df2 = pd.concat([df2,df_rows])
        df_year = pd.concat([df_year,df_rows])

        print("{0} to {1}  ({2} days)".format(strdate(d1,'-'), strdate(d2,'-'), ix2-ix1+1))

        if is_arg('contango'):
            print_contango_rows(df_rows)
        else:
            print_backtest_rows(df_rows)

    # Display distribution of contango values for the year
    #display_contango_distribution(df_year, dist_ranges=[0, 5, 10, 12.5, 15], description="CONTANGO FOR YEAR {0}".format(year))

print()
print("TOTAL MATCHES FOR {0} YEARS ({1}-{2}): {3}".format(y2-y1+1, y1, y2, total_match_count))
print()

# Display distribution of contango values for ALL years
display_contango_distribution(df2, dist_ranges=[0, 5, 10, 12.5, 15], description="")

fcout.close()




