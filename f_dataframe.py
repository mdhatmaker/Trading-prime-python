from __future__ import print_function
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
import pandas_datareader.data as web
import math
from os.path import basename, dirname

#-----------------------------------------------------------------------------------------------------------------------

from f_date import *    #sortable_symbol  # compare_calendar
from f_iqfeed import *
from f_calc import *

#-----------------------------------------------------------------------------------------------------------------------
pd.options.display.width = 160      # Pandas defaults to use only 80 text chars before word-wrapping

q_columns = ['d4','d3','d2','d1','unch','u1','u2','u3','u4']

def calc_quartile_hit_ratios(df, lookback):
    for col in q_columns:
        Qcol = 'Q' + col
        df[col] = df[Qcol].shift(1).rolling(lookback).mean()
    return

# Given a pathname to a csv file and a list of columns which should be treated as dates
# Return the pandas dataframe containing the data
# if no directory is given as part of pathname (only a filename is passed), use DF_DATA
def read_dataframe(pathname, date_columns=['DateTime'], display=True):
    if dirname(pathname) == '':
        pathname = join(df_folder, pathname)
    if display: print("Reading dataframe from '{0}' ... ".format(basename(pathname)), end="")
    df = pd.read_csv(pathname, parse_dates=date_columns)
    if display: print("Done.")
    return df

# Given dataframe and pathname of output csv file
# Write dataframe to csv file
# if no directory is given as part of pathname (only a filename is passed), use DF_DATA
def write_dataframe(df, pathname, myindex=False, date_only=False, display=True):
    if dirname(pathname) == '':
        pathname = join(df_folder, pathname)
    if date_only:
        date_format = '%Y-%m-%d'
    else:
        date_format = '%Y-%m-%d %H:%M:%S'
    df.to_csv(pathname, index=myindex, date_format=date_format)
    if display: print("Write dataframe to '{0}'".format(basename(pathname)))
    return

def append_dataframe(df, pathname, myindex=False, display=True):
    with open(pathname, 'a') as f:
        df.to_csv(f, index=myindex, header=False)
    if display: print("Append dataframe to '{0}'".format(basename(pathname)))
    return

# Given a dataframe and the Symbol column name within that dataframe that is used for the continuous contract
# Return a list of (symbol, last_date_before_roll)
def df_get_roll_dates(df, symbol_column='Symbol'):
    roll_dates = []
    symbols = df[symbol_column].unique()    # unique VX front-month symbols
    #np.delete(symbols, 0)                   # skip first symbol
    #symbols = np.delete(symbols, len(symbols)-1)      # skip last symbol (because last date of the dataframe is not necessarily a roll date)
    for symbol in symbols:
        dfx = df[df[symbol_column]==symbol]
        first, last = df_first_last(dfx)
        if symbol == symbols[0]:
            roll_dates.append((symbol, None, last['DateTime']))         # for first symbol, first date of the dataframe is not necessarily a roll date
        elif symbol == symbols[-1]:
            roll_dates.append((symbol, first['DateTime'], None))        # for last symbol, last date of the dataframe is not necessarily a roll date
        else:
            roll_dates.append((symbol, first['DateTime'], last['DateTime']))
    return roll_dates

# Given a dataframe with a 'DateTime' column
# Return a list of the unique DATES (date only, ignoring time)
def df_get_unique_dates(df):
    unique_dates = df['DateTime'].dt.date.unique()
    unique_dates.sort()
    return unique_dates

# Given dataframe and column_name and indicator of whether or not to sort results and (optional) sort compare_function
# Return list of UNIQUE values in specified column (sorted if do_sort is True)
def df_get_unique(df, column_name, do_sort=True, key_function=None): # compare_function=None):
    g = df.groupby(column_name).groups
    keys = g.keys()
    if do_sort:
        if key_function == None:
            return sorted(keys)
        else:
            #return sorted(keys, key=compare_function)
            return sorted(keys, key=lambda x: key_function(x))
            #return sorted(keys, compare_function)
    else:
        return keys

# Given a dataframe
# Return the first row in the dataframe
def df_first(df):
    if df.empty:
        return None
    else:
        return df.iloc[0]

# Given a dataframe
# Return the last row in the dataframe
def df_last(df):
    if df.empty:
        return None
    else:
        return df.iloc[df.shape[0]-1]

# Given a dataframe
# Return the first and last rows in the dataframe (as a tuple)
def df_first_last(df):
    return df_first(df), df_last(df)

# Given a dataframe
# Return the index of the first row and the last row (as a tuple)
# returns -1 for indexes if the dataframe is empty (there are no first/last row)
def df_get_first_last_index(df):
    if df.empty: return (-1, -1)
    lix = df.index
    return (lix[0], lix[-1])

# Given an existing dataframe (df) and a dataframe with (potentially) newer rows (df_new) AND a list of column names
# Return the original dataframe (df) with rows removed that match for the given column names
def df_remove_duplicates(df, df_new, columns):
    df1 = df.reset_index(drop=True).set_index(columns)
    df2 = df_new.reset_index(drop=True).set_index(columns)
    dfx = df1[df1.index.isin(df2.index)]
    return df1.drop(dfx.index).reset_index()

# Given a dataframe (df) and session start time, length, and intervals to create
# Return an array of the dataframe slices that belong to these sessions
def df_get_sessions(df, session_start=time(20,0), session_length=timedelta(hours=21), session_interval=timedelta(hours=1)):
    #print("Splitting data into sessions ...", end='')
    unique_dates = df_get_unique_dates(df)
    sessions = []
    for dt in unique_dates:
        dt1 = datetime(dt.year, dt.month, dt.day, session_start.hour, session_start.minute, 0)
        dt2 = dt1 + session_length - session_interval
        df_sess = df[(df['DateTime']>=dt1) & (df['DateTime']<=dt2)]     # all rows in session (between dt1 and dt2)
        if df_sess.shape[0] > 0:
            sessions.append(df_sess)
    #print("Done.")
    return sessions

# Given a dataframe (df) and an array of sessions (see df_get_sessions above)
# Return last row from each session (effective close)
def get_last_session_rows(df, sessions):
    #print("Getting last row from each session (effective close) ...", end='')
    df_day = pd.DataFrame()
    for df_sess in sessions:
        df_temp = df.loc[df_sess.index, :]
        df_day = pd.concat([df_day, df_temp.tail(1)])
    #print("Done.")
    return df_day

"""
# Given dataframe
# 
def df_get_unique_dates(df):
    unique_dates = []
    for dt in hist[symbol].keys():
        dateonly = datetime(dt.year, dt.month, dt.day, 0, 0, 0)
        if not dateonly in unique_dates:
            unique_dates.append(dateonly)
    return unique_dates
"""

# Given the pathname to a dataframe containing OHLC columns
# Return a tuple (passed, df_errant) where passed is True/False and df_errant is the dataframe of rows that fail sanity check
def df_pass_sanity_check(df):
    dfx = df[(df.High<df.Open) | (df.High<df.Low) | (df.High<df.Close) | (df.Low>df.Open) | (df.Low>df.High) | (df.Low>df.Close)]
    if dfx.empty:
        return True, dfx
    else:
        return False, dfx

# Given a dataframe and two lists of column names (old/existing and new/desired)
# Return a dataframe with the columns renamed as specified
# if you do not pass in old_names_list and new_names_list, this function will sort the columns so 'DateTime','Symbol' are first
def df_rename_columns(df, old_names_list=None, new_names_list=None):
    #old_names = ['$a', '$b', '$c', '$d', '$e']
    #new_names = ['a', 'b', 'c', 'd', 'e']
    if old_names_list is None and new_names_list is None:
        df.rename(columns={"Date": "DateTime"}, inplace=True)
        original_cols = df.columns.tolist()
        cols = ['DateTime', 'Symbol']
        cols.extend(original_cols[1:-1])
        df = df[cols]
        df.reset_index(drop=True, inplace=True)
    else:
        df.rename(columns=dict(zip(old_names_list, new_names_list)), inplace=True)
    return df

# Given a dataframe and a dictionary of old_column_name/new_column_name pairs
# Return the dataframe with the specified columns renamed
def df_rename_columns_dict(df, names_dict):
    #df = df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'})
    df = df.rename(columns=names_dict)
    return df

# Given a dataframe with a 'Symbol' column
# Return the same dataframe sorted by 'Symbol', 'DateTime' (where symbol is sorted correctly as 'XXXmYY')
def df_sort_mYY_symbols_by_date(df):
    df['sort'] = df['Symbol'].apply(lambda x: x[4:6] + x[3])
    df.sort_values(["sort", "DateTime"], ascending=True, inplace=True)
    df.drop(['sort'], axis=1, inplace=True)
    return

# Given dataframe
# Return list of (unique) symbols sorted by their calendar dates
def df_get_sorted_symbols(df, symbol_column='Symbol'):
    #g = df.groupby('Symbol').groups
    #keys = g.keys()
    #sorted_symbols = sorted(keys, compare_calendar)
    sorted_symbols = df_get_unique(df, symbol_column, do_sort=True, key_function=sortable_symbol)
    return sorted_symbols

# Given dataframe and start/end datetime values
# Return the dataframe filtered to contain only rows in datetime range (startdate <= DateTime < enddate)
def df_filter_dates(df, dt0, dt1):
    dfx = df[df['DateTime'] >= dt0]
    dfx = dfx[dfx['DateTime'] < dt1]
    return dfx

# Given dataframe and column_name
# Print summary statistics about the given column's data
def df_print_summary_stats(df, column_name):
    fmt = "0.2f"
    print()
    print(column_name.upper())
    print("min:", format(df[column_name].min(), fmt))
    print("max:", format(df[column_name].max(), fmt))
    print("mean:", format(df[column_name].mean(), fmt))
    print("median:", format(df[column_name].median(), fmt))
    print("stddev:", format(df[column_name].std(), fmt))
    #print "mode:"
    #print df[col].mode()
    #print "quantile:"
    #print df[col].quantile([0.25, 0.75])
    return

# Given a symbol root ("@ES", "QHO", etc.) and a roll function
# create a dataframe with the given output filename with the correct continuous front-month contracts
# input filename will be of format "XXX_futures.<interval>.DF.csv"
# output filename will be of format "XXX_continuous.<interval>.DF.csv"
# (function uses 'DateTime' and 'Symbol' column to do its magic)
def create_continuous_df(symbol_root, fn_roll_date, roll_day_adjust=0, interval=INTERVAL_DAILY, directory=df_folder, sanity_check=True):
    input_filename = "{0}_futures.{1}.DF.csv".format(symbol_root, str_interval(interval))
    output_filename = "{0}_continuous.{1}.DF.csv".format(symbol_root, str_interval(interval))

    # Read in the continuous futures file (ex: "contango_@VX.raw.DF.csv" or "@VX_futures.minute.DF.csv")
    df = read_dataframe(join(directory, input_filename))

    # We will build a new DataFrame that has the rows for which the front-month contract is correct
    # (check roll dates)
    dfz = pd.DataFrame()

    # The dataframe should be sorted by Symbol-then-Datetime, so these unique Symbols should be sorted also
    unique = df.Symbol.unique()

    dt1 = df.DateTime.min()
    for symbol in unique:
        roll_date = fn_roll_date(symbol, roll_day_adjust=roll_day_adjust)
        dfx = df[(df.DateTime >= dt1) & (df.DateTime < roll_date) & (df.Symbol==symbol)]
        if dfz.shape[0] == 0:
            dfz = dfx.copy()
        else:
            dfz = dfz.append(dfx)
        #print(symbol, dt1, roll_date)
        dt1 = roll_date
    print()
    print("Rows in RAW: {0}    Rows in CONTINUOUS: {1}".format(df.shape[0], dfz.shape[0]))

    #dfz = df_rename_columns(dfz)      # move 'DateTime','Symbol' to first columns in dataframe

    if sanity_check:
        passed, df_failed = df_pass_sanity_check(dfz)
        if not passed:
            raise ValueError("Continuous data for {0} failed sanity check:\n{1}".format(symbol_root, df_failed))

    # No longer RAW...
    # We now have a file in which the dates have the correct front-month contract (ex: 'contango_@VX.DF.csv')
    #if interval == INTERVAL_DAILY:
    #    date_only = True
    #else:
    #    date_only = False
    write_dataframe(dfz, join(directory, output_filename), date_only=(interval==INTERVAL_DAILY))
    return dfz

# Given a symbol root (ex: "@VX") and month numbers for the calendar front month (mx) and calendar back month (my)
# create a dataframe file containing the continuous calendar spread ETS prices (ex: "@VX_continuous_calendar-mx-my.hour.DF.csv")
# (expects existing continuous futures file which it uses to retrieve the appropriate front months ex:"@VX_continuous.hour.DF.csv")
# (expects existing calendar ETS futures file from which it will pull the appropriate calendar ETS for each date ex:"@VX_calendar-m0-m1.hour.DF.csv")
def create_continuous_calendar_ETS_df(symbol_root, mx, my, interval=INTERVAL_DAILY, directory=df_folder):
    continuous_filename = "{0}_continuous.{1}.DF.csv".format(symbol_root, str_interval(interval))
    calendar_filename = "{0}_calendar-m{1}-m{2}.{3}.DF.csv".format(symbol_root, mx, my, str_interval(interval))
    output_filename = "{0}_continuous_calendar-m{1}-m{2}.{3}.DF.csv".format(symbol_root, mx, my, str_interval(interval))
    df_cont = read_dataframe(join(directory, continuous_filename))
    df = read_dataframe(join(directory, calendar_filename))
    front_month_symbols = df_cont.Symbol.unique()
    dfz = pd.DataFrame()

    for symbol in front_month_symbols:
        symbol0 = next_month_symbol_x(symbol, mx)       # this will equal symbol if mx is zero
        symbol1 = next_month_symbol(symbol0)
        calendar_symbol = symbol0 + '-' + symbol1
        #print(symbol0, symbol1, mx, my, calendar_symbol)
        df0 = df_cont[df_cont.Symbol == symbol].drop(['Open','High','Low','Close','Volume','oi'], axis=1)
        df1 = df[df.Symbol == calendar_symbol]
        dfx = pd.merge(df0, df1, on="DateTime")
        if dfx.shape[0] > 0:
            print(dfx.head(1)['DateTime'].values[0], dfx.tail(1)['DateTime'].values[0], symbol0, symbol1, calendar_symbol, dfx.shape[0])
            if dfz.shape[0] == 0:
                dfz = dfx.copy()
            else:
                dfz = dfz.append(dfx)
        dt = datetime.now() - timedelta(days=3)
        last_row = df_last(dfz)
        if last_row is not None and last_row['DateTime'] >= dt: break   # TODO: this is not quite right...should probably use roll date
    # At this point, dfz should contain the ETS spread data
    dfz.drop(['Symbol_x'], axis=1, inplace=True)
    dfz.rename(columns={'Symbol_y':'Symbol'}, inplace=True)
    write_dataframe(dfz, join(directory, output_filename), date_only=(interval==INTERVAL_DAILY))
    return dfz

# Given a symbol root (ex: "@VX")
# create a dataframe file containing the contango values (ex: "@VX_contango.hour.DF.csv")
# this file includes "front" contango (1m0 x 1m1), "next" contango (1m1 x 1m2), and the 1m0x3m1x2m2 contango
# (expects existing continuous dataframe file has been created which it uses to calculate the appropriate months)
def create_contango_df(symbol_root, interval=INTERVAL_DAILY, directory=df_folder, round=2):
    continuous_filename = "{0}_continuous.{1}.DF.csv".format(symbol_root, str_interval(interval))
    futures_filename = "{0}_futures.{1}.DF.csv".format(symbol_root, str_interval(interval))
    output_filename = "{0}_contango.{1}.DF.csv".format(symbol_root, str_interval(interval))
    df_cont = read_dataframe(join(directory, continuous_filename))
    df = read_dataframe(join(directory, futures_filename))
    front_month_symbols = df_cont.Symbol.unique()
    dfz = pd.DataFrame()

    for symbol0 in front_month_symbols:
        symbol1 = next_month_symbol(symbol0)
        symbol2 = next_month_symbol(symbol1)
        df0 = df_cont[df_cont.Symbol == symbol0]
        df1 = df[df.Symbol == symbol1]
        df2 = df[df.Symbol == symbol2]
        dfx = pd.merge(df0, df1, on="DateTime")
        dfx = pd.merge(dfx, df2, on="DateTime")
        if dfx.shape[0] > 0:
            print(dfx.head(1)['DateTime'].values[0], dfx.tail(1)['DateTime'].values[0], symbol0, symbol1, symbol2, dfx.shape[0])
            if dfz.shape[0] == 0:
                dfz = dfx.copy()
            else:
                dfz = dfz.append(dfx)
    # At this point, dfz should contain the consolidated contango data (3 months at each DateTime)
    dfz.drop(['Open_x','High_x','Low_x','Volume_x','oi_x','Open_y','High_y','Low_y','Volume_y','oi_y','Open','High','Low','Volume','oi'], axis=1, inplace=True)
    dfz.rename(columns={'Close_x':'Close0', 'Symbol_x':'Symbol', 'Close_y':'Close1', 'Symbol_y':'Symbol1', 'Close':'Close2', 'Symbol':'Symbol2'}, inplace=True)
    dfz['m0_m1'] = (dfz.Close1 - dfz.Close0).round(round)
    dfz['m1_m2'] = (dfz.Close2 - dfz.Close1).round(round)
    dfz['1x3x2'] = ((2 * dfz['m1_m2']) - dfz['m0_m1']).round(round)
    dfz['contango'] = (dfz['m0_m1'] / dfz.Close0 * 100).round(round)
    dfz['contango_m1_m2'] = (dfz['m1_m2'] / dfz.Close1 * 100).round(round)
    #dfz['contango_1x3x2'] = ((2 * dfz['contango_m1_m2']) - dfz['contango']).round(round)
    dfz['contango_1x3x2'] = (((2 * dfz['m1_m2'] / dfz.Close0) - dfz['m0_m1'] / dfz.Close0) * 100).round(round)
    #dfz.contango1 = dfz.contango1.round(2)
    #dfz.contango2 = dfz.contango2.round(2)
    #dfz.contango1x3x2 = dfz.contango1x3x2.round(2)
    #dfz.contango = dfz.contango.round(2)
    write_dataframe(dfz, join(directory, output_filename), date_only=(interval==INTERVAL_DAILY))
    return dfz

# Given a symbol root ('@ES', '@VX', etc.)
# This function does the bulk of the work: It calculates the quartile hits/ratios and outputs them to a quartiles dataframe file.
# (output filename similar to "ES_VIX_quartiles (5-day lookback).DF.csv")
def create_quartile_df(symbol_root, vsymbol, sym, vsym, session_open_time, session_close_time, lookback_days=5, ticksize=0.0):
    filename1 = "{0}_continuous.daily.DF.csv".format(symbol_root)
    filename2 = "{0}_contract.daily.DF.csv".format(vsymbol)
    output_filename = "{0}_{1}_quartiles ({2}-day lookback).DF.csv".format(sym, vsym, lookback_days)

    #print "project folder: '{0}'".format(project_folder)
    print("input files: '{0}'  '{1}'".format(filename1, filename2))
    print("lookback days: {0}".format(lookback_days))
    print()

    df_es = read_dataframe(filename1)
    df_es = df_es.drop(['High', 'Low'], 1)
    df_es.rename(columns={'Symbol': 'Symbol_' + sym, 'Open': 'Open_' + sym, 'Close': 'Close_' + sym, 'Volume': 'Volume_' + sym, 'oi': 'oi_' + sym}, inplace=True)

    df_vix = read_dataframe(filename2)
    df_vix = df_vix.drop(['High', 'Low', 'Volume'], 1)
    df_vix.rename(columns={'Symbol': 'Symbol_' + vsym, 'Open': 'Open_' + vsym, 'Close': 'Close_' + vsym, 'oi': 'oi_' + vsym}, inplace=True)

    df = pd.merge(df_es, df_vix, on=['DateTime'])

    # Output this quartile summary data (for potential analysis or debugging)
    summary_filename = "{0}_{1}_quartile_summary.DF.csv".format(sym, vsym)
    write_dataframe(df, summary_filename)
    print("{0}/{1} summary (daily data) output to file: '{2}'".format(vsym, sym, summary_filename))
    print()

    symbol_column = 'Symbol_' + sym
    close_column = 'Close_' + sym
    vclose_column = 'Close_' + vsym

    symbols = df_get_sorted_symbols(df, symbol_column)  # get list of unique futures symbols in our data

    rows = []  # store our row tuples here (they will eventually be used to create a dataframe)

    df_1min = read_dataframe("{0}_futures.minute.DF.csv".format(symbol_root))  # read in the 1-minute data

    # For each specific future symbol, perform our quartile calculations
    for symbol in symbols:
        print("Processing future:", symbol)
        df_es = df_1min[df_1min.Symbol == symbol]  # read_dataframe(get_df_pathname(es))
        dfx = df[df[symbol_column] == symbol]
        for (ix, row) in dfx.iterrows():
            if not (ix + 1) in dfx.index:  # if we are at the end of the dataframe rows (next row doesn't exist)
                continue

            # Use Close of ES and VIX
            es_close = row[close_column]  # ES close
            vix_close = row[vclose_column]  # VIX close
            std = round(Calc_Std(vix_close), 4)  # calculate standard deviation
            dt_prev = row['DateTime']  # date of ES/VIX close to use
            dt = dfx.loc[ix + 1].DateTime  # following date (date of actual quartile calculation)

            # Get the 1-minute bars for the day session (for date following the date of ES/VIX close)
            exchange_open = dt.replace(hour=session_open_time.hour, minute=session_open_time.minute)
            exchange_close = dt.replace(hour=session_close_time.hour, minute=session_close_time.minute)
            df_day = df_es[(df_es.DateTime > exchange_open) & (
            df_es.DateTime <= exchange_close)]  # ES 1-minute bars for day session

            # Get OHLC for the day session 1-minute data
            day_open = df_day.iloc[0]['Open']
            day_high = max(df_day.High.max(), day_open)  # check in case the open is higher
            day_low = min(df_day.Low.min(), day_open)  # check in case the open is lower
            row_count = df_day.shape[0]
            day_close = df_day.iloc[row_count - 1]['Close']

            # For each quartile, determine if it was hit during the day session
            hit_quartile = {}
            (q_list, q_dict) = Calc_Quartiles(es_close, std, ticksize=ticksize)
            for i in range(+4, -5, -1):
                quartile = q_dict[i]
                if day_low <= quartile and day_high >= quartile:
                    hit_quartile[i] = 1
                else:
                    hit_quartile[i] = 0
                    # print i, quartile, hit_quartile[i]

            rows.append((dt, symbol, es_close, vix_close, std, day_open, day_high, day_low, day_close, hit_quartile[-4],
                         hit_quartile[-3], hit_quartile[-2], hit_quartile[-1], hit_quartile[0], hit_quartile[1],
                         hit_quartile[2], hit_quartile[3], hit_quartile[4]))

    # Create new Quartile dataframe from the rows we have calculated
    dfQ = pd.DataFrame(rows, columns=['DateTime', 'Symbol', 'Prev_Close', 'Prev_VClose', 'Std', 'Open_Session',
                                      'High_Session', 'Low_Session', 'Close_Session', 'Qd4', 'Qd3', 'Qd2', 'Qd1',
                                      'Qunch', 'Qu1', 'Qu2', 'Qu3', 'Qu4'])
    calc_quartile_hit_ratios(dfQ, lookback_days)
    dfQ.dropna(inplace=True)
    write_dataframe(dfQ, output_filename)
    print("Quartile analysis output to file: '{0}'".format(output_filename))
    print()
    return dfQ


# Given a dataframe of 1-minute data and the session open/close times, calculate the Open,High,Low,Close for each session
def get_ohlc_df(df, session_open='09:30:00', session_close='16:00:00', display=True):
    if display: print("Calculating OHLC from 1-minute data ... ", end="")
    df['just_date'] = df['DateTime'].dt.date
    unique_dates = df['just_date'].unique()
    rows = []
    for dt in unique_dates:
        (hh, mm, ss) = parse_timestr(session_open)
        dt1 = datetime(dt.year, dt.month, dt.day, hh, mm, ss)
        (hh, mm, ss) = parse_timestr(session_open)
        dt2 = datetime(dt.year, dt.month, dt.day, hh, mm, ss)
        dfx = df[(df.DateTime > dt1) & (df.DateTime <= dt2)]
        if dfx.shape[0] > 0:
            xopen = dfx.iloc[0]['Open']
            xhigh = dfx.High.max()
            xlow = dfx.Low.min()
            xclose = dfx.iloc[dfx.shape[0] - 1]['Close']
            rows.append([dt, xopen, xhigh, xlow, xclose])
    if display: print()
    return pd.DataFrame(rows, columns=['DateTime', 'Open', 'High', 'Low', 'Close'])


# Given a full dataset and a subset of the full dataset
# Return a list of df_subset rows where the rows are contiguous ranges
# this seems a bit difficult to understand, but it's actually quite simple:
# you have a dataframe (df_full), you use some analysis to pull only certain rows from that dataframe (df_subset)
#  this function will determine which rows from df_subset are contiguous ranges (based on rows in df_full)
def get_contiguous_ranges_df(df_full, df_subset):
    df_rows = []
    dt1 = None
    dt2 = None
    for ix, r in df_subset.iterrows():
        if dt1 == None:
            dt1 = r.DateTime
        else:  # dt2 == None:
            dt2 = r.DateTime

        if dt1 != None and dt2 != None:
            dfx = df_full[(df_full.DateTime >= dt1) & (df_full.DateTime <= dt2)]
            dfy = df_subset[(df_subset.DateTime >= dt1) & (df_subset.DateTime <= dt2)]
            nx = dfx.shape[0]
            ny = dfy.shape[0]
            if nx == ny:
                pass
            else:
                df_rows.append(dfy.iloc[0:dfy.shape[0] - 1].copy())
                dt1 = dt2
                dt2 = None
    if dt1 != None and dt2 == None:
        df_rows.append(df_subset[df_subset.DateTime >= dt1])
    return df_rows


############################## QUARTILE DATAFRAME FUNCTIONS #############################
# These are the base names of the dataframe columns used to store quartile data
quartile_columns = ['d4', 'd3', 'd2', 'd1', 'unch', 'u1', 'u2', 'u3', 'u4']


# Given two DAILY dataframes (dfx=underlying price, dfvol=volatility index) calculate the quartile values
# Return a dataframe containing the original data AND the quartile value columns ('Qd4'...'Qunch'...'Qu4')
# (optional) suffixes for merge of two dataframes defaults to standard ('_x','_y') but could be like ('_ES','_VIX')
def get_quartiles_df(dfx, dfvol, suffixes=('_x', '_y')):
    df = pd.merge(dfx, dfvol, on=["DateTime"], suffixes=suffixes)
    # Get column names of the two columns that contain closing prices for the underlying (x_col) and the volatility index (vol_col)
    x_col = "Close" + suffixes[0]
    vol_col = "Close" + suffixes[1]
    df['std'] = df[vol_col] / math.sqrt(252)  # calculate the standard deviation using the volatility number
    i = -4
    # Loop through each of the quartile columns calculating the value for each (i is incremented to act like range(-4, 5))
    column_names = ['Q'+col for col in quartile_columns]
    for qcol in column_names:
        df[qcol] = (df[x_col] + i * (df['std'] / 100.0 * df[x_col] / 4.0)).round(2)
        i += 1
    df.loc[:, column_names] = df.loc[:, column_names].shift(
        1)  # shift down because these represent the valid quartiles for the FOLLOWING day
    # df.loc[:,'Qd4':'Qu4'] = df.loc[:,'Qd4':'Qu4'].shift(1)
    df.drop(0, inplace=True)  # remove the first row (which contains NaN after shift down)
    return df


# Given a dataframe of Quartiles and a dataframe of 1-minute OHLC summary data
# Return a dataframe of Quartile Hits (0 or 1)
def get_quartile_hits_df(dfq, df_ohlc):
    df = pd.merge(dfq, df_ohlc, on='DateTime')
    for col in quartile_columns:
        if col == 'unch': continue              # skip the "unch" column--we'll handle that differently
        df[col] = (df.Low <= df['Q'+col])
        df[col] = df[col].astype('int')

    df['unch'] = (((df.Open <= df.Qunch) & (df.High >= df.Qunch)) | ((df.Open >= df.Qunch) & (df.Low <= df.Qunch)))
    df['unch'] = df['unch'].astype('int')
    """
    df['d4'] = df.Low<=df.Qd4
    df['d3'] = df.Low<=df.Qd3
    df['d2'] = df.Low<=df.Qd2
    df['d1'] = df.Low<=df.Qd1
    df['unch'] = (((df.Open<=df.Qunch)&(df.High>=df.Qunch)) | ((df.Open>=df.Qunch)&(df.Low<=df.Qunch)))
    df['u1'] = df.High>=df.Qu1
    df['u2'] = df.High>=df.Qu2
    df['u3'] = df.High>=df.Qu3
    df['u4'] = df.High>=df.Qu4
    df['d4'] = df['d4'].astype('int')
    df['d3'] = df['d3'].astype('int')
    df['d2'] = df['d2'].astype('int')
    df['d1'] = df['d1'].astype('int')
    df['unch'] = df['unch'].astype('int')
    df['u1'] = df['u1'].astype('int')
    df['u2'] = df['u2'].astype('int')
    df['u3'] = df['u3'].astype('int')
    df['u4'] = df['u4'].astype('int')
    """
    column_names = ['DateTime']
    column_names.extend(quartile_columns)
    dfz = df[column_names].copy()
    return dfz


# Given a dataframe of Quartile Hits (0 or 1 for columns 'd4'...'unch'...'u4')
# Return a dataframe of Quartile Hit Ratios (0.0-1.0 for columns 'hr_d4'...'hr_unch'...'hr_u4')
# (optional) lookback defaults to 5 (days to look back when calculating hit ratio mean)
def get_quartile_hit_ratios_df(df_vx, df_hit, lookback=5):
    df = pd.merge(df_vx, df_hit, on='DateTime')
    #column_names = ['hr_'+col for col in quartile_columns]
    for col in quartile_columns:
        hrcol = 'hr_'+col
        df[hrcol] = df[col].rolling(lookback).mean()
    """
    df['hr_d4'] = df.d4.rolling(lookback).mean()
    df['hr_d3'] = df.d3.rolling(lookback).mean()
    df['hr_d2'] = df.d2.rolling(lookback).mean()
    df['hr_d1'] = df.d1.rolling(lookback).mean()
    df['hr_unch'] = df.unch.rolling(lookback).mean()
    df['hr_u1'] = df.u1.rolling(lookback).mean()
    df['hr_u2'] = df.u2.rolling(lookback).mean()
    df['hr_u3'] = df.u3.rolling(lookback).mean()
    df['hr_u4'] = df.u4.rolling(lookback).mean()
    """
    column_names = ['DateTime','contango']
    column_names.extend(quartile_columns)
    column_names.extend(['hr_'+col for col in quartile_columns])
    dfz = df[column_names].copy()
    dfz.dropna(inplace=True)
    return dfz

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
def df_get_contango_range(df, column='contango', range=(None, None), padnan=True):
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
        dfx = df_get_contango_range(df, 'contango', contango_ranges[i], padnan=False)
        df.loc[dfx.index, 'contango_range'] = int(i)
    df['contango_range'] = df['contango_range'].astype('int')
    return df

# Given a dataframe (df) and a number of days lookback
# Create columns (names starting with 'Q') that contain the Quartile "hit ratio"
def df_calc_quartile_hit_ratios(df, lookback):
    for col in q_columns:
        Qcol = 'Q' + col
        df[col] = df[Qcol].shift(1).rolling(lookback).mean()
    return



