from __future__ import print_function


#-----------------------------------------------------------------------------------------------------------------------

from f_folders import *
from f_dataframe import *

#-----------------------------------------------------------------------------------------------------------------------


def calculate_spread_OHLC(df, sessions):
    print("Calculating spread OHLC for each session ...", end='')
    # Create 4 new columns in dataframe to hold spread open/high/low/close (for SESSION)
    df['spread_O'] = np.nan
    df['spread_H'] = np.nan
    df['spread_L'] = np.nan
    df['spread_C'] = np.nan
    for df_sess in sessions:
        r1, r2 = df_first_last(df_sess)
        df.loc[df_sess.index, 'spread_O'] = r1['Open_spread']
        df.loc[df_sess.index, 'spread_H'] = max(df_sess['Close_spread'].max(), r1['Open_spread'])
        df.loc[df_sess.index, 'spread_L'] = min(df_sess['Close_spread'].min(), r1['Open_spread'])
        df.loc[df_sess.index, 'spread_C'] = r2['Close_spread']
    print("Done.")
    return df.dropna()

def calculate_spread_ATR(df, df_day, lookback=10):
    df_day['spread_mean'] = df_day.spread_O.rolling(window=lookback).mean()
    df_day['range'] = df_day.spread_H - df_day.spread_L
    df_day['avg_range'] = df_day.range.rolling(window=lookback).mean()
    #df_day.dropna(inplace=True)
    df_day['atr_low'] = df_day.spread_O - df_day.avg_range
    df_day['atr_high'] = df_day.spread_O + df_day.avg_range
    #df_day['atr_low'] = df_day.spread_mean - df_day.avg_range
    #df_day['atr_high'] = df_day.spread_mean + df_day.avg_range

    df['spread_mean'] = np.nan
    df['atr_low'] = np.nan
    df['atr_high'] = np.nan

    print("Copying mean, ATR_low and ATR_high to each row ...", end='')
    for df_sess in sessions:
        dt = df_sess.tail(1).squeeze()['DateTime']
        row = df_day[df_day['DateTime'] == dt]
        if row.shape[0] > 0:
            df.loc[df_sess.index, 'spread_mean'] = row.squeeze()['spread_mean']
            df.loc[df_sess.index, 'atr_low'] = row.squeeze()['atr_low']
            df.loc[df_sess.index, 'atr_high'] = row.squeeze()['atr_high']
    print("Done.")
    return df

def spread_price_NQES(nq, es):
    return (nq * 120 - es * 250) / 100.0

def spread_price_HOGO(ho, go):
    return 126000 * ho - 400 * go

def get_spread_NQES(dfnq, dfes, fn_price, suffixes=('_x','_y')):
    df = pd.merge(dfnq, dfes, on="DateTime", suffixes=suffixes)
    df['spread'] = spread_price_NQES(df.Close_NQ, df.Close_ES)
    return df

def get_spread(df1, df2, fn_price, suffixes=('_x','_y')):    #, round=2):
    df = pd.merge(df1, df2, on='DateTime', suffixes=suffixes)
    open1 = 'Open' + suffixes[0]
    open2 = 'Open' + suffixes[1]
    close1 = 'Close' + suffixes[0]
    close2 = 'Close' + suffixes[1]
    df['Open_spread'] = fn_price(df[open1], df[open2])
    df['Close_spread'] = fn_price(df[close1], df[close2])
    #df['Open_spread'] = df['Open_spread'].round(round)
    #df['Close_spread'] = df['Close_spread'].round(round)
    return df

def calc_spread_moving_averages(df, col='spread'):
    df['SMA5'] = df[col].rolling(window=5,center=False).mean()
    df['SMA10'] = df[col].rolling(window=10,center=False).mean()
    df['SMA15'] = df[col].rolling(window=15,center=False).mean()
    df['SMA20'] = df[col].rolling(window=20,center=False).mean()
    df['EMA5'] = df[col].ewm(span=5).mean()
    df['EMA10'] = df[col].ewm(span=10).mean()
    df['EMA15'] = df[col].ewm(span=15).mean()
    df['EMA20'] = df[col].ewm(span=20).mean()
    return df
