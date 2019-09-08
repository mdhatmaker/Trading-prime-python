from f_date import *
from f_dataframe import *

#-------------------------------------------------------------------------------

class Bar:
    def __init__(self, df_bar, round_places=2):
        self.df_bar = df_bar
        self.o = round(df_bar.iloc[0].Open, round_places)
        self.h = round(max(df_bar.Close.max(), self.o), round_places)
        self.l = round(min(df_bar.Close.min(), self.o), round_places)
        self.c = round(df_bar.iloc[df_bar.shape[0] - 1].Close, round_places)
        self.session_date = df_bar.iloc[0].session_date


class BarSlice:
    def __init__(self, bar_list):
        self.temp = 'hello'
        self.bar_list = bar_list
        closes = []
        for bar in self.bar_list:
            closes.append(bar.c)
        self.closes = np.array(closes)

    def mean(self):
        return self.closes.mean()

    def get(self, i):
        return self.bar_list[i]

    def print_(self):
        for bar in self.bars:
            print("O:{0} H:{1} L:{2} C:{3}".format(bar.o, bar.h, bar.l, bar.c))
        return


class Bars:
    def __init__(self, df, bar_minutes):
        self.df = df
        self.bar_minutes = bar_minutes
        minutes_per_day = 60 * 24
        assert minutes_per_day % bar_minutes == 0  # ensure bar_minutes divides evely into minutes per day
        barix_count = minutes_per_day // bar_minutes
        df['bar_ix'] = np.nan
        for i in range(barix_count):
            t1, t2 = self.get_times(i)
            xt1 = add_time(t1, timedelta(minutes=-shift_minutes))
            if t2 == None:
                xt2 = None
            else:
                xt2 = add_time(t2, timedelta(minutes=-shift_minutes))
            print("{0:2d}  {1} {2}   ".format(i, strtime(xt1), strtime(xt2)),) # end='')
            if t2 == None:  # handle 24:00 differently since 24 is an illegal hour value
                dfx = df[(df['DateTime'].dt.time > t1) | (df['DateTime'].dt.time == time(0, 0))]
            else:
                dfx = df[(df['DateTime'].dt.time > t1) & (df['DateTime'].dt.time <= t2)]
            df.loc[dfx.index, 'bar_ix'] = i
            dfz = df[df.bar_ix == i]
            print(dfz['bar_ix'].count())
        # convert the bar index to integer (0-23 for hour bars, 0-47 for 30-minute bars, etc.)
        df['bar_ix'] = df['bar_ix'].astype('int')

    def get(self, i):
        ix1, ix2 = self.get_ix(i)
        return Bar(df.loc[ix1:ix2, :])

    def slice(self, i, j):
        bars = []
        for x in range(i, j + 1):
            bars.append(self.get(x))
        return BarSlice(bars)

    def get_ix(self, i):
        dt = self.get_dates()[i]
        dfx = self.df[self.df['session_date'] == dt]
        return dfx.index[0], dfx.index[-1]

    def get_times(self, i):
        minutes1 = i * self.bar_minutes
        minutes2 = (i + 1) * self.bar_minutes
        t1 = time(minutes1 // 60, minutes1 % 60)
        if minutes2 == 24 * 60:  # treat hour 24 differently (because it's considered hour zero)
            t2 = None
        else:
            t2 = time(minutes2 // 60, minutes2 % 60)
        return t1, t2

    def get_dates(self):
        return self.df['session_date'].unique()

    def filter_bars(self, x1, x2):
        self.df = self.df[(self.df.bar_ix >= x1) & (self.df.bar_ix <= x2)]
        return

    def filter_minimum_session_bars(self, minimum_bar_count):
        # The DAY SESSION should contain 12 bars (34-23+1)
        session_dates = df_get_unique(self.df, 'session_date')
        valid_dates = []
        for dt in session_dates:
            dfx = self.df[self.df.session_date == dt]
            bar_count = len(dfx.bar_ix.unique())
            # print(strdate(dt), bar_count)
            if bar_count >= minimum_bar_count:
                valid_dates.append(dt)
        self.df = self.df[self.df['session_date'].isin(valid_dates)]
        return

    def print_dates(self):
        for dt in self.get_dates():
            dfx = self.df[self.df.session_date == dt]
            bar_count = len(dfx.bar_ix.unique())
            print(strdate(pd.to_datetime(dt)), bar_count)
        return

# -----------------------------------------------------------------------------------------------------------------------
