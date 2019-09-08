from __future__ import print_function

#-----------------------------------------------------------------------------------------------------------------------

from f_folders import *
from f_dataframe import *
from f_iqfeed import *
from f_plot import *
from f_stats import *

#-----------------------------------------------------------------------------------------------------------------------

def create_risk_indicator_df(dt1, dt2, lookback=40, avg_close_days=3, data_interval="s1800", output_filename="risk_indicator.DF.csv"):
    # Attempt to build AlvinRisk indicator
    df1 = create_historical_contract_df("VIX.XO", dt1, dt2, interval=data_interval, beginFilterTime='093000', endFilterTime='160000')
    df2 = create_historical_contract_df("JYVIX.XO", dt1, dt2, interval=data_interval, beginFilterTime='093000', endFilterTime='160000')
    df3 = create_historical_contract_df("GVZ.XO", dt1, dt2, interval=data_interval, beginFilterTime='093000', endFilterTime='160000')
    df4 = create_historical_contract_df("TYX.XO", dt1, dt2, interval=data_interval, beginFilterTime='093000', endFilterTime='160000')

    df1.drop(['Symbol', 'Open', 'High', 'Low', 'Volume', 'oi'], axis=1, inplace=True)
    df2.drop(['Symbol', 'Open', 'High', 'Low', 'Volume', 'oi'], axis=1, inplace=True)
    df3.drop(['Symbol', 'Open', 'High', 'Low', 'Volume', 'oi'], axis=1, inplace=True)
    df4.drop(['Symbol', 'Open', 'High', 'Low', 'Volume', 'oi'], axis=1, inplace=True)

    df1.columns = ['DateTime', 'Close_VIX']
    df2.columns = ['DateTime', 'Close_JYVIX']
    df3.columns = ['DateTime', 'Close_GVZ']
    df4.columns = ['DateTime', 'Close_TYX']

    df1['Close_VIX'] = df1.Close_VIX.astype('float')
    df2['Close_JYVIX'] = df2.Close_JYVIX.astype('float')
    df3['Close_GVZ'] = df3.Close_GVZ.astype('float')
    df4['Close_TYX'] = df4.Close_TYX.astype('float')

    df = df1.merge(df2, on=['DateTime'])
    df = df.merge(df3, on=['DateTime'])
    df = df.merge(df4, on=['DateTime'])

    df['Close'] = df.Close_VIX + df.Close_JYVIX + df.Close_GVZ + df.Close_TYX

    df_ar, forecast = alvin_arima(df, lookback=lookback, avg_close_days=avg_close_days)

    write_dataframe(df_ar, output_filename)
    return forecast, df_ar


################################################################################

print("Testing ideas for Alvin RISK INDICATOR")
print()

dt1 = datetime(2017, 1, 1)
dt2 = datetime(2017, 12, 31)

df, forecast = create_risk_indicator_df(dt1, dt2, lookback=40, avg_close_days=3, data_interval="s1800", output_filename="risk_indicator.DF.csv")

STOP(forecast)

df = read_dataframe("risk_indicator.DF.csv")

# Some PLOTS of our ARIMA calculations
xl = lambda x: x.strftime('%m-%d') if (x.day % 2 == 0 and x.hour == 12 and x.minute == 0) else ''
plot_arima(df, title="Risk Indicator", tail_count=300, xaxis_lambda=xl)
xl = lambda x: x.strftime('%m/%d/%y') if (x.day % 10 == 0 and x.hour == 12 and x.minute == 0) else ''
plot_arima(df, title="Risk Indicator", xaxis_lambda=xl)

STOP(df)
