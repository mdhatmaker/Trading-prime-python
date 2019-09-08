import pandas as pd
import numpy as np
from os.path import join
import datetime

#import os
#path = os.path.dirname(os.path.realpath(__file__))
#path = "/Users/michael/CLionProjects/ALVIN/data_sd_analysis"        # macbook
#path = "C:\\Users\\Michael\\Dropbox\\ALVIN\\data_sd_analysis"       # lenovo laptop
path = "B:\\Users\\mhatmaker\\Dropbox\\ALVIN\\data_sd_analysis"     # apartment desktop


def get_datetime_from_d_t(d_str, t_str):
    vd = d_str.split('/')
    vt = t_str.split(':')
    return datetime.datetime(int(vd[2]), int(vd[0]), int(vd[1]), int(vt[0]), int(vt[1]), int(vt[2]))

def read_dataframe_csv(name):
    filename = name + ".csv"
    print "Reading CSV data file:", filename
    f = open(join(path, filename), 'r')
    line = f.readline()[:-1]
    col_names = line.split(',')
    f.close()
    col_names[0] = 'ID'
    df = pd.read_csv(join(path, filename), index_col='ID', names=col_names)
    return df

def read_dataframe_bars_csv(name, minutes):
    filename = name + " " + str(minutes) + " Minutes.csv"
    print "Reading CSV data file:", filename
    pathname = join(path, filename)
    df=pd.read_csv(pathname,
                   #header=None,
                   skiprows=1,
                   index_col="ID",
                   names=["ID", "DateTime","Open","High","Low","Close","Volume"],
                   dtype={"ID":"int", "DateTime":"str", "Open":"float", "High":"float", "Low":"float", "Close":"float", "Volume":"int"},
                   parse_dates = ['DateTime'],
                   sep=","
                )
    df.info()
    print
    return df

def read_dataframe_tick_csv(name):
    filename = name + " 1 Tick Bar.csv"
    print "Reading CSV data file:", filename, "(this may take a few minutes)"
    pathname = join(path, filename)
    df=pd.read_csv(pathname,
                   #header=None,
                   skiprows=1,
                   index_col="ID",
                   names=["ID", "DateTime","Price","Volume"],
                   dtype={"ID":"int", "DateTime":"str", "Price":"float", "Volume":"int"},
                   parse_dates = ['DateTime'],
                   sep=","
                )
    df.info()
    print
    return df

def get_min_date_str(df):
    x = str(df['DateTime'].min())
    return x[:10]

def get_max_date_str(df):
    x = str(df['DateTime'].max())
    return x[:10]

################################################################################


minutes = 30
symbol = "QCL#C"

# Read the dataframes from their respective CSV files
df = read_dataframe_bars_csv(symbol, minutes)
df_tick = read_dataframe_tick_csv("QCL#C")



#sLength = len(df['Date'])
#df['chg'] = pd.Series(np.zeros(sLength), index=df.index)



# Add a column that contains the price change (close - open)
df['chg'] = df['Close'] - df['Open']

# count will be the number of values to use in our mean and SD calculations
count = 40

dt0_str = get_min_date_str(df)
dt1_str = get_max_date_str(df)
filename = "SD" + str(count) + " " + symbol + " " + str(minutes) + " " + dt0_str + " " + dt1_str + ".csv"

pathname = join(path, filename)
fcout = open(pathname, 'w')

fcout.write("DateTime0,DateTime1,mean,std,hitcount_pos,hitcount_neg\n")
           
# Add two more columns (for mean and std)
df['mean'] = np.nan
df['std'] = np.nan

# For each row (price bar) calculate the mean and std (standard deviation) of the previous [count] bars
length = len(df['chg'])
for i in range(count,length):
    values = df['chg'].ix[i-count:i-1]
    mean = values.mean()
    std = values.std()
    #df.set_value(i, 'mean', np.std(values))    # shows n vs (n-1) difference in numpy vs pandas std calculation
    df.set_value(i, 'mean', mean)
    df.set_value(i, 'std', std)

#print "Running tick data",
#for i in range(count,length):
for i in range(count,length):
    #if i % 1000 == 0:
    #    print ".",
    row = df[['DateTime', 'Open', 'Close', 'mean', 'std']].ix[i:i]
    dt2 = row.stack()[0]
    dt1 = dt2 - datetime.timedelta(minutes=minutes)
    center_price = row.stack()[1]
    mean = row.stack()[3]
    std = row.stack()[4]
    up_one_sd = center_price + std
    down_one_sd = center_price - std
    ticks = df_tick.loc[(df_tick['DateTime'] > dt1) & (df_tick['DateTime'] <= dt2)]
    hitcount_pos = 0
    hitcount_neg = 0
    hit_upside = False
    hit_downside = False
    for (index,row) in ticks.iterrows():
        price = row['Price']
        if hit_upside == True:
            if price <= center_price:
                hit_upside = False
        elif hit_downside == True:
            if price >= center_price:
                hit_downside = False
        else:
            if price >= up_one_sd:
                hitcount_pos += 1
                hit_upside = True
            elif price <= down_one_sd:
                hitcount_neg -= 1
                hit_downside = True
    output_str = "%s, %s, %.4f, %.4f, %d, %d" % (dt1, dt2, mean, std, hitcount_pos, hitcount_neg)
    print output_str
    fcout.write(output_str + "\n")
    #print dt1, dt2, mean, std, hitcount_pos, hitcount_neg
    
print

fcout.close()

print "Results output to file: ", filename



#dt1 = datetime.datetime(2017, 6, 1, 8, 30, 0)
#dt2 = datetime.datetime(2017, 6, 1, 8, 45, 0)
#ticks = df_tick.loc[(df_tick['DateTime'] > dt1) & (df_tick['DateTime'] <= dt2)]     #df['DateTime'].isin(some_values)]
    




