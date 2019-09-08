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


bar_minutes = 15            # period of each bar
symbol = "QCL#C"            # symbol (will be used to determine from which CSV file to read data)
count = 40                  # number of values to use in our mean and SD calculations (lookback)

# Read the dataframes from their respective CSV files
df = read_dataframe_bars_csv(symbol, bar_minutes)


# Add a column that contains the price change (close - open)
df['diff'] = df['Close'] - df['Open']


# Construct the output file name
dt0_str = get_min_date_str(df)
dt1_str = get_max_date_str(df)
filename = "SD" + str(count) + " " + symbol + " " + str(bar_minutes) + " " + dt0_str + " " + dt1_str + ".csv"
pathname = join(path, filename)

# Open the output file and write the column headers
fcout = open(pathname, 'w')
fcout.write("DateTime0,DateTime1,Open,Close,diff,mean,std\n")
           
# Add two more columns (for mean and std)
df['mean'] = np.nan
df['std'] = np.nan

# For each row (price bar) calculate the mean and std (standard deviation) of the previous [count] bars
length = len(df['diff'])
for i in range(count,length):
    values = df['diff'].ix[i-count:i-1]
    mean = values.mean()
    std = values.std()
    #df.set_value(i, 'mean', np.std(values))    # shows n vs (n-1) difference in numpy vs pandas std calculation
    df.set_value(i, 'mean', mean)
    df.set_value(i, 'std', std)

print "Processing input file",
for i in range(count,length):
    if i % 100 == 0:
        print ".",
    row = df[['DateTime', 'Open', 'Close', 'diff', 'mean', 'std']].ix[i:i]
    dt2 = row.stack()[0]
    dt1 = dt2 - datetime.timedelta(minutes=bar_minutes)
    open_price = row.stack()[1]
    close_price = row.stack()[2]
    diff = row.stack()[3]
    mean = row.stack()[4]
    std = row.stack()[5]
    output_str = "%s, %s, %.4f, %.4f, %.4f, %.4f, %.4f" % (dt1, dt2, open_price, close_price, diff, mean, std)
    #print output_str
    fcout.write(output_str + "\n")
print

fcout.close()

print "Results output to file: ", filename


