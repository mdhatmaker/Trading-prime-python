import sys
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np

#-----------------------------------------------------------------------------------------------------------------------

execfile("f_folders.py")
execfile("f_date.py")

#folder = join(folder, "HOGO")

#-----------------------------------------------------------------------------------------------------------------------

bizday_count = 10           # business days before end of month (used for roll)

def print_trades(trade_column, trade_rows, side, entry_price, exit_price, df_rolls):
    average = 0.0
    for (t_entry, t_exit, cal_adjust, discount_adjust) in trade_rows:
        day_count = (t_exit.Date - t_entry.Date).days
        roll_count = roll_exists_in_date_range(t_entry.Date, t_exit.Date, df_rolls)
        roll_count_indicator = '*' * roll_count
        output = get_output(t_entry, t_exit, cal_adjust, discount_adjust)
        output += roll_count_indicator
        print output
        average += day_count
    average /= len(trades)
    print
    print len(trades), "trades"
    print "average holding period (days): %.1f" % (average)
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

def read_csv(pathname):
    #pathname = join(folder, filename)
    df = pd.read_csv(pathname, parse_dates=['DateTime'])
    return df

# If month_count is 1, then this function returns the same as next_month function
# pass in the count of the number of months following the given month
def get_next_month(m1, y1, month_count):
    m2 = m1
    y2 = y1
    for i in range(month_count):
        (m2, y2) = next_month(m2, y2)
    return (m2, y2)

# Return a symbol in the form 'XXXmYY' (ex: 'QHOM17')
# pass in the symbol root (ex: 'QHO', 'GAS', 'QHG') and integer month (1-12) and year (2-digit or 4-digit)
def get_symbol(symbol_root, m1, y1):
    return symbol_root + get_MYY(m1, y1)

# Return a calendar spread symbol in the form 'XXXmYY-XXXmYY' (ex: 'QHOM17-QHON17')
# pass in the symbol root (ex: 'QHO', 'GAS', 'QHG') and integer months (1-12) and years (2-digit or 4-digit)
def get_calendar_symbol(symbol_root, m1, y1, m2, y2):
    calendar_symbol = get_symbol(symbol_root, m1, y1) + '-' + get_symbol(symbol_root, m2, y2)
    return calendar_symbol

# Return the HO calendar symbol in the form 'QHOmYY-QHOmYY'
# pass in the integer month (1-12) and year (2-digit or 4-digit) of front month AND the front/back month counts
# using ifront=0 and iback=1will give you the front/next month calendar spread, etc.
def get_calendar_symbols(symbol_root, m1, y1, ifront, iback):
    (m2, y2) = get_next_month(m1, y1, iback)
    (m1, y1) = get_next_month(m1, y1, ifront)
    front_symbol = get_symbol(symbol_root, m1, y1)
    back_symbol = get_symbol(symbol_root, m2, y2)
    calendar_symbol = get_calendar_symbol(symbol_root, m1, y1, m2, y2)
    return (front_symbol, back_symbol, calendar_symbol)

# Return the path name of the file containing data for a dataframe
def get_df_pathname(symbol):
    filename = symbol + " 1 Minute.csv"
    #filename = symbol + " 1 Hour.csv"
    #return join(folder, "RAW_DATA", filename)
    return join(folder, "DF_DATA", filename)

def roll_date(m1, y1):
    return last_business_day(m1, y1, bizday_count)    # [bizday_count] business days before end of month
    
def get_roll_dates(m1, y1):
    (mp1, yp1) = prev_month(m1, y1)
    (mp2, yp2) = prev_month(mp1, yp1)
    return (roll_date(mp2, yp2), roll_date(mp1, yp1))

def filter_date_range(df, dt0, dt1):
    dfx = df[df['DateTime'] >= dt0]
    dfx = dfx[dfx['DateTime'] < dt1]
    return dfx
    
################################################################################

print "Creating prices data file and roll dates file using 1 Hour data..."
print "Output file will be comma-delimeted pandas-ready dataframe (.csv)"
print
print "Rolling at {0} business days before end-of-month".format(bizday_count)
print

#csv_files = [ f for f in listdir(folder) if (isfile(join(folder, f)) and f.endswith('.csv')) ]
#gas_files = [ f for f in csv_files if f.startswith('GAS') ]
#ho_files = [ f for f in csv_files if f.startswith('QHO') ]
#df = read_csv(csv_files[0])
#columns = ['Symbol', 'DateTime', 'Price']


symbol_root = 'QHO'
ifront = 3          # calendar front month index (0 = front month)
iback = 6           # calendar back month index (1 = next month)

year_list = [2013, 2017]
#from_year = 2013
#to_year = 2017

roll_filename = "{0}_roll_dates_{1}_{2}.csv".format(symbol_root, ifront, iback)
f = open(join(folder, "HOGO", roll_filename), 'w')
f.write("Symbol,StartDate,EndDate,PriceAdjust\n")

########## CREATE CALENDAR DATA FILE ##########
print "-----" + symbol_root + " CALENDARS-----"
prev_close = None
df_frames = []
for year in range(year_list[0], year_list[1]+1):
    for month in range(1, 12+1):
        count = 0
        rows_list = []
        print month, year,
        (front_symbol, back_symbol, calendar_symbol) = get_calendar_symbols(symbol_root, month, year, ifront, iback)
        f_pathname = get_df_pathname(front_symbol)
        b_pathname = get_df_pathname(back_symbol)
        if isfile(f_pathname) and isfile(b_pathname):
            df_front = read_csv(f_pathname)
            df_back = read_csv(b_pathname)
            (dt0, dt1) = get_roll_dates(month, year)                # get roll dates for the front month
            df_f = filter_date_range(df_front, dt0, dt1)
            df_b = filter_date_range(df_back, dt0, dt1)
            #df_f.set_index('DateTime')
            #df_b.set_index('DateTime')
            if df_f.empty or df_b.empty:
                print
                continue
            
            #df.insert(0, 'Symbol', calendar_symbol)
            #df_frames.append(df)

            #df = df_f.join(df_b, on='DateTime', how='outer', lsuffix='1', rsuffix='2', sort=True)
            df = pd.merge(df_f, df_b, on=['DateTime'])

            df.insert(0, 'Symbol', calendar_symbol)
            df['Open'] = (df['Open_x']-df['Open_y'])/df['Open_x'] * 100
            df['Close'] = (df['Close_x']-df['Close_y'])/df['Close_x'] * 100
            df = df.round({'Open': 4, 'Close': 4})
            #df['Volume1'] = df['Volume_x']
            #df['Volume2'] = df['Volume_y']
            #df = df.drop(['Open_x', 'High_x', 'Low_x', 'Close_x', 'Volume_x', 'Open_y', 'High_y', 'Low_y', 'Close_y', 'Volume_y'] , 1)
            df = df.drop(['High_x', 'Low_x', 'High_y', 'Low_y'] , 1)
            
            df = df.sort_values('DateTime')
            df_frames.append(df)
            #df.sort_values(['colA', 'colB'], ascending=[True, False])

            mindate = df.DateTime.min()
            maxdate = df.DateTime.max()

            if prev_close == None:
                prev_close = float(df.tail(1).Close)
                adjust = 0.0
            else:
                adjust = prev_close - float(df.head(1).Close)
                prev_close = float(df.tail(1).Close)

            output = "{0},{1},{2},{3}".format(calendar_symbol, mindate.strftime("%Y-%m-%d"), maxdate.strftime("%Y-%m-%d"), adjust)
            f.write(output + '\n')
            print output
            
            #df_frames.append(pd.DataFrame(rows_list, columns=columns))
        else:
            print

f.close()

df = pd.concat(df_frames, ignore_index=True)

filename = "{0}_calendar_{1}_{2}.csv".format(symbol_root, ifront, iback)
df.to_csv(join(folder, "HOGO", filename), index=False)  #, cols=('A','B','sum'))

print
print "Prices output to file:", filename
print "Roll dates output to file:", roll_filename
print




