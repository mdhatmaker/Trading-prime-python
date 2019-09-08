import sys
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np

#-----------------------------------------------------------------------------------------------------------------------

execfile("f_folders.py")
execfile("f_date.py")

folder = join(folder, "HOGO")

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
    calendar_symbol = symbol_root + get_MYY(m1, y1) + '-' + symbol_root + get_MYY(m2, y2)
    return calendar_symbol

# Return the HO calendar symbol in the form 'QHOmYY-QHOmYY'
# pass in the integer month (1-12) and year (2-digit or 4-digit) of front month AND the month count to calculate the back month
# using month_count = 1 will give you the front/next month calendar spread, etc.
def get_calendar(symbol_root, m1, y1, month_count):
    (m2, y2) = get_next_month(m1, y1, month_count)
    calendar_symbol = get_calendar_symbol(symbol_root, m1, y1, m2, y2)
    return calendar_symbol

# Return the path name of the file containing data for a dataframe
def get_df_pathname(symbol):
    #filename = symbol + " 1 Minute.csv"
    filename = symbol + " 1 Hour.csv"
    return join(folder, "RAW_DATA", filename)

def roll_date(m1, y1):
    return last_business_day(m1, y1, bizday_count)    # [bizday_count] business days before end of month
    
def get_roll_dates(m1, y1):
    (mp1, yp1) = prev_month(m1, y1)
    (mp2, yp2) = prev_month(mp1, yp1)
    return (roll_date(mp2, yp2), roll_date(mp1, yp1))
    
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
fi = 0              # calendar front month index (0 = front month)
bi = 1              # calendar back month index (1 = next month)

year_list = [2016, 2017]

roll_filename = symbol_root + "_calendar_rolls.csv"
f = open(join(folder, roll_filename), 'w')
f.write("Symbol,StartDate,EndDate,PriceAdjust\n")

########## CREATE ROLL DATES FILE ##########
print "-----" + symbol_root + " CALENDARS-----"
prev_close = None
df_frames = []
for year in year_list:
    for month in range(1, 12+1):
        count = 0
        rows_list = []
        print month, year,
        calendar_symbol = get_calendar(symbol_root, month, year, 1)
        data_pathname = get_df_pathname(calendar_symbol)
        if isfile(data_pathname):
            df_cal = read_csv(data_pathname)
            (dt0, dt1) = get_roll_dates(month, year)
            df = df_cal[df_cal['DateTime'] >= dt0]
            df = df[df['DateTime'] < dt1]
            if df.empty:
                print
                continue
            df.insert(0, 'Symbol', calendar_symbol)
            #df['Symbol'] = ho_symbol
            df_frames.append(df)

            #df.sort_values(['colA', 'colB'], ascending=[True, False])
            df = df.sort_values('DateTime')
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
            """
            for (index, ho_row) in df_ho.iterrows():
                dt = ho_row.DateTime
                gas_row = df_gas[df_gas.DateTime == dt]
                if not gas_row.empty:
                    gas_row = gas_row.squeeze()
                    count += 1
                    hogo_price = (ho_row.Close * 1) - (gas_row.Close * .3196)
                    dict1 = {'Month': month, 'Year': year, 'DateTime': dt, 'Price': round(hogo_price,2), 'VolumeHO': ho_row.Volume, 'VolumeGO': gas_row.Volume }
                    rows_list.append(dict1)
            print "    COUNTS >>>", "combined:", count, " ho:", len(df_ho.index), " go:", len(df_gas.index)
            """
            #df_frames.append(pd.DataFrame(rows_list, columns=columns))
        else:
            print

f.close()

df = pd.concat(df_frames, ignore_index=True)

filename = symbol_root + "_prices.csv"
df.to_csv(join(folder, filename), index=False)  #, cols=('A','B','sum'))

print
print "Prices output to file:", filename
print "Roll dates output to file:", roll_filename
print




