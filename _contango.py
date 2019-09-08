import sys
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np

#-----------------------------------------------------------------------------------------------------------------------

execfile("f_folders.py")
execfile("f_args.py")
execfile("f_date.py")
execfile("f_file.py")
execfile("f_dataframe.py")

#-----------------------------------------------------------------------------------------------------------------------

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
# pass in the integer month (1-12) and year (2-digit or 4-digit) of front month AND the month count to calculate the back month
# using month_count = 1 will give you the front/next month calendar spread, etc.
def get_calendar_symbols(symbol_root, m1, y1, month_count):
    (m2, y2) = get_next_month(m1, y1, month_count)
    front_symbol = get_symbol(symbol_root, m1, y1)
    back_symbol = get_symbol(symbol_root, m2, y2)
    calendar_symbol = get_calendar_symbol(symbol_root, m1, y1, m2, y2)
    return (front_symbol, back_symbol, calendar_symbol)


################################################################################

print "Creating data file of calendar spread contango (valued as returns)..."
print "Output file will be comma-delimeted pandas-ready dataframe (.csv)"
print

#csv_files = [ f for f in listdir(folder) if (isfile(join(folder, f)) and f.endswith('.csv')) ]
#gas_files = [ f for f in csv_files if f.startswith('GAS') ]
#ho_files = [ f for f in csv_files if f.startswith('QHO') ]
#df = read_csv(csv_files[0])
#columns = ['Symbol', 'DateTime', 'Price']


project_folder = join(data_folder, "vix_es")
symbol_root = "@VX"
timeframe = 'Daily'


#filename1 = "@VX_calendar_0_1.csv"
#filename2 = "QHO_calendar_0_6.csv"
filename1 = get_csv_filename(symbol_root + "_calendar", timeframe)
output_filename = "vix_contango.csv"

########## DIVIDE CALENDAR SPREAD BY FRONT MONTH PRICE ##########
print "-----CALCULATING CONTANGO-----"

df1 = read_dataframe(join(project_folder, filename1))
#df2 = read_csv(join(project_folder, filename2))

#df = pd.merge(df1, df2, on=['DateTime'])
df = df1

if symbol_root == "@VX":
    # For VIX only (invert = take negative of values)
    df['Contango_Open'] = -(df['Open'] / df['Open_x']) * 100    #df['Close_x'] / df['Close_y']
    df['Contango_Close'] = -(df['Close'] / df['Close_x']) * 100
else:
    df['Contango_Open'] = df['Open'] / df['Open_x'] * 100    #df['Close_x'] / df['Close_y']
    df['Contango_Close'] = df['Close'] / df['Close_x'] * 100

df = df.round({'Contango_Open': 2, 'Contango_Close': 2})
#df = df.round(4)
df = df.sort_values('DateTime')

#df = df.drop(['Symbol_x', 'Open_x', 'Close_x', 'Volume1_x', 'Volume2_x', 'Symbol_y', 'Open_y', 'Close_y', 'Volume1_y', 'Volume2_y'], 1)
df = df.drop(['High_x', 'Low_x', 'High_y', 'Low_y'], 1)



df.to_csv(join(project_folder, output_filename), index=False)  #, cols=('A','B','sum'))

print
print "Contango output to file:", output_filename
print




