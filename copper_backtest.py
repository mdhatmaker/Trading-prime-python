import sys
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta

#-----------------------------------------------------------------------------------------------------------------------

#execfile(r'..\..\..\python\f_analyze.py')
execfile(r'f_analyze.py')
import f_folders as folder
from f_args import *
from f_date import *
from f_file import *
from f_calc import *
from f_dataframe import *
from f_rolldate import *
from f_chart import *

project_folder = join(data_folder, "copper")

#-----------------------------------------------------------------------------------------------------------------------

def backtest_buy(df, column, buy_price, sell_price, start_date=None, end_date=None):
    li = []
    trade_active = False
    for index, row in df.iterrows():
        if trade_active == False:
            if row[column] <= buy_price:
                buy_row = row
                trade_active = True
                #print row.Date, "  Bought:", row.spread
        else:
            if row[column] >= sell_price:
                sell_row = row
                tuple = (buy_row, sell_row, 0.0, 0.0)
                li.append(tuple)
                trade_active = False
                #print row.Date, "  Sold:", row.spread
    return li

def backtest_sell(df, column, sell_price, buy_price, start_date=None, end_date=None):
    li = []
    trade_active = False
    for index, row in df.iterrows():
        if trade_active == False:
            if row[column] >= sell_price:
                sell_row = row
                trade_active = True
                #print row.Date, "  Sold:", row.spread
        else:
            if row[column] <= buy_price:
                buy_row = row
                tuple = (sell_row, buy_row, 0.0, 0.0)
                li.append(tuple)
                trade_active = False
                #print row.Date, "  Bought:", row.spread
    return li

def get_adjustments(symbol, prev_symbol, df_rolls):
    if symbol[:4] == prev_symbol[:4]:
        cal_adjust = 0.0
    else:
        cal1 = df_rolls.loc[df_rolls['Symbol'] == symbol, 'FirstCal'].values[0]
        cal0 = df_rolls.loc[df_rolls['Symbol'] == prev_symbol, 'LastCal'].values[0]
        cal_adjust = (cal1 - cal0)
    disc1 = df_rolls.loc[df_rolls['Symbol'] == symbol, 'FirstDiscount'].values[0]
    disc0 = df_rolls.loc[df_rolls['Symbol'] == prev_symbol, 'LastDiscount'].values[0]
    discount_adjust = (disc1 - disc0)
    return cal_adjust, discount_adjust

# return a list of trades discovered by backtesting
# column should be 'discount' or 'spread'
def backtest_buy_with_roll(df, column, buy_price, sell_price, df_rolls, start_date=None, end_date=None):
    li = []
    cal_adjust = 0.0
    discount_adjust = 0.0
    prev_symbol = None
    trade_active = False
    for index, row in df.iterrows():
        # Update price adjustment for calendar roll 
        symbol = row['symbol'].strip()
        if prev_symbol == None:
            prev_symbol = symbol
        elif symbol != prev_symbol:
            cadj, discount_adjust = get_adjustments(symbol, prev_symbol, df_rolls)
            cal_adjust += cadj
            #if trade_active:
            #    print buy_row.Date.strftime("%Y-%m-%d"), "cal_adjust =", cal_adjust
            prev_symbol = symbol
        
        # Check if we should enter a new trade or exit an existing trade (or neither)
        if trade_active == False:
            if row[column] <= buy_price:
                buy_row = row
                trade_active = True
                cal_adjust = 0.0
                discount_adjust = 0.0
                #print row.Date, "  Sold:", row.spread
        else:
            if column == 'discount':
                adjusted_price = row[column]-discount_adjust
            elif column == 'spread':
                adjusted_price = row[column]-cal_adjust

            if adjusted_price >= sell_price:
                sell_row = row
                tuple = (buy_row, sell_row, cal_adjust, discount_adjust)
                li.append(tuple)
                trade_active = False
                cal_adjust = 0.0
                discount_adjust = 0.0
                #print "----------------------------"
                #print row.Date, "  Bought:", row.spread
    return li

# return a list of trades discovered by backtesting
# column should be 'discount' or 'spread'
def backtest_sell_with_roll(df, column, sell_price, buy_price, df_rolls, start_date=None, end_date=None):
    li = []
    cal_adjust = 0.0
    discount_adjust = 0.0
    prev_symbol = None
    trade_active = False
    for index, row in df.iterrows():
        # Update price adjustment for calendar roll 
        symbol = row['symbol'].strip()
        if prev_symbol == None:
            prev_symbol = symbol
        elif symbol != prev_symbol:
            cadj, discount_adjust = get_adjustments(symbol, prev_symbol, df_rolls)
            cal_adjust += cadj
            #if trade_active:
            #    print sell_row.Date.strftime("%Y-%m-%d"), "cal_adjust =", cal_adjust
            prev_symbol = symbol
        
        # Check if we should enter a new trade or exit an existing trade (or neither)
        if trade_active == False:
            #if row['spread'] >= sell_price:
            if row[column] >= sell_price:
                sell_row = row
                trade_active = True
                cal_adjust = 0.0
                discount_adjust = 0.0
                #print row.Date, "  Sold:", row.spread
        else:
            if column == 'discount':
                adjusted_price = row[column]-discount_adjust
            elif column == 'spread':
                adjusted_price = row[column]-cal_adjust

            #if row['spread'] <= buy_price:
            if adjusted_price <= buy_price:
                buy_row = row
                tuple = (sell_row, buy_row, cal_adjust, discount_adjust)
                li.append(tuple)
                trade_active = False
                cal_adjust = 0.0
                discount_adjust = 0.0
                #print "----------------------------"
                #print row.Date, "  Bought:", row.spread
    return li

def backtest(dfall, column, side, enter_price, exit_price, df_rolls):
    print "Running backtest: ", column, side, enter_price, exit_price
    li = []
    df = dfall.sort_values('Date')
    if side.upper().startswith('B'):
        #li = backtest_buy(df, column, enter_price, exit_price)
        li = backtest_buy_with_roll(df, column, enter_price, exit_price, df_rolls)
    elif side.upper().startswith('S'):
        #li = backtest_sell(df, column, enter_price, exit_price)
        li = backtest_sell_with_roll(df, column, enter_price, exit_price, df_rolls)
    else:
        raise ValueError('Side passed to backtest function must be "BUY" or "SELL"')
    return li

def roll_exists_in_date_range(range_start, range_end, df_rolls):
    dfr1 = df_rolls[df_rolls.FirstDate > range_start]
    dfr2 = df_rolls[df_rolls.FirstDate > range_end]
    if len(dfr1.index) < 1 or len(dfr2.index) < 1:
        return 0
    else:
        d1 = dfr1.head(1)['FirstDate'].values[0]
        d2 = dfr2.head(1)['FirstDate'].values[0]
        #print d1, d2
        dfr = df_rolls[(df_rolls.FirstDate >= d1) & (df_rolls.FirstDate < d2)]
        #print dfr
        return len(dfr.index)

def get_output(t_entry, t_exit, cal_adjust, discount_adjust):
    day_count = (t_exit.Date - t_entry.Date).days
    output = '{0},{1:.2f},{2:.2f},{3:.2f},{4:.2f},{5:.2f},{6:.2f},{7},{8},{9},'.format(side, t_entry.discount, t_exit.discount, discount_adjust, t_entry.spread, t_exit.spread, cal_adjust, t_entry.Date.strftime("%Y-%m-%d"), t_exit.Date.strftime("%Y-%m-%d"), day_count)
    return output

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
    f = open(join(folder, "copper", filename), 'w')
    f.write("Side,EntryDiscount,ExitDiscount,AdjustDiscount,EntrySpread,ExitSpread,AdjustSpread,EntryDate,ExitDate,HoldingDays,RollCount\n")
    for (t_entry, t_exit, cal_adjust, discount_adjust) in trade_rows:
        day_count = (t_exit.Date - t_entry.Date).days
        roll_count = roll_exists_in_date_range(t_entry.Date, t_exit.Date, df_rolls)
        output = get_output(t_entry, t_exit, cal_adjust, discount_adjust)
        output += str(roll_count)
        f.write(output + '\n')
    f.close()
    return

"""
def print_calendar_roll_dates(df_rolls):
    print
    print "Calendar rolls (last date, discount-on-last-date):"
    for (index,row) in df_rolls.iterrows():
        symbol = row.values[0]
        lastdate = row.values[4]
        discount = row.values[6]
        d2_str = str(lastdate)[:10]
        output = '{0}, {1}, {2:.2f}'.format(d2_str, symbol, discount)
        print output
    return

def write_roll_file(filename, calendar_rolls):
    f = open(folder + filename, 'w')
    for (d1, d2) in calendar_rolls:
        d2_str = str(d2['Date'])[:10]
        output = '{0},{1},{2:.2f}\n'.format(d2_str, d2['symbol'], d2['discount'])
        f.write(output)
    f.close()
    return
"""



################################################################################


#copper_filename = "copper_premium_discount.csv"
copper_filename = "copper_discount.DF.csv"
df_copper = pd.read_csv(join(project_folder, copper_filename), parse_dates=['DateTime'])


"""
##### READ LME 1-MINUTE DATA AND TOSS ALL BUT THE CLOSEST DATAPOINT TO 1PM SETTLEMENT #####
df = read_dataframe(join(df_folder, "M.CU3=LX (1 Minute).csv"))
dt_first = df.DateTime.min()
dt_last = df.DateTime.max()
print dt_first, dt_last

df_rows = pd.DataFrame()

dt2 = datetime(dt_first.year, dt_first.month, dt_first.day, 13, 0, 0)
dt1 = dt2 - timedelta(days=7)
while dt2 <= dt_last:
    if dt2.weekday() >= 0 and dt2.weekday() <= 4:     # if weekday is Mon-Fri
        dfx = df[(df.DateTime>=dt1) & (df.DateTime <= dt2)]
        last1 = dfx.tail(1).copy()
        dtx = last1.squeeze()["DateTime"]
        #dt_1pm = datetime(dtx.year, dtx.month, dtx.day, 13, 0, 0)
        if dt2 - dtx > timedelta(minutes=15):
            print dtx, dt2-dtx
        elif df_rows.shape[0] == 0:
            #last1.set_value(0, "DateTime", dtx.replace(hour=13,minute=0,second=0))
            df_rows = last1
        else:
            #last1.set_value(0, "DateTime", dtx.replace(hour=13,minute=0,second=0))
            df_rows = df_rows.append(last1)
            
    dt1 += timedelta(days=1)
    dt2 += timedelta(days=1)

df_rows['DateTime'] = df_rows['DateTime'].apply(lambda x: x.replace(hour=13,minute=0,second=0))
df_rows.drop(['Open','High','Low','Volume'], axis=1, inplace=True)

sys.exit()
"""



"""
dt = datetime(2017, 7, 21)
df_copper = df_copper[df_copper.DateTime >= dt]
df_copper['dif'] = (df_copper.DateTime - dt)
#df_copper['days'] = int(df_copper.dif)
  
print df_copper[['DateTime', 'HG', 'dif']]
sys.exit()
"""


##### THIS CODE WAS TO TEST THE SETUP WITH ROLLING INTO NEXT MONTH AT HIGHER PREMIUM (premium >= 1.0 AND next month premium >= 1.0) ########
# Get unique Calendar symbols, then create a list of only the (unique, sorted-by-mYY) first symbols in the calendar pairs
unique = df_copper.Symbol.unique()
first_symbols = [cal_symbol[:6] for cal_symbol in unique]
first_symbols = set(first_symbols)
first_symbols = sorted(first_symbols, cmp=compare_calendar, reverse=False)

count = 0

# Iterate thru the symbols...
for i in range(0,len(first_symbols)-1):
    this_symbol = first_symbols[i]
    next_symbol = first_symbols[i+1]
    dfx = df_copper.loc[df_copper.Symbol.str.startswith(this_symbol, na=False)]
    # I could probably NOT add this 'dt' column and instead perform this format when I set dt1 and dt2, but...faster?
    dfx['dt'] =  pd.to_datetime(dfx['DateTime'], format='%d%b%Y:%H:%M:%S.%f')
    dt1 = dfx.iloc[0].DateTime
    dt2 = dfx.iloc[dfx.shape[0]-1].DateTime
    # Now we have a dataframe (dfx) with the rows that start with this_symbol.
    # So take some date range from the END of these rows and recalculate
    # the spread value with HG from the FOLLOWING month.
    dt3 = dt2 - timedelta(days=14)
    dfx_tail = dfx[dfx.dt >= dt3]
    this_filename = this_symbol + " (1 Hour).csv"
    next_filename = next_symbol + " (1 Hour).csv"
    df_this = pd.read_csv(join(df_folder, this_filename), parse_dates=['DateTime'])
    df_next = pd.read_csv(join(df_folder, next_filename), parse_dates=['DateTime'])
    df_merge = dfx_tail.merge(df_this, on='DateTime')
    df_merge = df_merge.merge(df_next, on='DateTime')
    #df_merge['Spread_temp'] = df_merge['LME'] * .000454
    df_merge['Spread_'] = 100 * (df_merge['Close_y'] - df_merge['LME'] * .000454)
    df_merge['Discount_'] = df_merge['Spread_'] - df_merge['Cal'] - (df_merge['Close_x'] - df_merge['Close_y'])
    df_merge['Symbol_'] = next_symbol
    #df_merge['Spread_next'] = df_merge['Close_y'] - df_merge['Spread_next']
    if df_merge.iloc[0].Discount >= 1.0 and df_merge.iloc[0].Discount_ >= 1.0:
        count += 1
        for j in range(df_merge.shape[0]):
            print "{0}    [discount: {1:5.2f} {2:5.2f}]   [spread: {3:5.2f} {4:5.2f}]   [cal: {5:5.2f}]   [HG: {6:.4f} {7:.4f}  {8:7.4f}]".format(df_merge.iloc[j].DateTime, df_merge.iloc[j].Discount, df_merge.iloc[j].Discount_, df_merge.iloc[j].Spread, df_merge.iloc[j].Spread_, df_merge.iloc[j].Cal, df_merge.iloc[j].Close_x, df_merge.iloc[j].Close_y, (df_merge.iloc[j].Close_x-df_merge.iloc[j].Close_y))
        dt1 = df_merge.iloc[df_merge.shape[0]-1].DateTime        
        dfz = df_copper[(df_copper.DateTime > dt1) & (df_copper.Discount <= 0.0)]
        if dfz.shape[0] > 0:
            dt2 = dfz.iloc[0].DateTime
            dfz = df_copper[(df_copper.DateTime > dt1) & (df_copper.DateTime <= dt2)]
            print "  ROLL: {0}".format(strdate(dfz.iloc[0].DateTime,'-'))
            days = (dt2 - dt1).days
        else:
            dfz = df_copper[df_copper.DateTime > dt1]
            days = -1
        for j in range(dfz.shape[0]):
            print "{0}    [discount:       {1:5.2f}]   [spread:       {2:5.2f}]   [cal: {3:5.2f}]".format(dfz.iloc[j].DateTime, dfz.iloc[j].Discount, dfz.iloc[j].Spread, dfz.iloc[j].Cal)
        print            
        max_discount = max(df_merge.Discount_.max(), dfz.Discount.max())
        if days == -1:
            print "Trade is still OPEN as of {0}.  Max discount (during entire period) was {1}.".format(dfz.tail(1).squeeze()['DateTime'], max_discount)
        else:
            print "Discount hit zero in {0} days of roll.  Max discount (during entire period) was {1}.".format(days, max_discount)
        print "-" * 116
print
print "Found {0} cases where premium was >= 1.0 AND premium using next month was >= 1.0".format(count)
sys.exit()
############################################################################################################################################


print "Use -e and -x command line args to specify the entry and exit prices  (ex: -e=-1 -x=0)"
print "Use -discount to use discount/premium entry and exit instead of spread prices  (ex: -discount)"
print

args['e'] = 1
args['x'] = 0
args['discount'] = ''

if not ('e' in args and 'x' in args):
    print "Error: Must provide -e and -x command line args for entry and exit prices."
    sys.exit()

entry_price = int(args['e'])
exit_price = int(args['x'])


print "date range:", min_date, "to", max_date
print


df = df_all.sort_values('Date')

side = "BUY"
if entry_price > exit_price:
    side = "SELL"

if 'discount' in args:
    trade_column = "discount"
else:
    trade_column = "spread"


########## READ CALENDAR ROLLS FILE ##########
roll_filename = "calendar_rolls.csv"

df_rolls = pd.read_csv(join(data_folder, "copper", roll_filename), parse_dates=['FirstDate','LastDate'])

count = 0
#for (ix,row) in df_rolls.iterrows():
for i in range(df_rolls.shape[0]):
    row = df_rolls.iloc[i]
    dt2 = row.FirstDate + timedelta(days=1)
    dt1 = dt2 - timedelta(days=14)
    dfx = df[(df.Date>=dt1)&(df.Date<dt2)]
    if dfx.discount.max() >= 1.0:
        if i == 0: continue
        
        prev = df_rolls.iloc[i-1]
        discount_chg = row.FirstDiscount - prev.LastDiscount
        #if discount_chg >= 1.0:
        #if discount_chg >= 0.0:
        if row.FirstDiscount >= 1.0:
            count += 1
            print "roll date: {0}    ".format(row.FirstDate.strftime('%Y-%m-%d')),      #, "       (", dt1.strftime('%Y-%m-%d'), "to", dt2.strftime('%Y-%m-%d'), ")"
            print "discount: {0:.2f}  next_discount: {1:.2f}    (discount_chg = {2:.2f})".format(prev.LastDiscount, row.FirstDiscount, discount_chg)
            dt1 = row.FirstDate
            dt2 = dt1 + timedelta(days=14)
            dfx = df[(df.Date>=dt1)&(df.discount<=0)]
            zero_date = dfx.iloc[0].Date
            days_to_zero = zero_date - dt1
            print "hits zero after {0} days   (on {1})".format(days_to_zero.days, zero_date.strftime('%Y-%m-%d'))
            


            #print dfx
            print

    
print "count =", count



sys.exit()
# Expected return value is a tuple: (sell_row, buy_row, roll_price_adjustment)
trades = backtest(df, trade_column, side, entry_price, exit_price, df_rolls)

print_trades(trade_column, trades, side, entry_price, exit_price, df_rolls)

trade_filename = "backtest {0} {1} {2}.csv".format(trade_column, entry_price, exit_price)
write_trades_file(trade_filename, trade_column, trades, side, entry_price, exit_price, df_rolls)

print
print "Roll dates used from file: ", '"' + roll_filename + '"'
print "Results output to file: ", '"' + trade_filename + '"'



#print_calendar_roll_dates(df_rolls)
#roll_filename = "calendar_rolls.csv".format(trade_column, entry_price, exit_price)
#write_roll_file(roll_filename, calendar_rolls)


