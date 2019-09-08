
#-----------------------------------------------------------------------------------------------------------------------

execfile("f_analyze.py")

#-----------------------------------------------------------------------------------------------------------------------


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

def get_output(row):
    roll_count_indicator = '*' * row.RollCount
    output = '{0:.2f},{1},{2:.2f},{3:.2f},{4:.2f},{5:.2f},{6:.2f},{7:.2f},{8},{9},{10},{11}'.format(row.SpreadProfit, row.Side, row.EntryDiscount, row.ExitDiscount, row.AdjustDiscount, row.EntrySpread, row.ExitSpread, row.AdjustSpread, row.EntryDate.strftime("%Y-%m-%d"), row.ExitDate.strftime("%Y-%m-%d"), row.HoldingDays, roll_count_indicator)
    return output

def print_trades(df_trades):
    avg_profit = 0.0
    avg_days = 0.0
    for (index, row) in df_trades.iterrows():
        output = get_output(row)
        print output
        avg_profit += row.SpreadProfit
        avg_days += row.HoldingDays
    avg_profit /= len(df_trades.index)
    avg_days /= len(df_trades.index)
    print
    print len(df_trades), "trades"
    print "average spread profit: %.2f" % (avg_profit)
    print "average holding period (days): %.1f" % (avg_days)
    return

"""
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
"""


################################################################################

#print "Use -e and -x command line args to specify the entry and exit prices  (ex: -e=-1 -x=0)"
#print "Use -discount to use discount/premium entry and exit instead of spread prices  (ex: -discount)"
print

#args['e'] = 1
#args['x'] = 0
#args['discount'] = ''

#if not ('e' in args and 'x' in args):
#    print "Error: Must provide -e and -x command line args for entry and exit prices."
#    sys.exit()

#entry_price = int(args['e'])
#exit_price = int(args['x'])


print "date range:", min_date, "to", max_date
print


df = df_all.sort_values('Date')


########## READ CALENDAR ROLLS FILE ##########
trade_filename = "backtest discount 1 0.csv"
df_trades = pd.read_csv(folder + trade_filename, parse_dates=['EntryDate','ExitDate'])

# When SELLING
df_trades['SpreadProfit'] = df_trades.EntrySpread - df_trades.ExitSpread + df_trades.AdjustSpread
# When BUYING
#df_trades['SpreadProfit'] = df_trades.EntrySpread - df_trades.ExitSpread + df_trades.AdjustSpread


print_trades(df_trades)


df_losers = df_trades[df_trades.SpreadProfit < 0]
df_winners = df_trades[df_trades.SpreadProfit > 0]
df_with_rolls = df_trades[df_trades.RollCount > 0]
df_without_rolls = df_trades[df_trades.RollCount == 0]

print "\nLosers:"
print df_losers.SpreadProfit.describe()
print "\nWinners:"
print df_winners.SpreadProfit.describe()
print "\nWith Rolls:"
print df_with_rolls.SpreadProfit.describe()
print "\nWithout Rolls:"
print df_without_rolls.SpreadProfit.describe()


print
#print "Roll dates used from file: ", '"' + roll_filename + '"'
#print "Results output to file: ", '"' + trade_filename + '"'



