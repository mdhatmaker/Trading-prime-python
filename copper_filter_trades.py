
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

def get_trade_row(t):
    output = '{0},{1:.2f},{2:.2f},{3:.2f},{4:.2f},{5:.2f},{6:.2f},{7},{8},{9},{10}'.format(t.Side, t.EntryDiscount, t.ExitDiscount, t.AdjustDiscount, t.EntrySpread, t.ExitSpread, t.AdjustSpread, t.EntryDate.strftime("%Y-%m-%d"), t.ExitDate.strftime("%Y-%m-%d"), t.HoldingDays, t.RollCount)
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

def write_trades_file(filename, df_trades):
    f = open(join(folder, "copper", filename), 'w')
    f.write("Side,EntryDiscount,ExitDiscount,AdjustDiscount,EntrySpread,ExitSpread,AdjustSpread,EntryDate,ExitDate,HoldingDays,RollCount\n")
    for index, row in df_trades.iterrows():
        output = get_trade_row(row)
        f.write(output + '\n')
    f.close()
    return


################################################################################

print "Use -tradefile command line arg to specify the file containing copper trades  (ex: -tradefile='backtest discount 1 0.csv')"
print "Use -days to specify number of days after roll dates  (ex: -days=7)"
print

args['tradefile'] = 'backtest discount 1 0.csv'
args['days'] = 7

if not ('tradefile' in args or 'days' in args):
    print "Error: Must provide -tradefile and -days command line args for file containing trades and days after roll dates."
    sys.exit()

trade_filename = args['tradefile']
days = int(args['days'])


print "date range:", min_date, "to", max_date
print


df = df_all.sort_values('Date')



########## READ CALENDAR ROLLS FILE ##########
roll_filename = "calendar_rolls.csv"

df_rolls = pd.read_csv(join(folder, "copper", roll_filename), parse_dates=['FirstDate','LastDate'])

########## READ TRADES FILE ##########
df_trades = pd.read_csv(join(folder, "copper", trade_filename), parse_dates=['EntryDate','ExitDate'])
df2 = pd.DataFrame(data=None, columns=df_trades.columns)    #,index=df_trades.index)
for index, row in df_trades.iterrows():
    trade_date = row['EntryDate']
    for ix2, r in df_rolls.iterrows():
        day_count = (trade_date - r['FirstDate']).days
        if day_count >= 0 and day_count <= days:
            #print day_count
            print get_trade_row(row)
            df2.loc[len(df2)] = row         # add row to df2 dataframe

df2 = df2.sort_values('Symbol')
total_trade_count = df_trades.shape[0]
filter_trade_count = df2.shape[0]

print
print "total trades:", total_trade_count
print "filtered trades:", filter_trade_count


filename, file_extension = splitext(trade_filename)
output_filename = filename + ".filter" + file_extension

write_trades_file(join(folder, "copper", output_filename), df2)


print
print "Roll dates read from file: ", '"' + roll_filename + '"'
print "Copper trades read from file: ", '"' + trade_filename + '"'
print "Results output to file: ", '"' + output_filename + '"'


