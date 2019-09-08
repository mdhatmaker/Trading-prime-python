from f_date import *
from f_dataframe import *


BUY = 'B'
SELL = 'S'

TRD_ENTRY = 'ENTRY'
TRD_EXIT = 'EXIT'
TRD_STOP = 'STOP'
TRD_EXPIRE = 'EXPIRE'
TRD_OTHER = 'OTHER'

class Trade:
    # symbol : instrument symbol
    # side : BUY or SELL (string constants, 'B' or 'S')
    # qty: number of contracts bought/sold (int)
    # price : price of trade (float)
    # dt: datetime of trade (datetime)
    # trade_type: TRD_ENTRY, TRD_EXIT, TRD_STOP, TRD_EXPIRE, TRD_OTHER (int constants)
    # id: if provided, will be this trade's "id" (otherwise will generate a new, unique id)
    def __init__(self, symbol, side, qty, price, dt, trade_type, id=None):
        self.symbol = symbol
        self.side = side
        self.qty = qty
        self.price = price
        self.dt = dt
        self.ttype = trade_type
        if id == None:
            self.id = to_unixtime()
        else:
            self.id = id

    def __repr__(self):
        return self.to_string()

    def trade_type(self):
        if self.ttype == TRD_ENTRY:
            return "ENTRY"
        elif self.ttype == TRD_EXIT:
            return "EXIT"
        elif self.ttype == TRD_STOP:
            return "STOP"
        elif self.ttype == TRD_EXPIRE:
            return "EXPIRE"
        elif self.ttype == TRD_OTHER:
            return "OTHER"
        else:
            return "(INVALID)"

    def to_string(self):
        str = "{0},{1},{2},{3},{4:.4f},{5},{6}".format(self.dt.strftime("%Y-%m-%d"), self.symbol, self.side, self.qty, self.price, self.trade_type(), self.id)
        return str

    def to_list(self):
        return [self.dt, self.symbol, self.side, self.qty, self.price, self.ttype, self.id]


# This class holds two trades (ENTRY/EXIT)
# it can return calculations such as days (held), profit, etc.
class TradeRoundTrip:
    def __init__(self, trade_entry, trade_exit):
        self.trade_entry = trade_entry
        self.trade_exit = trade_exit

    def __repr__(self):
        return "{0}\n{1}".format(self.trade_entry, self.trade_exit)

    def holding_period(self):
        return (self.trade_exit.dt - self.trade_entry.dt)

    def days(self):
        return self.holding_period().days

    def profit(self):
        if self.trade_entry.side == BUY:
            profit = self.trade_exit.price - self.trade_entry.price
        else:
            profit = self.trade_entry.price - self.trade_exit.price
        return profit

    def to_df(self):
        return get_trades_df([self.trade_entry, self.trade_exit])

#--------------------- MISC TRADE-RELATED FUNCTIONS ----------------------------
def get_trades_df(trade_list):
    lst = [t.to_list() for t in trade_list]
    df = pd.DataFrame(lst, columns=['DateTime','Symbol','Side','Qty','Price','TradeType','ID'])
    return df

def get_trades_list(df):
    trade_list = []
    for ix,row in df.iterrows():
        print(row[0].to_datetime(),row[1],row[2],row[3],row[4],row[5],row[6])
        trade_list.append(Trade(row[1],row[2],row[3],row[4],row[0],row[5],row[6]))
    return trade_list

# ".TRADES.csv" is automatically appended to 'name' provided
# if 'name' is None, name is automatically generated with current datetime
def write_trades_file(trade_list, name=None):
    df = get_trades_df(trade_list)
    if name is None:
        name = "trades_{0}".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    filename = "{0}.TRADES.csv".format(name)
    write_dataframe(df, filename)
    return

def read_trades_file(name):
    filename = "{0}.TRADES.csv".format(name)
    return read_dataframe(filename)

def buy(symbol, qty, price, dt, trade_type, associated_entry_trade=None):
    return Trade(symbol, BUY, qty, price, dt, trade_type, associated_entry_trade)

def sell(symbol, qty, price, dt, trade_type, associated_entry_trade=None):
    return Trade(symbol, SELL, qty, price, dt, trade_type, associated_entry_trade)



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

