# -*- coding: cp1252 -*-
import quandl
import json
import pandas as pd
from datetime import datetime

#-----------------------------------------------------------------------------------------------------------------------

#execfile(r'..\..\..\python\f_analyze.py')
#execfile(r'f_analyze.py')
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


quandl.ApiConfig.api_key = "gCbpWopzuxctHw6y-qq5"


quandl_data = {}


    
def get_hg_calendar(lme_date):
    dt_hg_front = get_front_month_hg(lme_date)
    dt_hg_back = get_spread_back_month_hg(lme_date)
    # Kind of a hack, but if the front and back month are the same, just push the back-month to the following month
    while dt_hg_back <= dt_hg_front:
        dt_hg_back = add_one_month(dt_hg_back)    
    symbol = "QHG{0}-QHG{1}".format(get_mYY(dt_hg_front), get_mYY(dt_hg_back))        
    return symbol

# expiration for HG is 3rd from last business day of month
def get_expiration_date_hg(month, year):
    return last_business_day(month, year, 3)
    #int max_day = tools::get_max_day(month, year);
    #SimpleTime day(year, month, max_day);                           // start at last day of the month
    #int count = 0;                                                  // count the number of business days encountered
    #do {
    #    if (is_business_day(day))
    #        ++count;
    #    if (count < 3) day.add_day(-1);                             // move to previous day
    #} while (count < 3);
    #return day;

def subtract_one_month(dt):
    (m, y) = add_month(dt.month, dt.year, -1)
    return datetime(y, m, dt.day)

def add_one_month(dt):
    (m, y) = add_month(dt.month, dt.year, +1)
    return datetime(y, m, dt.day)

# cal has format "XXXmYY-XXXmYY"
def previous_calendar_hg(cal):
    mo_combos = ["HJ", "HK", "KM", "KN", "NQ", "NU", "UV", "UX", "UZ", "ZF", "ZG", "ZH"]
    mm = cal[3] + cal[10]
    i = mo_combos.index(mm)
    yy1 = cal[4:6]
    yy2 = cal[-2:]
    if i == 0:
        pmm = "ZH"
        yy1 = str(int(yy1)-1)
    else:
        pmm = mo_combos[i-1]
    return cal[0:3] + pmm[0] + yy1 + cal[7:10] + pmm[1] + yy2

# cal has format "XXXmYY-XXXmYY"
def next_calendar_hg(cal):
    mo_combos = ["HJ", "HK", "KM", "KN", "NQ", "NU", "UV", "UX", "UZ", "ZF", "ZG", "ZH"]
    mm = cal[3] + cal[10]
    i = mo_combos.index(mm)
    yy1 = cal[4:6]
    yy2 = cal[-2:]
    if i == len(mo_combos)-1:
        nmm = "HJ"
        yy1 = str(int(yy1)+1)
    else:
        nmm = mo_combos[i+1]
    if nmm=="ZF":
        yy2 = str(int(yy2)+1)
    return "{0}{1}{2}-{3}{4}{5}".format(cal[0:3], nmm[0], yy1, cal[7:10], nmm[1], yy2)
        
        
def previous_front_month_hg(mYY):
    hg_months = "HKNUZ"
    mc = mYY[0]
    yy = mYY[1:]
    i = hg_months.index(mc)
    if i == 0:
        pm = "Z"
        pyy = str(int(yy)-1)
    else:
        pm = hg_months[i-1]
        pyy = yy
    return pm + pyy;

# cal has format "XXXmYY-XXXmYY"
def get_hg_calendar_expiry(cal):
    mYY1 = cal[3:6]
    mYY2 = cal[-3:]
    exp1 = get_expiry(mYY1)
    exp2 = get_expiry(mYY2) - timedelta(days=90)
    print "{0}  {1}  {2}".format(exp1.strftime("%Y-%m-%d"), exp2.strftime("%Y-%m-%d"), cal)
    return min(exp1, exp2)                          # return expiration of either front or back month, whichever is earlier
    
def get_expiry(mYY):
    mc = mYY[0]
    yy = mYY[1:]
    m = get_month_number(mc)
    y = 2000 + int(yy)
    expiry = get_expiration_date_hg(m, y)
    return expiry    
    
def get_front_month_hg(dt):
    # Here are the front months we use for HG
    # H   K   N   U   Z
    # Mar May Jul Sep Dec

    ystr = str(dt.year)[-2:]
    expiry_mar = subtract_one_month(get_expiry("H" + ystr))
    expiry_may = subtract_one_month(get_expiry("K" + ystr))
    expiry_jul = subtract_one_month(get_expiry("N" + ystr))
    expiry_sep = subtract_one_month(get_expiry("U" + ystr))
    expiry_dec = subtract_one_month(get_expiry("Z" + ystr))

    if dt < expiry_mar:
        return datetime(dt.year, 3, 1)
    elif dt < expiry_may:
        return datetime(dt.year, 5, 1)
    elif dt < expiry_jul:
        return datetime(dt.year, 7, 1)
    elif dt < expiry_sep:
        return datetime(dt.year, 9, 1)
    elif dt < expiry_dec:
        return datetime(dt.year, 12, 1)
    else:
        return datetime(dt.year+1, 3, 1)

def get_spread_back_month_hg(dt):
    # return �back month� (2nd contract in calendar spread) for the given date
    # I *think* this means date + 90 days (as a starting point), but again, this could take into account LME holidays, etc...
    #//TimeTm lme_back_date = add_day(dt, +90);
    lme_back_date = dt;
    lme_back_date += timedelta(days=90)                         # +90 days
    if lme_back_date > get_expiry(get_mYY(lme_back_date)):
        lme_back_date = lme_back_date.replace(day=1)
        lme_back_date = add_one_month(lme_back_date)
    else:
        lme_back_date = lme_back_date.replace(day=1)            # return the 1st day of the month
    return lme_back_date

def get_copper_spread(df_lme):
    unique_dates = pd.DatetimeIndex(df_lme.DateTime).normalize().unique()

    
# given futures date in "mY" format ("H6"), return Quandl symbol ("CME/HGH2016")
# instrument is Quandl symbol (i.e. "CL")
def get_quandl_symbol(fut, instrument):
    m1 = fut[0]
    y1 = "20" + fut[1:3]
    symbol = "CME/" + instrument + m1 + y1
    return (m1, y1, symbol)

def get_quandl_cal_symbols(cal, instrument="HG"):
    mc1 = cal[0]
    ystr1 = "20" + cal[1:3]
    mc2 = cal[3]
    ystr2 = "20" + cal[4:6]
    symbol1 = "CME/" + instrument + mc1 + ystr1
    symbol2 = "CME/" + instrument + mc2 + ystr2
    return (mc1, ystr1, mc2, ystr2, symbol1, symbol2)

# given calendar symbol in "mYmY" format ("H6J6"), return Multichart symbol ("QHGH16-QHGJ16")
def get_multichart_symbol(cal, instrument="QHG"):
    m1 = cal[0]
    y1 = cal[1:3]
    m2 = cal[3]
    y2 = cal[4:6]
    symbol = instrument + m1 + y1 + "-" + instrument + m2 + y2
    return (m1, y1, m2, y2, symbol)

# given symbol in "QHGmYY-QHGmYY" format, return cal symbol in "mYmY" format
def multichart_to_cal(qcal):
    m1 = qcal[3]
    y1 = qcal[5]
    m2 = qcal[10]
    y2 = qcal[12]
    symbol = m1 + y1 + m2 + y2
    return (m1, y1, m2, y2, symbol)

# data is a dictionary and fut is the futures date in "mY" format (i.e. "H6")
# instrument is Quandl symbol (i.e. "CL")
def get_futures_data(data, fut, instrument):
    (m1, y1, symbol) = get_quandl_symbol(fut, instrument)
    y_prev = previous_year(y1)
    dts = ""
    dte = ""
    if m1 == "F":
        dts = y_prev + "-7-1"
        dte = y1 + "-1-31"
    elif m1 == "G":
        dts = y_prev + "-8-1"
        dte = y1 + "-2-28"
    elif m1 == "H":
        dts = y_prev + "-9-1"
        dte = y1 + "-3-31"
    elif m1 == "J":
        dts = y_prev + "-11-1"
        dte = y1 + "-4-30"
    elif m1 == "K":
        dts = y_prev + "-12-1"
        dte = y1 + "-5-31"
    elif m1 == "M":
        dts = y1 + "-2-1"
        dte = y1 + "-6-30"
    elif m1 == "N":
        dts = y1 + "-3-1"
        dte = y1 + "-7-31"
    elif m1 == "Q":
        dts = y1 + "-4-1"
        dte = y1 + "-8-31"
    elif m1 == "U":
        dts = y1 + "-5-1"
        dte = y1 + "-9-30"
    elif m1 == "V":
        dts = y1 + "-6-1"
        dte = y1 + "-10-31"
    elif m1 == "X":
        dts = y1 + "-7-1"
        dte = y1 + "-11-30"
    elif m1 == "Z":
        dts = y1 + "-7-1"
        dte = y1 + "-12-31"
    else:
        raise ValueError

    print(symbol, dts, dte)
    
    data[fut] = quandl.get(symbol, start_date=dts, end_date=dte)
    
    return

def get_futures_data_for_year(data, year, instrument):
    y = str(year)[-2:]
    get_futures_data(data, "F" + y, instrument)
    get_futures_data(data, "G" + y, instrument)
    get_futures_data(data, "H" + y, instrument)
    get_futures_data(data, "J" + y, instrument)
    get_futures_data(data, "K" + y, instrument)
    get_futures_data(data, "M" + y, instrument)
    get_futures_data(data, "N" + y, instrument)
    get_futures_data(data, "Q" + y, instrument)
    get_futures_data(data, "U" + y, instrument)
    get_futures_data(data, "V" + y, instrument)
    get_futures_data(data, "X" + y, instrument)
    get_futures_data(data, "Z" + y, instrument)
    return

# data is a dictionary and cal is the calendar symbol in "mYmY" format (i.e. "H6J6")
#def get_calendar_data(data, cal, instrument="HG"):
def get_calendar_data(cal, instrument="HG"):
    (mc1, ystr1, mc2, ystr2, symbol1, symbol2) = get_quandl_cal_symbols(cal)
    """
    dt2 = datetime(y1, get_month_number(mc1), 31)
    dt1 = dt2.replace(day=1)
    dt1 = dt1 - timedelta(months=6)
    print dt1, dt2
    sys.exit()
    """
    y_prev = previous_year(ystr1)
    dts = ""
    dte = ""
    if (mc1 == "H" and (mc2 == "J" or mc2 == "K")):
        dts = y_prev + "-9-1"
        dte = ystr2 + "-3-31"
    elif (mc1 == "K" and (mc2 == "M" or mc2 == "N")):
        dts = y_prev + "-12-1"
        dte = ystr2 + "-5-31"
    elif (mc1 == "N" and (mc2 == "Q" or mc2 == "U")):
        dts = ystr1 + "-3-1"
        dte = ystr2 + "-7-31"
    elif (mc1 == "U" and (mc2 == "V" or mc2 == "X" or mc2 == "Z")):
        dts = ystr1 + "-5-1"
        dte = ystr2 + "-9-30"
    elif (mc1 == "Z" and (mc2 == "F" or mc2 == "G" or mc2 == "H")):
        dts = ystr1 + "-7-1"
        dte = ystr2 + "-12-31"
    else:
        raise ValueError

    print(symbol1, symbol2, dts, dte)
    
    #data[cal[0:3]] = quandl.get(symbol1, start_date=dts, end_date=dte)
    #data[cal[3:]] = quandl.get(symbol2, start_date=dts, end_date=dte)

    df1 = quandl.get(symbol1, start_date=dts, end_date=dte)
    df2 = quandl.get(symbol2, start_date=dts, end_date=dte)
    #print df1
    #print df2
    df1.rename(columns={"Prev. Day Open Interest":"OpenInterest", "Open Interest":"OpenInterest", "Previous Day Open Interest":"OpenInterest"}, inplace=True)
    df2.rename(columns={"Prev. Day Open Interest":"OpenInterest", "Open Interest":"OpenInterest", "Previous Day Open Interest":"OpenInterest"}, inplace=True)
    df1.index.names = ["DateTime"]
    df2.index.names = ["DateTime"]
    df1 = df1.astype({"Volume":"int", "OpenInterest":"int"})
    df2 = df2.astype({"Volume":"int", "OpenInterest":"int"})
    if df1.shape[0] > 0:
        mYY1 = cal[0] + cal[1:3]
        filename1 = "quandl_QHG" + mYY1 + ".DF.csv"
        df1 = df1.reset_index()
        write_dataframe(df1, join(quandl_folder, filename1))
    if df2.shape[0] > 0:
        mYY2 = cal[3] + cal[4:6]
        filename2 = "quandl_QHG" + mYY2 + ".DF.csv"
        df2 = df2.reset_index()
        write_dataframe(df2, join(quandl_folder, filename2))
    if df1.shape[0] > 0 and df2.shape[0] > 0:
        dfx = pd.merge(df1, df2, on="DateTime")
        dfx['Cal'] = (dfx.Settle_x - dfx.Settle_y) * 100
        dfx.sort_values('DateTime', inplace=True)
        dfx['DateTime'] = dfx.DateTime.apply(update_hour)
        dfx.drop(['Open_x','High_x','Low_x','Last_x','Change_x','Open_y','High_y','Low_y','Last_y','Change_y'], axis=1, inplace=True)
        filenamex = "quandl_QHG{0}-QHG{1}.DF.csv".format(mYY1, mYY2)
        write_dataframe(dfx, join(quandl_folder, filenamex))        
    return

#def get_calendar_data_for_year(data, year, instrument="HG"):
def get_calendar_data_for_year(year, instrument="HG"):
    y = str(year)[-2:]
    y2 = str(year+1)[-2:]
    get_calendar_data("H" + y + "J" + y)
    get_calendar_data("H" + y + "K" + y)
    get_calendar_data("K" + y + "M" + y)
    get_calendar_data("K" + y + "N" + y)
    get_calendar_data("N" + y + "Q" + y)
    get_calendar_data("N" + y + "U" + y)
    get_calendar_data("U" + y + "V" + y)
    get_calendar_data("U" + y + "X" + y)
    get_calendar_data("U" + y + "Z" + y)
    get_calendar_data("Z" + y + "F" + y2)
    get_calendar_data("Z" + y + "G" + y2)
    get_calendar_data("Z" + y + "H" + y2)
    return

# Update only the more recent Quandl data for HG settlement prices and (derived) HG calendar spreads
def update_quandl_data(force_update_all=False):
    y1 = datetime.now().year
    y2 = y1 + 1
    if force_update_all == True: y1 = 2010
    for y in range(y1, y2+1):
        get_calendar_data_for_year(y)
    return

# Given a datetime, return the same datetime but with the hour set to 13 (1pm)
# (used for setting the time on HG settlements)
def update_hour(dt):
    hour = 13
    #print dt,
    dt = datetime(dt.year, dt.month, dt.day, hour, 0, 0)
    #print dt
    return datetime(dt.year, dt.month, dt.day, hour, 0, 0)

#def get_calendar_price(data, cal, dt, tm):
# expects cal in form "QHGmYY-QHGmYY"
def get_calendar_price(cal, dt):
    #print "   {0} {1}".format(cal, dt)
    cal1 = cal[3:6]
    cal2 = cal[10:13]
    """
    if not quandl_data.has_key(cal1):
        filename = "quandl_HG" + cal1[0] + "20" + cal1[1:3] + ".DF.csv"
        quandl_data[cal1] = read_dataframe(join(quandl_folder, filename))        
    if not quandl_data.has_key(cal2):
        filename = "quandl_HG" + cal2[0] + "20" + cal2[1:3] + ".DF.csv"
        quandl_data[cal2] = read_dataframe(join(quandl_folder, filename))        
        #return (False, 0.0000)
    """
    if not cal in quandl_data:
        filename = "quandl_{0}.DF.csv".format(cal)
        if os.path.exists(join(quandl_folder, filename)):
            quandl_data[cal] = read_dataframe(join(quandl_folder, filename))
            print("Loaded '{0}'".format(filename))

    #if not quandl_data.has_key(cal):
    if not cal in quandl_data:
        return np.nan
    
    dfx = quandl_data[cal]
    dfx = dfx[dfx.DateTime < dt]
    #print "{0} {1} {2}".format(cal, dfx.DateTime.min(), dfx.DateTime.max())
    if dfx.shape[0] == 0:
        #print "dataframe has NO DATA: '{0}'".format(cal)
        return np.nan
    else:
        last_ix = dfx.shape[0]-1
        return round(dfx.iloc[last_ix].Cal, 4)  #, cal, dfx.iloc[last_ix].DateTime, dt
        #return dfx.iloc[dfx.shape[0]-1].Cal
        #return round((settle1 - settle2) * 100, 4)


def lbd(m, y):
    return last_business_day(m, y, 3)

def get_hg_front_month(dt):
    m = dt.month
    y = dt.year
    if dt >= lbd(11, y):
        return "H" + str(y+1)[-2:]
    elif dt >= lbd(8, y):
        return "Z" + str(y)[-2:]
    elif dt >= lbd(6, y):
        return "U" + str(y)[-2:]
    elif dt >= lbd(4, y):
        return "N" + str(y)[-2:]
    elif dt >= lbd(2, y):
        return "K" + str(y)[-2:]
    else:
        return "H" + str(y)[-2:]

def get_hg_back_month(dt):
    m = dt.month
    y = dt.year
    if dt >= lbd(11, y):
        return "J" + str(y+1)[-2:]
    elif dt >= lbd(10, y):
        return "G" + str(y+1)[-2:]
    elif dt >= lbd(8, y):
        return "F" + str(y+1)[-2:]
    elif dt >= lbd(7, y):
        return "X" + str(y)[-2:]
    elif dt >= lbd(6, y):
        return "V" + str(y)[-2:]
    elif dt >= lbd(5, y):
        return "U" + str(y)[-2:]
    elif dt >= lbd(4, y):
        return "Q" + str(y)[-2:]
    elif dt >= lbd(3, y):
        return "N" + str(y)[-2:]
    elif dt >= lbd(2, y):
        return "M" + str(y)[-2:]
    elif dt >= lbd(1, y):
        return "K" + str(y)[-2:]
    else:
        return "J" + str(y)[-2:]


def get_hg_calendar_symbol(dt):
    return "QHG{0}-QHG{1}".format(get_hg_front_month(dt), get_hg_back_month(dt))

def get_copper_discount(df_lme, start_year=2010, end_year=datetime.now().year):
    unique_dates = pd.DatetimeIndex(df_lme.DateTime).normalize().unique()

    df_all = get_copper_spread(df_lme)
    min_date = unique_dates.min()
    min_date = datetime(start_year, 1, 1)
    max_date = unique_dates.max()
    max_date = datetime(end_year, 12, 31)

    df_all = pd.DataFrame()

    y1 = min_date.year
    y2 = max_date.year
    print(y1, y2)
    for y in range(y1, y2+1):
        dt1 = datetime(y, 1, 1,)
        dt2 = datetime(y, 12, 31)
        df_year = df_lme[(df_lme.DateTime >= dt1) & (df_lme.DateTime <= dt2)].copy()
        df_year = df_year.sort_values("DateTime")

        for m in range(11, -1, -1):
            if m == 0:            
                dt2 = lbd(1, y)
                dfx = df_year[df_year.DateTime < dt2].copy()
                calendar_symbol = get_hg_calendar_symbol(dt2 - timedelta(days=1))
                dfx['Symbol'] = calendar_symbol
            elif m == 11:
                dt1 = lbd(11, y)
                dfx = df_year[df_year.DateTime >= dt1].copy()
                calendar_symbol = get_hg_calendar_symbol(dt1)
                dfx['Symbol'] = calendar_symbol        
            else:
                dt1 = lbd(m, y)
                dt2 = lbd(m+1, y)
                dfx = df_year[(df_year.DateTime >= dt1) & (df_year.DateTime < dt2)].copy()
                calendar_symbol = get_hg_calendar_symbol(dt1)
                dfx['Symbol'] = calendar_symbol

            filename1 = calendar_symbol[:6] + " (1 Hour).csv"
            hg_filename1 = join(df_folder, filename1)
            df_hg = read_dataframe(hg_filename1)
            df_hg = df_hg.sort_values("DateTime")
            dt = df_hg.iloc[df_hg.shape[0]-1].DateTime
            x = get_calendar_price(calendar_symbol, dt) 
            print(calendar_symbol, dt, x)
            df_hg['Cal'] = np.nan
            for ix,row in df_hg.iterrows():
                x = get_calendar_price(calendar_symbol, row.DateTime)
                df_hg.set_value(ix, 'Cal', x)
            df_hg = df_hg.dropna()
            dfx = pd.merge(dfx, df_hg, on="DateTime", suffixes=('_LME', '_HG'), sort=False).dropna()

            if df_all.shape[0] == 0:
                df_all = dfx.copy()
            else:
                df_all = df_all.append(dfx, ignore_index=True)
            
    df_all.sort_values('DateTime', ascending=True, inplace=True)

    df_all['Spread'] = (df_all.Close_HG - (.000454 * df_all.Close_LME)) * 100
    df_all.Spread = df_all.Spread.round(4)
    df_all['Discount'] = df_all.Spread - df_all.Cal

    #df_all.drop(['Open_LME', 'Open_HG', 'High_LME', 'High_HG', 'Low_LME', 'Low_HG'], axis=1, inplace=True)
    #df_all.rename(columns={ 'Close_LME':'LME', 'Close_HG':'HG' }, inplace=True)
    df_all.drop(['Open_LME','High_LME','Low_LME','Volume_LME','Open_HG','High_HG','Low_HG','Volume_HG'], axis=1, inplace=True)
    df_all.rename(columns={ 'Close_LME':'LME', 'Close_HG':'HG' }, inplace=True)
    sequence = ['DateTime', 'HG', 'LME', 'Spread', 'Cal', 'Discount', 'Symbol'] #, 'Volume_HG', 'Volume_LME']
    df_all = df_all.reindex(columns=sequence)
    return df_all

def write_df(df, filename):
    write_dataframe(df, join(project_folder, filename))
    print("Output to file: '{0}'".format(filename))
    print()
    return


################################################################################

#symbol = "CME/HGK2017-HGM2017"
#data = quandl.get(symbol)                       # get data for symbol
# Change formats
#data = quandl.get(symbol, returns="numpy")      # get data in NumPy array
# Make a filtered time-series call
#data = quandl.get(symbol, start_date="2001-12-31", end_date="2005-12-31")   # get data for date range
#data = quandl.get(["NSE/OIL.1", "WIKI/AAPL.4"]) # get specific columns
#data = quandl.get("WIKI/AAPL", rows=5)          # get last 5 rows
# Preprocess the data
#data = quandl.get("EIA/PET_RWTC_D", collapse="monthly") # change sampling frequency
#data = quandl.get("FRED/GDP", transformation="rdiff")   # perform elementary calculations on the data


"""
cl_data = {}

get_futures_data_for_year(cl_data, 2015, "CL")
get_futures_data_for_year(cl_data, 2016, "CL")
get_futures_data_for_year(cl_data, 2017, "CL")
"""

#dt = datetime(2017, 8, 1, 12, 1, 3)
#cal = get_calendar_price("QHGZ17-QHGG18", dt)
#get_calendar_data("Z17G18")
#sys.exit()

"""
########## CLEAN UP OUTPUT DATA -- LOOK FOR SPREAD OUTLIERS ##########
filename = join(project_folder, "copper_discount.DF.csv")
df = read_dataframe(filename)

#df['avg'] = pd.rolling_mean(df.Spread, 10)
df['avg'] = df.Spread.rolling(window=7,center=True).mean()
df['diff'] = (df.Spread - df.avg) / df.avg * 100

dfx = df[(df['diff'] > -10.0) & (df['diff'] < 10.0)].copy()
dfx.drop(['avg','diff'], axis=1, inplace=True)
#dfx = dfx[['DateTime','Symbol','Spread','diff']]
print dfx
output_filename = "copper_discount.2.DF.csv"
write_dataframe(dfx, join(project_folder, output_filename))
sys.exit()
"""

"""
########## READ HG DATA FROM QUANDL AND CONSTRUCT HG CALENDARS ##########
for y in range(2010, 2018+1):
    get_calendar_data_for_year(y)
sys.exit()
"""

"""
# TEST RUN THROUGH A RANGE OF CONSECUTIVE HG CALENDARS TO CHECK THEIR START/END DATE ACCURACY
cal = "QHGH10-QHGJ10"
while True:
    exp = get_hg_calendar_expiry(cal)
    hgcal1 = get_hg_calendar(exp)
    hgcal2 = get_hg_calendar(exp+timedelta(days=1))
    print " ", cal, strdate(exp), "  ", hgcal1[3:6] + " " + hgcal1[-3:], "   ", hgcal2[3:6] + " " + hgcal2[-3:]
    #print
    cal = next_calendar_hg(cal)
    if cal == "QHGU10-QHGV10": break

base = datetime(2009, 10, 1)
num_days = 365
last = None
date_list = [base + timedelta(days=x) for x in range(0, num_days)]
for dt in date_list:
    hgcal = get_hg_calendar(dt)
    hgcal = hgcal[3:6] + " " + hgcal[-3:]
    if hgcal != last:
        print strdate(dt), "  ", hgcal
        last = hgcal
    
sys.exit()
"""

# If we need to update the HG settlement price data from Quandl...
#update_quandl_data()

########## (RE)CREATE COPPER DISCOUNT FILE USING DATA FILES FOR LME AND HG ##########
lme_filename = join(df_folder, "M.CU3=LX (1 Hour).csv")
#lme_filename = join(data_folder, "misc", "LME_settlement.csv")
df_lme = read_dataframe(lme_filename)
df = get_copper_discount(df_lme) #, start_year=2017)
write_df(df, "copper_discount.2.DF.csv")

df_lme = df_lme[df_lme['DateTime'].dt.hour == 13]       # get only LME datapoints at time 13:00:00 (to match settlement time for HG)
df = get_copper_discount(df_lme) #, start_year=2017)
write_df(df, "copper_settle_discount.2.DF.csv")



sys.exit()


########## READ COPPER SPREAD DATA AND CALCULATE DISCOUNT/PREMIUM ##########
mydata = {}

in_filename = "copper_spread.csv"
out_filename = "copper_premium_discount.csv"
print("\nReading from copper spread data file: " + in_filename)
print("Writing to updated premium/discount file: " + out_filename,)
out_file = open(join(project_folder, out_filename), 'w')
out_file.write("DateTime,HG,LME,Spread,Cal,Discount,Symbol\n")

in_file = open(join(project_folder, in_filename), 'r')
line = in_file.readline()     # ignore first line of file
line = in_file.readline()     # start processing with second line of file
count = 0
while line:
    if count % 100 == 0: print(".",)
    count+=1
    #print line,
    columns = line.split(',')
    dt = columns[0]
    tm = columns[1]
    hg = columns[2]
    lme = columns[3]
    spread = columns[4]
    calendar_symbol = columns[5]

    (m1, y1, m2, y2, cal) = multichart_to_cal(calendar_symbol)
    (is_valid, cal_price) = get_calendar_price(data, cal, dt, tm)

    if is_valid:
        #columns[4] = str(float(spread) * 100)
        #columns.append(calendar_symbol)             # add 7th column (columns[6])
        #columns.append(calendar_symbol)             # add 8th column (columns[7])
        #columns[5] = str(cal_price * 100)
        #discount = float(spread) * 100 - cal_price * 100
        cal_price *= 100
        discount = float(spread) - cal_price
        #columns[6] = str(discount)

        out_columns = []
        dtm = get_datetime(dt, tm)
        dtm_str = str(dtm)
        out_columns.append(dtm_str)                 # column 0
        out_columns.append(hg)                      # column 1
        out_columns.append(lme)                     # column 2
        out_columns.append(spread)                  # column 3
        out_columns.append(str(cal_price))          # column 4
        out_columns.append(str(discount))           # column 5
        out_columns.append(calendar_symbol)         # column 6
        out_file.write(",".join(out_columns))

        #print ",".join(columns),
        #my_datetime = get_datetime(columns[0], columns[1])
        #my_spread = float(spread) * 100
        #my_discount = float(spread) * 100 - cal_price * 100
        my_datetime = dtm
        my_spread = float(spread)
        my_discount = float(spread) - cal_price
        mydata[str(my_datetime)] = {"spread":my_spread, "discount":my_discount, "symbol":calendar_symbol}
        
    line = in_file.readline()

out_file.close()
in_file.close()
print

# Write processed data to JSON file
json_filename = "quandl_hg.json"
print("Writing JSON file: " + json_filename)
f = open(join(project_folder, json_filename), 'w')
f.write(json.dumps(mydata))
f.close()


########## CREATE CALENDAR ROLLS FILE ##########
#df = pd.read_csv(folder + "copper_premium_discount.csv", parse_dates=[0], index_col=0)
df = pd.read_csv(join(project_folder, "copper_premium_discount.csv"), parse_dates=['DateTime'])

#calendar_rolls = []

roll_filename = "calendar_rolls.csv"
print("Writing calendar roll file: " + roll_filename)
fout = open(join(project_folder, roll_filename), 'w')
fout.write("Symbol,FirstDate,FirstCal,FirstDiscount,LastDate,LastCal,LastDiscount\n")

# Group the data by the calendar spread symbol
sorted_symbols = df_get_sorted_symbols(df)
for symbol in sorted_symbols:
    df_bysymbol = df[df.Symbol == symbol].sort_values('DateTime')
    first = df_bysymbol.iloc[0]
    last = df_bysymbol.iloc[-1]
    #tuple = (
    #    { "Date": first.DateTime, "discount": first.Discount, "symbol": symbol.strip() },       
    #    { "Date": last.DateTime, "discount":last.Discount, "symbol": symbol.strip() }
    #)     
    #calendar_rolls.append(tuple)
    d1_str = first.DateTime.strftime("%Y-%m-%d")
    d2_str = last.DateTime.strftime("%Y-%m-%d")
    output = "{0},{1},{2},{3},{4},{5},{6}".format(symbol.strip(), d1_str, first.Cal, first.Discount, d2_str, last.Cal, last.Discount)
    #print output
    fout.write(output + "\n")

fout.close()



print("Done.")






