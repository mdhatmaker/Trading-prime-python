from datetime import datetime, date, time, timedelta
import os.path
import pandas as pd
from pandas.tseries.offsets import BDay
from dateutil import relativedelta

monthcodes = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']
weekdays = { 'Mon':0, 'Tue':1, 'Wed':2, 'Thu':3, 'Fri':4, 'Sat':5, 'Sun':6 }

# TODO: Do we need this with the above weekdays dictionary?
# constants representing the Python datetime.weekday() numbering of weekdays
WEEKDAY_MON = 0
WEEKDAY_TUE = 1
WEEKDAY_WED = 2
WEEKDAY_THU = 3
WEEKDAY_FRI = 4
WEEKDAY_SAT = 5
WEEKDAY_SUN = 6

# given a year string in "YYYY" format, return a string containing the previous year
def previous_year(yr_str):
    yr = int(yr_str)
    return str(yr-1)

def get_date_from_yyyymmdd(d_str):
    # ex: "20160831"
    yyyy = int(d_str[0:4])
    mm = int(d_str[4:6])
    dd = int(d_str[6:8])
    return datetime(yyyy, mm, dd, 0, 0, 0)

def get_datetime_from_dt(dt_str):
    # ex: "2016-07-01 11:00:00"
    yyyy = int(dt_str[0:4])
    mm = int(dt_str[5:7])
    dd = int(dt_str[8:10])
    HH = int(dt_str[11:13])
    MM = int(dt_str[14:16])
    SS = int(dt_str[17:19])
    return datetime(yyyy, mm, dd, HH, MM, SS)

def get_datetime_from_d_t(d_str, t_str):
    vd = d_str.split('/')
    vt = t_str.split(':')
    return datetime(int(vd[2]), int(vd[0]), int(vd[1]), int(vt[0]), int(vt[1]), int(vt[2]))

def get_datetime(dt, tm):
    d = dt.split('/')
    t = tm.split(':')
    return datetime(int(d[2]), int(d[0]), int(d[1]), int(t[0]), int(t[1]), int(t[2]))

def get_quandl_date(dt):
    d = dt.split('/')
    d1 = d[2]
    d2 = "0" + d[0]
    d3 = "0" + d[1]
    #return dt1 + "-" + dt2[:2] + "-" + dt3[:2]
    return datetime(int(d1), int(d2), int(d3), 0, 0, 0)

def sortable_symbol(symbol):
    if '-' in symbol:
        return symbol[4:6] + symbol[3] + symbol[11:13] + symbol[10]
    else:
        return symbol[4:6] + symbol[3]

#def sortable_calendar(calendar_symbol):
#    return calendar_symbol[4:6] + calendar_symbol[3] + calendar_symbol[11:13] + calendar_symbol[10]

# Function that can compare either symbols ('XXXmYY') or calendar symbols ('XXXmYY-XXXmYY')
def compare_calendar(symbol1, symbol2):
    cmp1 = sortable_symbol(symbol1)
    cmp2 = sortable_symbol(symbol2)
    if cmp1 < cmp2:
        return -1
    elif cmp1 > cmp2:
        return 1
    else:
        return 0

# Return the integer month (1-12) for a given monthcode
# pass in a monthcode ('F'...'Z')
def get_month_number(monthcode):
    return monthcodes.index(monthcode)+1
    
# Return a "CME-style" monthcode (F, G, H, J, K, M, N, Q, U, V, X, Z)
# pass in an integer month (1-12)
def get_monthcode(month):
    return monthcodes[month-1]

# Return the integer month and year that results from adding month_count months to the month provided (-1=previous month, +1=next month, etc.)
# pass in an integer month (1-12) and year (2-digit or 4-digit) and a month_count of any positive or negative integer
def add_month(month, year, month_count):
    if not month in range(1,12+1):
        raise Exception("Illegal month passed to add_month function: " + str(month))
    (m, y) = (month, year)
    m += month_count
    if m >= 1:
        y += int((m-1) / 12)    # adjust the year; m=12 will NOT add 1 whereas m=13 WILL add 1, etc...
        m = 1 + (m-1) % 12      # (m-1) % 12 gives 0 for m=1, 1 for m=2, 2 for m=3, ..., 11 for m=12, 0 for m=13, 1 for m=14, etc...
        return (m, y)
    else:
        m -= 1                  # subtract one more because we go through zero to get to negative
        y += int(m / 12)        # this will subtract from the year because (m/12) is negative
        m = 1 + (m % 12)        # this gives us the correct month
        return (m, y)

# Return the integer month and year for the month immediately preceding the one provided
# pass in an integer month (1-12) and year (2-digit or 4-digit)
def prev_month(month, year):
    return add_month(month, year, -1)

# Return the integer month and year for the month immediately following the one provided
# pass in an integer month (1-12) and year (2-digit or 4-digit)
def next_month(month, year):
    return add_month(month, year, +1)

# Given a symbol ending with a monthcode and 2-digit year (ex: QHOF17, @VXK15, GASX12, etc.)
# Return that same symbol with the monthcode and 2-digit year of the previous month
def prev_month_symbol(symbol):
    mYY = symbol[-3:]
    (m, y) = get_month_year(mYY)
    (mn, yn) = prev_month(m, y)
    mYY = get_MYY(mn, yn)
    return symbol[:-3] + mYY

# Given a symbol ending with a monthcode and 2-digit year (ex: QHOF17, @VXK15, GASX12, etc.)
# Return that same symbol with the monthcode and 2-digit year of the following month
def next_month_symbol(symbol):
    mYY = symbol[-3:]
    (m, y) = get_month_year(mYY)
    (mn, yn) = next_month(m, y)
    mYY = get_MYY(mn, yn)
    return symbol[:-3] + mYY

# Given a symbol ending with a monthcode and 2-digit year (ex: QHOF17, @VXK15, GASX12, etc.) AND an integer x
# Return that same symbol with the monthcode and 2-digit year of the "xth" previous month
def prev_month_symbol_x(symbol, x):
    pms_symbol = symbol
    for i in range(x):
        pms_symbol = prev_month_symbol(pms_symbol)
    return pms_symbol

# Given a symbol ending with a monthcode and 2-digit year (ex: QHOF17, @VXK15, GASX12, etc.) AND an integer x
# Return that same symbol with the monthcode and 2-digit year of the "xth" following month
def next_month_symbol_x(symbol, x):
    nms_symbol = symbol
    for i in range(x):
        nms_symbol = next_month_symbol(nms_symbol)
    return nms_symbol

# Return a "CME-style" month/year in the form of monthcode and 2-digit year (ex: F17, K15, X12, etc.)
# pass in an integer month (1-12) and an integer year (can be 2-digit or 4-digit year: 2017, 17, etc.)
def get_MYY(month, year):
    year_str = str(year)
    return get_monthcode(month) + year_str[-2:]

# Return a "CME-style" month/year in the form of monthcode and 2-digit year (ex: F17, K15, X12, etc.)
# pass in a datetime (only month and year of datetime are used--day is ignored)
def get_mYY(dt):
    return get_MYY(dt.month, dt.year)

# Given a symbol root (ex: "@VX") and a datetime object
# Return the mYY futures contract symbol (ex: "@VXM16")
def mYY(symbol_root, dt):
    m = dt.month
    y = dt.year
    return symbol_root + monthcodes[m-1] + str(y)[-2:]

# Given a symbol root ("@ES", "QHO", etc.) and a datetime
# Return the future symbol in the format "XXXmYY"
# (only the month and year are used from the given datetime dt)
def future_symbol(symbol_root, dt):
    return symbol_root + get_mYY(dt)

# Given a symbol in XXXmYY format (ex: "@VXM16")
# Return the datetime of the first day of that symbol month
# TODO: For now, we assume a XXXmYY (6-char) future symbol
# TODO: Also, we assume year is >= 2000
def get_symbol_date(symbol):
    m = monthcodes.index(symbol[3])+1
    y = 2000 + int(symbol[-2:])
    return datetime(y, m, 1)

# Given a monthcode and 2-digit year (i.e. 'X15') return the integer month (1-12) and year (4-digit)
def get_month_year(mYY):
    monthcode = mYY[:1]
    year_str = "20" + mYY[-2:]
    m = get_month_number(monthcode)
    y = int(year_str)
    return (m, y)

# Given a specific date, return the last day of month containing that date
def last_day_of_month(any_date):
    next_month = any_date.replace(day=28) + timedelta(days=4)  # this will never fail
    return next_month - timedelta(days=next_month.day)

# Given an integer month (1-12) and year (4-digit) return the last business day of the month
# (or day_count business days from last day of the month)
def last_business_day(month, year, day_count=0):
    (m, y) = next_month(month, year)
    dt = pd.datetime(y, m, 1)               # first day of month of next month
    lbd = dt - BDay(day_count+1)            # last business day of month (or x-to-last business day if day_count > 0)
    return lbd

# Given an integer month (1-12) and year (4-digit) return the xth instance of the specified weekday (Mon=0, Sun=6)
# Return the date of this xth instance of the specified weekday
def x_weekday_of_month(month, year, x, weekday):
    count = 0
    xdt = datetime(year, month, 1)
    while count < x:
        if xdt.weekday() == weekday:
            count += 1
        xdt += timedelta(days=1)
    return xdt - timedelta(days=1)

# Given two datetime dates
# Return the number of BUSINESS days between the two dates
def diff_business_days(dt1, dt2):
    if (dt1 > dt2):
        return -diff_business_days(dt2, dt1)
    else:
        count = 0
        dt = dt1
        while dt < dt2:
            count += 1
            dt = dt + BDay(1)
    return count

# Given a symbol (including monthcode and 2-digit year) return the expiration date for that contract
# symbol is a symbol ending with monthcode and year in format 'SYMmYY'
def get_expiration(symbol):
    s = symbol.upper()
    mYY = s[-3:]
    x = len(symbol)-3
    s = s[:x]
    (m, y) = get_month_year(mYY)
    (m_prev, y_prev) = prev_month(m, y)
    if s == 'QHO':
        return last_business_day(m_prev, y_prev)        
    elif s == 'GAS':
        dt = datetime(y, m, 14)
        return dt - BDay(2)
    else:
        return None

# Return a calendar spread symbol in the form 'XXXmYY-XXXmYY' (ex: 'QHOM17-QHON17')
# pass in the symbol root (ex: 'QHO', 'GAS', 'QHG') and integer months (1-12) and years (2-digit or 4-digit)
def get_calendar_symbol(symbol_root, m1, y1, m2, y2):
    calendar_symbol = get_symbol(symbol_root, m1, y1) + '-' + get_symbol(symbol_root, m2, y2)
    return calendar_symbol

# Return the symbol in the form 'XXXmYY' (ex: 'QHOM17')
# pass in the integer month (1-12) and year (2-digit or 4-digit)
def get_symbol(symbol_root, m1, y1):
    (m2, y2) = next_month(m1, y1)
    symbol = symbol_root + get_MYY(m1, y1)
    return symbol

# Given a symbol root (ex: "@VX") and a month (1-12) and a year (4-digit)
# Return the future symbols for the front month, one month out, and two months out (tuple of 3 values)
def get_contango_symbols(symbol_root, m, y):
    dt0 = datetime(y, m, 1)
    dt1 = dt0 + relativedelta.relativedelta(months=1)
    dt2 = dt1 + relativedelta.relativedelta(months=1)
    return future_symbol(symbol_root, dt0), future_symbol(symbol_root, dt1), future_symbol(symbol_root, dt2)

# Return front month symbol in form 'XXXmYY', back month symbol in form 'XXXmYY', and calendar spread symbol in form 'XXXmYY-XXXmYY'
# ex: tuple ('QHOM17', 'QHON17', 'QHOM17-QHON17')
# pass in the symbol root (ex: 'QHO', 'GAS', 'QHG') and integer months (1-12) and years (2-digit or 4-digit)
# pass in the integers for front month (ifmonth) and back month (ibmonth)  (ex: ifmonth=0, ibmonth=1 is front-to-next-month calendar)
def get_calendar_symbols(symbol_root, m1, y1, ifmonth, ibmonth):
    (mf, yf) = add_month(m1, y1, ifmonth)
    (mb, yb) = add_month(m1, y1, ibmonth)
    front_symbol = get_symbol(symbol_root, mf, yf)
    back_symbol = get_symbol(symbol_root, mb, yb)
    calendar_symbol = get_calendar_symbol(symbol_root, mf, yf, mb, yb)
    return (front_symbol, back_symbol, calendar_symbol)

# Given a datetime
# Return a default string represenation of a DATE (date only, no time component)
def strdate(dt, separator='-'):
    fmt = "%b{0}%d{0}%Y".format(separator)
    return dt.strftime(fmt)

# Given a datetime
# Return a default string representation of a DATETIME (date AND time components)
# (optional) include_time is True by default but can be set to false to return string of date only
def strdt(dt, include_time=True):
    fmt = "%Y-%m-%d %H:%M:%S"
    if include_time == False:
        fmt = "%Y-%m-%d"
    return dt.strftime(fmt)

# Given a datetime.time object
# Return a default string representation of a TIME in format "HH:mm"
# (optional) include_seconds defaults to False but setting to True will get return value in format "HH:mm:ss"
def strtime(tm, include_seconds=False):
    if tm == None: return ("??:??" if include_seconds==False else "??:??:??")
    str = "{0:02d}:{1:02d}".format(tm.hour, tm.minute)
    if include_seconds == True:
        str = "{0:02d}:{1:02d}:00".format(tm.hour, tm.minute)
    return str

# Given time string in format "HH:mm:SS"
# Return the hour, minute, seconds as integers
def parse_timestr(timestr):
    HH = timestr[:2]
    mm = timestr[3:5]
    SS = timestr[6:]
    return int(HH), int(mm), int(SS)

# Given a datetime.time and a datetime.timedelta
# Return a new datetime.time representing the given timedelta added to the given time
# (this gets around an issue where you cannot add/subtract timedelta to/from time)
def add_time(tm, dlta):
    dt = datetime.combine(date.today(), tm) + dlta
    return dt.time()

# Given two datetime.time objects (tm2 >= tm1)
# Return the timedelta representing the subtraction of these two times
# (python does not allow us to directly subtract two datetime.time objects)
def subtract_times(tm2, tm1):
    dt2 = datetime.combine(date.today(), tm2)
    dt1 = datetime.combine(date.today(), tm1)
    dlta = dt2 - dt1
    seconds = dlta.total_seconds()
    seconds_per_hour = 60 * 60
    hours = int(seconds // seconds_per_hour)
    minutes = int(seconds % seconds_per_hour)
    return time(hours, minutes)

# TODO: Make sure this works using our new xth_weekday() function
# Given a datetime
# Return the datetime of the third Friday of that month
def third_friday(dt):
    return xth_weekday(dt, 3, WEEKDAY_FRI)

# Given a datetime, x representing xth, and weekday (0=MON to 6=SUN or use constants WEEKDAY_SUN, WEEKDAY_MON, etc.
# Return the datetime of the xth "weekday" of that month
# example: xth_weekday(dt, 1, WEEKDAY_THU) returns date of first Thursday of the month
# example: xth_weekday(dt, 3, WEEKDAY_FRI) returns date of the third Friday of the month
def xth_weekday(dt, x, weekday=WEEKDAY_FRI):
    dtx = dt.replace(day=1)
    if dtx.weekday() == weekday:                    # is the first day of the month the weekday we are seeking?
        count = 1
    else:
        count = 0
    while count < x:                                # use count to reach the "xth" given weekday of the month
        dtx += timedelta(days=1)
        if dtx.weekday() == weekday:
            count += 1
    return dtx

# Given a datetime (or use current datetime if none provided)
# Return the number of seconds since datetime(1970,1,1) (as float)
def to_unixtime(dt=None):
    if dt == None:
        td = datetime.now() - datetime(1970, 1, 1)
    else:
        td = dt - datetime(1970, 1, 1)
    return td.total_seconds()

# Given a number of seconds since datetime(1970,1,1) (as float)
# Return the corresponding datetime
def from_unixtime(seconds):
    td = timedelta(seconds=seconds)
    return datetime(1970, 1, 1) + td

def to_unix_time(dt):
    epoch =  datetime.utcfromtimestamp(0)
    return (dt - epoch).total_seconds() * 1000

# Given an integer number of days (numdays)
# Return a list containing all the dates (list of datetime) that make up those previous <numdays> dates (starting from current date)
# (optional) sort defaults to True, which will return the days in typical date order (earliest to most recent); otherwise order is reversed
def date_range_previous_x_days(numdays, sort=True):
    base = datetime.datetime.today()
    date_list = [base - datetime.timedelta(days=x) for x in range(0, numdays)]
    if sort: date_list.sort()
    return date_list

