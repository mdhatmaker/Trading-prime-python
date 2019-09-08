from f_date import *

bizday_count_qgc = 1        # business days before expiry (used for roll)
bizday_count_qho = 10       # business days before expiry (used for roll)


# Given a symbol root (ex: '@VX')
# Return a tuple containing roll date function and roll rule description
def get_roll_function(symbol_root):
    sym = symbol_root.strip().upper()
    if sym == '@ES':  # TODO: check if we should continue using VIX roll rule
        roll_description = "Rolling one day prior to first Wed on or before 30 days prior to 3rd Friday of month immediately following expiration month"
        return (get_roll_dates_es, roll_description)
    elif sym == '@JY':
        roll_description = "Rolling one day prior to two business days prior to 3rd Wednesday of delivery month"
        return (get_roll_dates_jy, roll_description)
    elif sym == '@TY':
        roll_description = "Rolling one day prior to seventh business day prior last business day of delivery month"
        return (get_roll_dates_ty, roll_description)
    elif sym == '@VX':
        roll_description = "Rolling one day prior to first Wed on or before 30 days prior to 3rd Friday of month immediately following expiration month"
        return (get_roll_dates_vx, roll_description)
    elif sym == 'QGC':
        roll_description = "Rolling {0} business days before the 3rd-last business day of the expiration month".format(bizday_count_qgc)
        return (get_roll_dates_gc, roll_description)    
    elif sym == 'QHO':
        roll_description = "Rolling {0} business days before the last business day of the expiration month".format(bizday_count_qho)
        return (get_roll_dates_ho, roll_description)    
    else:
        return (None, None)

#-----------------------------------------------------------------------------------------------------------------------
# Calculate @VX expiration
def expiry_date_vx(m1, y1):
    #bizday_count = 10                                   # business days before end of month (used for roll)
    #return last_business_day(m1, y1, bizday_count)      # [bizday_count] business days before end of month
    (mn1, yn1) = next_month(m1, y1)    
    xdt = x_weekday_of_month(mn1, yn1, 3, weekdays['Fri'])  # get the 3rd Friday of the (next) month
    xdt -= timedelta(days=30)                               # 30 days previous
    while xdt.weekday() != weekdays['Wed']:                 # find the first Wed on or before this date
        xdt -= timedelta(days=1)
    return xdt

# Calculate @VX roll date
def roll_date_vx(m1, y1):
    xdt = expiry_date_vx(m1, y1) - BDay(1)                  # roll one business day prior to expiration
    #dt = pd.datetime(y, m, 1)
    #lbd = dt - BDay(1) 
    return xdt

# Return @VX roll dates for given month and previous month
def get_roll_dates_vx(m1, y1):
    (mp1, yp1) = prev_month(m1, y1)
    return (roll_date_vx(mp1, yp1), roll_date_vx(m1, y1))

#-----------------------------------------------------------------------------------------------------------------------
# Ccalculate QHO roll date
def roll_date_ho(m1, y1):
    return last_business_day(m1, y1, bizday_count_qho)      # [bizday_count] business days before end of month

# Return QHO roll dates for given mmonth and previous month
def get_roll_dates_ho(m1, y1):
    (mp1, yp1) = prev_month(m1, y1)
    (mp2, yp2) = prev_month(mp1, yp1)
    return (roll_date_ho(mp2, yp2), roll_date_ho(mp1, yp1))

#-----------------------------------------------------------------------------------------------------------------------
# Calculate @ES (S&P e-mini) expiration
def expiry_date_es(m1, y1):
    #(mn1, yn1) = next_month(m1, y1)
    #xdt = x_weekday_of_month(mn1, yn1, 3, weekdays['Fri'])  # get the 3rd Friday of the (next) month
    xdt = x_weekday_of_month(m1, y1, 3, weekdays['Fri'])    # get the 3rd Friday of the month
    #xdt -= timedelta(days=30)                               # 30 days previous
    while xdt.weekday() != weekdays['Wed']:                 # find the first Wed on or before this date
        xdt -= timedelta(days=1)
    return xdt

# Calculate @ES roll date
def roll_date_es(m1, y1):
    xdt = expiry_date_es(m1, y1) - BDay(1)                  # roll one business day prior to expiration
    #dt = pd.datetime(y, m, 1)
    #lbd = dt - BDay(1)
    return xdt

# Return @ES roll dates for given month and previous month
def get_roll_dates_es(m1, y1):
    if not monthcodes[m1-1] in ['H', 'M', 'U', 'Z']:
        return (None, None)
    (mp1, yp1) = add_month(m1, y1, -3)
    return (roll_date_es(mp1, yp1), roll_date_es(m1, y1))

#-----------------------------------------------------------------------------------------------------------------------
# Calculate QGC (gold) expiration
def expiry_date_gc(m1, y1):
    xdt = last_business_day(m1, y1, 3)                      # 3rd-last business day of month
    #xdt -= BDay(7)                                          # seventh biz day preceding
    return xdt

# Calculate QGC roll date
def roll_date_gc(m1, y1):
    xdt = expiry_date_gc(m1, y1)
    #xdt = expiry_date_gc(m1, y1) - BDay(1)                  # roll one business day prior to expiration
    return xdt

# Return QGC roll dates for given month and previous month
def get_roll_dates_gc(m1, y1):
    (mp1, yp1) = prev_month(m1, y1)
    return (roll_date_gc(mp1, yp1), roll_date_gc(m1, y1))

#-----------------------------------------------------------------------------------------------------------------------
# Calculate @JY (Japanese Yen) expiration
def expiry_date_jy(m1, y1):
    xdt = x_weekday_of_month(m1, y1, 3, weekdays['Wed'])    # get the 3rd Wednesday of the month
    xdt -= BDay(2)                                          # 2nd business day immediately preceding 3rd Wed
    return xdt

# Calculate @JY roll date
def roll_date_jy(m1, y1):
    xdt = expiry_date_jy(m1, y1) - BDay(1)                  # roll 1 business day prior to expiration
    return xdt

# Return @JY roll dates for given month and previous month
def get_roll_dates_jy(m1, y1):
    if not monthcodes[m1-1] in ['H', 'M', 'U', 'Z']:
        return (None, None)
    (mp1, yp1) = add_month(m1, y1, -3)
    return (roll_date_jy(mp1, yp1), roll_date_jy(m1, y1))

#-----------------------------------------------------------------------------------------------------------------------
# Calculate @TY (10-Year T-Note) expiration
def expiry_date_ty(m1, y1):
    xdt = last_business_day(m1, y1)                         # last business day of month
    xdt -= BDay(7)                                          # seventh biz day preceding
    return xdt

# Calculate @TY roll date
def roll_date_ty(m1, y1):
    xdt = expiry_date_ty(m1, y1) - BDay(1)                  # roll one business day prior to expiration
    return xdt

# Return @TY roll dates for given month and previous month
def get_roll_dates_ty(m1, y1):
    if not monthcodes[m1-1] in ['H', 'M', 'U', 'Z']:
        return (None, None)
    (mp1, yp1) = add_month(m1, y1, -3)
    return (roll_date_ty(mp1, yp1), roll_date_ty(m1, y1))




"""
# TODO: DO I NEED THIS? OR SHOULD I JUST CREATE CONSTANTS TO REPRESENT PYTHON WEEKDAY NUMBERING?
# Given a "standard" day of the week (1=SUN, 2=MON, ..., 7=SAT)
# Return the integer used by python to represent that weekday (see datetime.weekday() function)
# python numbers weekdays as 0=MON, 1=TUE, ..., 6=SUN 
def get_python_weekday(day_of_week):
    if day_of_week == 1:
        return 6
    else:
        return day_of_week-2
"""

"""
# Given a datetime
# Return the datetime of the third Friday of that month
def third_friday(dt):
    dtx = dt.replace(day=1)
    if dtx.weekday() == WEEKDAY_FRI:                  # weekday 4 is FRIDAY
        count = 1
    else:
        count = 0
    while count < 3:
        dtx += timedelta(days=1)
        if dtx.weekday() == 4:
            count += 1
    return dtx
"""

###################### THIS IS NEWER CODE THAT IS NOT AS GENERAL BUT SIMPLER ######################
# The Final Settlement Date for a contract with the "VX" ticker symbol is on the Wednesday
# that is 30 days prior to the third Friday of the calendar month immediately following the
# month in which the contract expires.
# http://cfe.cboe.com/cfe-products/vx-cboe-volatility-index-vix-futures/contract-specifications
def get_final_settlement_date_VX(symbol):
    dt = get_symbol_date(symbol)
    dt_nextmonth = dt + relativedelta.relativedelta(months=1)
    dt_thirdfri = third_friday(dt_nextmonth)
    dt = dt_thirdfri - timedelta(days=30)
    while dt.weekday() != 2:                # weekday 2 is WEDNESDAY
        dt -= timedelta(days=1)
    return dt.replace(hour=8, minute=0, second=0)       # 8am? (TODO)

# Given a VX future symbol (ex: "@VXmYY")
# Return the roll date for this VX future
def get_roll_date_VX(symbol, roll_day_adjust=0):
    return get_final_settlement_date_VX(symbol) - BDay(1) + BDay(roll_day_adjust)            # 1 business day before settlement

# Given an ES future symbol (ex: "@ESmYY")
# Return the roll date for this ES future
def get_roll_date_ES(symbol, roll_day_adjust=0):
    dt = get_symbol_date(symbol)
    return xth_weekday(dt, 1, WEEKDAY_THU) + BDay(roll_day_adjust)                          # roll 1st Thursday of the expiration month

def get_final_settlement_date_NQ(symbol):
    dt = get_symbol_date(symbol)
    return xth_weekday(dt, 3, WEEKDAY_FRI)

def get_roll_date_NQ(symbol, roll_day_adjust=0):
    dt = get_symbol_date(symbol)
    return get_final_settlement_date_NQ(symbol) - BDay(1) + BDay(roll_day_adjust)          # 1 business day before settlement

def get_roll_date_GAS(symbol, roll_day_adjust=0):
    dt = get_symbol_date(symbol)
    dt = dt.replace(day=14)
    dt - BDay(2) + BDay(roll_day_adjust)                                           # 2 biz days prior to 14th
    return dt

def get_roll_date_QHO(symbol, roll_day_adjust=0):
    dt = get_symbol_date(symbol)
    dt = dt - BDay(1)+ BDay(roll_day_adjust)                                       # last biz day of month preceding delivery month
    return dt

#--------------------------- Lean Hog Futures Contract (HE) ------------------------------------------------------------
def get_final_settlement_date_HE(symbol):
    dt = get_symbol_date(symbol)
    dt = dt - timedelta(days=1)                             # last day of previous month
    dt = dt + BDay(10)                                      # 10th business day of contract month
    return dt.replace(hour=12, minute=0, second=0)          # 12pm? (TODO)

def get_roll_date_HE(symbol):
    return get_final_settlement_date_HE(symbol) - BDay(1)   # roll 1 business day before settlement
# ----------------------------------------------------------------------------------------------------------------------

#--------------------------- Live Cattle Futures Contract (LE) ---------------------------------------------------------
def get_final_settlement_date_LE(symbol):
    dt = get_symbol_date(symbol)
    dt = last_business_day(dt.month, dt.year)               # last business day of contract month
    return dt.replace(hour=12, minute=0, second=0)          # 12pm? (TODO)

def get_roll_date_LE(symbol):
    return get_final_settlement_date_LE(symbol) - BDay(1)   # roll 1 business day before settlement
# ----------------------------------------------------------------------------------------------------------------------

