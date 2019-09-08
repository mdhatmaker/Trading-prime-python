import pytz
from iqfeed import get_bars

instrument = 'GLD'
start_date = '20150101'
end_date = '20151231'
tz = pytz.timezone('US/Eastern')
seconds_per_bar = 60  # For 1M data
iqfeed_host = 'localhost'
iqfeed_port = 9100

bars = get_bars(instrument, start_date, end_date, tz, seconds_per_bar, iqfeed_host, iqfeed_port)

