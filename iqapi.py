"""
Credit: Michael Halls-Moore
Url:    https://www.quantstart.com/articles/Downloading-Historical-Intraday-US-
        Equities-From-DTN-IQFeed-with-Python
        
I simply wrapped the logic into a class. 
Will possibly extend for live feeds.
@author: Luke Patrick James Tighe
"""

import datetime
import socket
import os.path
import pandas as pd

"""
IQ DTN Feed Historical Symbol Download.
Downloads the symbol data in CSV format and stores it in a local directroy.
If we already have the symbol data downloaded, it will not hit IQ DTN Feed again,
it will simple use the local data.
To flush the local CSV file, simply delete the directory.
Constructor enables to specify a start and end date for the symbol data as well
as the frequency. Great for making sure data is consistent.
Simple usage example:
    from iqfeed import historicData
    dateStart = datetime.datetime(2014,10,1)
    dateEnd = datetime.datetime(2015,10,1)        
        
    iq = historicData(dateStart, dateEnd, 60)
    symbolOneData = iq.download_symbol(symbolOne)
    
"""


# The following would be used if you wanted to request a certain number of days rather than a begin/end datetime (which we don't currently do):
# HTD,[Symbol],[Days],[MaxDatapoints],[BeginFilterTime],[EndFilterTime],[DataDirection],[RequestID],[DatapointsPerSend]<CR><LF>
# HID,[Symbol],[Interval],[Days],[MaxDatapoints],[BeginFilterTime],[EndFilterTime],[DataDirection],[RequestID],[DatapointsPerSend],[IntervalType]<CR><LF>


class IQHistoricData:
    # interval begins with char (followed immediately by integer value):
    # 'd' = daily               (ex: 'd1', 'd5')
    # 's' = seconds interval    (ex: 's60', 's3600')
    # 'v' = volume interval     (ex: 'v2000', 'v100000')
    # 't' = tick interval       (ex: 't20', 't1000')        TODO: I don't think tick interval is implemented yet...or is it?
    def __init__(self, startDate, endDate, interval='d', beginFilterTime='', endFilterTime=''):
        self.startDate = startDate.strftime("%Y%m%d %H%M%S")
        self.endDate = endDate.strftime("%Y%m%d %H%M%S")
        #self.timeFrame = str(timeFrame)
        self.interval = interval.strip()                # interval is now expected to be a char+integer combo (as a string--ex: 's60' or 'v2000' or 't50')
        self.interval_type = self.interval[0]           # break up interval into 'interval_type' and 'interval_value'
        if len(self.interval) > 1:
            self.interval_value = self.interval[1:]     # (ex: 's60' -> interval_type='s' and interval_value='60')
        else:
            self.interval_value = None
        #print("INTERVALS: {0} {1} {2}".format(self.interval, self.interval_type, self.interval_value))
        self.beginFilterTime = beginFilterTime          # format "HHmmSS"
        self.endFilterTime = endFilterTime              # format "HHmmSS"
        # We dont want the download directory to be in our source control
        #self.downloadDir = "./MarketData/"
        self.downloadDir = ""
        self.host = "127.0.0.1"  # Localhost
        self.port = 9100  # Historical data socket port


    def read_historical_data_socket(self, sock, recv_buffer=4096):
        #Read the information from the socket, in a buffered
        #fashion, receiving only 4096 bytes at a time.
        #
        #Parameters:
        #sock - The socket object
        #recv_buffer - Amount in bytes to receive per read
        buffer = ""
        data = ""
        while True:
            data = sock.recv(recv_buffer)
            buffer += data
    
            # Check if the end message string arrives
            if "!ENDMSG!" in buffer:
                break
       
        # Remove the end message string
        buffer = buffer[:-12]
        return buffer


    # HIT,[Symbol],[Interval],[BeginDate BeginTime],[EndDate EndTime],[MaxDatapoints],[BeginFilterTime],[EndFilterTime],[DataDirection],[RequestID],[DatapointsPerSend],[IntervalType]<CR><LF>
    # [DataDirection]    '0' (default) for "newest to oldest"  or  '1' for "oldest to newest"
    # [IntervalType]    's'=seconds, 'v'=volume, 't'=ticks


    def get_message_seconds_interval(self, symbol):
        message = "HIT,{0},{1},{2},{3},,{4},{5},1,IQFEED_HITs,5000,'s'\n".format(symbol, self.interval_value, self.startDate, self.endDate, self.beginFilterTime, self.endFilterTime)
        return message

    def get_message_volume_interval(self, symbol):
        message = "HIT,{0},{1},{2},{3},,{4},{5},1,IQFEED_HITv,5000,'v'\n".format(symbol, self.interval_value, self.startDate, self.endDate, self.beginFilterTime, self.endFilterTime)
        return message

    #def get_message_tick_interval(self, symbol):
    #    message = "HIT,{0},{1},{2},{3},,{4},{5},1,IQFEED_HITt,5000,'t'\n".format(symbol, self.interval_value, self.startDate, self.endDate, self.beginFilterTime, self.endFilterTime)
    #    return message

    def get_message_ticks(self, symbol):
        message = "HTT,{0},{1},{2},{3},,{4},{5},1,IQFEED_HTT,5000\n".format(symbol, self.interval_value, self.startDate, self.endDate, self.beginFilterTime, self.endFilterTime)
        return message

    def get_message_daily(self, symbol):
        message = "HDT,{0},{1},{2},,1,IQFEED_HDT,2500\n".format(symbol, self.startDate[:8], self.endDate[:8])
        return message

    def get_message_composite_weekly(self, symbol):
        message = "HWX,{0},{1},1,IQFEED_HWX,2500\n".format(symbol, self.interval_value)
        return message

    def get_message_composite_monthly(self, symbol):
        message = "HMX,{0},{1},1,IQFEED_HMX,2500\n".format(symbol, self.interval_value)
        return message

    def reorder_ohlc(self, data):
        # REORDER the columns to OHLC (I think by default they come HLOC)
        lines = []
        for line in data.split("\n"):
            x = line.split(',')
            if len(x) == 8:
                if self.interval_type == 'd':
                    text = "{1},{2},{3},{4},{5},{6},{7}".format(x[0], x[1][:10], x[4], x[2], x[3], x[5], x[6], x[7])
                else:
                    text = "{1},{2},{3},{4},{5},{6},{7}".format(x[0], x[1], x[4], x[2], x[3], x[5], x[6], x[7])
                lines.append(text)
        return "\n".join(lines)


    def iqfeed_request(self, symbol):
        if self.interval_type == 'd':
            message = self.get_message_daily(symbol)
        elif self.interval_type == 'v':
            message = self.get_message_volume_interval(symbol)
        elif self.interval_type == 't':
            message = self.get_message_ticks(symbol)
        else:   # 's'
            message = self.get_message_seconds_interval(symbol)
        print(" " + message.strip())
        
        # Open a streaming socket to the IQFeed server locally
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.host, self.port))
        
        sock.sendall(message)
        data = self.read_historical_data_socket(sock)
        sock.close
        
        # Remove all the endlines and line-ending
        # comma delimiter from each record
        data = "".join(data.split("\r"))
        data = data.replace(",\n","\n")[:-1]
        
        return data
        
 
    def download_symbol(self, symbol, write_file=False):
        # Construct the message needed by IQFeed to retrieve data
        #[bars in seconds],[beginning date: CCYYMMDD HHmmSS],[ending date: CCYYMMDD HHmmSS],[empty],[beginning time filter: HHmmSS],[ending time filter: HHmmSS],[old or new: 0 or 1],[empty],[queue data points per second]
        if self.interval_type == 'd':
            fileName = "{0}{1}-{2}-{3}-{4}.csv".format(self.downloadDir, symbol, "daily", self.startDate[:8], self.endDate[:8])
        elif self.interval_type == 's':
            fileName = "{0}{1}-{2}-{3}-{4}.csv".format(self.downloadDir, symbol, self.interval_value, self.startDate[:8], self.endDate[:8])
        else:
            fileName = "{0}{1}-{2}-{3}-{4}.csv".format(self.downloadDir, symbol, self.interval, self.startDate[:8], self.endDate[:8])
        #print(fileName)

        """
        override = True
        exists = os.path.isfile(fileName)
        if exists == False or override == True:       
            data = self.iqfeed_request(symbol)
            # Write the data stream to disk
            f = open(fileName, "w")
            f.write("DateTime,Open,High,Low,Close,Volume,oi\n")
            #f.write(data)
            f.write(self.reorder_ohlc(data))
            f.close()
        """
        # This is an attempt to avoid writing out the data to a file and reading it back into a dataframe
        # (instead it creates the dataframe on the fly)
        data = self.iqfeed_request(symbol)
        if '!NO_DATA!' in data:
            df = pd.DataFrame([], columns=['DateTime','Symbol','Open','High','Low','Close','Volume','oi'])
        else:
            data = self.reorder_ohlc(data)
            rows = []
            for line in data.split("\n"):
                x = line.split(',')
                rows.append(x)
            df = pd.DataFrame(rows, columns=['DateTime','Open','High','Low','Close','Volume','oi'])

            #df = pd.io.parsers.read_csv(fileName, header=0, index_col=0, names=['DateTime','Open','High','Low','Close','Volume','oi'], parse_dates=True)
            #df = pd.io.parsers.read_csv(fileName, header=0, names=['DateTime','Open','High','Low','Close','Volume','oi'], parse_dates=True)
            df['Symbol'] = symbol
            df = df[['DateTime', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 'oi']]
            """
            if write_file == False:
                try:
                    os.remove(fileName)
                except:
                    print("Could not remove file")  # '{0}'".format(filename))
            """
        return df

########################################################################################################################

# Class to encapsulate some of the IQFeed API capabilities to search within symbols, descriptions, etc.
class IQSymbolSearch:
    def __init__(self):
        # We dont want the download directory to be in our source control
        # self.downloadDir = "./MarketData/"
        self.downloadDir = ""
        self.host = "127.0.0.1"  # Localhost
        self.port = 9100  # Historical data socket port

    # Read the information from the socket, in a buffered fashion, receiving only 4096 bytes at a time.
    # Parameters:
    # sock - The socket object
    # recv_buffer - Amount in bytes to receive per read
    def read_socket(self, sock, recv_buffer=4096):
        buffer = ""
        data = ""
        while True:
            data = sock.recv(recv_buffer)
            buffer += data
            # Check if the end message string arrives
            if "!ENDMSG!" in buffer:
                buffer = buffer[:-12] + '\n'
                break
            elif "!SYNTAX_ERROR!" in buffer:
                break
        # Remove the end message string
        #buffer = buffer[:-12]
        return buffer

    def iqfeed_request(self, message):
        print("IQFeed Request: [{0}]".format(message.strip()))
        # Open a streaming socket to the IQFeed server locally and send our request message
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.host, self.port))
        sock.sendall(message)
        data = self.read_socket(sock)
        sock.close
        # Remove all the endlines and line-ending comma delimeter from each record
        data = "".join(data.split("\r"))
        data = data.replace(",\n", "\n")[:-1]
        return data

    # Search SYMBOLS for the given search text
    def symbol_search(self, search_text):
        message = "SBF,s,{0},,,SYM_SEARCH\n".format(search_text)
        data = self.iqfeed_request(message)
        return data

    # Search DESCRIPTIONS for the given search text
    def description_search(self, search_text):
        message = "SBF,d,{0},,,DESC_SEARCH\n".format(search_text)
        data = self.iqfeed_request(message)
        return data

    # You can search for full SIC code ('8361' for RESIDENTIAL CARE)
    # or a partial SIC code of at least 2 digits ('83' to match all SIC codes starting with those two digits)
    def sic_code_search(self, search_code):
        message = "SBS,{0},SIC_SEARCH\n".format(search_code)
        data = self.iqfeed_request(message)
        return data

    # Given at least the first two digits of an existing NIAC code
    # Return a list of symbols that match that NIAC code (ex: "928110" gives list of symbols in the National Security NIAC code)
    def niac_code_search(self, search_code):
        message = "SBN,{0},NIAC_SEARCH\n".format(search_code)
        data = self.iqfeed_request(message)
        return data

    # Requests (via IQFeed) Listed Markets, Security Types, Trade Conditions, SIC codes and NIAC codes
    # Returns these as a dictionary with a descriptive text key for each list
    def request_lists(self):
        d = {}
        d['LISTED_MARKETS'] = self.request_list("SLM\n")
        d['SECURITY_TYPES'] = self.request_list("SST\n")
        d['TRADE_CONDITIONS'] = self.request_list("STC\n")
        d['SIC_CODES'] = self.request_list("SSC\n")
        d['NIAC_CODES'] = self.request_list("SNC\n")
        # TODO: It may be better for me to turn these into dataframes (and write them to a .CSV file, etc.)
        return d

    # generic function to submit a request message to the IQFeed API
    def request_list(self, message):
        data = self.iqfeed_request(message)
        return data.split('\n')

    # Set Protocol message (shouldn't be needed)
    #message = "S,SET PROTOCOL,5.2\n"

