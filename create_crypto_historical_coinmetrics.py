from os.path import join, basename, splitext
import pandas as pd
import sys

#-----------------------------------------------------------------------------------------------------------------------
from f_folders import *
from f_date import *
from f_plot import *
from f_stats import *
from f_dataframe import *
from f_file import *
#-----------------------------------------------------------------------------------------------------------------------

# Perform the necessary tweaks to convert coinmetrics raw data to DF_DATA dataframe format
def convert_coinmetrics_to_DF(csv_filename, raw_path, df_path):
    root, ext = splitext(csv_filename)
    df_pathname = join(df_path, "coinmetrics."+root+".daily.DF.csv")
    print "Converting to dataframe '{0}' ...".format(df_pathname),
    with open(df_pathname, 'w') as fw:
        with open(join(raw_path, csv_filename)) as f:
            row_count = 0
            for line in f:
                if row_count == 0:
                    #print "DateTime,TxVolumeUSD,TxCount,MarketCapUSD,PriceUSD,ExchangeVolumeUSD,GeneratedCoins,Fees"
                    fw.write("DateTime,TxVolumeUSD,TxCount,MarketCapUSD,PriceUSD,ExchangeVolumeUSD,GeneratedCoins,Fees\n")
                else:
                    #print line.strip()
                    substrings = line.split(',')
                    substrings[0] = substrings[0].replace('/','-')      # change date format to use dashes instead of slashes
                    fw.write(','.join(substrings))
                row_count += 1
    print "Done."
    return

# Given the URL of the coinmetrics historical data file
# Download (to RAW_DATA) and convert (to DF_DATA) historical data for a single coinmetrics symbol
def coinmetrics_download(csv_url, raw_path, df_path):
    download_csv(csv_url, raw_path)
    convert_coinmetrics_to_DF(get_url_filename(csv_url), raw_path, df_path)
    return

# For all coinmetrics symbols, download the (daily) historical data and convert to dataframe in DF_DATA
def download_and_convert_coinmetrics_historical():
    for symbol in coinmetrics_symbols:
        coinmetrics_download("https://coinmetrics.io/data/{0}.csv".format(symbol), raw_folder, df_folder)
    return

########################################################################################################################


#-------------------------------------------------------------------------------
# https://coinmetrics.io/data-downloads/

# -------------------------------------------------------------------------------
# COMMAND LINE ARGUMENTS (arg[0] is script name)
#sys.argv.append('COINMETRICS_HISTORICAL')      # FOR TESTING ONLY!!!
args = sys.argv
if (len(args) > 0):
    print(len(args))
    print(get_arg(1), get_arg(2), get_arg(3))
    # sys.exit()

    # ARG: "COINMETRICS_HISTORICAL"
    # UPDATE COINMETRICS HISTORICAL DATA (output to DF_DATA folder)
    if get_arg(1) == 'COINMETRICS_HISTORICAL':
        download_and_convert_coinmetrics_historical()
        STOP()

# command line arguments -------------------------------------------------------

sys.exit()

download_and_convert_coinmetrics_historical()








