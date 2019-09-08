from __future__ import print_function
from urllib2 import urlopen
import urllib3
#import StringIO
import io
from bs4 import BeautifulSoup
from os import listdir, rename
from os.path import basename, splitext, join, isfile
from datetime import date, time, datetime
import pandas as pd
import numpy as np
import sys


#-----------------------------------------------------------------------------------------------------------------------
from f_folders import *
from f_date import *
from f_plot import *
from f_stats import *
from f_dataframe import *
from f_file import *
from f_date import from_unixtime

pd.set_option('display.width', 160)
#-----------------------------------------------------------------------------------------------------------------------

#btcdata_folder = "/Users/michael/Dropbox/dev/data/MISC/bitcoincharts"
btcdata_folder = join(misc_folder, "bitcoincharts")
scraped_data_info_filename = "bitcoincharts_scraped_data_info.DF.csv"

# After finding an element using BeautifulSoup (an 'a' element in this case), parse the immediately following text
# into date/time of last file modification and file size (in bytes)
# Return these as a tuple (date, time, bytesize)
def get_info_after(a):
    pre_text = a.next_sibling.strip().split()
    dt = datetime.strptime(pre_text[0], '%d-%b-%Y').date()
    tm = datetime.strptime(pre_text[1], '%H:%M').time()
    bytesize = int(pre_text[2])
    return (dt, tm, bytesize)

# Perform a similar "scrape" operation as the one to download data files, but instead retrieve last updated and file size
# (we can use this to determine if any of the data sets have been updated and to only re-download these updated data sets)
def get_bitcoinchart_data_last_updated(url="http://api.bitcoincharts.com/v1/csv/", write_to_file=False):
    #html = urlopen(url).read()
    http = urllib3.PoolManager()
    response = http.request('GET', url)
    html = response.data
    soup = BeautifulSoup(html, "html.parser")
    li = soup.findAll('a')
    if write_to_file:
        f_out = open(join(btcdata_folder, scraped_data_info_filename), 'w')
        f_out.write("Filename,LastUpdated,ByteSize\n")
    row_list = []
    for a in li:
        gz_filename = a['href']
        if gz_filename.endswith('.gz'):
            dt, tm, bytesize = get_info_after(a)
            #pre_text = a.next_sibling.strip().split()
            #dt = datetime.strptime(pre_text[0], '%d-%b-%Y').date()
            #tm = datetime.strptime(pre_text[1], '%H:%M').time()
            #size_in_bytes = int(pre_text[2])
            row_list.append([gz_filename, datetime.combine(dt,tm), bytesize])
            #print("{0:25} {1} {2} {3:10}".format(gz_filename, dt, tm, size_in_bytes))
            if write_to_file: f_out.write("{0},{1} {2},{3}\n".format(gz_filename, dt, tm, bytesize))
    if write_to_file:
        f_out.close()
        print("\nWrote bitcoincharts scraped data info to file: '{0}'".format(scraped_data_info_filename))
    df = pd.DataFrame(row_list, columns=['Filename','LastUpdated','ByteSize']) #, dtype={'Filename':np.str_, 'LastUpdated':np.datetime64, 'ByteSize':np.int64})
    df['LastUpdated'] = df['LastUpdated'].astype(datetime)
    return df

# Given a GZIP file (with extension ".gz") and the URL for the bitcoincharts api data website
# download and decompress this file to the 'btcdata_folder'
def download_bitcoinchart_data(gz_filename, url="http://api.bitcoincharts.com/v1/csv/"):
    if not gz_filename.endswith('.gz'): return
    dt, tm, size_in_bytes = get_info_after(a)
    print("Downloading {0}  ({1} bytes) ...".format(gz_filename, size_in_bytes))
    response = urlopen(url + gz_filename)
    #compressedFile = StringIO.StringIO(response.read())
    compressedFile = io.StringIO(response.read())
    decompressedFile = gzip.GzipFile(fileobj=compressedFile)
    output_filename = splitext(gz_filename)[0]
    with open(join(btcdata_folder, output_filename), 'w') as f_out:
        f_out.write(decompressedFile.read())
    return

# Grab ALL the available data from the given bitcoincharts url
# these are ".gz" GZIP files, but we download them to the 'btcdata_folder' provided AND decompress them into .CSV files
# the three columns (which are not labeled) are 1) Unixtime timestamp 2) Price 3) Amount
def download_all_bitcoinchart_data(url="http://api.bitcoincharts.com/v1/csv/"):
    html = urlopen(url).read()
    soup = BeautifulSoup(html, "html.parser")
    li = soup.findAll('a')
    for a in li:
        gz_filename = a['href']
        download_bitcoinchart_data(gz_filename)
    return

# Given the full path/filename of a data file we have downloaded from bitcoincharts
# Create a dataframe with our familiar column structure
# the three columns originally provided by bitcoincharts 1) Unixtime timestamp 2) Price 3) Amount
def create_bitcoinchart_df(filepath):
    output_filename = splitext(filepath)[0] + ".DF.csv"
    symbol = splitext(basename(filepath))[0]
    print("Creating converted dataframe for {0}".format(symbol))
    with open(filepath, 'rt') as f_in, open(output_filename, 'wt') as f_out:
        f_out.write("DateTime,Symbol,Price,Amount\n")                           # output the column names
        line = f_in.readline().strip()
        while line:
            split = line.split(',')
            timestamp = float(split[0])
            dt = from_unixtime(timestamp)
            price = split[1]
            amount = split[2]
            f_out.write("{0},{1},{2},{3}\n".format(dt, symbol, price, amount))
            line = f_in.readline().strip()
    return

# Given the 'folder' location of the data files we have downloaded from bitcoincharts
# Create a dataframe with our familiar column structure
# the three columns originally provided by bitcoincharts are 1) Unixtime timestamp 2) Price 3) Amount
def create_all_bitcoinchart_df(folder):
    csvfiles = [f for f in listdir(folder) if isfile(join(folder, f)) and f.endswith(".csv") and not f.endswith(".DF.csv")]
    for f in csvfiles:
        create_bitcoinchart_df(join(folder, f))
    print("Done.")
    return

# Re-scrape the bitcoinchart data api website to obtain the latest file modification date/time for each GZIP data file
# for any datasets whose file modification date/time has changed from the one we previously stored, download the updated data
def check_for_bitcoinchart_data_updates():
    # Re-scrape the bitcoincharts website to see if the file modification date has changed for any of the datasets
    # (In other words, do any of the datasets on the bitcoincharts website have new data that we need to fetch?)
    df_new = get_bitcoinchart_data_last_updated()
    df_old = read_dataframe(join(btcdata_folder, scraped_data_info_filename), date_columns=['LastUpdated'])
    need_update_list = []
    for ix, row in df_new.iterrows():
        dfx = df_old[df_old.Filename == row['Filename']]
        # if not dfx.empty: print("{0:25}  {1}  {2}    {3}".format(row['Filename'], dfx.iloc[0]['LastUpdated'], row['LastUpdated'], dfx.iloc[0]['LastUpdated'] < row['LastUpdated']))
        if dfx.empty or (dfx.iloc[0]['LastUpdated'] < row['LastUpdated']):
            need_update_list.append(row['Filename'])
    if len(need_update_list) == 0:
        print("No bitcoincharts datasets have been updated since we last retrieved data.")
    else:
        # Go through each filename in 'need_update_list' and
        for f in need_update_list:
            no_gz_filename = splitext(f)[0]
            print("Retrieving updated data for '{0}'".format(no_gz_filename))
            download_bitcoinchart_data(no_gz_filename)
            create_bitcoinchart_df(join(btcdata_folder, no_gz_filename))
        write_dataframe(df_new, join(btcdata_folder, scraped_data_info_filename))  # write out the updated dataset info
    return

# Runs some simple evaluation of the bitcoincharts datasets and discards (moves to "bad_data" folder) those that weak
# DEPRECATED: (optional) alphabetic_start can be set to any string, and only those filenames that begin with text >= 'alphabetic_start' are included
def evaluate_bitcoincharts_datasets(folder):    #, alphabetic_start=""):
    dffiles = [f for f in listdir(folder) if isfile(join(folder, f)) and f.endswith("DF.csv") and not f==scraped_data_info_filename]
    #n = len(alphabetic_start)
    #dffiles = filter(lambda x: x[:n] >= alphabetic_start, dffiles)
    for f in dffiles:
        try:
            filepath = join(folder, f)
            bad_data_filepath = join(folder, "bad_data", f)
            df = read_dataframe(filepath, display=False)
            if df.empty:
                print("{0:25} EMPTY".format(f))
                print("Moving '{0}' to 'bad_data' folder".format(f))
                rename(filepath, bad_data_filepath)
            else:
                dtmin = df['DateTime'].min()
                dtmax = df['DateTime'].max()
                data_count = df.shape[0]
                print("{0:25}  {1}  {2}   {3:>10,d} rows".format(f, dtmin, dtmax, data_count))
                if dtmax < datetime(2017,1,1):
                    #print("'{0}'  to  '{1}'".format(filepath, bad_data_filepath))
                    print("Moving '{0}' to 'bad_data' folder".format(f))
                    rename(filepath, bad_data_filepath)
        except:
            print("ERROR: Exception occurred attempting to load '{0}'".format(f))
    return

########################################################################################################################

#---------------------------------------------------------------------------------------------------
# http://api.bitcoincharts.com/v1/csv/

#---------------------------------------------------------------------------------------------------
# These functions download/decompress the GZIP data files from bitcoincharts web site
#download_all_bitcoinchart_data()
#download_bitcoinchart_data("bitbayEUR.csv.gz")

#---------------------------------------------------------------------------------------------------
# These functions take existing downloaded bitcoincharts data and reformat it to DF_DATA format
#create_all_bitcoinchart_df(btcdata_folder)
#create_bitcoinchart_df(join(btcdata_folder, "bitbayEUR.csv"))

#---------------------------------------------------------------------------------------------------
# Do a quick run-through of the bitcoincharts datasets to identify those with stale or limited data
# (those with poor data are moved to "bad_data" folder)
#evaluate_bitcoincharts_datasets(btcdata_folder)

#-------------------------------------------------------------------------------
# Scrape website to pull in 'LastUpdated' date for each bitcoincharts file
df = get_bitcoinchart_data_last_updated()
print(df)

