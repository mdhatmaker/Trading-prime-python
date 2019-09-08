import f_folders as folder
from os import listdir
from os.path import isfile, join, split, splitext, basename
import os
import errno
import requests
import csv
import urlparse
import json
import gzip
import shutil
import sys

#-----------------------------------------------------------------------------------------------------------------------
coinmetrics_symbols = ['btc', 'bch', 'ltc', 'eth', 'xem', 'dcr', 'zec', 'dash', 'doge', 'etc', 'pivx', 'xmr']
#-----------------------------------------------------------------------------------------------------------------------

# Given a timeframe in abbreviated format (ex: '1m', '1h', 'daily')
# Return the text used by data filenames (ex: '1 Minute', '1 Hour', 'Daily')
def get_timeframe(tf):
    tfx = tf.strip().lower()
    if tfx == 'daily':
        return 'Daily'
    elif tfx == '1m':
        return '1 Minute'
    elif tfx == '1h':
        return '1 Hour'
    elif tfx == '1d':
        return '1 Day'
    else:
        return 'Unknown'
    
# Given any filename and timeframe (default is '1m')
# Return the csv filename with the appropriate timeframe appended
def get_csv_filename(filename, timeframe='1m'):
    tf = get_timeframe(timeframe)
    result = filename + ' (' + tf + ').csv'
    #print result
    return result

# Given a symbol and timeframe (default is '1m')
# Return the csv filename containing the price data (from the DF_DATA folder)
def get_df_pathname(symbol, timeframe='1m'):
    filename = get_csv_filename(symbol, timeframe)
    return join(folder.df_folder, filename)

# Given a full pathname (path/file.ext)
# Return a tuple containing (folder, filename, extension)
def split_pathname(full_pathname):
    (pathname, extension) = splitext(full_pathname)
    #(drive, path) = os.path.splitdrive(line)
    (folder, filename)  = split(pathname)
    return (folder, filename, extension)

# Given a full pathname (path/file.ext)
# Return the filename (including file extension)
def get_split_filename(full_pathname):
    (folder, filename, extension) = split_pathname(full_pathname)
    return filename + extension

# Given the pathname to a JSON file
# Return the json data from the file
def read_json(pathname):
    print("Reading JSON:", pathname)
    f = open(pathname, 'r')
    json_data = json.loads(f.read())
    f.close()
    return json_data

# Given a text string
# Return that text surrounded by quotation marks
def quoted(text, mark='"'):
    return mark + text + mark

# Get only the filename from a complete file URL
def get_url_filename(url):
    a = urlparse.urlparse(url)
    return basename(a.path)

# Download a CSV file from a specified URL
def download_csv(csv_url, local_path, display=True):
    url_filename = get_url_filename(csv_url)
    with requests.Session() as s:
        download = s.get(csv_url)
        decoded_content = download.content.decode('utf-8')
        cr = csv.reader(decoded_content.splitlines(), delimiter=',')
        local_pathname = join(local_path, url_filename)
        if display: print("Downloading to '{0}' ...".format(local_pathname),)
        with open(local_pathname, 'wb') as f:
            cw = csv.writer(f, dialect='excel')
            row_count = 0
            for row in cr:
                if row_count == 0:
                    cw.writerow(row)
                else:
                    cw.writerow(row[:-1])
                row_count += 1
        if display: print("Done.")
    return

# Compress an input file to a '.gz' GZIP file
def compress(input_filename):
    output_filename = input_filename + ".gz"
    with open(input_filename, 'rb') as f_in, gzip.open(output_filename, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    return

# Decompress a '.gz' GZIP file
def decompress(input_filename, output_filename=None):
    if not input_filename.endswith(".gz"):
        print("decompress: input_filename must end with '.gz'")
        return
    output_filename = splitext(input_filename)[0]
    with gzip.open(input_filename, 'rb') as f_in, open(output_filename, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    return

# Given full pathname of TXT file (containing one item per line)
# Return a list of these items (as strings)
def read_list(txt_pathname):
    li = []
    with open(txt_pathname, 'r') as f:
        for line in f:
            li.append(line.strip())
    return li

# Given list and full pathname of TXT file
# Write the list items to the file (one item per line)
def write_list(li, txt_pathname):
    with open(txt_pathname, 'w') as f:
        for item in li:
            f.write("{}\n".format(item))

# Delete a file (if it exists)
def silent_remove(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred
