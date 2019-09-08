from __future__ import print_function
from os.path import join, exists
import os.path
import sys


def STOP(x=None):
    print("****SUCCESS****")
    if x is not None: print(x)
    sys.exit()

def get_arg(i):
    args = sys.argv
    if (len(args) > i):
        return args[i]
    else:
        return None

"""
# OLD STYLE = SET THE ROOT_FOLDER MANUALLY
#root_folder = "/Users/michael/Dropbox/alvin/"
#root_folder = "C:\\Users\\Michael\\Dropbox\\alvin\\"
#root_folder = "D:\\Users\\mhatmaker\\Dropbox\\alvin\\"
root_folder = "X:\\Users\\Trader\\Dropbox\\alvin\\"

# NEW STYLE = CHECK FOR EXISTENCE OF VARIOUS ROOT_FOLDER VALUES
if exists("X:\\Users\\Trader\\Dropbox\\alvin\\"):
    root_folder = "X:\\Users\\Trader\\Dropbox\\alvin\\"
elif exists("C:\\Users\\Trader\\Dropbox\\alvin\\"):
    root_folder = "C:\\Users\\Trader\\Dropbox\\alvin\\"
elif exists("D:\\Users\\mhatmaker\\Dropbox\\alvin\\"):
    root_folder = "D:\\Users\\mhatmaker\\Dropbox\\alvin\\"
elif exists("/Users/michael/Dropbox/alvin/"):
    root_folder = "/Users/michael/Dropbox/alvin/"
else:
    sys.exit("No valid root_folder found!")
"""

# EVEN NEWER STYLE - CHECK 'DROPBOXPATH' ENVIRONMENT VARIABLE
dropboxpath = os.getenv('DROPBOXPATH')
if dropboxpath is not None and exists(join(dropboxpath, 'alvin')):
    root_folder = join(dropboxpath, 'alvin')
else:
    print("No valid root_folder found!")
    sys.exit()


python_folder = join(root_folder, "python")
args_folder = join(python_folder, "args")
data_folder = join(root_folder, "data")
raw_folder = join(data_folder, "RAW_DATA")
df_folder = join(data_folder, "DF_DATA")
html_folder = join(data_folder, "charts")
quandl_folder = join(data_folder, "DF_QUANDL")
crypto_folder = join(data_folder, "DF_CRYPTO")
misc_folder = join(data_folder, "MISC")
system_folder = join(data_folder, "SYSTEM")
excel_folder = join(data_folder, "EXCEL")
charts_folder = join(data_folder, "CHARTS")

# Add to the PYTHONPATH the folder containing python function modules
li = ["X:\\Users\\Trader\\Dropbox\\alvin\\python", "C:\\Users\\Trader\\Dropbox\\alvin\\python", "D:\\Users\\mhatmaker\\Dropbox\\alvin\\python", "/Users/michael/Dropbox/alvin/python"]
sys.path.extend( [path for path in li if os.path.exists(path)] )

print("")
print("DATA folder:", data_folder)
print("")
