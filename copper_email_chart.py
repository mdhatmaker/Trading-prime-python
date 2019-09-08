import plotly.plotly as py
import plotly.graph_objs as go
import plotly
from plotly.graph_objs import Scatter, Layout
from os.path import join

#-----------------------------------------------------------------------------------------------------------------------

#execfile("f_analyze.py")
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




################################################################################

#print("Use -1 to display alternate chart 1 (histograms)")
#print("Use -2 to display alternate chart 2 (show 3-month HG calendars)")
print("")
    
#if not ('e' in args and 'x' in args):
#    print "Error: Must provide -e and -x command line args for entry and exit prices."
#    sys.exit()

#min_date = df.DateTime.min()
#max_date = df.DateTime.max()
#print("date range:", min_date, "to", max_date)
#print()

onlyfiles = [f for f in listdir(charts_folder) if isfile(join(charts_folder, f)) and f.startswith("copper_spread_") and f.endswith(".html")]

fdict = {}
for f in onlyfiles:
    print(f)
    stat = os.stat(join(charts_folder, f))
    fdict[stat.st_mtime] = f
    #print(stat.st_mtime)

dates = fdict.keys()
dates.sort()
dates.reverse()
print(dates)
print("\n" + fdict[dates[0]])

sys.exit()

if '1' in args:
    # histograms
    display_chart1()
elif '2' in args:
    # chart overlaid with 3-month HG calendars
    display_chart2()
else:
    # default chart (copper spread/discount)
    display_default_chart(filename)


