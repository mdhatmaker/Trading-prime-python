execfile(r'..\..\..\python\f_analyze.py')
import f_folders as folder
from f_args import *
from f_date import *
from f_file import *
from f_calc import *
from f_dataframe import *
from f_rolldate import *
from f_chart import *

project_folder = join(data_folder, "copper")

################################################################################


print

print "date range:", min_date, "to", max_date
print

#print "data count:", df['spread'].value_counts()
#print_stats("spread")
#print_stats("discount")

print df_all.describe()

#dfg.plot.hist()

print
print "spread ranges"
print "---------------"
print_ranges(df_hist['g_spread'].value_counts())

print
print "discount ranges"
print "---------------"
print_ranges(df_hist['g_discount'].value_counts())

print

