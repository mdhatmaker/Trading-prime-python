
#-----------------------------------------------------------------------------------------------------------------------

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

#-----------------------------------------------------------------------------------------------------------------------


def get_output_row(t):
    output = '{0},{1:.2f},{2:.2f},{3:.2f},{4:.2f},{5:.2f},{6:.2f},{7},{8},{9},{10}'.format(t.Side, t.EntryDiscount, t.ExitDiscount, t.AdjustDiscount, t.EntrySpread, t.ExitSpread, t.AdjustSpread, t.EntryDate.strftime("%Y-%m-%d"), t.ExitDate.strftime("%Y-%m-%d"), t.HoldingDays, t.RollCount)
    return output

def write_file(filename, column_names, df):
    f = open(join(folder, "copper", filename), 'w')
    f.write(column_names + '\n')
    for index, row in df.iterrows():
        output = get_output_row(row)
        f.write(output + '\n')
    f.close()
    return


################################################################################

#print "Use -tradefile command line arg to specify the file containing copper trades  (ex: -tradefile='backtest discount 1 0.csv')"
#print "Use -days to specify number of days after roll dates  (ex: -days=7)"
print

#args['tradefile'] = 'backtest discount 1 0.csv'
#args['days'] = 7

#if not ('tradefile' in args or 'days' in args):
#    print "Error: Must provide -tradefile and -days command line args for file containing trades and days after roll dates."
#    sys.exit()

#trade_filename = args['tradefile']
#days = int(args['days'])


print "date range:", min_date, "to", max_date
print


df = df_all.sort_values('Date')



########## READ CALENDAR ROLLS FILE ##########
copper_data_filename = "premium_discount_updated.csv"

df_data = pd.read_csv(join(project_folder, copper_data_filename), parse_dates=['DateTime'])

df_symbols = df_data['Symbol'].drop_duplicates()

xdict = {}
for symbol in df_symbols:
    s1 = symbol[0:4]
    s2 = symbol[6:11]
    dfx = df_data[df_data['Symbol'].str.contains(s2)==True]
    dfx = dfx[dfx['Symbol'].str.contains(s1)==True]
    xsymbol = s1 + "xx" + s2 + "xx"
    xcount = dfx.shape[0]
    xmin = float(dfx['Cal'].min())
    xmax = float(dfx['Cal'].max())
    xmean = round(float(dfx['Cal'].mean()),2)
    xmedian = float(dfx['Cal'].median())
    #xmode = dfx['Cal'].mode().values
    xstd = round(float(dfx['Cal'].std()),2)
    xdict[xsymbol] = [xcount, xmin, xmax, xmean, xmedian, xstd]

output_filename = "copper_calendars.csv"
f = open(join(project_folder, output_filename), 'w')

column_names = "Symbol,Count,Min,Max,Mean,Median,Std"
print column_names
f.write(column_names + '\n')


for k in sorted(xdict.keys()):
    x = xdict[k]
    output = '{0},{1},{2:.2f},{3:.2f},{4:.2f},{5:.2f},{6:.2f}'.format(k, x[0], x[1], x[2], x[3], x[4], x[5])
    print output
    f.write(output + '\n')

f.close()

print
print "Data read from file: '{0}'".format(copper_data_filename)
print "Output to file: '{0}'".format(output_filename)


