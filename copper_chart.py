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

def get_trace(df, x_column, y_column, line_name, line_color, line_width=1, line_dash=''):
    trace0 = go.Scatter(
        x=df[x_column],
        y=df[y_column],
        name=line_name,
        mode = 'lines',
        #mode = 'lines+markers',
        #mode = 'markers',
        #line = dict(color=(line_color), width=line_width)
        line = dict(color=(line_color), width=line_width, dash=line_dash)
        #line = dict(color=(line_color), width=line_width, dash='dash')
        #line = dict(color=(line_color), width=line_width, dash='dot')
        )
    return trace0

def get_plot_lines(df, red, blue):
    trace0 = go.Scatter(
        x=df.DateTime,
        y=df.Spread,
        name='spread',
        line = dict(color=(red), width=1)
        )
    trace1 = go.Scatter(
        x=df.DateTime,
        y=df.Discount,
        name='discount',
        line = dict(color=(blue), width=1)
        )
    return [trace0, trace1]

def create_chart(chart_data, chart_name):
    layout = go.Layout(showlegend=False)
    fig = go.Figure(data=chart_data, layout=layout)
    plotly.offline.plot(fig, filename=join(html_folder, chart_name + ".html"))
    return

# Create the default copper spread/discount chart
def display_default_chart(data_filename):
    #df_all = pd.read_csv(folder + "copper_premium_discount.csv", parse_dates=['DateTime'])
    #df_all = pd.read_csv(data_filename, parse_dates=['DateTime'])
    df_all = read_dataframe(data_filename)
    chart_data = []
    count = 0
    sorted_symbols = df_get_sorted_symbols(df_all)
    for symbol in sorted_symbols:
        df = df_all[df_all.Symbol == symbol].sort_values('DateTime')
        i = count % 2
        chart_data.extend(get_plot_lines(df, colors_reds[i], colors_blues[i]))
        count += 1
    create_chart(chart_data, 'copper_spread_{0}'.format(datetime.now().strftime("%Y%m%d")))
    return

# Create the histogram charts
def display_chart1():
    chart_data = []
    chart_data = [go.Histogram(x=df_hist['g_spread'])]
    plotly.offline.plot(chart_data, filename='spread_hist.html')
    chart_data = [go.Histogram(x=df_hist['g_discount'])]
    plotly.offline.plot(chart_data, filename='discount_hist.html')
    return

# Create the chart overlaid with the 3-month HG calendar prices
def display_chart2():
    #df = pd.read_csv(folder + "copper_premium_discount.csv", parse_dates=['DateTime'])
    df = pd.read_csv(join(project_folder, "copper_discount.DF.csv"), parse_dates=['DateTime'])
    chart_data = []
    chart_data.append(get_trace(df, 'DateTime', 'Spread', 'spread', reds[1], 1, 'dot'))
    chart_data.append(get_trace(df, 'DateTime', 'Discount', 'discount', blues[1], 1, 'dot'))

    dfx = df.copy()
    dfx.Symbol = dfx.Symbol.apply(lambda x: x if x.startswith("QHGZ") else "")
    dfx.loc[dfx.Symbol == '', 'Cal'] = np.nan
    chart_data.append(get_trace(dfx, 'DateTime', 'Cal', 'Z', 'gray', 3))

    dfx = df.copy()
    dfx.Symbol = dfx.Symbol.apply(lambda x: x if x.startswith("QHGH") else "")
    dfx.loc[dfx.Symbol == '', 'Cal'] = np.nan
    chart_data.append(get_trace(dfx, 'DateTime', 'Cal', 'H', 'black', 3))

    create_chart(chart_data, 'spread_cals')
    return


################################################################################

print("Use -1 to display alternate chart 1 (histograms)")
print("Use -2 to display alternate chart 2 (show 3-month HG calendars)")
print()
    
#if not ('e' in args and 'x' in args):
#    print "Error: Must provide -e and -x command line args for entry and exit prices."
#    sys.exit()

#entry_price = int(args['e'])
#exit_price = int(args['x'])

#filename = join(project_folder, "copper_discount.2.DF.csv")
filename = "copper_settle_discount.2.DF.csv"
df = read_dataframe(filename)

min_date = df.DateTime.min()
max_date = df.DateTime.max()
print("date range:", min_date, "to", max_date)
print()


if '1' in args:
    # histograms
    display_chart1()
elif '2' in args:
    # chart overlaid with 3-month HG calendars
    display_chart2()
else:
    # default chart (copper spread/discount)
    display_default_chart(filename)



#df = web.DataReader('aapl', 'yahoo', datetime(2015, 1, 1), datetime(2016, 7, 1))
#data = [go.Scatter(x=df.index, y=df.High)]
#py.iplot(data)

#df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv")
#data = [go.Scatter(x=df.Date, y=df['AAPL.Close'])]
#plotly.offline.plot(data)
#py.iplot(data)

#plotly.offline.plot({
#    "data": [Scatter(x=[1, 2, 3, 4], y=[4, 3, 2, 1])],
#    "layout": Layout(title="hello world")
#})
