import plotly.plotly as py
import plotly.graph_objs as go
import plotly
from plotly.graph_objs import Scatter, Layout

#-----------------------------------------------------------------------------------------------------------------------

execfile("f_analyze.py")
html_folder = folder + "charts\\"
folder += "HOGO\\RAW_DATA\\"

#-----------------------------------------------------------------------------------------------------------------------

def get_trace(df, x_column, y_column, line_name, line_color, line_width=1, line_dash=''):
    trace0 = go.Scatter(
        x=df[x_column],
        y=df[y_column],
        name=line_name,
        mode = 'lines',
        #mode = 'lines+markers',
        #mode = 'markers',
        line = dict(color=(line_color), width=line_width, dash=line_dash)
        #line = dict(color=(line_color), width=line_width, dash='dash')
        #line = dict(color=(line_color), width=line_width, dash='dot')
        )
    return trace0

def get_plot_line(df, color):
    #ylim = [0,2]
    trace0 = go.Scatter(
        x=df.DateTime,
        y=df.Price,
        name='hogo price',
        line = dict(color=(color), width=1),
        #yaxis = dict(range = ylim)
        )
    return [trace0]

def create_chart(chart_data, chart_name, chart_range, chart_title=''):
    if chart_title == '':
        layout = go.Layout(showlegend=False, yaxis=dict(range=chart_range))
    else:
        layout = go.Layout(title=chart_title, showlegend=False, yaxis=dict(range=chart_range))
    fig = go.Figure(data=chart_data, layout=layout)
    plotly.offline.plot(fig, filename=html_folder + chart_name + ".html")
    return

# Create the default copper spread/discount chart
def display_default_chart(month, year, line_color, chart_name):
    if chart_name == 'hogo':
        chart_range = [-2, 2]
    elif chart_name == 'goho':
        chart_range = [-5, 1]
    filename = chart_name + "_prices.csv"
    df_all = pd.read_csv(folder + filename, parse_dates=['DateTime'])
    
    chart_data = []
    df = df_all[(df_all.Month==month) & (df_all.Year==year)]
    if not df.empty:
        chart_data.extend(get_plot_line(df, line_color))
        name = chart_name + '-' + str(month) + '-' + str(year)
        (m2, y2) = next_month(month, year)
        (m3, y3) = next_month(m2, y2)
        d1 = datetime(year, month, 1)
        d2 = datetime(y2, m2, 1)
        d3 = datetime(y3, m3, 1)
        title = chart_name.upper() + ' ' + d1.strftime("%Y %b")
        title += d2.strftime("-%b")
        title += d3.strftime("-%b")
        create_chart(chart_data, name, chart_range, title)
        #dt_max = df.DateTime.max()

    #chart_data = []
    #df = df_all[(df_all.Month==2) & (df_all.Year==2017)]
    #df = df[df.DateTime > dt_max]
    #chart_data.extend(get_plot_line(df, blues[0]))
    #create_chart(chart_data, 'hogo_2-2017')
    #dt_max = df.DateTime.max()    

    return

"""
# Create the chart overlaid with the 3-month HG calendar prices
def display_chart2():
    df = pd.read_csv(folder + "copper_premium_discount.csv", parse_dates=['DateTime'])
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
"""

def create_hogo_charts(year_list):
    for y in year_list:
        for m in range(1, 12+1):
            if m % 2 == 0:
                color = reds[0]
            else:
                color = blues[0]
            display_default_chart(m, y, color, 'hogo')
    return

def create_goho_charts(year_list):
    for y in year_list:
        for m in range(1, 12+1):
            if m % 2 == 0:
                color = reds[1]
            else:
                color = blues[1]
            display_default_chart(m, y, color, 'goho')
    return


################################################################################

print "HOGO Charts"
#print "Use -1 to display alternate chart 1 (histograms)"
#print "Use -2 to display alternate chart 2 (show 3-month HG calendars)"
print


year_list = [2016, 2017]

#create_hogo_charts(year_list)
create_goho_charts(year_list)

                      
"""    
print "date range:", min_date, "to", max_date
print

if '1' in args:
    # histograms
    display_chart1()
elif '2' in args:
    # chart overlaid with 3-month HG calendars
    display_chart2()
else:
    # default chart (copper spread/discount)
    display_default_chart()
"""


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
