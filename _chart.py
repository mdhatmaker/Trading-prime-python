import plotly.plotly as py
import plotly.graph_objs as go
import plotly
from plotly.graph_objs import Scatter, Layout

#-----------------------------------------------------------------------------------------------------------------------

#execfile("f_analyze.py")
execfile("f_folders.py")
execfile("f_tools.py")

#-----------------------------------------------------------------------------------------------------------------------

reds = ['rgb(164, 10, 19)', 'rgb(246, 15, 29)', 'rgb(255, 30, 58)']
blues = ['rgb(18, 77, 134', 'rgb(31, 161, 234)', 'rgb(47, 242, 255)']


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

def get_plot_line(df, price_column, series_name, color):
    #ylim = [0,2]
    trace0 = go.Scatter(
        x=df.DateTime,
        y=df[price_column],
        name=series_name,
        line = dict(color=(color), width=1),
        #yaxis = dict(range = ylim)
        )
    return [trace0]

def create_chart(chart_data, chart_name, chart_title='', chart_range=[]):
    if chart_title == '':
        if len(chart_range) == 2:
            layout = go.Layout(showlegend=False, yaxis=dict(range=chart_range))
        else:
            layout = go.Layout(showlegend=False)
    else:
        if len(chart_range) == 2:
            layout = go.Layout(title=chart_title, showlegend=False, yaxis=dict(range=chart_range))
        else:
            layout = go.Layout(title=chart_title, showlegend=False)
    fig = go.Figure(data=chart_data, layout=layout)
    plotly.offline.plot(fig, filename=join(html_folder, chart_name + ".html"))
    return

def prices_info(prices_filename, price_column, series_name):
    pdict = {}
    pdict['filename'] = prices_filename
    pdict['column'] = price_column
    pdict['series'] = series_name
    return pdict

# Create the default copper spread/discount chart
def display_chart(prices_info, line_color, chart_name, chart_title):
    #chart_range = [-0.5, 3.5]
    df_all = pd.read_csv(join(folder, prices_info['filename']), parse_dates=['DateTime'])
    
    chart_data = []
    #df = df_all[(df_all.Month==month) & (df_all.Year==year)]
    df = df_all
    
    if not df.empty:
        chart_data.extend(get_plot_line(df, prices_info['column'], prices_info['series'], line_color))
        create_chart(chart_data, chart_name, chart_title) #, chart_range)
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



################################################################################

if get_arg('project') == None:
    set_args_from_file("chart_vix_contango.args")
    #set_args_from_file("chart_ho_continuous.args")
    #set_args_from_file("chart_ho_contango_ratio.args")
    #set_args_from_file("chart_vix_calendar_0_1.args")
    #set_args_from_file("chart_ho_calendar_3_6.args")
    #set_args_from_file("chart_vix_calendar.args")



if get_arg('input_filename') == None:
    set_args( {'input_filename': join(data_folder, get_arg('project'), get_arg('filename'))} )
              
(folder, filename, ext) = split_pathname(get_arg('input_filename'))

print filename + " Chart"
#print "Use -1 to display alternate chart 1 (histograms)"
#print "Use -2 to display alternate chart 2 (show 3-month HG calendars)"
print


#pinfo = prices_info(filename + ext, 'Close', series_name)
pinfo = prices_info(filename + ext, get_arg('column_name'), get_arg('series_name'))
display_chart(pinfo, reds[0], filename, get_arg('chart_title'))

                      
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

