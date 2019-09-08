import plotly.plotly as py
import plotly.graph_objs as go
import plotly
from plotly.graph_objs import Scatter, Layout
from os.path import join
import pandas as pd 
import sys


#-----------------------------------------------------------------------------------------------------------------------

import f_folders as folder

#execfile("f_analyze.py")
#execfile("f_folders.py")
#execfile("f_tools.py")

#-----------------------------------------------------------------------------------------------------------------------

colors_reds = ['rgb(164, 10, 19)', 'rgb(246, 15, 29)', 'rgb(255, 30, 58)']
colors_blues = ['rgb(18, 77, 134', 'rgb(31, 161, 234)', 'rgb(47, 242, 255)']
colors_vga = ['navy', 'green 1', 'red 1', 'gray', 'magenta', 'purple', 'black', 'blue', 'cyan / aqua', 'teal', 'yellow 1', 'maroon', 'silver']

df_colors = None

# Given an integer i that can be incremented in the calling code loop
# Return the next color in the vga_colors list
# this is a good way to just "get a different color" as you append lines to a chart
def chart_color(i):
    return colors_vga[i % len(colors_vga)]

# Load a dataframe with a set of colors from a specified file
# Format of dataframe columns should be: ['Color_Name','Hex','R','G','B']
def load_colors(colors_filename):
    global df_colors
    df_colors = pd.read_csv(colors_filename)
    return

def get_color(color_name):
    global df_colors
    if type(df_colors) != 'pandas.core.frame.DataFrame':
        #current_script_path = os.path.dirname(os.path.realpath(__file__))
        pathname = join(folder.data_folder, 'MISC', 'web_colors.csv')
        load_colors(pathname)
    rows = df_colors[df_colors['Color_Name']==color_name]
    if rows.empty:
        return (None, None, None, None)
    else:
        row = rows.iloc[0]
        return (row['Hex'], row['R'], row['G'], row['B'])

def get_trace(df, line_name, y_column, x_column='DateTime', line_color='rgb(0,0,0)', line_width=1, line_dash='', mode='lines', marker_labels=None):
    if marker_labels == None:
        trace0 = go.Scatter(
            x=df[x_column],
            y=df[y_column],
            name=line_name,
            #mode = 'lines',
            #mode = 'lines+markers',
            #mode = 'markers',
            line = dict(color=(line_color), width=line_width, dash=line_dash)
            #line = dict(color=(line_color), width=line_width, dash='dash')
            #line = dict(color=(line_color), width=line_width, dash='dot')
            )
    else:
        trace0 = go.Scatter(
            x=df[x_column],
            y=df[y_column],
            text=df[marker_labels],
            name=line_name,
            line = dict(color=(line_color), width=line_width, dash=line_dash)
            )
    return trace0

def get_trace_ohlc(df, x_column='DateTime'):
    trace = go.Ohlc(
        x=df[x_column],
        open=df.Open,
        high=df.High,
        low=df.Low,
        close=df.Close
    )
    return trace

def create_chart(chart_data, chart_filename, chart_title='', chart_range=[]):
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
    plotly.offline.plot(fig, filename=chart_filename)
    return


# Create the default copper spread/discount chart
def display_chart(trace_list, chart_filename, chart_title):
    #chart_range = [-0.5, 3.5]
    #df_all = pd.read_csv(join(folder, prices_info['filename']), parse_dates=['DateTime'])
    
    create_chart(trace_list, chart_filename, chart_title) #, chart_range)
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

# ---------- NEW AND IMPROVED CHART FUNCTIONS ----------
# Example use:
# df = (some dataframe)
# lines = []
# lines.append(get_chart_line(df, "Close"))
# (append any other lines here)
# show_chart(lines, "my_chart")
# -------------------------------------------------------

# Given a dataframe (df) and at minimum, a y_column name (ex: 'Close')
# (optional) x_column name (defaults to 'DateTime')
# (optional) line_name (defaults to the y_column name)
# (optional) line_color (defaults to 'rgb(0,0,0)')
# (optional) line_width (defaults to 1)
# (optional) line_dash (defaults to 'solid' -- can also be 'dash' or 'dot' or 'dashdot')
# (optional) mode (defaults to 'lines' -- can also be 'lines+markers' or 'markers')
# (optional) marker_labels (pandas Series defaults to using y_column values)
# Return a chart line -- that can be added to a list of lines passed to the show_chart function
def get_chart_line(df, y_column, x_column='DateTime', line_name=None, line_color='rgb(0,0,0)', line_width=1, line_dash='solid', mode='lines', marker_labels=None):
    if line_name == None: line_name = y_column
    if marker_labels == None: marker_labels = df[y_column]
    trace0 = go.Scatter(
        x=df[x_column],
        y=df[y_column],
        name=line_name,
        line = dict(color=(line_color), width=line_width, dash=line_dash)
        )
    return trace0

# Given a list of chart lines and at minimun, a chart title
# Show a plot.ly chart containing the given lines
def show_chart(chart_lines_list, chart_title, chart_filename=None, yaxis_range=[]):
    if chart_filename == None: chart_filename = join(folder.html_folder, chart_title + ".html")
    if len(yaxis_range) == 2:
        layout = go.Layout(title=chart_title, showlegend=False, yaxis=dict(range=yaxis_range))
    else:
        layout = go.Layout(title=chart_title, showlegend=False)
    fig = go.Figure(data=chart_lines_list, layout=layout)
    plotly.offline.plot(fig, filename=chart_filename)
    return

# Given a dataframe and a list of column names
# Show a chart that displays a line for each of these columns (in different colors)
# (optional) chart title defaults to 'Quick Chart'
def quick_chart(df, columns_list, title='Quick Chart'):
    lines = []
    i = 0
    for col in columns_list:
        lines.append(get_chart_line(df, col, line_color=chart_color(i)))
        i += 1
    show_chart(lines, title)
    return
################################################################################


