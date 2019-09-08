import plotly.plotly as py
import plotly.graph_objs as go
import plotly
from plotly.graph_objs import Scatter, Layout
from os.path import join
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#-----------------------------------------------------------------------------------------------------------------------
import f_folders as folder
from f_dataframe import *
from f_stats import *
from f_file import *

#-----------------------------------------------------------------------------------------------------------------------

plot_colors = { 'b':'blue', 'g':'green', 'r':'red', 'c':'cyan', 'm':'magenta', 'y':'yellow', 'k':'black', 'w':'white' }
plot_line_styles = { '_':'solid', '-':'dashed', '.':'dotted' }

# Given a single-character string
# Return True if this character represents one of the built-in MatPlotLib colors; otherwise, return False
def is_plot_color(char1):
    return char1 in plot_colors.keys()

# Given a 2-character string containing a symbol followed by color-identifying letter ('_r', '-b', '.g', etc.)
# Return a tuple containing (line_style, line_color) to be used in plots
def get_line_style_and_color(char2):
    line_style = plot_line_styles[char2[0]]
    line_color = plot_colors[char2[1]]
    return line_style, line_color

# Begin the plot of a figure for the given dataframe with a horizontal line at the given yvalue
# (optional) figure defaults to 1
# (optional) subplot defaults to None
# (optional) style defaults to 'dark_background' (plt.style.available is list of style strings)
# sample plt.style.available list:
# [u'seaborn-darkgrid', u'seaborn-notebook', u'classic', u'seaborn-ticks', u'grayscale', u'bmh',
# u'seaborn-talk', u'dark_background', u'ggplot', u'fivethirtyeight', u'_classic_test', u'seaborn-colorblind',
# u'seaborn-deep', u'seaborn-whitegrid', u'seaborn-bright', u'seaborn-poster', u'seaborn-muted',
# u'seaborn-paper', u'seaborn-white', u'seaborn-pastel', u'seaborn-dark', u'seaborn', u'seaborn-dark-palette']
def plot_with_hline(df, yvalue, chartx=None, figure=1, subplot=None, style='dark_background'):
    if chartx is None: chartx = df['DateTime']
    plt.style.use(style)
    plt.figure(figure)
    if subplot is not None: plt.subplot(subplot)
    plt.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.05)
    #xmin = df['DateTime'].min()
    #xmax = df['DateTime'].max()
    plot_hline(df, yvalue, chartx)
    return

# Plot a horizontal line at the given yvalue (default yvalue=0)
def plot_hline(df, yvalue, chartx=None, alpha=0.75):
    if chartx is None: chartx = df['DateTime']
    #xmin = df.index.min()
    #xmax = df.index.max()
    #xmin = df['DateTime'].min()
    #xmax = df['DateTime'].max()
    xmin = chartx.min()
    xmax = chartx.max()
    plt.hlines(yvalue, xmin, xmax, colors='gray', linestyles='dashed', alpha=alpha, label='')
    return

# Begin the plot of a figure for the given dataframe
# (optional) figure defaults to 1
def plot_without_hline(figure=1):
    plt.style.use('dark_background')
    plt.figure(figure)
    plt.subplots_adjust(left=0.03, right=0.98, top=0.98, bottom=0.03)
    return

# Plot vertical lines at each roll date for the given dataframe[symbol_column]
def plot_roll_dates(df, symbol_column, ymin=-10, ymax=+25, alpha=0.5):
    roll_dates = df_get_roll_dates(df, symbol_column)
    for symbol,first_date,last_date in roll_dates:
        plt.vlines(last_date, ymin, ymax, colors='gray', linestyles='dotted', alpha=alpha, label='')
    return

# Calculate (and plot) a best-fit line using points in X and Y
def plot_fit_line(X, Y, N=100):     # N could be just 2 if you are only drawing a straight line
    m, b = fit_line2(X, Y)
    points = np.linspace(X.min(), X.max(), N)
    plt.plot(points, m*points + b)
    return

# Given a dataframe on which we have run the alvin_arima function
# Plot the ARIMA value, EMA (exponential moving average), and the underlying value (in a separate subplot)
# (optional) column_name defaults to 'Close', but you can change this to match the name of the underlying data column
def plot_arima(df0, title="ARIMA", tail_count=None, xaxis_lambda=None, column_name='Close'):
    if (tail_count is None):
        df = df0.copy()
    else:
        df = df0.tail(tail_count).copy()
    abs_ema_max = max(abs(df.EMA.max()), abs(df.EMA.min()))
    abs_arima_max = max(abs(df.ARIMA.max()), abs(df.ARIMA.min()))
    df['ARIMA'] = abs_ema_max * df.ARIMA / abs_arima_max

    idx = df.index

    plot_with_hline(df, 0, chartx=idx, subplot=211)

    plt.plot(idx, df['EMA'], color='yellow', linewidth=1)
    plt.bar(idx, df['ARIMA'], color='white')

    if xaxis_lambda is None:
        df['xtick'] = df['DateTime']
    else:
        df['xtick'] = df['DateTime'].apply(xaxis_lambda)
    plt.xticks(idx, df['xtick'])

    plt.subplot(212)
    plt.title(title)
    plt.plot(idx, df[column_name], color='red', linewidth=2)
    #plot_hline(df, 0, chartx=idx)
    plt.xticks(idx, df['xtick'])

    # plt.figure(2)
    # plt.subplot(211)
    # plt.plot(df.index, df['EMA'], color='yellow', linewidth=1)
    # plt.bar(df.index, df['AR2'], color='white')
    # plt.subplot(212)
    # plt.plot(df.index, df['Close'], color='red', linewidth=2)

    plt.show()
    return

# Given a dataframe containing columns 'EMA' and 'ARIMA'
# scale the ARIMA values to match the scale of the EMA
def scale_arima_plot(df):
    abs_ema_max = max(abs(df.EMA.max()), abs(df.EMA.min()))
    abs_arima_max = max(abs(df.ARIMA.max()), abs(df.ARIMA.min()))
    arima_max = abs_ema_max * df.ARIMA.max() / abs_arima_max
    arima_min = abs_ema_max * df.ARIMA.min() / abs_arima_max
    df['ARIMA'] = abs_ema_max * df.ARIMA / abs_arima_max
    return df

# Given a dataframe containing columns 'EMA' and 'ARIMA' (and 'Close' for closing price)
# plot the EMA/ARIMA on the top subplot and underlying price on the bottom subplot
def plot_ARIMA(df, title="", scale_arima=False):
    if scale_arima == True:
        scale_arima_plot(df)
    plot_with_hline(df, subplot=211)
    ind = df.index
    plt.plot(ind, df['EMA'], color='yellow', linewidth=1)
    plt.bar(ind, df['ARIMA'], color='white')
    df['xtick'] = df['DateTime'].apply(lambda x: x.strftime('%m-%d') if (x.day % 2 == 0 and x.hour == 12 and x.minute == 0) else '')
    plt.xticks(ind, df['xtick'])
    plt.subplot(212)
    plt.title(title)
    plt.plot(ind, df['Close'], color='red', linewidth=2)
    plot_hline(df)
    plt.xticks(ind, df['xtick'])
    plt.show()
    return

def savefig(plt, pathname):
    silent_remove(pathname)
    plt.savefig(pathname)
    return

# Given a dataframe and a dictionary of key/value pairs representing column/style
# plot the given column values in the style (line color and line style) provided
# (optional) if a column name is provided for 'roll_symbol', this column will be used to
# determine roll dates and roll dates will be plotted
# TODO: for now, we assume that xvalues are ALWAYS 'DateTime'
def plot_(df, arg_dict, roll_symbol=None):
    #plot_with_hline(df, 0)
    patches = []
    for column in arg_dict.keys():
        style = arg_dict[column]
        if is_plot_color(style[0]):                         # style is like 'ro' or 'bx' or 'g.' or...
            plt.plot(df['DateTime'], df[column], style)
            patches.append(mpatches.Patch(color=plot_colors[style[0]], label=column))
        else:                                               # style is like '_r' or '-b' or '.g' or...
            line_style, line_color = get_line_style_and_color(style)
            plt.plot(df['DateTime'], df[column], color=line_color, linestyle=line_style, linewidth=1)
            patches.append(mpatches.Patch(color=line_color, label=column))
    if roll_symbol is not None:
        plot_roll_dates(df, roll_symbol, ymin=-10, ymax=+30)
    plt.legend(handles=patches)
    plt.show()  #block=False)
    return

"""
def plot_roll_dates(df, symbol_column, value_column, ymin=None, ymax=None, alpha=0.5):
    if ymin is None:
    roll_dates = df_get_roll_dates(df, symbol_column)
    for dt in roll_dates:
        plt.vlines(dt, ymin, ymax, colors='gray', linestyles='dotted', alpha=alpha, label='')
    return
"""


"""
====================================================================================================
Here are the basic built-in MatPlotLib colors (along with their 1-character abbreviation):
b: blue
g: green
r: red
c: cyan
m: magenta
y: yellow
k: black
w: white

And the system for lines would have the symbol first:
_b = blue line
-b = blue dashed line
.b = blue dotted lines

While the "marker" style would preserve the MatPlotLib format with color first:
rx = red "X"
ro = red "o"

================    ================================================================================
character           description
================    ================================================================================
   -                solid line style
   --               dashed line style
   -.               dash-dot line style
   :                dotted line style
   .                point marker
   ,                pixel marker
   o                circle marker
   v                triangle_down marker
   ^                triangle_up marker
   <                triangle_left marker
   >                triangle_right marker
   1                tri_down marker
   2                tri_up marker
   3                tri_left marker
   4                tri_right marker
   s                square marker
   p                pentagon marker
   *                star marker
   h                hexagon1 marker
   H                hexagon2 marker
   +                plus marker
   x                x marker
   D                diamond marker
   d                thin_diamond marker
   |                vline marker
   _                hline marker
================    ================================================================================
"""
