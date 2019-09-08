from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys


#-----------------------------------------------------------------------------------------------------------------------

import f_folders as folder
from f_dataframe import read_dataframe
from f_plot import *

#-----------------------------------------------------------------------------------------------------------------------

# Parse the command-line arguments for any with an equal sign ('col=2', 'season=winter', 'doit=True')
# Return a dictionary of key/value pairs from these args containing an equal sign
def get_args():
    arg_dict = {}
    for arg in sys.argv:
        if '=' in arg:
            parts = arg.split('=')
            arg_dict[parts[0]] = parts[1]
    return arg_dict

# Given a dataframe and a dictionary of key/value pairs representing column/style
# plot the given column values in the style (line color and line style) provided
# (optional) if a column name is provided for 'roll_symbol', this column will be used to
# determine roll dates and roll dates will be plotted
# TODO: for now, we assume that xvalues are ALWAYS 'DateTime'
def plot(df, arg_dict, roll_symbol=None):
    begin_plot(df)
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

# Plot 'contango' and 'contango_1x3x2'
def begin_plot(df):
    plot_with_hline(df, 0)
    return

########################################################################################################################

# TODO: Add ability to pass ROLL_SYMBOL_COLUMN which will display the roll dates on the chart
if len(sys.argv) <= 1:
    sys.argv = ['_plot.py', '@VX_contango.EXPANDED.daily.DF.csv', 'contango=_r', 'VX_1x3x2=-b', 'Close_VX=gx']

# COMMAND-LINE ARGUMENTS EXPECTED:
# 1. name of DATAFRAME ("*.DF.csv") file
# 2. [column]=CS or [column]=SC
# 3. (and on and on...) You can have as many of these #2 arguments as you want. They will all be plotted.

#python _plot.py "@VX_contango.EXPANDED.daily.DF.csv" contango=_r VX_1x3x2=-b Close_VX=gx
#-----------------------------------------------------------------------------------------------------------------------

data_filename = sys.argv[1]
df = read_dataframe(data_filename)

arg_dict = get_args()

print("Showing plot ... ", end='')
plot(df, arg_dict)
print("Done.")


