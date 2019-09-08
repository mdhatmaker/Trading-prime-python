import sys
import os
from os import listdir
from os.path import isfile, join
import glob, os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------------------------------------------------

import f_folders as folder
from f_args import *
from f_date import *
from f_file import *
from f_calc import *
from f_dataframe import *
from f_rolldate import *
from f_chart import *

project_folder = join(data_folder, "vix_es")

plt.style.use('ggplot')
#-----------------------------------------------------------------------------------------------------------------------


# Given a TRADE as a string (see trade_dict in other analysis)
# Return a list of the values that make up the trade
# (which can then be appended as rows that can create a dataframe)
def get_trade_as_list(trd):
    splits = trd.split('[')
    trd = [x.strip('] ') for x in splits][1:]
    tentry = trd[0]
    texit = trd[1]
    tresult = trd[2]
    tprofit = trd[3][trd[3].index('=')+1:].strip()
    tdrawdown = trd[4][trd[4].index('=')+1:].strip()

    splits = tentry.split()
    dt_str = splits[1]+" "+splits[2]
    #dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
    tentry = [splits[0], dt_str, splits[3], float(splits[4])]
    splits = texit.split()
    dt_str = splits[1]+" "+splits[2]
    #dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
    texit = [splits[0], dt_str, splits[3], float(splits[4])]
    #tentry_str = "{0},{1},{2},{3}".format(tentry[0], tentry[1], tentry[2], tentry[3])
    #texit_str = "{0},{1},{2},{3}".format(texit[0], texit[1], texit[2], texit[3])
    #row_str = "{0},{1},{2},{3},{4}".format(tentry_str, texit_str, tresult, float(tprofit), float(tdrawdown))
    li = []
    li.extend(tentry)
    li.extend(texit)
    li.extend([tresult, float(tprofit), float(tdrawdown)])
    return li

def print_results(df, filter_description):
    print filter_description
    df_losers = df[df.profit<0]
    df_losers = df_losers.sort_values('profit', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
    #print "LOSERS"
    #print df_losers[['Contango', 'profit']]
    #print

    df_winners = df[df.profit>0]
    df_winners = df_winners.sort_values('profit', axis=0, ascending=False, inplace=False, kind='quicksort', na_position='last')
    #print "WINNERS"
    #print df_winners[['Contango', 'profit']]
    #print

    print "{0} winners   {1} losers".format(df_winners.shape[0], df_losers.shape[0])
    winner_sum = df_winners.profit.sum()
    loser_sum = df_losers.profit.sum()
    net_sum = winner_sum + loser_sum
    print "winner_sum={0:8.2f}   loser_sum={1:8.2f}        net={2:8.2f}".format(winner_sum, loser_sum, net_sum)
    print
    return

############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################

quartiles = ['d4', 'd3', 'd2', 'd1', 'unch', 'u1', 'u2', 'u3', 'u4']



# for 'iq': 0=d4  1=d3  2=d2  3=d1  4=unch  5=u1  6=u2  7=u3  8=u4
iq = 3
Q_colname = quartiles[iq]       # text of selected quartile (for use as column_name in dataframe)


# GET THE setup+trades FILES FROM THE PROJECT FOLDER
filenames = []
i = 0
os.chdir(project_folder)
print "FILES"
for file in glob.glob("setup+trades.vix_es.*.csv"):
    i += 1
    print(i, file)
    filenames.append(file)
print

trades_filename = filenames[0]
df_all = pd.read_csv(join(project_folder, trades_filename))


# THIS IS A META-ANALYSIS OF VIX/ES TRADE DATA

# For each row, parse the TRADE into its component parts and put each of these in its own column
li_rows = []
for i in range(df_all.shape[0]):
    trd = df_all.iloc[i].Trade
    li_rows.append(get_trade_as_list(trd))

columns=["entry_symbol","entry_datetime","entry_side","entry_price","exit_symbol","exit_datetime","exit_side","exit_price","result","profit","drawdown"]
dfx = pd.DataFrame(li_rows, index=df_all.index, columns=columns)

df = pd.merge(df_all, dfx, left_index=True, right_index=True)

#print "DATAFRAMES: df_losers    df_winners"
#print


print_results(df, "NO FILTER")
print_results(df[df.Contango<=0], "FILTER FOR CONTANGO <= 0")
print_results(df[df.Contango>0], "FILTER FOR CONTANGO > 0")
print_results(df[df.Contango>=5], "FILTER FOR CONTANGO >= 5")
print_results(df[df.Contango>=10], "FILTER FOR CONTANGO >= 10")
print_results(df[df.Contango>=15], "FILTER FOR CONTANGO >= 15")




