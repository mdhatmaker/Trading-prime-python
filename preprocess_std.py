import pandas as pd
import numpy as np
from os.path import join
import datetime

#import os
#path = os.path.dirname(os.path.realpath(__file__))
#path = "/Users/michael/CLionProjects/ALVIN/data_sd_analysis"        # macbook
#path = "C:\\Users\\Michael\\Dropbox\\ALVIN\\data_sd_analysis"       # lenovo laptop
path = "B:\\Users\\mhatmaker\\Dropbox\\ALVIN\\data_sd_analysis"     # apartment desktop


def get_datetime_from_d_t(d_str, t_str):
    vd = d_str.split('/')
    vt = t_str.split(':')
    return datetime.datetime(int(vd[2]), int(vd[0]), int(vd[1]), int(vt[0]), int(vt[1]), int(vt[2]))

def convert_mc_to_dataframe_with_index(name):
    print "Converting Multicharts to Dataframe:", name,
    f = open(join(path, name + ".txt"), 'r')
    fout = open(join(path, name + ".csv"), 'w')
    line = f.readline()
    col = line.split(',')
    col_csv = []
    for s in col:
        s = s.strip().strip("<>")
        col_csv.append(s)
    # Combine "Date" and "Time" into "DateTime"
    col_csv[0] = "DateTime"
    col_csv.remove(col_csv[1])
    fout.write('ID,' + ','.join(col_csv) + '\n')
    
    i = 0
    line = f.readline()
    while (line):
        if i % 10000 == 0:
            print ".",
        col = line.split(',')
        dt = get_datetime_from_d_t(col[0], col[1])
        col[1] = str(dt)
        #col.remove(col[2])
        col[0] = str(i)
        #fout.write(str(i) + "," + line)
        fout.write(','.join(col))
        i += 1
        line = f.readline()
    print
    
    fout.close()
    f.close()
    return

def convert_mc_to_dataframe(name):
    print "Converting Multicharts to Dataframe:", name
    f = open(join(path, name + ".txt"), 'r')
    fout = open(join(path, name + ".csv"), 'w')
    line = f.readline()
    col = line.split(',')
    col_csv = []
    for s in col:
        s = s.strip().strip("<>")
        col_csv.append(s)
    #fout.write(',' + ','.join(col_csv) + '\n')
    fout.write(','.join(col_csv) + '\n')
   
    #i = 0
    line = f.readline()
    while (line):
        #fout.write(str(i) + "," + line)
        fout.write(line)
        #i += 1
        line = f.readline()

    fout.close()
    f.close()
    return


################################################################################


#convert_mc_to_dataframe_with_index("QCL#C 15 Minutes")
#convert_mc_to_dataframe_with_index("QCL#C 30 Minutes")
convert_mc_to_dataframe_with_index("QCL#C 45 Minutes")
#convert_mc_to_dataframe_with_index("QCL#C 60 Minutes")

#convert_mc_to_dataframe_with_index("QCL#C 1 Tick Bar")







