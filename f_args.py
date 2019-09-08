from datetime import datetime
from os.path import join
from f_folders import *
from f_date import strdate
import sys

#-----------------------------------------------------------------------------------------------------------------------

args = {}

#-----------------------------------------------------------------------------------------------------------------------

# Print the possible arguments and their usage (the args that are generic to most or all Python scripts)
def print_generic_usage():
    """
    print "Use -input_file (-if) to load arguments from a .args file (.args extension is optional)"
    print " ex: -if=create_continuous_vx"
    print "Use -project (-p) to specify project (and project folder) name"
    print " ex: -project=vix_es"
    print "Use -symbol (-s) to specify symbol root"
    print " ex: -symbol=@VX"
    print "Use -filename (-f) to specify the input filename (within the project folder)"
    print " ex: -filename=vix_contango.csv"
    print "Use -timeframe (-t) to speccify the timeframe ('1m', '1h', 'Daily')"
    print " ex: -t=Daily"
    print "Use -column_name (-cn) to specify name of a data column to process (i.e. column to chart)"
    print " ex: -column_name=Close"
    print "Use -start_date (-sd) and -end_date (-ed) to specify start/end dates in YYYYMMDD format"
    print " ex: -sd=20170128 -ed=20171205"
    print "Use -start_year (-sy) and -end_year (-ey) to specify start/end year in YYYY format"
    print " ex: -sy=2013 -ey=2017"
    """
    return

def remove_quotes(text):
    txt = text.strip()
    if (txt.startswith("'") and txt.endswith("'")) or (txt.startswith('"') and txt.endswith('"')):
        return txt[1:-1]
    else:
        return txt
    
# Given dictionary of argument names/values
# Set the global arguments to the specified values (in global 'args' dictionary)
def set_args(args_dict):
    for k in args_dict:
        args[k] = args_dict[k]
    return

# Given a filename containing argument names/values (.args extension will be added if not provided)
# Add these to the global args as if they were entered individually on the command line
def set_args_from_file(args_filename):
    pathname = join(args_folder, args_filename)
    if not pathname.lower().endswith('.args'):
        pathname += '.args'
    lines = list(open(pathname))
    for line in lines:
        line = line.strip()
        if line.startswith('#') or line == '':
            continue
        if line.endswith(','):
            line = line[:-1]
        #print line
        splits = line.split(':', 1)
        arg_name = remove_quotes(splits[0])
        arg_value = remove_quotes(splits[1])
        args[arg_name] = arg_value
    return

# Given an argument name (ex: 'start_year')
# Return the argument value (or None if the argument does not exist)
# NOTE: This function will attempt to identify shortened versions of argument names
# (ex: -project or -p    -filename or -f    -column_name or -cn)
def get_arg(arg_name):
    if arg_name in args:
        return args[arg_name]
    else:
        splits = arg_name.split('_')
        abbrev = splits[0][0]
        for i in range(1, len(splits)):
            abbrev += splits[i][0]
        if abbrev in args:
            return args[abbrev]
    return None

# Given an argument name (ex: 'end_date')
# Return True if the argument (or its equivalent abbreviation) exists; otherwise return False
def is_arg(arg_name):
    return get_arg(arg_name) != None

########################################################################################################################

# The start_date and end_date should be modifiable from the command line
argv = sys.argv
argc = len(argv)

# Create dictionary of command-line arguments
if argc > 1:
    for xarg in argv:
        if xarg.startswith("-"): # and xarg.find("=") != -1:
            split = xarg.split('=')
            arg_id = split[0][1:].lower()
            if len(split) < 2:
                arg_value = ""
            else:
                arg_value = split[1]
            args[arg_id] = arg_value
else:
    pass
    #print "GENERAL COMMAND-LINE ARGUMENTS"
    #print_generic_usage()
    
# Some arguments (like dates) might need special processing applied...

# input_file (load arguments from .args file)
if is_arg('input_file'):
    input_filename = get_arg('input_file')
    print("Using arguments from args file: '{0}'".format(input_filename))
    set_args_from_file(input_filename)
    
# Project folder
if is_arg('project'):
    project_folder = join(data_folder, get_arg('project'))
    print("project folder:", project_folder)
          
# Start/End dates
start_date = datetime(1900, 1, 1, 0, 0, 0)
end_date = datetime(2100, 1, 1, 0, 0, 0)
if is_arg('start_date'):
    start_date = get_date_from_yyyymmdd(get_arg('start_date'))
if is_arg('end_date'):
    end_date = get_date_from_yyyymmdd(get_arg('end_date'))
set_args({'start_date': start_date, 'end_date': end_date})

# Start/End years
start_year = 1900
end_year = 2100
if is_arg('start_year'):
    start_year = int(get_arg('start_year'))
if is_arg('end_year'):
    end_year = int(get_arg('end_year'))
set_args({'start_year': start_year, 'end_year': end_year})
