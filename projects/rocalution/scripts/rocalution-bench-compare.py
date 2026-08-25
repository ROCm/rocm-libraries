#!/usr/bin/env python3

# ########################################################################
# Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

import argparse
import subprocess
import os
import re # regexp package.
import sys
import tempfile
import json
import xml.etree.ElementTree as ET
import rocalution_bench_gnuplot_helper

def export_gnuplot(obasename,xargs, yargs, case_results,case_titles,verbose = False,debug = False):
    index_time_analyze_median = 2
    index_time_analyze_low = 3
    index_time_analyze_up = 4
    index_time_analyze_median_ratio = 5

    index_time_solve_median = 6
    index_time_solve_low = 7
    index_time_solve_up = 8
    index_time_solve_median_ratio = 9

    index_iter_median = 10
    index_iter_low = 11
    index_iter_up = 12
    index_iter_median_ratio = 13

    index_nrm_residual_median = 14
    index_nrm_residual_low = 15
    index_nrm_residual_up = 16
    index_nrm_residual_median_ratio = 17

    num_cases = len(case_results)
    datafile = open(obasename + ".dat", "w+")
    len_xargs = len(xargs)
    for iplot in range(len(yargs)):
        for case_index in range(num_cases):
            samples = case_results[case_index]
            for ixarg  in range(len_xargs):
                isample = iplot * len_xargs + ixarg
                tg = samples[isample]["timing"]
                tg0 = case_results[0][isample]["timing"]
                datafile.write(os.path.basename(os.path.splitext(xargs[ixarg])[0]) + " " +
                               #
                               tg["median"]["time_analyze"] + " " +
                               tg["low"]["time_analyze"] + " " +
                               tg["up"]["time_analyze"] + " " +
                               str(float(tg0["median"]["time_analyze"]) / float(tg["median"]["time_analyze"]))  + " " +
                               #
                               tg["median"]["time_solve"] + " " +
                               tg["low"]["time_solve"] + " " +
                               tg["up"]["time_solve"] + " " +
                               str(float(tg0["median"]["time_solve"]) / float(tg["median"]["time_solve"]))  + " " +
                               #
                               tg["median"]["iter"] + " " +
                               tg["low"]["iter"] + " " +
                               tg["up"]["iter"] + " " +
                               str(float(tg0["median"]["iter"]) / float(tg["median"]["iter"]))  + " " +
                               #
                               tg["median"]["norm_residual"] + " " +
                               tg["low"]["norm_residual"] + " " +
                               tg["up"]["norm_residual"] + " " +
                               str(float(tg0["median"]["norm_residual"]) / float(tg["median"]["norm_residual"]))  + " " +
                               #
                               "\n")
            datafile.write("\n")
            datafile.write("\n")
    datafile.close();

    if verbose:
        print('//rocalution-bench-compare  -  write gnuplot file : \'' + obasename + '.gnuplot\'')

    cmdfile = open(obasename + ".gnuplot", "w+")
    # for each plot
    num_plots=len(yargs)
    for iplot in range(len(yargs)):
        fyarg = yargs[iplot]
        fyarg = fyarg.replace("=","")
        fyarg = fyarg.replace(",","_")
        if num_plots==1:
            filename_extension= ".pdf"
        else:
            filename_extension= "."+fyarg+".pdf"
        #
        # Reminder, files is what we want to compare.
        #
        plot_index=iplot * num_cases

        rocalution_bench_gnuplot_helper.simple_histogram(cmdfile,
                                                         obasename + "_msec_ratio"+ filename_extension,
                                                         'Time ratio',
                                                         range(plot_index,plot_index + num_cases),
                                                         obasename + ".dat",
                                                         [-0.5,len_xargs + 0.5],
                                                         "",
                                                         index_time_solve_median_ratio,
                                                         case_titles)

        rocalution_bench_gnuplot_helper.histogram(cmdfile,
                                                 obasename + "_msec"+ filename_extension,
                                                 'Time',
                                                 range(plot_index,plot_index + num_cases),
                                                 obasename + ".dat",
                                                 [-0.5,len_xargs + 0.5],
                                                 "milliseconds",
                                                  index_time_solve_median,
                                                  index_time_solve_low,
                                                  index_time_solve_up,
                                                  case_titles,True)

        rocalution_bench_gnuplot_helper.simple_histogram(cmdfile,
                                                         obasename + "_iter_ratio"+ filename_extension,
                                                         'Iteration ratio',
                                                         range(plot_index,plot_index + num_cases),
                                                         obasename + ".dat",
                                                         [-0.5,len_xargs + 0.5],
                                                         "",
                                                         index_iter_median_ratio,
                                                         case_titles)

        rocalution_bench_gnuplot_helper.histogram(cmdfile,
                                                 obasename + "_iter"+ filename_extension,
                                                 'Iterations',
                                                 range(plot_index,plot_index + num_cases),
                                                 obasename + ".dat",
                                                 [-0.5,len_xargs + 0.5],
                                                 "iterations",
                                                  index_iter_median,
                                                  index_iter_low,
                                                  index_iter_up,
                                                  case_titles,False)


    cmdfile.close();

    rocalution_bench_gnuplot_helper.call(obasename + ".gnuplot")
    if verbose:
        print('//rocalution-bench-compare CLEANING')

    if not debug:
        os.remove(obasename + '.dat')
        os.remove(obasename + '.gnuplot')


#
#
# MAIN
#
#
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--obasename',    required=False, default = 'a')
    parser.add_argument('-v', '--verbose',         required=False, default = False, action = "store_true")
    parser.add_argument('-d', '--debug',         required=False, default = False, action = "store_true")
    user_args, case_names = parser.parse_known_args()
    if len(case_names) < 2:
        print('//rocalution-bench-compare.error number of filenames provided is < 2, (num_cases = '+str(len(case_names))+')')
        exit(1)

    verbose=user_args.verbose
    debug=user_args.debug
    obasename = user_args.obasename

    cases = []
    num_cases = len(case_names)

    case_titles = []
    for case_index in range(num_cases):
        case_titles.append(os.path.basename(os.path.splitext(case_names[case_index])[0]))

    for case_index in range(num_cases):
        with open(case_names[case_index],"r") as f:
            cases.append(json.load(f))

    cmd = [case['cmdline'] for case in cases]
    xargs = [case['xargs'] for case in cases]
    yargs = [case['yargs'] for case in cases]
    case_results = [case['results'] for case in cases]
    num_samples = len(case_results[0])
    len_xargs = len(xargs[0])

    if verbose:
        print('//rocalution-bench-compare INPUT CASES')
        for case_index in range(num_cases):
            print('//rocalution-bench-compare  - case'+str(case_index) +'      : \'' + case_names[case_index] + '\'')
        print('//rocalution-bench-compare CHECKING')

####
    for case_index in range(1,num_cases):
        if xargs[0] != xargs[case_index]:
            print('xargs\'s must be equal, xargs from case \''+case_names[case_index]+'\' is not equal to xargs from case \''+case_names[0]+'\'')
            exit(1)

    if verbose:
        print('//rocalution-bench-compare  -  xargs checked.')
####
    for case_index in range(1,num_cases):
        if yargs[0] != yargs[case_index]:
            print('yargs\'s must be equal, yargs from case \''+case_names[case_index]+'\' is not equal to yargs from case \''+case_names[0]+'\'')
            exit(1)
    if verbose:
        print('//rocalution-bench-compare  -  yargs checked.')
####
    for case_index in range(1,num_cases):
        if num_samples != len(case_results[case_index]):
            print('num_samples\'s must be equal, num_samples from case \''+case_names[case_index]+'\' is not equal to num_samples from case \''+case_names[0]+'\'')
            exit(1)
    if verbose:
        print('//rocalution-bench-compare  -  num samples checked.')
####
    if verbose:
        print('//rocalution-bench-compare  -  write data    file : \'' + obasename + '.dat\'')

    export_gnuplot(obasename,
                   xargs[0],
                   yargs[0],
                   case_results,
                   case_titles,
                   verbose,
                   debug)

if __name__ == "__main__":
    main()

