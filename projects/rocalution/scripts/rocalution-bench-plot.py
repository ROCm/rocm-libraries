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
import re # regexp package
import sys
import tempfile
import json



import rocalution_bench_gnuplot_helper

#
# EXPORT TO PDF WITH GNUPLOT
# arg plot: "all", "gflops", "time", "bandwidth"
#
#
def export_gnuplot(plot, obasename,xargs, yargs, results,verbose = False,debug = False):

    index_time_analyze_median = 2
    index_time_analyze_low = 3
    index_time_analyze_up = 4

    index_time_solve_median = 5
    index_time_solve_low = 6
    index_time_solve_up = 7

    index_iter_median = 8
    index_iter_low = 9
    index_iter_up = 10

    index_norm_residual_median = 11
    index_norm_residual_low = 12
    index_norm_residual_up = 13

    datafile = open(obasename + ".dat", "w+")
    len_xargs = len(xargs)
    for iplot in range(len(yargs)):
        for ixarg  in range(len_xargs):
            isample = iplot * len_xargs + ixarg
            tg = results[isample]["timing"]
            datafile.write(os.path.basename(os.path.splitext(xargs[ixarg])[0]) + " " +
                           tg["median"]["time_analyze"] + " " +
                           tg["low"]["time_analyze"] + " " +
                           tg["up"]["time_analyze"] + " " +
                           tg["median"]["time_solve"] + " " +
                           tg["low"]["time_solve"] + " " +
                           tg["up"]["time_solve"] + " " +
                           tg["median"]["iter"] + " " +
                           tg["low"]["iter"] + " " +
                           tg["up"]["iter"] + " " +
                           tg["median"]["norm_residual"] + " " +
                           tg["low"]["norm_residual"] + " " +
                           tg["up"]["norm_residual"] + " " +
                           "\n")
        datafile.write("\n")
        datafile.write("\n")
    datafile.close();

    if verbose:
        print('//rocalution-bench-plot  -  write gnuplot file : \'' + obasename + '.gnuplot\'')

    cmdfile = open(obasename + ".gnuplot", "w+")
    # for each plot
    num_curves=len(yargs)
    filetype="pdf"
    filename_extension= "." + filetype
    if plot == "time":
        rocalution_bench_gnuplot_helper.histogram(cmdfile,
                                                 obasename + filename_extension,
                                                 'Time solve',
                                                 range(num_curves),
                                                 obasename + ".dat",
                                                 [-0.5,len_xargs + 0.5],
                                                  "milliseconds",
                                                  index_time_solve_median,
                                                  index_time_solve_low,
                                                  index_time_solve_up,
                                                  yargs,True)
    else if plot == "norm_residual":
        rocalution_bench_gnuplot_helper.histogram(cmdfile,
                                                 obasename + filename_extension,
                                                 'Residual',
                                                 range(num_curves),
                                                 obasename + ".dat",
                                                 [-0.5,len_xargs + 0.5],
                                                  "norm residual",
                                                  index_norm_residual_median,
                                                  index_norm_residual_low,
                                                  index_norm_residual_up,
                                                  yargs,True)
    elif plot == "iter":
        rocalution_bench_gnuplot_helper.histogram(cmdfile,
                                                 obasename + filename_extension,
                                                 'Iterations',
                                                 range(num_curves),
                                                 obasename + ".dat",
                                                 [-0.5,len_xargs + 0.5],
                                                 "iterations",
                                                  index_iter_median,
                                                  index_iter_low,
                                                  index_iter_up,
                                                  yargs, False)
    elif plot == "all":
        rocalution_bench_gnuplot_helper.histogram(cmdfile,
                                                 obasename + "_msec"+ filename_extension,
                                                 'Time solve',
                                                 range(num_curves),
                                                 obasename + ".dat",
                                                 [-0.5,len_xargs + 0.5],
                                                  "milliseconds",
                                                  index_time_solve_median,
                                                  index_time_solve_low,
                                                  index_time_solve_up,
                                                  yargs, True)
        rocalution_bench_gnuplot_helper.histogram(cmdfile,
                                                 obasename + "_iter" + filename_extension,
                                                 'Iterations',
                                                 range(num_curves),
                                                 obasename + ".dat",
                                                 [-0.5,len_xargs + 0.5],
                                                 "iterations",
                                                  index_iter_median,
                                                  index_iter_low,
                                                  index_iter_up,
                                                  yargs, False)
        rocalution_bench_gnuplot_helper.histogram(cmdfile,
                                                 obasename + "_residual" + filename_extension,
                                                 'Residual',
                                                 range(num_curves),
                                                 obasename + ".dat",
                                                 [-0.5,len_xargs + 0.5],
                                                  "norm residual",
                                                  index_norm_residual_median,
                                                  index_norm_residual_low,
                                                  index_norm_residual_up,
                                                  yargs,True)
    else:
        print("//rocalution-bench-plot::error invalid plot keyword '"+plot+"', must be 'all' (default), 'time', 'iter' or 'norm_residual' ")
        exit(1)
    cmdfile.close();

    rocalution_bench_gnuplot_helper.call(obasename + ".gnuplot")
    if verbose:
        print('//rocalution-bench-plot CLEANING')

    if not debug:
        os.remove(obasename + '.dat')
        os.remove(obasename + '.gnuplot')




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--workingdir',     required=False, default = './')
    parser.add_argument('-o', '--obasename',    required=False, default = 'a')
    parser.add_argument('-p', '--plot',    required=False, default = 'all')
    parser.add_argument('-v', '--verbose',         required=False, default = False, action = "store_true")
    parser.add_argument('-d', '--debug',         required=False, default = False, action = "store_true")
    user_args, unknown_args = parser.parse_known_args()
    verbose=user_args.verbose
    debug=user_args.debug
    obasename = user_args.obasename
    plot = user_args.plot
    if len(unknown_args) > 1:
        print('expecting only one input file.')
    with open(unknown_args[0],"r") as f:
        case=json.load(f)

    cmd = case['cmdline']
    xargs = case['xargs']
    yargs = case['yargs']
    results = case['results']
    num_samples = len(results)
    len_xargs = len(xargs)

    if verbose:
        print('//rocalution-bench-plot')
        print('//rocalution-bench-plot  - file : \'' + unknown_args[0] + '\'')

    export_gnuplot(plot, obasename, xargs,yargs, results, verbose,debug)

if __name__ == "__main__":
    main()

