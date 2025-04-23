#!/usr/bin/env python3

# ########################################################################
# Copyright (C) 2021-2022 Advanced Micro Devices, Inc. All rights Reserved.
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



import rocsparse_bench_gnuplot_helper

#
# EXPORT TO PDF WITH GNUPLOT
# arg plot: "all", "gflops", "time", "bandwidth"
#
#
def export_gnuplot(plot, obasename,xargs, yargs, results,verbose = False,debug = False,linear=False):

    datafile = open(obasename + ".dat", "w+")
    len_xargs = len(xargs)
    for iplot in range(len(yargs)):
        for ixarg  in range(len_xargs):
            isample = iplot * len_xargs + ixarg
            tg = results[isample]["timing"]

            min_allowed_yvalue = 0.0
            if not linear:
                min_allowed_yvalue = 1.0

            time0 = max(float(tg["time"][0]), min_allowed_yvalue)
            time1 = max(float(tg["time"][1]), min_allowed_yvalue)
            time2 = max(float(tg["time"][2]), min_allowed_yvalue)
            flops0 = max(float(tg["flops"][0]), min_allowed_yvalue)
            flops1 = max(float(tg["flops"][1]), min_allowed_yvalue)
            flops2 = max(float(tg["flops"][2]), min_allowed_yvalue)
            bandwidth0 = max(float(tg["bandwidth"][0]), min_allowed_yvalue)
            bandwidth1 = max(float(tg["bandwidth"][1]), min_allowed_yvalue)
            bandwidth2 = max(float(tg["bandwidth"][2]), min_allowed_yvalue)

            datafile.write(os.path.basename(os.path.splitext(xargs[ixarg])[0]) + " " +
                           str(time0) + " " +
                           str(time1) + " " +
                           str(time2) + " " +
                           str(flops0) + " " +
                           str(flops1) + " " +
                           str(flops2) + " " +
                           str(bandwidth0) + " " +
                           str(bandwidth1) + " "+
                           str(bandwidth2) + "\n")
        datafile.write("\n")
        datafile.write("\n")
    datafile.close();

    if verbose:
        print('//rocsparse-bench-plot  -  write gnuplot file : \'' + obasename + '.gnuplot\'')

    cmdfile = open(obasename + ".gnuplot", "w+")
    # for each plot
    num_curves=len(yargs)
    filetype="pdf"
    filename_extension= "." + filetype
    if plot == "time":
        rocsparse_bench_gnuplot_helper.histogram(cmdfile,
                                                 obasename + filename_extension,
                                                 'Time',
                                                 range(num_curves),
                                                 obasename + ".dat",
                                                 [-0.5,len_xargs + 0.5],
                                                 "milliseconds",
                                                 2,3,4,
                                                 yargs,
                                                 linear)
    elif plot == "gflops":
        rocsparse_bench_gnuplot_helper.histogram(cmdfile,
                                                 obasename + filename_extension,
                                                 'Performance',
                                                 range(num_curves),
                                                 obasename + ".dat",
                                                 [-0.5,len_xargs + 0.5],
                                                 "GFlops",
                                                 5,6,7,
                                                 yargs,
                                                 linear)
    elif plot == "bandwidth":
        rocsparse_bench_gnuplot_helper.histogram(cmdfile,
                                                 obasename + filename_extension,
                                                 'Bandwidth',
                                                 range(num_curves),
                                                 obasename + ".dat",
                                                 [-0.5,len_xargs + 0.5],
                                                 "GBytes/s",
                                                 8,9,10,
                                                 yargs,
                                                 linear)
    elif plot == "all":
        rocsparse_bench_gnuplot_helper.histogram(cmdfile,
                                                 obasename + "_msec"+ filename_extension,
                                                 'Time',
                                                 range(num_curves),
                                                 obasename + ".dat",
                                                 [-0.5,len_xargs + 0.5],
                                                 "milliseconds",
                                                 2,3,4,
                                                 yargs,
                                                 linear)


        rocsparse_bench_gnuplot_helper.histogram(cmdfile,
                                                 obasename + "_gflops"+ filename_extension,
                                                 'Performance',
                                                 range(num_curves),
                                                 obasename + ".dat",
                                                 [-0.5,len_xargs + 0.5],
                                                 "GFlops",
                                                 5,6,7,
                                                 yargs,
                                                 linear)

        rocsparse_bench_gnuplot_helper.histogram(cmdfile,
                                                 obasename + "_bandwidth"+ filename_extension,
                                                 'Bandwidth',
                                                 range(num_curves),
                                                 obasename + ".dat",
                                                 [-0.5,len_xargs + 0.5],
                                                 "GBytes/s",
                                                 8,9,10,
                                                 yargs,
                                                 linear)
    else:
        print("//rocsparse-bench-plot::error invalid plot keyword '"+plot+"', must be 'all' (default), 'time', 'gflops' or 'bandwidth' ")
        exit(1)
    cmdfile.close();

    rocsparse_bench_gnuplot_helper.call(obasename + ".gnuplot")
    if verbose:
        print('//rocsparse-bench-plot CLEANING')

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
    parser.add_argument('-l', '--linear',         required=False, default = False, action = "store_true")
    user_args, unknown_args = parser.parse_known_args()
    verbose=user_args.verbose
    debug=user_args.debug
    obasename = user_args.obasename
    plot = user_args.plot
    linear = user_args.linear

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
        print('//rocsparse-bench-plot')
        print('//rocsparse-bench-plot  - file : \'' + unknown_args[0] + '\'')

    export_gnuplot(plot, obasename, xargs,yargs, results, verbose, debug, linear)

if __name__ == "__main__":
    main()

