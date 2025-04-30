#!/usr/bin/env python3

# ########################################################################
# Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights Reserved.
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
import rocsparse_bench_gnuplot_helper

def export_gnuplot(obasename,xargs, yargs, case_results,case_titles,verbose = False,debug = False,linear=False):
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

                time_ratio = 0.0
                flops_ratio = 0.0
                bandwidth_ratio = 0.0

                if float(tg["time"][0]) != 0:
                    time_ratio = float(tg0["time"][0]) / float(tg["time"][0])

                if float(tg["flops"][0]) != 0:
                    flops_ratio = float(tg0["flops"][0]) / float(tg["flops"][0])

                if float(tg["bandwidth"][0]) != 0:
                    bandwidth_ratio = float(tg0["bandwidth"][0]) / float(tg["bandwidth"][0])

                datafile.write(os.path.basename(os.path.splitext(xargs[ixarg])[0]) + " " +
                               str(time0) + " " +
                               str(time1) + " " +
                               str(time2) + " " +
                               str(time_ratio)  + " " +
                               str(flops0) + " " +
                               str(flops1) + " " +
                               str(flops2) + " " +
                               str(flops_ratio)  + " " +
                               str(bandwidth0) + " " +
                               str(bandwidth1) + " "+
                               str(bandwidth2) + " " +
                               str(bandwidth_ratio)  + " " +
                               "\n")
            datafile.write("\n")
            datafile.write("\n")
    datafile.close();

    if verbose:
        print('//rocsparse-bench-compare  -  write gnuplot file : \'' + obasename + '.gnuplot\'')

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

#        rocsparse_bench_gnuplot_helper.curve(cmdfile,
#                                             obasename + "_msec"+ filename_extension,
#                                             'Time',
#                                             range(plot_index,plot_index + num_cases),
#                                             obasename + ".dat",
#                                             [-0.5,len_xargs + 0.5],
#                                             "milliseconds",
#                                             2,
#                                             case_titles)


        rocsparse_bench_gnuplot_helper.simple_histogram(cmdfile,
                                                 obasename + "_msec_ratio"+ filename_extension,
                                                 'Time ratio',
                                                 range(plot_index,plot_index + num_cases),
                                                 obasename + ".dat",
                                                 [-0.5,len_xargs + 0.5],
                                                 "",
                                                 5,
                                                 case_titles)

        rocsparse_bench_gnuplot_helper.simple_histogram(cmdfile,
                                                 obasename + "_gflops_ratio"+ filename_extension,
                                                 'Performance ratio',
                                                 range(plot_index,plot_index + num_cases),
                                                 obasename + ".dat",
                                                 [-0.5,len_xargs + 0.5],
                                                 "",
                                                 9,
                                                 case_titles)

        rocsparse_bench_gnuplot_helper.simple_histogram(cmdfile,
                                                 obasename + "_bandwitdh_ratio"+ filename_extension,
                                                 'Bandwidth ratio',
                                                 range(plot_index,plot_index + num_cases),
                                                 obasename + ".dat",
                                                 [-0.5,len_xargs + 0.5],
                                                 "",
                                                 13,
                                                 case_titles)

        rocsparse_bench_gnuplot_helper.histogram(cmdfile,
                                                 obasename + "_msec"+ filename_extension,
                                                 'Time',
                                                 range(plot_index,plot_index + num_cases),
                                                 obasename + ".dat",
                                                 [-0.5,len_xargs + 0.5],
                                                 "milliseconds",
                                                 2,3,4,
                                                 case_titles,
                                                 linear)

        rocsparse_bench_gnuplot_helper.histogram(cmdfile,
                                                 obasename + "_gflops"+ filename_extension,
                                                 'Performance',
                                                 range(plot_index,plot_index + num_cases),
                                                 obasename + ".dat",
                                                 [-0.5,len_xargs + 0.5],
                                                 "GFlops",
                                                 6,7,8,
                                                 case_titles,
                                                 linear)

        rocsparse_bench_gnuplot_helper.histogram(cmdfile,
                                                 obasename + "_bandwitdh"+ filename_extension,
                                                 'Bandwidth',
                                                 range(plot_index,plot_index + num_cases),
                                                 obasename + ".dat",
                                                 [-0.5,len_xargs + 0.5],
                                                 "GBytes/s",
                                                 10,11,12,
                                                 case_titles,
                                                 linear)
    cmdfile.close();

    rocsparse_bench_gnuplot_helper.call(obasename + ".gnuplot")
    if verbose:
        print('//rocsparse-bench-compare CLEANING')

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
    parser.add_argument('-l', '--linear',         required=False, default = False, action = "store_true")
    user_args, case_names = parser.parse_known_args()
    if len(case_names) < 2:
        print('//rocsparse-bench-compare.error number of filenames provided is < 2, (num_cases = '+str(len(case_names))+')')
        exit(1)

    verbose=user_args.verbose
    debug=user_args.debug
    obasename = user_args.obasename
    linear = user_args.linear

    cases = []
    num_cases = len(case_names)

    case_titles = []
    for case_index in range(num_cases):
        case_titles.append(os.path.basename(os.path.splitext(case_names[case_index])[0]))

    for case_index in range(num_cases):
        with open(case_names[case_index],"r") as f:
            cases.append(json.load(f))


#    mytree = ET.parse('rocsparse-bench-csrmv.xml')
#    myroot = mytree.getroot()
#    print(len(myroot))
#    for i in range(len(myroot)):
#        for j in range(len(myroot[i])):
#            print(myroot[i][j].attrib['cmd'])
#            proc=subprocess.Popen(['bash', '-c', myroot[i][j].attrib['cmd']])
#            proc.wait()
#            rc = proc.returncode
#            if rc != 0:
#                print('//rocsparse-bench-compare.error running cmd')
#                exit(1)
#    return

    cmd = [case['cmdline'] for case in cases]
    xargs = [case['xargs'] for case in cases]
    yargs = [case['yargs'] for case in cases]
    case_results = [case['results'] for case in cases]
    num_samples = len(case_results[0])
    len_xargs = len(xargs[0])

    if verbose:
        print('//rocsparse-bench-compare INPUT CASES')
        for case_index in range(num_cases):
            print('//rocsparse-bench-compare  - case'+str(case_index) +'      : \'' + case_names[case_index] + '\'')
        print('//rocsparse-bench-compare CHECKING')

####
#    for i in range(1,num_cases):
#        if cmd[0] != cmd[i]:
#            print('cmdlines must be equal, cmdline from file \''+case_names[i]+'\' is not equal to cmdline from file \''+case_names[0]+'\'')
#            exit(1)

#    if verbose:
#        print('//rocsparse-bench-compare  -  cmdlines checked.')

####
    for case_index in range(1,num_cases):
        if xargs[0] != xargs[case_index]:
            print('xargs\'s must be equal, xargs from case \''+case_names[case_index]+'\' is not equal to xargs from case \''+case_names[0]+'\'')
            exit(1)

    if verbose:
        print('//rocsparse-bench-compare  -  xargs checked.')
####
    for case_index in range(1,num_cases):
        if yargs[0] != yargs[case_index]:
            print('yargs\'s must be equal, yargs from case \''+case_names[case_index]+'\' is not equal to yargs from case \''+case_names[0]+'\'')
            exit(1)
    if verbose:
        print('//rocsparse-bench-compare  -  yargs checked.')
####
    for case_index in range(1,num_cases):
        if num_samples != len(case_results[case_index]):
            print('num_samples\'s must be equal, num_samples from case \''+case_names[case_index]+'\' is not equal to num_samples from case \''+case_names[0]+'\'')
            exit(1)
    if verbose:
        print('//rocsparse-bench-compare  -  num samples checked.')
####
    if verbose:
        print('//rocsparse-bench-compare  -  write data    file : \'' + obasename + '.dat\'')

    export_gnuplot(obasename,
                   xargs[0],
                   yargs[0],
                   case_results,
                   case_titles,
                   verbose,
                   debug,
                   linear)

if __name__ == "__main__":
    main()

