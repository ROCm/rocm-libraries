#!/usr/bin/env python3

# ########################################################################
# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights Reserved.
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
import os
import json
import xml.etree.ElementTree as ET
import rocsparse_bench_gnuplot_helper

def multiplot(out,ofilename,title,curve_indeces,plot_indeces,curves_per_plot,ifilename,y_range,y_label,col_index,curve_titles,plot_titles,xtic_label,linear=False):
    ncurves = len(curve_indeces)
    nplots = len(plot_indeces)

    if (ncurves == 0) or (nplots == 0):
        return

    xtic_label = xtic_label if len(xtic_label) != 0 else '1'

    out.write("reset\n")
    out.write("set term pdfcairo enhanced color font 'Helvetica,5'\n")
    out.write("set output \"" + ofilename + "\"\n")
    out.write("set termoption noenhanced\n")

    out.write("set ylabel \"" + y_label + "\"\n")

    cols = min(nplots, 3)
    rows = min(int((nplots - 1) / cols + 1), 2)

    plots_per_page = cols*rows
    pages = int((nplots - 1) / (plots_per_page) + 1)

    for p in range(pages):
        out.write("set multiplot title \"" + title + "\" font 'Helvetica,9' layout " + str(rows) + "," + str(cols) + " scale 1,1\n")

        if not linear:
            out.write("set logscale y\n")

        for i in range(plots_per_page):
            plot_index = p * plots_per_page + i

            if plot_index >= nplots:
                break

            if(plot_index % cols == 0):
                out.write("set ylabel \"" + y_label + "\"\n")
            else:
                out.write("set ylabel \" \"\n")

            out.write("set yrange [" + str(y_range[0]) + ":" + str(y_range[1]) + "]\n")
            out.write("set xtics rotate by -45\n")

            out.write("set title \"" + plot_titles[plot_index] + "\"\n")

            index = plot_index * curves_per_plot + curve_indeces[0]
            out.write("plot '"+ifilename+"' index "+str(index)+" using 1:"+str(col_index)+":xtic("+xtic_label+") with lines title '"+ curve_titles[0] +"'")
            for j in range(1,ncurves):
                index = plot_index * curves_per_plot + curve_indeces[j]
                out.write(",\\\n '' index "+str(index)+" using 1:"+str(col_index) + ":xtic("+xtic_label+") with lines title '"+curve_titles[j]+"'")

            out.write("\n")
        out.write("unset multiplot\n")

def export_gnuplot(obasename,xargs, yargs, case_results,case_titles,xtic_label = '',verbose = False,debug = False,linear=False):
    num_cases = len(case_results)
    datafile = open(obasename + ".dat", "w+")
    len_xargs = len(xargs)

    min_allowed_yvalue = 0.0
    max_time = float(case_results[0][0]["timing"]["time"][2])
    max_flops = float(case_results[0][0]["timing"]["flops"][2])
    max_bandwidth = float(case_results[0][0]["timing"]["bandwidth"][2])
    min_time = float(case_results[0][0]["timing"]["time"][0])
    min_flops = float(case_results[0][0]["timing"]["flops"][0])
    min_bandwidth = float(case_results[0][0]["timing"]["bandwidth"][0])

    for iplot in range(len(yargs)):
        for case_index in range(num_cases):
            samples = case_results[case_index]
            for ixarg  in range(len_xargs):
                isample = iplot * len_xargs + ixarg
                tg = samples[isample]["timing"]

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

                max_time = max(max_time, time2)
                max_flops = max(max_flops, flops2)
                max_bandwidth = max(max_bandwidth, bandwidth2)
                min_time = min(min_time, time0)
                min_flops = min(min_flops, flops0)
                min_bandwidth = min(min_bandwidth, bandwidth0)

                datafile.write(os.path.basename(os.path.splitext(xargs[ixarg])[0]) + " " +
                               str(time0) + " " +
                               str(time1) + " " +
                               str(time2) + " " +
                               str(flops0) + " " +
                               str(flops1) + " " +
                               str(flops2) + " " +
                               str(bandwidth0) + " " +
                               str(bandwidth1) + " "+
                               str(bandwidth2) + " " +
                               "\n")
            datafile.write("\n")
            datafile.write("\n")
    datafile.close();

    # pad y-axis
    pad_ratio = 0.05
    min_time = (1 + pad_ratio) * min_time - pad_ratio * max_time
    min_flops = (1 + pad_ratio) * min_flops - pad_ratio * max_flops
    min_bandwidth = (1 + pad_ratio) * min_bandwidth - pad_ratio * max_bandwidth

    max_time = (1 + pad_ratio) * max_time - pad_ratio * min_time
    max_flops = (1 + pad_ratio) * max_flops - pad_ratio * min_flops
    max_bandwidth = (1 + pad_ratio) * max_bandwidth - pad_ratio * min_bandwidth

    # fix y-axis for log scale
    if not linear:
        min_time = max(min_time, 1)
        min_flops = max(min_flops, 1)
        min_bandwidth = max(min_bandwidth, 1)

    if verbose:
        print('//rocsparse-bench-multiplot  -  write gnuplot file : \'' + obasename + '.gnuplot\'')

    cmdfile = open(obasename + ".gnuplot", "w+")
    # for each plot
    num_plots=len(yargs)
    filetype="pdf"
    filename_extension= "." + filetype

    multiplot(cmdfile,
              obasename + "_msec"+ filename_extension,
              'Time',
              range(num_cases),
              range(num_plots),
              num_cases,
              obasename + ".dat",
              [min_time, max_time],
              "milliseconds",
              2,
              case_titles,
              yargs, xtic_label, linear)
    multiplot(cmdfile,
              obasename + "_gflops"+ filename_extension,
              'Performance',
              range(num_cases),
              range(num_plots),
              num_cases,
              obasename + ".dat",
              [min_flops, max_flops],
              "GFlops",
              5,
              case_titles,
              yargs, xtic_label, linear)
    multiplot(cmdfile,
              obasename + "_bandwidth"+ filename_extension,
              'Bandwidth',
              range(num_cases),
              range(num_plots),
              num_cases,
              obasename + ".dat",
              [min_bandwidth, max_bandwidth],
              "GBytes/s",
              8,
              case_titles,
              yargs, xtic_label, linear)
    cmdfile.close();

    rocsparse_bench_gnuplot_helper.call(obasename + ".gnuplot")
    if verbose:
        print('//rocsparse-bench-multiplot CLEANING')

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
    parser.add_argument('-c', '--casenames',    required=False, default = "", help='name of curves separated by \';\'. e.g., case1;case2')
    parser.add_argument('-y', '--yargnames',    required=False, default = "", help='plot titles separated by \';\'. e.g., plot1;plot2')
    parser.add_argument('-t', '--xticlabel',    required=False, default = "", help='formula for xtic()')
    parser.add_argument('-v', '--verbose',         required=False, default = False, action = "store_true")
    parser.add_argument('-d', '--debug',         required=False, default = False, action = "store_true")
    parser.add_argument('-l', '--linear',         required=False, default = False, action = "store_true")
    user_args, case_names = parser.parse_known_args()

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

    cmd = [case['cmdline'] for case in cases]
    xargs = [case['xargs'] for case in cases]
    yargs = [case['yargs'] for case in cases]
    case_results = [case['results'] for case in cases]
    num_samples = len(case_results[0])

    case_names = user_args.casenames.split(";") if len(user_args.casenames) != 0 else case_titles
    yarg_names = user_args.yargnames.split(";") if len(user_args.yargnames) != 0 else yargs[0]

    if verbose:
        print('//rocsparse-bench-multiplot INPUT CASES')
        for case_index in range(num_cases):
            print('//rocsparse-bench-multiplot  - case'+str(case_index) +'      : \'' + case_names[case_index] + '\'')
        print('//rocsparse-bench-multiplot CHECKING')

####
    for case_index in range(1,num_cases):
        if xargs[0] != xargs[case_index]:
            print('xargs\'s must be equal, xargs from case \''+case_names[case_index]+'\' is not equal to xargs from case \''+case_names[0]+'\'')
            exit(1)

    if verbose:
        print('//rocsparse-bench-multiplot  -  xargs checked.')
####
    for case_index in range(1,num_cases):
        if yargs[0] != yargs[case_index]:
            print('yargs\'s must be equal, yargs from case \''+case_names[case_index]+'\' is not equal to yargs from case \''+case_names[0]+'\'')
            exit(1)
    if verbose:
        print('//rocsparse-bench-multiplot  -  yargs checked.')

    if len(yargs[0]) != len(yarg_names):
        print('yargnames length does not equal yargs length')
        exit(1)
    if verbose:
        print('//rocsparse-bench-multiplot  -  yargsnames checked.')
####
    for case_index in range(1,num_cases):
        if num_samples != len(case_results[case_index]):
            print('num_samples\'s must be equal, num_samples from case \''+case_names[case_index]+'\' is not equal to num_samples from case \''+case_names[0]+'\'')
            exit(1)
    if verbose:
        print('//rocsparse-bench-multiplot  -  num samples checked.')

    if num_cases != len(case_names):
        print('num_cases does not equal case_names length')
        exit(1)
    if verbose:
        print('//rocsparse-bench-multiplot  -  casenames checked.')
####
    if verbose:
        print('//rocsparse-bench-multiplot  -  write data    file : \'' + obasename + '.dat\'')

    export_gnuplot(obasename,
                   xargs[0],
                   yarg_names,
                   case_results,
                   case_names,
                   user_args.xticlabel,
                   verbose,
                   debug,
                   linear)

if __name__ == "__main__":
    main()

